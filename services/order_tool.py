"""
The action layer: turns a confirmed customer intent into a real
WooCommerce order.

Deliberately two tools, not one -- this split is load-bearing, not
stylistic:

- propose_order() is a pure, deterministic lookup + arithmetic, exactly
  like generate_quote() in quote_service.py. It never talks to
  WooCommerce. It builds a priced summary and stores it as this
  session's one pending order.
- confirm_order() is the only function in this file that writes
  anything. It takes no product/price/address arguments of its own --
  it only ever acts on whatever propose_order() already stored for this
  session. That means there is no path from "the model parsed one
  ambiguous message" straight to "a real order got created": the LLM
  can call propose_order freely (a wrong guess there just shows the
  customer the wrong proposal, which self-corrects like every other
  read-only tool), but confirm_order only does something once a
  customer has explicitly said yes to a specific, already-priced
  proposal it can see.

This mirrors the LLM-decides / deterministic-code-executes split the
rest of this codebase already uses (see tool_registry.py's module
docstring and services/tool_executor.py) -- it adds a second gate,
session-scoped confirmation, on top of it, because this is the first
tool with a real-world, hard-to-undo side effect the five read-only
tools don't have.

Price vs payment, on purpose: this file creates an order in WooCommerce
with status "on-hold" (WooCommerce's own "awaiting payment" status), not
"processing" or "completed". Nothing here collects or verifies payment
-- that's a genuinely separate integration (a mobile money rail, most
likely, for this store's customers) that doesn't exist yet. Creating
the order is "write data KasaFlow already has to a system it already
talks to"; collecting payment is "wait for and verify a real-world event
outside this system", a different, harder problem, not implemented
here.

Idempotency, and what's actually handled: WooCommerce's REST API has no
built-in idempotency-key support. The "status == submitting" guard stops
a second confirm_order call from firing a second POST *while the first
is still in flight in the same process*. Separately, a timed-out create
(the POST may have reached WooCommerce and created the order, but the
response never reached us) is handled by _find_existing_order_by_token():
before treating a timeout as a failure, look the order up by the token
already embedded in its customer_note and meta_data, and use it if
found, rather than risk creating a duplicate on retry. That lookup
relies on WooCommerce's default order search covering customer_note
(stored internally as the order's post_excerpt) -- not yet verified
against a real store, since this environment has no live WooCommerce
credentials. Confirm it actually returns a match before relying on it
in production.

TODO(next payment milestone): this whole file still infers order state
from the synchronous HTTP response (or the fallback lookup above).
WooCommerce can push an `order.created`/`order.updated` webhook instead
-- a more correct, event-driven replacement for both the timeout lookup
here and the eventual "payment confirmed" signal once a mobile money
rail exists. Worth building as one piece when payment confirmation is
tackled, not before -- a webhook receiver is real, standing
infrastructure (a public HTTPS endpoint, signature verification), not
justified for idempotency alone while the fallback above covers it.
"""

import logging
import uuid

import requests

from app.config import settings
from services.delivery_tool import delivery_option_label, delivery_options_phrase, is_valid_delivery_option
from services.memory import get_session_store
from services.product_tool import get_product_price
from services.whatsapp_client import WhatsAppError, send_text_message

logger = logging.getLogger(__name__)

_store = get_session_store()

_PENDING_ORDER_KEY = "pending_order"
_LAST_CONFIRMED_KEY = "last_confirmed_order"


def propose_order(
    product_name: str,
    material: str,
    quantity,
    delivery_address: str,
    delivery_option: str,
    session_id: str,
) -> dict:
    """Price a specific order and hold it against this session, pending
    the customer's explicit confirmation. Never writes to WooCommerce.

    Deliberately doesn't add a delivery cost to the total -- see
    delivery_tool.py's module docstring: this business delivers by rider
    from Accra/Kumasi or ships internationally, and the real cost/timing
    for any of those isn't something this system can price on its own.
    total is product cost only; a human finalises delivery after
    confirm_order() hands the order off (see that function's staff
    notification)."""

    quantity_int = _parse_quantity(quantity)
    if quantity_int is None:
        return {"error": "How many would you like?"}

    address_stripped = str(delivery_address).strip() if delivery_address else ""
    if not address_stripped or address_stripped.lower() == "unknown":
        return {"error": "What address should this be delivered to?"}

    delivery_key = str(delivery_option).strip().lower() if delivery_option else ""
    if not is_valid_delivery_option(delivery_key):
        return {"error": f"Would you like {delivery_options_phrase()}?"}

    product = get_product_price(product_name, material)
    if not product:
        return {"error": "Sorry, we couldn't find that product."}

    if "id" not in product:
        # products.json was synced before this feature shipped -- see
        # woocommerce_sync.py's build_catalogue(). Re-sync the catalogue
        # before this product can be ordered, rather than letting this
        # fail later inside confirm_order() once a customer has already
        # said yes to a proposal that can never actually be placed.
        logger.error(
            "Product %r has no WooCommerce id -- re-sync the catalogue "
            "(python -m services.woocommerce_sync) before taking orders for it",
            product.get("product"),
        )
        return {"error": "Sorry, I can't take orders for that item right now."}

    subtotal = product["price"] * quantity_int

    proposal = {
        "token": str(uuid.uuid4()),
        "status": "pending",
        "product_id": product["id"],
        "variation_id": product.get("variation_id"),
        "product": product["product"],
        "material": product["material"],
        "unit_price": product["price"],
        "quantity": quantity_int,
        "subtotal": subtotal,
        "total": subtotal,
        "delivery_address": delivery_address,
        "delivery_option": delivery_key,
        "delivery_option_label": delivery_option_label(delivery_key),
    }

    _store.set(session_id, _PENDING_ORDER_KEY, proposal)
    return {"proposal": proposal}


def get_pending_order_summary(session_id: str) -> dict | None:
    """Read-only peek at this session's pending proposal, if any.

    Used by router.py to hand the LLM's tool-selection prompt (see
    llm.py's _pending_order_state_line()) an actual answer to "does this
    customer have anything to confirm right now", rather than leaving it
    to guess blind from a bare "yh"/"yeah" with no conversation history.
    Read-only and side-effect free -- never mutates session state, only
    the confirm_order flow above does that."""
    pending = _store.get(session_id, _PENDING_ORDER_KEY)
    if pending is None:
        return None
    return {
        "product": pending["product"],
        "material": pending["material"],
        "quantity": pending["quantity"],
        "total": pending["total"],
    }


def confirm_order(session_id: str) -> dict:
    """Create the real WooCommerce order for whatever propose_order()
    last stored against this session. Does nothing, and asks nothing,
    beyond the session id -- see module docstring for why."""

    pending = _store.get(session_id, _PENDING_ORDER_KEY)

    if pending is None:
        last = _store.get(session_id, _LAST_CONFIRMED_KEY)
        if last is not None:
            # The customer confirmed again after it already went through
            # (a duplicated WhatsApp webhook delivery is the realistic
            # cause). Resend the same confirmation rather than "nothing
            # to confirm", which would read as if their order vanished.
            return {"order_confirmation": last}
        # Deliberately open-ended, not "want a quote?" -- this also fires
        # when a customer says a bare "yh"/"yeah" in reply to something
        # that was never an order proposal at all (see llm.py's
        # confirm_order guidance: it has no visibility into what this
        # assistant last asked, only the customer's current message), so
        # assuming they meant a quote specifically would often be wrong.
        return {
            "error": "Hmm, I don't have anything pending to confirm right now -- "
            "were you looking to go ahead with one of the pieces we discussed?"
        }

    if pending.get("status") == "submitting":
        # A previous confirm_order call for this exact proposal is still
        # in flight. See module docstring for why this can't be a full
        # idempotency guarantee, only a same-process race guard.
        return {"error": "Still placing your order, one moment."}

    pending["status"] = "submitting"
    _store.set(session_id, _PENDING_ORDER_KEY, pending)

    try:
        order = _create_woocommerce_order(pending)
    except requests.exceptions.Timeout:
        # Genuinely ambiguous, unlike every other failure below: the POST
        # may have reached WooCommerce and created the order before the
        # response was lost on the way back to us. Check before deciding
        # anything, rather than assume failure (risking a customer-
        # triggered retry that double-creates the order) or assume
        # success (risking telling the customer "confirmed" for an order
        # that was never actually placed).
        logger.warning(
            "Order creation timed out for token=%s -- checking WooCommerce before retrying",
            pending.get("token"),
        )
        existing = _find_existing_order_by_token(pending["token"])
        if existing is not None:
            return _finalize_confirmation(session_id, pending, existing["id"])

        # Checked and found nothing -- genuinely unresolved, not just
        # unlucky timing. Safe to hand back to "pending" for a real retry.
        pending["status"] = "pending"
        _store.set(session_id, _PENDING_ORDER_KEY, pending)
        return {"error": "Something went wrong placing your order -- let's try that again."}
    except Exception:
        # Every other failure (connection refused, a 4xx/5xx WooCommerce
        # actually returned, malformed response) is NOT ambiguous -- the
        # request either never reached WooCommerce or WooCommerce clearly
        # rejected it, so there's nothing to look up before retrying.
        logger.exception("WooCommerce order creation failed for token=%s", pending.get("token"))
        pending["status"] = "pending"
        _store.set(session_id, _PENDING_ORDER_KEY, pending)
        return {"error": "Something went wrong placing your order -- let's try that again."}

    return _finalize_confirmation(session_id, pending, order["id"])


def _finalize_confirmation(session_id: str, pending: dict, order_id) -> dict:
    confirmation = {
        "order_id": order_id,
        "total": pending["total"],
        "delivery_address": pending["delivery_address"],
        "delivery_option_label": pending.get("delivery_option_label"),
    }
    _store.set(session_id, _LAST_CONFIRMED_KEY, confirmation)
    _store.set(session_id, _PENDING_ORDER_KEY, None)
    _notify_staff_of_new_order(session_id, pending, order_id)
    return {"order_confirmation": confirmation}


def _notify_staff_of_new_order(session_id: str, pending: dict, order_id) -> None:
    """Delivery isn't automated (see delivery_tool.py) -- a human has to
    actually see this order and its chosen delivery option to arrange
    the rider or shipping. Best-effort and non-fatal, on purpose: the
    order has already been created in WooCommerce by the time this
    runs, so a failed notification is a real problem worth logging
    loudly, but it must never turn into a failure the customer sees for
    an order that, from their side, genuinely went through."""
    if not settings.staff_notification_phone:
        logger.warning(
            "STAFF_NOTIFICATION_PHONE not configured -- order #%s was created but no "
            "rider coordinator was notified. Set it in .env so deliveries actually get "
            "arranged.",
            order_id,
        )
        return

    label = pending.get("delivery_option_label") or pending.get("delivery_option") or "not specified"
    message = (
        f"New KasaFlow order #{order_id} -- please arrange delivery.\n"
        f"{pending['quantity']} x {pending['material']} {pending['product']}\n"
        f"Delivery: {label}\n"
        f"Address: {pending['delivery_address']}\n"
        f"Customer WhatsApp: {session_id}\n"
        f"Product total: GH₵{pending['total']:,.2f} (delivery cost to be arranged separately)"
    )
    try:
        send_text_message(settings.staff_notification_phone, message)
    except WhatsAppError as e:
        logger.error("Failed to notify staff about order #%s: %s", order_id, e)


def _find_existing_order_by_token(token: str) -> dict | None:
    """Look up whether an order referencing this token already exists,
    for the one case where that's genuinely ambiguous: a create request
    that timed out on the way back to us (see confirm_order()). Relies
    on WooCommerce's default order search covering customer_note (stored
    internally as the order's post_excerpt) -- not yet verified against
    a real store, since this environment has no live WooCommerce
    credentials. Confirm this actually surfaces a match before relying
    on it in production; if it doesn't, filter the response client-side
    against meta_data (also written by _create_woocommerce_order())
    instead of the search param.
    """
    try:
        auth = (settings.woocommerce_orders_consumer_key, settings.woocommerce_orders_consumer_secret)
        response = requests.get(
            f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/orders",
            params={"search": token},
            auth=auth,
            timeout=15,
        )
        response.raise_for_status()
        matches = response.json()
    except Exception:
        # The lookup itself failing is not the same as "no order exists"
        # -- log it distinctly so a flaky lookup doesn't get read as
        # confirmation the order was never created.
        logger.exception("Lookup for existing order (token=%s) failed", token)
        return None

    return matches[0] if matches else None


def _parse_quantity(quantity) -> int | None:
    try:
        value = int(quantity)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _require_orders_config() -> None:
    missing = [
        name
        for name, value in [
            ("WOOCOMMERCE_URL", settings.woocommerce_url),
            ("WOOCOMMERCE_ORDERS_CONSUMER_KEY", settings.woocommerce_orders_consumer_key),
            ("WOOCOMMERCE_ORDERS_CONSUMER_SECRET", settings.woocommerce_orders_consumer_secret),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing WooCommerce order config: {', '.join(missing)}. Add them to .env."
        )


def _create_woocommerce_order(pending: dict) -> dict:
    _require_orders_config()

    line_item = {"product_id": pending["product_id"], "quantity": pending["quantity"]}
    if pending.get("variation_id"):
        line_item["variation_id"] = pending["variation_id"]

    delivery_label = pending.get("delivery_option_label") or pending.get("delivery_option") or "not specified"
    payload = {
        # "on-hold" = WooCommerce's own "awaiting payment" status. Never
        # "processing"/"completed" here -- see module docstring, nothing
        # in this file has collected payment yet.
        "status": "on-hold",
        "line_items": [line_item],
        "shipping": {"address_1": pending["delivery_address"]},
        # Delivery isn't priced/arranged automatically (see
        # delivery_tool.py) -- the chosen option is written onto the
        # order itself, not just sent as a one-off WhatsApp ping (see
        # _notify_staff_of_new_order()), so it's still visible to
        # whoever looks at the order in WooCommerce later, not only to
        # whoever happened to see the notification when it arrived.
        "customer_note": (
            f"Placed via KasaFlow WhatsApp assistant. Reference: {pending['token']}. "
            f"Delivery: {delivery_label}."
        ),
        # Structured, purpose-built lookup key -- _find_existing_order_by_token()
        # currently searches customer_note (more certain to work with
        # WooCommerce's default search), but this is the more robust key
        # to switch to once that's verified against a real store.
        "meta_data": [
            {"key": "kasaflow_order_token", "value": pending["token"]},
            {"key": "kasaflow_delivery_option", "value": pending.get("delivery_option") or ""},
        ],
    }

    auth = (settings.woocommerce_orders_consumer_key, settings.woocommerce_orders_consumer_secret)
    response = requests.post(
        f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/orders",
        json=payload,
        auth=auth,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
