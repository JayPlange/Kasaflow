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
from services.delivery_tool import (
    delivery_option_label,
    delivery_options_phrase,
    is_valid_delivery_option,
)
from services.geocoding_tool import infer_delivery_option, resolve_delivery_match
from services.memory import clear_order_state, get_session_store, set_last_action_outcome
from services.product_tool import _extract_karat, get_product_price, get_product_price_by_id
from services.whatsapp_client import WhatsAppError, send_text_message

logger = logging.getLogger(__name__)

_store = get_session_store()

_PENDING_ORDER_KEY = "pending_order"
_LAST_CONFIRMED_KEY = "last_confirmed_order"

# See memory.set_last_action_outcome()'s docstring. Shared by both
# confirm_order() failure branches below -- a genuine WooCommerce write
# failure, as opposed to the customer having anything to fix themselves.
# Reused rather than reset every call: the order is still recoverable
# (status goes back to "pending"), unlike propose_order's no-id case, so
# this deliberately doesn't claim it can't be tried again.
_CONFIRM_ORDER_FAILURE_OUTCOME = {
    "action": "confirm_order",
    "customer_safe_explanation": (
        "There was a temporary hiccup placing your order just now -- nothing "
        "was charged or lost. We can just try that again."
    ),
}


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
    notification).

    Argument checks run in this order deliberately: product/material
    first, then quantity/address/delivery_option, then the actual
    product lookup last. A customer who says "place an order" with no
    item named at all still has product_name="unknown" -- without this
    check first, they'd be walked through "how many?" then "what
    address?" for a product that was never identified, only to hit a
    context-free "couldn't find that product" at the very end (confirmed
    live, 2026-08-12: a customer answered quantity and address in full
    before the system finally admitted it never knew what they were
    ordering). A human salesperson asks "what would you like to order?"
    before anything else -- this makes the code do the same, while still
    only paying for the real product lookup once everything else checks
    out (see the quantity/address/delivery_option tests in
    test_order_tool.py, which rely on the lookup never being reached for
    those failures)."""

    product_stripped = str(product_name).strip() if product_name else ""
    if not product_stripped or product_stripped.lower() == "unknown":
        return {"error": "Sure -- which item would you like to order?"}

    # Clear a stale pending order the moment a genuinely different
    # product is named -- independent of, and earlier than, whatever
    # this call eventually returns. Without this, an old, fully-priced
    # proposal for the ABANDONED product stayed in _PENDING_ORDER_KEY
    # for however many turns it took to build the NEW product's own
    # proposal (karat, then quantity, ...): still exactly what
    # get_pending_order_summary() handed the LLM's own prompt as "you
    # have a pending order" for a product the conversation had already
    # moved past (see llm.py's _pending_order_state_line()), and --
    # before confirmation_allowed below closed the other half of this --
    # still fully confirmable the whole time. Found via
    # test_active_product_never_resurrects_after_an_explicit_product_
    # switch (test_order_conversations.py), 2026-08-21 (Webb, P0.3).
    # Mirrors memory.py's _names_a_different_product() guard for
    # _PRODUCT_SPECIFIC_KEYS, but pending_order lives in this file's own
    # store key, untouched by remember_context().
    # active_product_id/active_product_name are captured before the
    # stale-clear below (which may blank _PENDING_ORDER_KEY this same
    # call) so they survive as a fallback identity for the product_id
    # recovery further down -- see that block's comment for why this
    # exists and what live bug it closes.
    active_product_id = None
    active_product_name = None
    existing_pending = _store.get(session_id, _PENDING_ORDER_KEY)
    if existing_pending is not None:
        active_product_id = existing_pending.get("product_id")
        active_product_name = existing_pending.get("product")
        existing_product = str(active_product_name or "").strip().lower()
        if existing_product and existing_product != product_stripped.lower():
            _store.set(session_id, _PENDING_ORDER_KEY, None)

    material_stripped = str(material).strip() if material else ""
    if not material_stripped or material_stripped.lower() == "unknown":
        # awaiting_field tags exactly which single detail this question
        # is asking for -- see memory.set_awaiting_field()'s docstring
        # and router.py's _try_resolve_awaiting_field(): a deterministic
        # bare-karat reply next turn ("14k") can be resolved without
        # ever routing back through the general LLM tool-selection call
        # at all. Requested by Webb, 2026-08-21 (P0.4).
        return {"error": "What karat would you like that in?", "awaiting_field": "material"}

    quantity_int = _parse_quantity(quantity)
    if quantity_int is None:
        return {"error": "How many would you like?", "awaiting_field": "quantity"}

    address_stripped = str(delivery_address).strip() if delivery_address else ""
    if not address_stripped or address_stripped.lower() == "unknown":
        return {"error": "What address should this be delivered to?", "awaiting_field": "delivery_address"}

    delivery_key = str(delivery_option).strip().lower() if delivery_option else ""
    team_confirm = False
    if not is_valid_delivery_option(delivery_key):
        # The model hasn't stated (or couldn't confidently infer) one of
        # the three real arrangements -- try to infer it from the address
        # itself before falling back to asking a question the customer
        # may have already effectively answered. See
        # geocoding_tool.infer_delivery_option()'s docstring: this is
        # what actually closes the "east legon" case (Webb, 2026-08-19,
        # live: told "east legon", still got asked "Would you like rider
        # delivery within Accra, rider delivery within Kumasi, or
        # shipping outside Ghana?", which the address already answered).
        inferred = infer_delivery_option(address_stripped)
        if inferred == "ghana_other":
            # A real Ghanaian place, just not Accra or Kumasi -- there is
            # no rider zone that fits and "international" would be
            # actively wrong (the same category error resolve_delivery_match()
            # exists to catch from the other direction). Proceed without
            # asking; a human confirms the actual arrangement, exactly
            # what Webb originally asked for delivery gaps like this.
            team_confirm = True
        elif inferred:
            delivery_key = inferred
        else:
            return {"error": f"Would you like {delivery_options_phrase()}?"}

    product = get_product_price(product_name, material)
    if not product and active_product_id and product_stripped:
        # Correction-recovery fallback: the primary name-based lookup
        # above found nothing at all in the catalogue for this exact
        # product_name -- but this session already has a verified,
        # previously-priced product (active_product_id, captured above
        # from _PENDING_ORDER_KEY before it could be cleared). If this
        # call's product_name is a case-insensitive PREFIX of that
        # already-active product's exact catalogue name, treat it as the
        # same product restated incompletely, not a new, unmatched one --
        # and reprice against the known id instead of the possibly-broken
        # name string.
        #
        # This is deliberately narrower than a catalogue-wide prefix
        # search (which Webb/GPT explicitly rejected, 2026-08-21, and
        # which real data in data/products.json proves is genuinely
        # unsafe -- "Custom Gye Nyame Gold Necklace with Earrings, 12g"
        # and ", 20g" are different WooCommerce products, id 6520 vs
        # 6800; "Sparkling Crown Gold Ring, 12g" exists twice under two
        # different ids, 6119 and 3802). It only ever checks the restated
        # name against the ONE product already active in this session,
        # never against the catalogue as a whole, so it can't pick the
        # wrong SKU out of a lineup -- it can only recognise a truncation
        # of the one product this session already committed to.
        #
        # Confirmed live, 2026-08-21 (Webb, real OpenAI-backed /demo
        # run): "actually change the karat to 18" for a previously
        # selected "Big White Crown Stone Gold Ring, 14g" (product_id
        # 5892) came back with product_name restated as just "Big White
        # Crown Stone Gold Ring" (the ", 14g" weight suffix dropped) --
        # get_product_price()'s exact-match-first design (see that
        # function's own docstring) can never match a truncated name, so
        # this returned "Sorry, we couldn't find that product." for a
        # customer who had done nothing wrong. id is stable across every
        # karat/size variant of one named product in this catalogue
        # (confirmed against the real data, 2026-08-21 -- all 33
        # karat+size rows for this exact ring share id=5892), so
        # get_product_price_by_id() can resolve the new karat correctly
        # even with a mangled name, exactly the way get_product_price()
        # already resolves a karat correction when the name IS restated
        # correctly.
        #
        # DELIBERATE SCOPE BOUNDARY -- do not broaden this (Webb,
        # 2026-08-21, reviewing the live-verification replay):
        #
        #   Karat correction + truncated name  -> recover by active_product_id
        #   Explicit weight/product-variant change -> do NOT recover by
        #       active_product_id; require a lookup that actually
        #       resolves the new identity
        #
        # id is stable across karat (confirmed: all 33 karat+size rows
        # for one ring share one id), which is exactly why recovering by
        # id is safe for a karat correction. It is NOT stable across a
        # weight/size variant of the same base name -- "Custom Gye Nyame
        # Gold Necklace with Earrings, 12g" (id 6520) and ", 20g" (id
        # 6800) are different WooCommerce products sharing one base
        # name. If a customer explicitly says "actually make it the 20g
        # one" and the model drops the weight suffix entirely, this
        # prefix check cannot tell that apart from a karat-style
        # truncation, and will currently keep resolving the OLD id
        # (confirmed via a live replay against the real 6520/6800 pair,
        # 2026-08-21 -- see
        # test_propose_order_does_not_recover_a_dropped_weight_suffix_as_
        # a_product_switch in test_order_tool.py). That is a known,
        # accepted limitation, not a bug this fallback is meant to
        # close: making the prefix check "smarter" so it also resolves
        # weight-variant switches would turn a narrowly-safe recovery
        # mechanism into a product-substitution mechanism, and reopen
        # exactly the wrong-SKU risk the catalogue-wide-prefix-match
        # idea was rejected for above. A real fix for the weight-variant
        # case needs an explicit product-identity-change detection step
        # upstream of this, not a looser match here.
        if active_product_name and str(active_product_name).strip().lower().startswith(product_stripped.lower()):
            recovered = get_product_price_by_id(active_product_id, material)
            if recovered:
                logger.info(
                    "product_name %r didn't match the catalogue directly but is a prefix of "
                    "this session's active product %r (id=%s) -- recovered via product_id "
                    "rather than refusing the correction.",
                    product_name, active_product_name, active_product_id,
                )
                product = recovered

    if not product:
        return {"error": "Sorry, we couldn't find that product."}

    # Hard, LLM-independent price/variant invariant: refuse to build a
    # customer-facing proposal if the karat actually priced disagrees
    # with the karat the customer selected -- regardless of which path
    # inside get_product_price() produced `product`, and regardless of
    # whether that function's own internals change later. This is a
    # second, independent layer on top of get_product_price()'s own
    # karat-mismatch guard in its semantic-search fallback (see that
    # function's docstring) -- that guard protects one code path inside
    # one function; this one protects every path, at the one place a
    # proposal is actually assembled, which is what "the data boundary"
    # means here. Requested explicitly by Webb, 2026-08-21, after this
    # exact split was seen live: a correction_note said "updated the
    # karat to 18k" while the proposal underneath it silently kept
    # pricing at 12k (GH₵132,000 for 6 items at the 12k rate). A
    # customer should never be shown a price whose variant doesn't match
    # what they were just told, and that must not depend on the model
    # getting its own confirmation sentence right.
    #
    # Imports _extract_karat from product_tool.py rather than
    # duplicating it a fourth time (response_formatter.py and
    # recommendation_service.py each already carry their own copy, by
    # design -- see response_formatter.py's module docstring for why
    # those two are meant to stay standalone). This check is different:
    # it exists specifically to catch product_tool.py disagreeing with
    # itself, so it has to parse karat exactly the way
    # get_product_price() does, not a second, potentially-drifting
    # definition of "karat".
    selected_karat = _extract_karat(material)
    priced_karat = _extract_karat(product.get("material"))
    if selected_karat and priced_karat and selected_karat != priced_karat:
        logger.error(
            "Refusing to build a proposal: customer selected karat=%s but the priced "
            "product %r came back as karat=%s (product_name=%r, material argument=%r). "
            "get_product_price() has its own karat guard already -- if this fires, that "
            "guard has a gap somewhere and needs investigating, not just this check.",
            selected_karat, product.get("product"), priced_karat, product_name, material,
        )
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
        # Unlike every error above (all "still missing a detail" prompts
        # that are self-explanatory on their own), this one is a genuine,
        # unrecoverable dead end the customer did nothing wrong to cause
        # -- see memory.set_last_action_outcome()'s docstring for why a
        # customer's likely next message ("why?") needs this recorded,
        # not just the fact that something went wrong.
        set_last_action_outcome(session_id, {
            "action": "propose_order",
            "customer_safe_explanation": (
                "I found that item in the catalogue, but it isn't linked to our "
                "ordering system yet, so I can't take an order for it right now."
            ),
        })
        return {"error": "Sorry, I can't take orders for that item right now."}

    subtotal = product["price"] * quantity_int

    # The customer picked one of the three real delivery arrangements,
    # but that doesn't guarantee it actually matches the address they
    # gave -- see geocoding_tool.resolve_delivery_match()'s docstring
    # for the live cases (Tamale address/kumasi_rider, Cape Coast
    # address/international) this closes. Rather than silently building
    # a confirmation sentence that names two different places, or
    # blocking the order outright over a heuristic that can
    # false-positive on a real Accra/Kumasi address, this only ever
    # softens the *label* shown to the customer and staff --
    # delivery_option itself (the raw key) is stored unchanged, since it
    # still reflects what the customer actually chose.
    if team_confirm:
        resolved_label = (
            "a delivery arrangement to be confirmed by our team (we don't have automatic "
            "rider coverage for that address yet)"
        )
    elif resolve_delivery_match(delivery_key, address_stripped):
        resolved_label = delivery_option_label(delivery_key)
    else:
        resolved_label = (
            f"a delivery arrangement to be confirmed by our team (this address doesn't "
            f"match our usual {delivery_option_label(delivery_key)} zone)"
        )

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
        "delivery_option": "team_confirm" if team_confirm else delivery_key,
        "delivery_option_label": resolved_label,
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


def confirm_order(session_id: str, confirmation_allowed: bool = True) -> dict:
    """Create the real WooCommerce order for whatever propose_order()
    last stored against this session. Does nothing, and asks nothing,
    beyond the session id -- see module docstring for why.

    confirmation_allowed is the confirmation invariant Webb asked for,
    2026-08-21: a bare "yes" may only confirm the proposal that was the
    LAST thing this session did -- not a stale one sitting untouched
    from earlier. router.py passes memory.is_awaiting_confirmation()'s
    value, captured at the very start of the turn, before its own
    per-turn reset clears it. Defaults to True so every other, non-router
    caller (every existing test in this file, and any future direct
    call) keeps working unchanged -- this is deliberately something the
    caller has to opt OUT of trusting, not something confirm_order()
    guesses at itself: it has no way to know, from session_id alone,
    whether anything happened between the proposal and this call.

    Before this, awaiting_confirmation was tracked (see
    memory.set_awaiting_confirmation()) but only ever READ by the LLM's
    own prompt (llm.py's _pending_order_state_line()) -- nothing in the
    deterministic code path actually enforced it. That gap is real, not
    theoretical: test_active_product_never_resurrects_after_an_explicit_
    product_switch (test_order_conversations.py) found it directly --
    after "actually I'll take Product B instead" got asked for karat
    (propose_order for B failed before storing a new proposal), the OLD
    pending_order for Product A was still sitting in
    _PENDING_ORDER_KEY, fully intact and confirmable, even though
    nothing about it was true anymore for what the customer was
    actually asking for now."""

    pending = _store.get(session_id, _PENDING_ORDER_KEY)

    if pending is not None and not confirmation_allowed:
        # Same customer-facing wording as "nothing pending" below --
        # from the customer's side there's nothing to distinguish these
        # two cases, and pretending a stale, superseded proposal is
        # still live would be worse than just asking again.
        return {
            "error": "Hmm, I don't have anything pending to confirm right now -- "
            "were you looking to go ahead with one of the pieces we discussed?"
        }

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
        set_last_action_outcome(session_id, _CONFIRM_ORDER_FAILURE_OUTCOME)
        return {"error": "Something went wrong placing your order -- let's try that again."}
    except Exception:
        # Every other failure (connection refused, a 4xx/5xx WooCommerce
        # actually returned, malformed response) is NOT ambiguous -- the
        # request either never reached WooCommerce or WooCommerce clearly
        # rejected it, so there's nothing to look up before retrying.
        logger.exception("WooCommerce order creation failed for token=%s", pending.get("token"))
        pending["status"] = "pending"
        _store.set(session_id, _PENDING_ORDER_KEY, pending)
        set_last_action_outcome(session_id, _CONFIRM_ORDER_FAILURE_OUTCOME)
        return {"error": "Something went wrong placing your order -- let's try that again."}

    return _finalize_confirmation(session_id, pending, order["id"])


_CANCELLABLE_STATUSES = {"pending", "on-hold"}


class _OrderNotFound(Exception):
    pass


def cancel_order(session_id: str, order_id=None) -> dict:
    """Cancel a customer's order.

    Prefers an order number the customer stated explicitly; falls back
    to this session's last confirmed order if they didn't give one (the
    LLM passes "unknown" for order_id in that case -- see llm.py's
    cancel_order guidance). This deliberately doesn't rely on session
    memory alone for anything beyond finding *which* order to look at:
    the order's actual status is re-checked live against WooCommerce
    every time, because staff can move an order forward -- or cancel it
    themselves -- directly in WooCommerce at any point after handoff, so
    what this session last knew about the order is not assumed to still
    be true.

    Three distinct outcomes, not one -- see response_formatter.py:
    - order_cancellation: found it, it was still cancellable, cancelled it.
    - order_already_cancelled: found it, it was already cancelled (a
      repeated "cancel" message, most likely -- see the module's
      idempotency discussion above for why WhatsApp delivery can
      duplicate a message).
    - order_escalation: found it, but its status (shipped, completed,
      refunded, ...) means this tool won't touch it automatically --
      handed to staff instead of guessed at.

    WooCommerce PUT-to-update-status is the documented, standard way to
    change an order's status via the REST API -- not yet verified
    against a real store, same caveat as _find_existing_order_by_token()
    above: confirm it actually works before relying on it in
    production."""

    resolved_id = _resolve_order_id(session_id, order_id)
    if resolved_id is None:
        return {
            "error": "I don't have an order on file for you right now -- "
            "what's the order number you'd like to cancel?"
        }

    try:
        order = _get_woocommerce_order(resolved_id)
    except _OrderNotFound:
        return {"error": f"I couldn't find order #{resolved_id} -- could you double-check the number?"}
    except Exception:
        logger.exception("Order lookup failed for #%s during cancellation", resolved_id)
        return {"error": "Something went wrong looking up that order -- let's try again in a moment."}

    if order["status"] == "cancelled":
        return {"order_already_cancelled": {"order_id": resolved_id}}

    if order["status"] not in _CANCELLABLE_STATUSES:
        _notify_staff_of_cancel_request(session_id, resolved_id, order["status"])
        return {"order_escalation": {"order_id": resolved_id, "status": order["status"]}}

    try:
        _cancel_woocommerce_order(resolved_id)
    except Exception:
        logger.exception("WooCommerce cancellation failed for order #%s", resolved_id)
        return {"error": "Something went wrong cancelling that order -- let's try again in a moment."}

    _notify_staff_of_cancellation(session_id, resolved_id)

    last = _store.get(session_id, _LAST_CONFIRMED_KEY)
    if last is not None and last.get("order_id") == resolved_id:
        # No longer the active order to fall back to -- see
        # _resolve_order_id()'s fallback below.
        _store.set(session_id, _LAST_CONFIRMED_KEY, None)

    return {"order_cancellation": {"order_id": resolved_id}}


def _resolve_order_id(session_id: str, order_id) -> int | None:
    given = str(order_id).strip() if order_id else ""
    if given and given.lower() != "unknown":
        try:
            return int(given)
        except ValueError:
            return None
    last = _store.get(session_id, _LAST_CONFIRMED_KEY)
    return last["order_id"] if last else None


def _get_woocommerce_order(order_id: int) -> dict:
    _require_orders_config()
    auth = (settings.woocommerce_orders_consumer_key, settings.woocommerce_orders_consumer_secret)
    response = requests.get(
        f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/orders/{order_id}",
        auth=auth,
        timeout=15,
    )
    if response.status_code == 404:
        raise _OrderNotFound(order_id)
    response.raise_for_status()
    return response.json()


def _cancel_woocommerce_order(order_id: int) -> None:
    _require_orders_config()
    auth = (settings.woocommerce_orders_consumer_key, settings.woocommerce_orders_consumer_secret)
    response = requests.put(
        f"{settings.woocommerce_url.rstrip('/')}/wp-json/wc/v3/orders/{order_id}",
        json={"status": "cancelled"},
        auth=auth,
        timeout=30,
    )
    response.raise_for_status()


def _notify_staff_of_cancellation(session_id: str, order_id: int) -> None:
    if not settings.staff_notification_phone:
        logger.warning(
            "STAFF_NOTIFICATION_PHONE not configured -- order #%s was cancelled but no "
            "rider coordinator was notified.",
            order_id,
        )
        return
    message = f"KasaFlow order #{order_id} was cancelled by the customer (WhatsApp: {session_id})."
    try:
        send_text_message(settings.staff_notification_phone, message)
    except WhatsAppError as e:
        logger.error("Failed to notify staff about cancellation of order #%s: %s", order_id, e)


def _notify_staff_of_cancel_request(session_id: str, order_id: int, status: str) -> None:
    """A customer asked to cancel an order that's past the point this
    tool will touch automatically. Staff still need to know -- silently
    telling the customer "no" without anyone finding out they wanted to
    cancel would be worse than the automatic case just not existing."""
    if not settings.staff_notification_phone:
        logger.warning(
            "STAFF_NOTIFICATION_PHONE not configured -- customer requested cancellation of "
            "order #%s (status: %s) but no rider coordinator was notified.",
            order_id,
            status,
        )
        return
    message = (
        f"Customer (WhatsApp: {session_id}) asked to cancel order #{order_id}, "
        f"but its status is \"{status}\" -- please handle manually."
    )
    try:
        send_text_message(settings.staff_notification_phone, message)
    except WhatsAppError as e:
        logger.error("Failed to notify staff about cancel request for order #%s: %s", order_id, e)


def _finalize_confirmation(session_id: str, pending: dict, order_id) -> dict:
    confirmation = {
        "order_id": order_id,
        "total": pending["total"],
        "delivery_address": pending["delivery_address"],
        "delivery_option_label": pending.get("delivery_option_label"),
    }
    _store.set(session_id, _LAST_CONFIRMED_KEY, confirmation)
    _store.set(session_id, _PENDING_ORDER_KEY, None)
    # This order is done -- product/material/quantity/address/delivery
    # option and last_priced_product must not survive to bleed into
    # whatever the customer asks for next. See clear_order_state()'s
    # docstring (2026-08-20 architecture audit, failure #3).
    clear_order_state(session_id)
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
