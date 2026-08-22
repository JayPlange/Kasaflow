"""
Golden-conversation regression tests: whole multi-turn orders end to
end, through the real /process endpoint (real FastAPI routing, real
session memory, real router/order_tool/delivery/geocoding/response-
formatter wiring), with only the LLM's tool-selection call mocked per
turn (same boundary test_integration.py already documents: "we mock at
the router level, one layer below the HTTP boundary") and WooCommerce's
order-creation POST mocked for the one test that confirms.

Why this file exists, separate from test_router.py's wiring tests and
test_order_tool.py's per-function tests: those prove each piece works
in isolation, but neither catches a regression in how the pieces snap
together across a whole conversation -- which is exactly what slipped
through before this file existed. Webb's real live transcript,
2026-08-19: "12k -> order that one -> qty 7 -> wait, 14k rather ->
east legon" surfaced two bugs (a silent, unacknowledged correction, and
a redundant delivery-option question for an address that already
answered it) that no single-function unit test caught, because each
individual function was working exactly as designed -- the gap was
only visible across the full sequence. These tests reproduce that
transcript verbatim, plus close variants (a compound single-message
correction, a no-correction control), as permanent regression coverage
for the two fixes built in response to it (see services/router.py's
_describe_order_corrections() and services/geocoding_tool.py's
infer_delivery_option()).
"""

from dataclasses import replace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import main
from app.config import settings
from services import order_tool
from services import router as router_module
from services.response_formatter import format_for_customer

client = TestClient(main.app)
AUTH_HEADERS = {"X-API-Key": settings.app_api_key}


def _send(message: str, session_id: str) -> dict:
    response = client.post(
        "/process",
        json={"message": message, "session_id": session_id},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _mock_understand_customer(monkeypatch, *tool_requests: dict) -> None:
    """Feeds one canned understand_customer() return value per turn, in
    order -- the same per-call sequencing unittest.mock.MagicMock's
    side_effect already gives us, just named for readability at each
    call site below."""
    monkeypatch.setattr(router_module, "understand_customer", MagicMock(side_effect=list(tool_requests)))


def _mock_woocommerce(monkeypatch, order_id=9001):
    monkeypatch.setattr(
        order_tool,
        "settings",
        replace(
            order_tool.settings,
            woocommerce_url="https://adomdejeweller.com",
            woocommerce_orders_consumer_key="ck_test",
            woocommerce_orders_consumer_secret="cs_test",
            staff_notification_phone=None,
        ),
    )
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"id": order_id, "status": "on-hold"}
    monkeypatch.setattr(order_tool.requests, "post", MagicMock(return_value=fake_response))


_PRODUCT = {
    "id": 555,
    "product": "Gye Nyame White Necklace with Earrings, 30g",
    "material": "12k",
    "price": 39000.0,
}


def _priced(material: str, price: float) -> dict:
    return {**_PRODUCT, "material": material, "price": price}


def test_full_transcript_correction_and_delivery_inference(monkeypatch):
    # Reproduces Webb's real live transcript, 2026-08-19, verbatim --
    # both fixes exercised together: the material correction (12k -> 14k)
    # must be acknowledged, not applied silently, and "east legon" alone
    # must resolve straight to accra_rider without a menu question.
    #
    # Three of this transcript's own turns ("7", "east legon", "confirm")
    # are now resolved by router.py's awaiting_field deterministic
    # short-circuit (P0.4, 2026-08-21) rather than going through
    # understand_customer() at all -- see _try_resolve_awaiting_field().
    # This mock list only has to cover the turns that still reach the
    # LLM: the browsing questions, the initial order attempt, and the
    # "wait i want to order the 14k rather" correction (a full sentence,
    # not a bare pattern match, and explicitly excluded from the address
    # short-circuit by the "wait" prefix guard -- it still needs the LLM
    # to recognise it as a material correction).
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0),  # "how much ... in 14k"
        _priced("18k", 51000.0),  # "what about in 18k"
        _priced("12k", 39000.0),  # "do you have 12k?"
    ]))
    _mock_woocommerce(monkeypatch)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "converse", "arguments": {"reply": "Hey! How can I help you today?"}},
        {"tool": "get_product_price", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k"}},
        {"tool": "get_product_price", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "18k"}},
        {"tool": "get_product_price", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "12k"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "unknown", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        # "7" (quantity) is resolved deterministically -- no canned entry.
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "14k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        # "east legon" (delivery_address) and "confirm" (confirmation)
        # are also resolved deterministically -- no canned entries.
    )
    session_id = "golden-transcript-1"

    _send("hey", session_id)
    r2 = _send("How much is the Gye Nyame White Necklace with Earrings, 30g in 14k", session_id)
    assert r2["price"] == 45000.0
    r3 = _send("what about in 18k", session_id)
    assert r3["price"] == 51000.0
    r4 = _send("do you have 12k?", session_id)
    assert r4["price"] == 39000.0

    r5 = _send("i want to order that one", session_id)
    assert r5["error"] == "How many would you like?"

    r6 = _send("7", session_id)
    assert r6["error"] == "What address should this be delivered to?"

    # The correction: material changes 12k -> 14k. Must be acknowledged,
    # and quantity (7) must survive from the previous turn, not reset.
    r7 = _send("wait i want to order the 14k rather", session_id)
    assert r7["correction_note"] == "Got it, I've updated the karat to 14k."
    assert r7["error"] == "What address should this be delivered to?"

    # The delivery inference: "east legon" alone must resolve straight
    # to accra_rider, no "Would you like rider delivery within Accra,
    # ..." menu question (the exact bug this transcript surfaced). Now
    # resolved by the awaiting_field deterministic short-circuit (P0.4),
    # which stores the customer's own text verbatim -- same as a real
    # WhatsApp reply typed in lowercase, and unrelated to the actual bug
    # this test locks in (delivery_option inference from the address).
    r8 = _send("east legon", session_id)
    proposal = r8["proposal"]
    assert proposal["material"] == "14k"
    assert proposal["quantity"] == 7
    assert proposal["delivery_address"] == "east legon"
    assert proposal["delivery_option"] == "accra_rider"
    assert proposal["delivery_option_label"] == "rider delivery within Accra"
    assert proposal["total"] == 45000.0 * 7
    # Prose check too, not just the raw shape -- format_for_customer()
    # is what a real customer actually reads.
    assert "rider delivery within Accra" in format_for_customer(r8)

    r9 = _send("confirm", session_id)
    confirmation = r9["order_confirmation"]
    assert confirmation["order_id"] == 9001
    assert confirmation["delivery_option_label"] == "rider delivery within Accra"


def test_compound_correction_in_one_message_is_acknowledged_together(monkeypatch):
    # "Actually 14k, make it 6 instead" -- both karat and quantity
    # change in the same message. Acknowledged in one sentence, not two
    # separate corrections, and no extra confirmation round trip forced.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(return_value=_priced("14k", 45000.0)))
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "12k", "quantity": 7,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "14k", "quantity": 6,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
    )
    session_id = "golden-compound-correction"

    r1 = _send("order the Gye Nyame White Necklace with Earrings, 30g in 12k, 7 of them", session_id)
    assert r1["error"] == "What address should this be delivered to?"

    r2 = _send("actually 14k, make it 6 instead", session_id)
    assert r2["correction_note"] == "Got it, I've updated the karat to 14k and the quantity to 6."
    assert r2["error"] == "What address should this be delivered to?"


def test_correction_after_a_full_proposal_confirms_the_corrected_order(monkeypatch):
    # Webb, 2026-08-20: "the confirmation flag cannot simply mean 'there
    # is some pending order' -- it needs to mean 'the exact current draft
    # that was most recently proposed is awaiting confirmation.'" This is
    # the end-to-end proof: a full proposal exists (awaiting_confirmation
    # = True), the customer corrects a field, which produces a whole NEW
    # proposal (awaiting_confirmation goes back to True for THAT one),
    # and confirming afterwards must place the CORRECTED order -- not
    # silently keep confirming the pre-correction 14k proposal.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0),  # original proposal
        _priced("18k", 51000.0),  # corrected proposal
    ]))
    _mock_woocommerce(monkeypatch, order_id=9003)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 2,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "18k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "confirm_order", "arguments": {}},
    )
    session_id = "golden-correction-then-confirm"

    r1 = _send("2 Gye Nyame White Necklace with Earrings, 30g in 14k, deliver to Accra, rider delivery", session_id)
    assert r1["proposal"]["material"] == "14k"
    assert r1["proposal"]["total"] == 45000.0 * 2

    r2 = _send("actually make it 18k", session_id)
    assert r2["correction_note"] == "Got it, I've updated the karat to 18k."
    proposal = r2["proposal"]
    assert proposal["material"] == "18k"
    assert proposal["quantity"] == 2  # carried over, not lost by the correction
    assert proposal["delivery_address"] == "Accra"
    assert proposal["total"] == 51000.0 * 2

    r3 = _send("yes", session_id)
    confirmation = r3["order_confirmation"]
    assert confirmation["order_id"] == 9003
    # The actual WooCommerce order reflects the CORRECTED (18k) proposal,
    # not the original 14k one -- proves confirm_order() confirmed the
    # current pending_order, which propose_order's correction call
    # overwrote in session state, not a stale copy.
    assert confirmation["total"] == 51000.0 * 2


def test_rapid_sequential_answers_then_a_late_correction_land_in_the_right_fields(monkeypatch):
    # Webb, 2026-08-20: "14k, 6, East Legon in very rapid succession,
    # then 'actually 18k'" -- proves the per-session turn lock protects
    # state SEQUENCING (each turn's write must be based on the previous
    # turn's completed state), not just concurrent execution safety.
    # "14k", "6", and "East Legon" are now each resolved by router.py's
    # awaiting_field deterministic short-circuit (P0.4, 2026-08-21) --
    # each one unambiguously answers the single field propose_order's
    # own previous error just asked for, so none of them reach
    # understand_customer() at all any more. Only the initial order
    # attempt and the final "actually 18k" correction (a full sentence,
    # and not a bare confirmation phrase given awaiting_field is
    # "confirmation" by that point) still need canned LLM responses.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0), _priced("18k", 51000.0),
    ]))
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "unknown",
            "quantity": "unknown", "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "18k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
    )
    session_id = "golden-rapid-sequence"

    _send("order the Gye Nyame White Necklace with Earrings, 30g", session_id)
    _send("14k", session_id)
    _send("6", session_id)
    r4 = _send("East Legon", session_id)
    proposal = r4["proposal"]
    assert proposal["material"] == "14k"
    assert proposal["quantity"] == 6
    assert proposal["delivery_address"] == "East Legon"
    assert proposal["delivery_option"] == "accra_rider"

    r5 = _send("actually 18k", session_id)
    assert r5["correction_note"] == "Got it, I've updated the karat to 18k."
    final = r5["proposal"]
    assert final["material"] == "18k"
    assert final["quantity"] == 6
    assert final["delivery_address"] == "East Legon"
    assert final["delivery_option"] == "accra_rider"
    assert final["total"] == 51000.0 * 6


def test_confirmed_order_does_not_bleed_into_the_next_unrelated_order(monkeypatch):
    # Webb, 2026-08-20, check #2: right after confirming, the customer
    # starts an entirely different, unrelated order -- must be built
    # cleanly from what THIS message actually states, with no stale
    # correction_note and no leftover product/karat/address/delivery
    # inherited from the order that was just placed.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("18k", 51000.0),  # first order
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "14k", "price": 24000.0},
    ]))
    _mock_woocommerce(monkeypatch, order_id=9004)
    # "confirm" is now resolved by the awaiting_field deterministic
    # short-circuit (P0.4, 2026-08-21) -- it's a bare confirmation
    # phrase and awaiting_field is "confirmation" right after the first
    # full proposal, so it never reaches understand_customer(). Only the
    # initial proposal and the brand-new, fully-specified second order
    # still need canned LLM responses.
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "18k", "quantity": 1,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "14k", "quantity": 1,
            "delivery_address": "Kumasi", "delivery_option": "kumasi_rider"}},
    )
    session_id = "golden-confirm-then-unrelated-order"

    _send("Gye Nyame White Necklace with Earrings, 30g in 18k, deliver to Accra, rider delivery", session_id)
    r2 = _send("confirm", session_id)
    assert r2["order_confirmation"]["order_id"] == 9004

    # A brand-new, unrelated propose_order call -- must not carry a
    # correction_note (a genuinely different, explicitly-named product
    # is a fresh order, not a correction -- see
    # _describe_order_corrections()), and must reflect ONLY this
    # message's own details, nothing left over from the confirmed order.
    r3 = _send(
        "1 Big White Crown Stone Gold Ring, 14g in 14k, deliver to Kumasi, rider delivery within Kumasi",
        session_id,
    )
    assert "correction_note" not in r3
    proposal = r3["proposal"]
    assert proposal["product"] == "Big White Crown Stone Gold Ring, 14g"
    assert proposal["material"] == "14k"
    assert proposal["delivery_address"] == "Kumasi"
    assert proposal["delivery_option"] == "kumasi_rider"


def test_changing_product_after_a_full_proposal_starts_a_clean_order_for_the_new_one(monkeypatch):
    # Webb, 2026-08-20, check #6 -- the real validation of the
    # product-identity guard: order Product A in 14k, qty 6, full
    # proposal shown, then "actually I'll take Product B" instead.
    #
    # While designing this test, tracing it by hand surfaced a second,
    # live-reachable bug beyond the correction_note wording: fill_missing_
    # context() would have silently completed Product B's order using
    # Product A's exact karat/quantity, since neither was restated for B
    # and the function had no concept of "this is a different product
    # now". Fixed in memory.py's fill_missing_context() (see
    # _PRODUCT_SPECIFIC_KEYS) alongside this test -- this proves both
    # fixes together, end to end. (delivery_address/delivery_option are
    # order-level, not item-level, and are expected to keep carrying
    # forward across a product switch -- see
    # test_product_switch_preserves_delivery_address_already_given below.)
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0),  # Product A
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "12k", "price": 21000.0},
    ]))
    _mock_woocommerce(monkeypatch, order_id=9005)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        # "actually I'll take the ring instead" -- names a different
        # product; nothing else in the message states a karat, quantity,
        # or address for it.
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "unknown",
            "quantity": "unknown", "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "12k", "quantity": 1,
            "delivery_address": "Kumasi", "delivery_option": "kumasi_rider"}},
        {"tool": "confirm_order", "arguments": {}},
    )
    session_id = "golden-product-switch"

    r1 = _send(
        "Gye Nyame White Necklace with Earrings, 30g in 14k, 6 of them, deliver to Accra, "
        "rider delivery within Accra", session_id,
    )
    assert r1["proposal"]["product"] == "Gye Nyame White Necklace with Earrings, 30g"
    assert r1["proposal"]["material"] == "14k"
    assert r1["proposal"]["quantity"] == 6

    # No correction note referring to Product A -- this is a fresh
    # order, not a correction (see _describe_order_corrections()). And
    # none of Product A's karat/quantity leak into the question this now
    # asks about Product B (delivery_address/option are order-level, not
    # item-level, and DO carry over -- see
    # test_product_switch_preserves_delivery_address_already_given).
    r2 = _send("actually I'll take the Big White Crown Stone Gold Ring, 14g instead", session_id)
    assert "correction_note" not in r2
    assert r2["error"] == "What karat would you like that in?"

    # This message also names a genuinely different delivery address
    # (Kumasi) than the one already on file for this order (Accra, from
    # the first product). That's a real change to order-level state, not
    # a leftover from the product switch -- it SHOULD be flagged as a
    # correction, same as changing the address mid-order for a single
    # product always has been.
    r3 = _send("12k, 1 of them, deliver to Kumasi, rider delivery within Kumasi", session_id)
    assert "correction_note" in r3
    assert "Kumasi" in r3["correction_note"]
    proposal = r3["proposal"]
    assert proposal["product"] == "Big White Crown Stone Gold Ring, 14g"
    assert proposal["material"] == "12k"
    assert proposal["quantity"] == 1
    assert proposal["delivery_address"] == "Kumasi"
    assert proposal["delivery_option"] == "kumasi_rider"
    assert proposal["total"] == 21000.0

    # Confirmation applies to Product B only.
    r4 = _send("confirm", session_id)
    confirmation = r4["order_confirmation"]
    assert confirmation["order_id"] == 9005
    assert confirmation["total"] == 21000.0
    assert confirmation["delivery_address"] == "Kumasi"


def test_product_switch_preserves_delivery_address_already_given(monkeypatch):
    # Webb, 2026-08-20, check #6 follow-up: a product switch must NOT
    # wipe an address the customer already gave earlier in the same
    # conversation. Unlike karat/quantity (see the product-switch test
    # above), delivery_address/delivery_option describe the customer's
    # order as a whole, not the specific item, so they should keep
    # resolving from memory -- the customer should be asked for karat on
    # the new product, but never asked to repeat an address they already
    # answered.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0),  # Product A
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "12k", "price": 21000.0},
    ]))
    _mock_woocommerce(monkeypatch, order_id=9006)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        # "actually I'll take the ring instead" -- names a different
        # product, restates nothing else. The LLM correctly reports
        # delivery_address/option as unknown here too -- it's fill_missing_
        # context()'s job, not the model's, to recognise these should
        # still carry over.
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "unknown",
            "quantity": "unknown", "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "12k", "quantity": 1,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
    )
    session_id = "golden-product-switch-keeps-address"

    r1 = _send(
        "Gye Nyame White Necklace with Earrings, 30g in 14k, 6 of them, deliver to Accra, "
        "rider delivery within Accra", session_id,
    )
    assert r1["proposal"]["delivery_address"] == "Accra"

    # Switching product must ask for karat -- but must NOT ask for the
    # address again, since it's order-level and already known.
    r2 = _send("actually I'll take the Big White Crown Stone Gold Ring, 14g instead", session_id)
    assert r2["error"] == "What karat would you like that in?"

    # This message restates karat/quantity for the new item but says
    # nothing about delivery -- the address/option from the FIRST
    # product's order must still fill in, unprompted.
    r3 = _send("12k, 1 of them", session_id)
    assert "correction_note" not in r3
    proposal = r3["proposal"]
    assert proposal["product"] == "Big White Crown Stone Gold Ring, 14g"
    assert proposal["material"] == "12k"
    assert proposal["quantity"] == 1
    assert proposal["delivery_address"] == "Accra"
    assert proposal["delivery_option"] == "accra_rider"


def test_active_product_never_resurrects_after_an_explicit_product_switch(monkeypatch):
    # Named, permanent regression test for the exact failure pattern
    # Webb traced live, 2026-08-21: Product A is started, the customer
    # explicitly names Product B instead, then answers karat and
    # quantity as SEPARATE bare replies (not one combined message, the
    # scenario test_changing_product_after_a_full_proposal_starts_a_
    # clean_order_for_the_new_one already covers) -- exactly the shape
    # of reply that misrouted to recommend_products before #92's
    # order_draft fix, because a stale pending_order for Product A
    # suppressed the new, in-progress draft for Product B. #92 already
    # fixes the underlying bug; this test exists so nobody can
    # accidentally break that guarantee later without a test failing --
    # see #92's docstrings in memory.py/router.py for the mechanism.
    #
    # Asserts non-resurrection two ways at every step: the HTTP response
    # never names Product A again, AND the session's actual pending-order
    # state (order_tool.get_pending_order_summary(), read directly, not
    # inferred from the response shape) never points back to it either.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0),  # Product A, full proposal
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "12k", "price": 21000.0},
    ]))
    _mock_woocommerce(monkeypatch, order_id=9007)
    _mock_understand_customer(
        monkeypatch,
        # 1: start and fully price an order for Product A.
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        # 2: customer explicitly names Product B instead -- nothing else
        # restated.
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "unknown",
            "quantity": "unknown", "delivery_address": "unknown", "delivery_option": "unknown"}},
        # 3: bare karat reply -- the exact shape that misrouted live.
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "12k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        # 4: bare quantity reply.
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "unknown", "quantity": 2,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "confirm_order", "arguments": {}},
    )
    session_id = "golden-no-product-resurrection"

    r1 = _send(
        "Gye Nyame White Necklace with Earrings, 30g in 14k, 6 of them, deliver to Accra, "
        "rider delivery within Accra", session_id,
    )
    assert r1["proposal"]["product"] == "Gye Nyame White Necklace with Earrings, 30g"
    assert order_tool.get_pending_order_summary(session_id)["product"] == (
        "Gye Nyame White Necklace with Earrings, 30g"
    )

    r2 = _send("actually I'll take the Big White Crown Stone Gold Ring, 14g instead", session_id)
    assert r2["error"] == "What karat would you like that in?"
    assert "Gye Nyame" not in str(r2)
    # The old proposal must no longer be the session's active pending
    # order -- a genuinely new, different product was just named.
    assert order_tool.get_pending_order_summary(session_id) is None

    r3 = _send("12k", session_id)
    assert r3["error"] == "How many would you like?"
    assert "Gye Nyame" not in str(r3)
    assert order_tool.get_pending_order_summary(session_id) is None

    r4 = _send("2", session_id)
    assert "Gye Nyame" not in str(r4)
    proposal = r4["proposal"]
    assert proposal["product"] == "Big White Crown Stone Gold Ring, 14g"
    assert proposal["material"] == "12k"
    assert proposal["quantity"] == 2
    assert order_tool.get_pending_order_summary(session_id)["product"] == "Big White Crown Stone Gold Ring, 14g"

    r5 = _send("confirm", session_id)
    confirmation = r5["order_confirmation"]
    assert confirmation["order_id"] == 9007
    assert confirmation["total"] == 42000.0


# ---------------------------------------------------------------------
# Confirmation invariant: a bare "yes" may only confirm the proposal
# that was the actual last thing this session did. Webb, 2026-08-21 --
# exact scenarios requested. See order_tool.confirm_order()'s
# confirmation_allowed parameter and router.py's was_awaiting_confirmation.
# ---------------------------------------------------------------------

def test_a_bare_yes_after_an_unrelated_question_does_not_confirm_a_stale_proposal(monkeypatch):
    _mock_woocommerce(monkeypatch, order_id=9008)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        # Something unrelated is asked/said in between -- nothing about
        # the order itself.
        {"tool": "converse", "arguments": {"reply": "We're open every day except Sunday."}},
        # A bare "yeah" that the model (wrongly, but this is exactly the
        # kind of mistake the invariant exists to survive) reads as
        # confirming the order from two turns ago.
        {"tool": "confirm_order", "arguments": {}},
    )
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(return_value=_priced("14k", 45000.0)))
    session_id = "golden-confirmation-invariant-unrelated"

    r1 = _send(
        "Gye Nyame White Necklace with Earrings, 30g in 14k, 6 of them, deliver to Accra, "
        "rider delivery within Accra", session_id,
    )
    assert "proposal" in r1

    r2 = _send("are you open on Sundays?", session_id)
    assert "conversation_reply" in r2

    r3 = _send("yeah", session_id)
    assert "order_confirmation" not in r3
    assert "anything pending to confirm" in r3["error"].lower()
    # The proposal itself is still there, genuinely unconfirmed -- a
    # real "yes" right now should still be able to confirm it. This only
    # refuses a confirmation that wasn't actually the next thing said
    # after the proposal.
    assert order_tool.get_pending_order_summary(session_id) is not None


def test_confirming_after_a_correction_confirms_the_corrected_proposal(monkeypatch):
    _mock_woocommerce(monkeypatch, order_id=9009)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "14k", "quantity": 6,
            "delivery_address": "Accra", "delivery_option": "accra_rider"}},
        # Corrects quantity to 5 -- still the same product, a genuine
        # correction, not a switch.
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "unknown", "quantity": 5,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "confirm_order", "arguments": {}},
    )
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        _priced("14k", 45000.0), _priced("14k", 45000.0),
    ]))
    session_id = "golden-confirmation-invariant-correction"

    r1 = _send(
        "Gye Nyame White Necklace with Earrings, 30g in 14k, 6 of them, deliver to Accra, "
        "rider delivery within Accra", session_id,
    )
    assert r1["proposal"]["quantity"] == 6
    assert r1["proposal"]["total"] == 270000.0

    r2 = _send("actually make it 5", session_id)
    assert "correction_note" in r2
    assert r2["proposal"]["quantity"] == 5
    assert r2["proposal"]["total"] == 225000.0

    # "yes" now must confirm the CORRECTED (5-item) proposal -- this is
    # the last thing propose_order actually stored, and was the last
    # thing that happened in the session, so the invariant allows it.
    r3 = _send("yes", session_id)
    confirmation = r3["order_confirmation"]
    assert confirmation["order_id"] == 9009
    assert confirmation["total"] == 225000.0


def test_delivery_option_is_rederived_when_the_address_moves_to_a_different_zone(monkeypatch):
    # Webb, 2026-08-20, live: after a Kumasi order, later Accra/Kasoa
    # addresses kept coming back "doesn't match our usual rider delivery
    # within Kumasi zone" -- delivery_option ("kumasi_rider") was
    # silently carried over from the OLD address instead of being
    # re-derived for the new one, because order_tool.propose_order()
    # only re-infers it when the current value isn't already valid.
    # Reproduces the exact address sequence end to end: a Kumasi order,
    # then a fresh order to a Kasoa address with no delivery_option
    # restated -- must resolve to accra_rider, not the Kumasi-zone
    # mismatch message.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "14k", "price": 20857.2},
        _priced("18k", 51000.0),
    ]))
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "14k", "quantity": 1,
            "delivery_address": "Kumasi", "delivery_option": "kumasi_rider"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "18k", "quantity": 1,
            "delivery_address": "Kasoa", "delivery_option": "unknown"}},
    )
    session_id = "golden-address-zone-change"

    r1 = _send(
        "1 Big White Crown Stone Gold Ring, 14g in 14k, deliver to Kumasi, rider delivery within Kumasi",
        session_id,
    )
    assert r1["proposal"]["delivery_option"] == "kumasi_rider"

    r2 = _send("Gye Nyame White Necklace with Earrings, 30g in 18k, deliver to Kasoa", session_id)
    proposal = r2["proposal"]
    assert proposal["delivery_option"] == "accra_rider", (
        f"Expected delivery_option to be re-derived as accra_rider for the new Kasoa address, "
        f"got {proposal['delivery_option']!r} -- the old Kumasi arrangement carried over stale."
    )
    assert "doesn't match our usual" not in proposal["delivery_option_label"]


def test_delivery_option_is_rederived_when_the_address_was_never_captured_the_first_time(monkeypatch):
    # Webb's own first live trace run against the awaiting_field
    # instrumentation, 2026-08-21: a compound order message ("1 Big
    # White Crown Stone Gold Ring, 14g in 12k, deliver to Kumasi, rider
    # delivery within Kumasi") had delivery_option ("kumasi_rider")
    # extracted correctly but delivery_address failed to extract at all
    # that same turn (stayed "unknown") -- so nothing was ever
    # remembered for it. Quantity was answered next (a bare "5"), then
    # several turns later an address finally arrived ("deliver to
    # Kasoa"). Because _names_a_different_address() previously only
    # fired when a DIFFERENT address had genuinely been remembered
    # before, and here NO address had ever been remembered at all, the
    # stale "kumasi_rider" (never actually derived from any address)
    # carried straight through to the Kasoa order -- live-visible as
    # "this address doesn't match our usual rider delivery within
    # Kumasi zone" for an address nowhere near Kumasi. See
    # memory._names_a_different_address()'s docstring for the fix.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(side_effect=[
        {"id": 5892, "product": "Big White Crown Stone Gold Ring, 14g", "material": "12k", "price": 17648.4},
    ]))
    _mock_understand_customer(
        monkeypatch,
        # Turn 1: delivery_option extracts, delivery_address and
        # quantity both fail to extract -- reproducing the live gap
        # verbatim, not assuming a cause beyond what was observed.
        {"tool": "propose_order", "arguments": {
            "product_name": "Big White Crown Stone Gold Ring, 14g", "material": "12k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "kumasi_rider"}},
    )
    session_id = "golden-address-never-captured-first-time"

    r1 = _send(
        "1 Big White Crown Stone Gold Ring, 14g in 12k, deliver to Kumasi, rider delivery within Kumasi",
        session_id,
    )
    assert r1["error"] == "How many would you like?"

    # Bare quantity -- resolved by the awaiting_field deterministic
    # short-circuit, no LLM call.
    r2 = _send("5", session_id)
    assert r2["error"] == "What address should this be delivered to?"

    # The address finally lands -- also resolved deterministically.
    # delivery_option must be re-derived fresh for Kasoa, not silently
    # keep the "kumasi_rider" that was never actually grounded in any
    # real address.
    r3 = _send("deliver to Kasoa", session_id)
    proposal = r3["proposal"]
    assert proposal["delivery_option"] == "accra_rider", (
        f"Expected delivery_option to be re-derived as accra_rider for Kasoa, got "
        f"{proposal['delivery_option']!r} -- a delivery_option that was never grounded in any "
        f"address carried over stale."
    )
    assert "doesn't match our usual" not in proposal["delivery_option_label"]


def test_straight_through_order_with_no_corrections_has_no_correction_note(monkeypatch):
    # Control case: a clean order with nothing changed mid-flow must
    # never carry a correction_note -- guards against the note firing
    # on ordinary "answering the next missing question" turns.
    monkeypatch.setattr(order_tool, "get_product_price", MagicMock(return_value=_priced("18k", 51000.0)))
    _mock_woocommerce(monkeypatch, order_id=9002)
    _mock_understand_customer(
        monkeypatch,
        {"tool": "propose_order", "arguments": {
            "product_name": "Gye Nyame White Necklace with Earrings, 30g", "material": "18k", "quantity": 2,
            "delivery_address": "Kumasi", "delivery_option": "unknown"}},
        {"tool": "confirm_order", "arguments": {}},
    )
    session_id = "golden-no-correction"

    r1 = _send("2 Gye Nyame White Necklace with Earrings, 30g in 18k, deliver to Kumasi", session_id)
    assert "correction_note" not in r1
    proposal = r1["proposal"]
    assert proposal["delivery_option"] == "kumasi_rider"
    assert proposal["delivery_option_label"] == "rider delivery within Kumasi"

    r2 = _send("confirm", session_id)
    assert "correction_note" not in r2
    assert r2["order_confirmation"]["order_id"] == 9002
