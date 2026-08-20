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
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "unknown", "quantity": 7,
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "14k", "quantity": "unknown",
            "delivery_address": "unknown", "delivery_option": "unknown"}},
        {"tool": "propose_order", "arguments": {
            "product_name": "unknown", "material": "unknown", "quantity": "unknown",
            "delivery_address": "East Legon", "delivery_option": "unknown"}},
        {"tool": "confirm_order", "arguments": {}},
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

    # The delivery inference: "East Legon" alone must resolve straight
    # to accra_rider, no "Would you like rider delivery within Accra,
    # ..." menu question (the exact bug this transcript surfaced).
    r8 = _send("east legon", session_id)
    proposal = r8["proposal"]
    assert proposal["material"] == "14k"
    assert proposal["quantity"] == 7
    assert proposal["delivery_address"] == "East Legon"
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
