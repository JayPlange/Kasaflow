"""
Integration tests for POST /webhook/whatsapp's signature verification.

SECURITY (2026-09-05): before this, the endpoint processed any POST body
that merely parsed as JSON in the expected shape -- no check it actually
came from Meta. These tests go through the real FastAPI app (TestClient),
like test_integration.py, because the thing being verified is the HTTP
boundary itself: raw body bytes in, header checked, 403 or 200 out. A
unit test calling _signature_is_valid() directly wouldn't prove the route
actually wires it in before touching the payload.

Never uses the real WHATSAPP_APP_SECRET from .env -- monkeypatches a
known value on app.whatsapp_routes.settings so the expected signature can
be computed independently in the test itself.
"""

import hashlib
import hmac
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from app import main, whatsapp_routes

client = TestClient(main.app)

_TEST_SECRET = "test-app-secret-for-signature-verification"


def _sign(body: bytes, secret: str = _TEST_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _no_op_payload() -> bytes:
    # Shape doesn't matter for these tests -- entry/changes/value present
    # but no "messages" key resolves to an empty list, so the loop that
    # would schedule _handle_message() as a BackgroundTask never fires and
    # nothing downstream needs mocking. Confirms signature verification
    # runs before any payload-shape handling, not what that handling does.
    return json.dumps({"entry": [{"changes": [{"value": {}}]}]}).encode("utf-8")


def test_valid_signature_is_accepted(monkeypatch):
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_app_secret=_TEST_SECRET))
    body = _no_op_payload()

    response = client.post(
        "/webhook/whatsapp",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body)},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "received"}


def test_missing_signature_header_is_rejected(monkeypatch):
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_app_secret=_TEST_SECRET))
    body = _no_op_payload()

    response = client.post("/webhook/whatsapp", content=body)

    assert response.status_code == 403


def test_wrong_signature_is_rejected(monkeypatch):
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_app_secret=_TEST_SECRET))
    body = _no_op_payload()

    response = client.post(
        "/webhook/whatsapp",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body, secret="wrong-secret")},
    )

    assert response.status_code == 403


def test_signature_computed_over_a_different_body_is_rejected(monkeypatch):
    # A forged request can't just replay a signature it saw on some other
    # (or earlier, tampered) payload -- the signature must match THIS
    # exact body.
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_app_secret=_TEST_SECRET))
    real_body = _no_op_payload()
    tampered_body = json.dumps({"entry": [{"changes": [{"value": {"tampered": True}}]}]}).encode("utf-8")

    response = client.post(
        "/webhook/whatsapp",
        content=tampered_body,
        headers={"X-Hub-Signature-256": _sign(real_body)},
    )

    assert response.status_code == 403


def test_no_app_secret_configured_fails_closed(monkeypatch):
    # If WHATSAPP_APP_SECRET somehow isn't set at request time (shouldn't
    # happen given load_settings()'s startup check, but this is the
    # actual runtime behaviour if it did), every request must be
    # refused, not silently accepted because there's nothing to compare.
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_app_secret=None))
    body = _no_op_payload()

    response = client.post(
        "/webhook/whatsapp",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body)},
    )

    assert response.status_code == 403


def test_get_verification_handshake_is_unaffected(monkeypatch):
    # The GET verification handshake is a separate, pre-existing check
    # (hub.verify_token) -- confirm this fix didn't touch it.
    monkeypatch.setattr(whatsapp_routes, "settings", replace(whatsapp_routes.settings, whatsapp_verify_token="expected-token"))

    response = client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "expected-token", "hub.challenge": "12345"},
    )

    assert response.status_code == 200
    assert response.text == "12345"
