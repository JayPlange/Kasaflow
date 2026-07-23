"""
Integration tests for app/main.py

Different from the unit tests: these go through the real FastAPI app
via TestClient, meaning real HTTP request parsing, real Pydantic
validation, and the real exception handler -- not just a Python
function called directly. We're checking the pieces snap together
correctly end-to-end.

We still never call the real OpenAI API here -- that's what prompt
regression tests are for. We mock at the router level, one layer below
the HTTP boundary, so this test proves "does a request in produce the
right response out," without paying for or depending on the network.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import main
from app.config import settings
from services.llm import ToolSelectionError

client = TestClient(main.app)

# /process is authenticated, so every call that should reach route_customer
# needs the real shared key from this machine's .env. Tests never hardcode
# or print the key itself, only reference settings.app_api_key.
AUTH_HEADERS = {"X-API-Key": settings.app_api_key}


def test_home_endpoint():
    # Arrange: nothing to set up -- "/" is intentionally unauthenticated
    # so uptime checks / Docker's HEALTHCHECK don't need a key

    # Act
    response = client.get("/")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "KasaFlow is running"


def test_process_rejects_missing_api_key():
    # Arrange: no X-API-Key header at all

    # Act
    response = client.post("/process", json={"message": "how much is a gold ring?"})

    # Assert: rejected before Pydantic validation or routing ever runs
    assert response.status_code == 401


def test_process_rejects_wrong_api_key():
    # Arrange

    # Act
    response = client.post(
        "/process",
        json={"message": "how much is a gold ring?"},
        headers={"X-API-Key": "definitely-not-the-real-key"},
    )

    # Assert
    assert response.status_code == 401


def test_process_rejects_empty_message():
    # Arrange: no mocking -- this should be rejected by Pydantic before
    # route_customer is ever called

    # Act
    response = client.post("/process", json={"message": ""}, headers=AUTH_HEADERS)

    # Assert: FastAPI's automatic validation kicks in
    assert response.status_code == 422


def test_process_rejects_missing_message_field():
    # Arrange

    # Act
    response = client.post("/process", json={}, headers=AUTH_HEADERS)

    # Assert
    assert response.status_code == 422


def test_process_happy_path_end_to_end(monkeypatch):
    # Arrange: mock one layer below the HTTP boundary so this test
    # exercises real request parsing + real routing wiring, without a
    # real OpenAI call
    monkeypatch.setattr(
        main,
        "route_customer",
        MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200}),
    )

    # Act
    response = client.post(
        "/process", json={"message": "how much is a gold ring?"}, headers=AUTH_HEADERS
    )

    # Assert: the tool result comes through untouched, plus a session_id
    # every response carries so a client can continue the conversation
    body = response.json()
    assert response.status_code == 200
    assert body["product"] == "ring"
    assert body["material"] == "gold"
    assert body["price"] == 1200
    assert "session_id" in body and body["session_id"]


def test_process_reuses_the_session_id_the_caller_sends(monkeypatch):
    # Arrange
    mock_route = MagicMock(return_value={"product": "ring", "material": "gold", "price": 1200})
    monkeypatch.setattr(main, "route_customer", mock_route)

    # Act: caller supplies its own session_id from a previous response
    response = client.post(
        "/process",
        json={"message": "how much is that ring again?", "session_id": "existing-session-123"},
        headers=AUTH_HEADERS,
    )

    # Assert: the same session_id is echoed back, not replaced with a new one
    assert response.json()["session_id"] == "existing-session-123"
    mock_route.assert_called_once_with("how much is that ring again?", "existing-session-123")


def test_process_returns_200_with_friendly_error_when_llm_cant_understand(monkeypatch):
    # Arrange: route_customer already catches ToolSelectionError internally
    # and returns a friendly dict rather than raising -- confirm that
    # behavior survives the trip through the real HTTP layer
    monkeypatch.setattr(
        main,
        "route_customer",
        MagicMock(return_value={"error": "I couldn't understand that request. Could you rephrase it?"}),
    )

    # Act
    response = client.post(
        "/process", json={"message": "asdkjaslkdj"}, headers=AUTH_HEADERS
    )

    # Assert: this is a "handled" failure, so it's still a 200 with an
    # error payload, not a 500
    assert response.status_code == 200
    assert "error" in response.json()


def test_process_returns_500_for_truly_unexpected_failures(monkeypatch):
    # Arrange: simulate a genuine bug -- something route_customer's own
    # error handling didn't anticipate
    monkeypatch.setattr(
        main, "route_customer", MagicMock(side_effect=RuntimeError("something nobody expected"))
    )

    # Act
    response = client.post(
        "/process", json={"message": "how much is a gold ring?"}, headers=AUTH_HEADERS
    )

    # Assert: caught by main.py's catch-all, returns a generic message --
    # never leaks the raw exception text to the customer
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
