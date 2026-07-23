"""
Unit tests for the API key dependency in app/auth.py.

Uses a lightweight stand-in for Settings rather than the real settings
object, so these tests never depend on (or need to know) whatever
APP_API_KEY happens to be set in the developer's own .env.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import auth


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setattr(auth, "settings", SimpleNamespace(app_api_key="test-secret-key"))


def test_verify_api_key_accepts_correct_key():
    assert auth.verify_api_key(provided_key="test-secret-key") == "test-secret-key"


def test_verify_api_key_rejects_wrong_key():
    with pytest.raises(HTTPException) as exc_info:
        auth.verify_api_key(provided_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_verify_api_key_rejects_missing_key():
    with pytest.raises(HTTPException) as exc_info:
        auth.verify_api_key(provided_key=None)
    assert exc_info.value.status_code == 401


def test_verify_api_key_rejects_empty_string_key():
    # Guards against a client sending the header with an empty value
    # rather than omitting it, which would otherwise skip the "missing"
    # check but should still be rejected.
    with pytest.raises(HTTPException) as exc_info:
        auth.verify_api_key(provided_key="")
    assert exc_info.value.status_code == 401
