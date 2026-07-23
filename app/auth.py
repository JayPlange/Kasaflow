"""
API key authentication for endpoints that trigger paid LLM calls.

Why this exists: /process was previously reachable by anyone who found
the URL, meaning anyone could trigger unlimited paid OpenAI calls with
no way to tell who was calling. This adds a single shared-secret check
via a request header.

Deliberately simple: one shared key, not per-customer accounts. That's
the right level of complexity for a single client's internal/webhook
traffic right now, not a public multi-tenant product. Revisit (real
per-client keys or OAuth) if/when multiple distinct customers need to
be told apart.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Depends(_api_key_header)) -> str:
    # secrets.compare_digest instead of `==` so a mistyped key can't be
    # brute-forced faster by measuring how long the comparison took.
    if not provided_key or not secrets.compare_digest(provided_key, settings.app_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return provided_key
