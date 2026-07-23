"""
Central configuration for KasaFlow.

Why this file exists:
Before, settings were read with os.getenv() scattered across llm.py and
product_tool.py. That means every new setting requires hunting through
the codebase, there's no single place to see what's configurable, and
there's no validation that required values (like the API key) are
actually present before the app starts handling traffic.

This module reads the environment once, validates it, and exposes a
single `settings` object that every other module imports from.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root, computed from this file's location so it works no matter
# what directory the app is launched from (Docker, systemd, your terminal).
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    products_path: Path
    llm_max_retries: int
    llm_timeout_seconds: float
    log_level: str
    app_api_key: str
    rate_limit_per_minute: int


def load_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before starting the app."
        )

    # Shared secret clients must send in the X-API-Key header to reach
    # /process. Required (not optional) for the same reason as the OpenAI
    # key above: fail loudly at startup, not by silently running an
    # unauthenticated endpoint that can trigger unlimited paid LLM calls.
    app_api_key = os.getenv("APP_API_KEY")
    if not app_api_key:
        raise RuntimeError(
            "APP_API_KEY is not set. Add it to your .env file before starting the app "
            "-- this is the shared secret clients must send in the X-API-Key header."
        )

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        products_path=Path(
            os.getenv("PRODUCTS_PATH", str(BASE_DIR / "data" / "products.json"))
        ),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        app_api_key=app_api_key,
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20")),
    )


# Loaded once at import time, deliberately. If the config is broken (e.g.
# missing API key), we want the app to fail immediately on startup, not
# on the first customer request.
settings = load_settings()
