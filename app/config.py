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
    image_embeddings_path: Path
    llm_max_retries: int
    llm_timeout_seconds: float
    log_level: str
    app_api_key: str
    rate_limit_per_minute: int
    policies_path: Path
    embedding_model: str

    # Optional integrations below. Not required at startup like
    # openai_api_key/app_api_key are -- someone should be able to run and
    # test the core /process endpoint without WhatsApp, Khaya, or
    # WooCommerce credentials configured. Each module that actually needs
    # one of these validates it's present at the point of use, not here.
    woocommerce_url: str | None
    woocommerce_consumer_key: str | None
    woocommerce_consumer_secret: str | None

    # Deliberately separate from the sync key above. The sync key is
    # read-only and runs unattended on a schedule (see
    # services/woocommerce_sync.py) -- giving it write access just to
    # support order creation would mean a bug in an unattended cron job
    # could create/modify real orders. This key is scoped Read/Write in
    # WooCommerce admin and is only ever used from the live, per-request
    # order path in services/order_tool.py.
    woocommerce_orders_consumer_key: str | None
    woocommerce_orders_consumer_secret: str | None

    khaya_api_key: str | None
    khaya_api_base: str

    whatsapp_verify_token: str | None
    whatsapp_access_token: str | None
    whatsapp_phone_number_id: str | None

    # The rider coordinator's WhatsApp number -- delivery isn't priced or
    # scheduled automatically (see services/delivery_tool.py's module
    # docstring), so once a customer confirms an order, a human needs to
    # actually be told about it to arrange the rider/shipping. Optional,
    # like the integrations above: confirm_order() logs a warning and
    # still completes the order if this isn't set, rather than blocking
    # a real sale on a notification channel not being configured yet.
    staff_notification_phone: str | None

    # Cohere, not OpenAI -- OpenAI has no hosted image-embedding
    # endpoint (confirmed against their API reference, 2026-08-18);
    # text-embedding-3-* is text-only. Optional like the other
    # integrations above: identify_product_from_photo() falls back to
    # the ordinary text pipeline when this isn't configured, a photo
    # customers send just won't get exact-item identification without
    # it. See services/image_embed_tool.py.
    cohere_api_key: str | None

    # Optional, like cohere_api_key above: services/geocoding_tool.py
    # falls back to delivery_tool.delivery_option_matches_address()'s
    # offline curated-city-name heuristic when this isn't configured,
    # rather than blocking order flow on a geocoding vendor. See
    # geocoding_tool.py's module docstring for why a real geocoder
    # (rather than a hardcoded city list) is needed at all: customers
    # give neighbourhood-level addresses ("East Legon", "Suame",
    # "Mankessim"), not region names, and no fixed list can cover those.
    google_maps_api_key: str | None


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
        image_embeddings_path=Path(
            os.getenv("IMAGE_EMBEDDINGS_PATH", str(BASE_DIR / "data" / "image_embeddings.json"))
        ),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        app_api_key=app_api_key,
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20")),
        policies_path=Path(
            os.getenv("POLICIES_PATH", str(BASE_DIR / "data" / "policies.json"))
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        woocommerce_url=os.getenv("WOOCOMMERCE_URL"),
        woocommerce_consumer_key=os.getenv("WOOCOMMERCE_CONSUMER_KEY"),
        woocommerce_consumer_secret=os.getenv("WOOCOMMERCE_CONSUMER_SECRET"),
        woocommerce_orders_consumer_key=os.getenv("WOOCOMMERCE_ORDERS_CONSUMER_KEY"),
        woocommerce_orders_consumer_secret=os.getenv("WOOCOMMERCE_ORDERS_CONSUMER_SECRET"),
        khaya_api_key=os.getenv("KHAYA_API_KEY"),
        khaya_api_base=os.getenv("KHAYA_API_BASE", "https://translation-api.ghananlp.org"),
        whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN"),
        whatsapp_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN"),
        whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID"),
        staff_notification_phone=os.getenv("STAFF_NOTIFICATION_PHONE"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY"),
    )


# Loaded once at import time, deliberately. If the config is broken (e.g.
# missing API key), we want the app to fail immediately on startup, not
# on the first customer request.
settings = load_settings()
