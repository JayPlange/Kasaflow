"""
One place to configure logging for the whole app.

Why: `print()` statements can't be filtered by severity, can't be
redirected to a log aggregator, and don't include timestamps or the
module they came from. Swapping to `logging` now costs nothing and
means every log line is filterable and shippable to something like
CloudWatch/Datadog later without touching call sites.
"""

import logging

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Quiet down noisy third-party loggers unless we're debugging.
    if settings.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
