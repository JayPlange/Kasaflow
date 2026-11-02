"""
Optional, per-attempt metrics for the model call in llm.py's _call_llm().

Off by default. Nothing is recorded, logged or changed unless
LLM_METRICS_ENABLED is set to 1/true/yes/on, or a sink is registered
(the benchmark in scripts/bench/ does that). With it off, the only cost
is one env lookup per attempt, and _call_llm() behaves exactly as before.

What one record means (one record per HTTP attempt, not per customer
message, so a retried message produces several):

    latency_ms          Request sent -> complete response, for THIS attempt
                        only. Excludes the retry back-off sleeps, which
                        sit outside the attempt. Includes building the
                        prompt string (scripts/bench measures that cost).
    outcome             ok | timeout | connection_error | api_error |
                        other_error. "ok" means the API returned a
                        response. It does not mean the reply parsed as a
                        tool request, or that the business action worked.
    input_tokens        Tokens sent, as reported by the API.
    output_tokens       Tokens generated, as reported by the API.
    cached_tokens       Part of input_tokens the provider served from its
                        own prompt cache (None when not reported).
    reasoning_tokens    Part of output_tokens spent on hidden reasoning
                        (None when not reported).
    output_chars        Length of the reply text. 0 means the API call
                        succeeded but returned nothing usable.
    in_flight_at_start  Model calls running in this process when this one
                        started, counting itself.

This is non-streaming, so it cannot give TTFT. TTFT comes from the
separate streaming probe in scripts/bench/stream_probe.py.

Records never contain prompts, customer messages or model output.
Recording is best effort: nothing in here may raise into the request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable

from openai import APIConnectionError, APIError, APITimeoutError

logger = logging.getLogger(__name__)

ENV_FLAG = "LLM_METRICS_ENABLED"
_TRUE_VALUES = {"1", "true", "yes", "on"}

_lock = threading.Lock()
_in_flight = 0
_sinks: list[Callable[[dict], None]] = []


def _flag_on() -> bool:
    return os.getenv(ENV_FLAG, "").strip().lower() in _TRUE_VALUES


def is_enabled() -> bool:
    return _flag_on() or bool(_sinks)


def add_sink(sink: Callable[[dict], None]) -> None:
    """Receive every record as a dict. Called on the thread that made the call."""
    with _lock:
        _sinks.append(sink)


def remove_sink(sink: Callable[[dict], None]) -> None:
    with _lock:
        if sink in _sinks:
            _sinks.remove(sink)


def classify_error(exc: BaseException) -> str:
    # APITimeoutError subclasses APIConnectionError, so test it first.
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection_error"
    if isinstance(exc, APIError):
        return "api_error"
    return "other_error"


def _count(value: Any) -> int | None:
    # Only real integers. Test doubles (MagicMock etc.) become None.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def extract_usage(response: Any) -> dict:
    """Token counts from a Responses API result. Missing or odd -> None."""
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": _count(getattr(usage, "input_tokens", None)),
        "output_tokens": _count(getattr(usage, "output_tokens", None)),
        "cached_tokens": _count(getattr(input_details, "cached_tokens", None)),
        "reasoning_tokens": _count(getattr(output_details, "reasoning_tokens", None)),
    }


class _Attempt:
    """Context manager around one model call. Never suppresses exceptions."""

    def __init__(self, attempt: int, max_attempts: int, model: str) -> None:
        self._attempt = attempt
        self._max_attempts = max_attempts
        self._model = model
        self._active = False
        self._t0 = 0.0
        self._in_flight_at_start = 0
        self._response: Any = None

    def succeeded(self, response: Any) -> None:
        """Hand over the API response so its usage can be recorded."""
        self._response = response

    def __enter__(self) -> "_Attempt":
        global _in_flight
        if is_enabled():
            self._active = True
            with _lock:
                _in_flight += 1
                self._in_flight_at_start = _in_flight
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _in_flight
        if not self._active:
            return False
        elapsed = time.perf_counter() - self._t0
        with _lock:
            _in_flight -= 1
        try:
            self._record(elapsed, exc)
        except Exception:  # best effort: metrics must never break a request
            logger.debug("LLM metrics record failed", exc_info=True)
        return False

    def _record(self, elapsed: float, exc: BaseException | None) -> None:
        record: dict[str, Any] = {
            "event": "llm_attempt",
            "ts": round(time.time(), 3),
            "thread_id": threading.get_ident(),
            "model": self._model,
            "attempt": self._attempt,
            "max_attempts": self._max_attempts,
            "latency_ms": round(elapsed * 1000, 1),
            "in_flight_at_start": self._in_flight_at_start,
        }
        if exc is None:
            text = getattr(self._response, "output_text", None)
            record["outcome"] = "ok"
            record["output_chars"] = len(text) if isinstance(text, str) else None
            record.update(extract_usage(self._response))
        else:
            record["outcome"] = classify_error(exc)
            record["error_type"] = type(exc).__name__
            record["status_code"] = _count(getattr(exc, "status_code", None))
            record["output_chars"] = None
            record.update(extract_usage(None))

        if _flag_on():
            logger.info("KASAFLOW_LLM_METRICS %s", json.dumps(record, separators=(",", ":")))
        with _lock:
            sinks = list(_sinks)
        for sink in sinks:
            try:
                sink(record)
            except Exception:
                logger.debug("LLM metrics sink failed", exc_info=True)


def attempt(attempt_number: int, max_attempts: int, model: str) -> _Attempt:
    return _Attempt(attempt_number, max_attempts, model)
