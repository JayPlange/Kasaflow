"""
Streaming timing probe for the model call. Benchmark only: nothing in the
app imports this, and it never touches the production request path.

It sends the same prompt KasaFlow builds, with stream=True, and timestamps
each event so the phases of one request can be told apart:

    dispatch      Taken immediately before client.responses.create() is
                  called. Includes SDK request building and, on the first
                  request on a connection, TCP/TLS set-up. Compare the
                  first requests with later ones to see that cost.
    first_event   First stream event of any kind (metadata, not text).
    created       The response.created event.
    TTFT          Dispatch -> first response.output_text.delta event whose
                  text is non-empty. The first ACTUAL generated text, not
                  the first event.
    decode        First text delta -> last text delta.
    total         Dispatch -> the terminal event (response.completed).
    tokens/sec    output_tokens / decode seconds. The API reports tokens
                  only as a total, and the first token arrives at TTFT, so
                  output_tps_decode_excl_first = (output_tokens - 1) /
                  decode seconds is also given. With ~40 output tokens
                  the two differ by about 2 percent.

Limits to keep in mind. A stream chunk can carry more than one token, so
n_text_events (chunks) is not the token count, and decode timing has the
resolution of the chunks. KasaFlow's reply is a short JSON tool call, so
decode is a few milliseconds to a few hundred, and tokens/sec from it is
noisy. Usage comes from the response.completed event.

Run one request and print the raw event timeline:
    python3 scripts/run_llm_baseline.py --probe-one --show-events
"""

from __future__ import annotations

import threading
import time
from typing import Any

from services.llm_metrics import classify_error, extract_usage

TEXT_DELTA = "response.output_text.delta"
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


class Gauge:
    """Counts probe requests in flight (same meaning as llm_metrics' counter)."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def enter(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def leave(self) -> None:
        with self._lock:
            self._n -= 1


def probe_once(
    client: Any,
    model: str,
    prompt: str,
    *,
    timeout: float,
    gauge: Gauge | None = None,
    keep_events: bool = False,
) -> dict:
    in_flight = gauge.enter() if gauge else None
    events: list[dict] = []
    t_first_event = t_created = t_first_text = t_last_text = t_terminal = None
    n_text_events = 0
    text_chars = 0
    usage = extract_usage(None)
    outcome = "ok"
    error_type = None
    terminal_type = None
    stream = None

    wall_ts = time.time()
    t_dispatch = time.perf_counter()
    try:
        stream = client.responses.create(model=model, input=prompt, stream=True, timeout=timeout)
        for event in stream:
            now = time.perf_counter()
            etype = getattr(event, "type", "unknown")
            if keep_events:
                events.append({"type": etype, "t_ms": round((now - t_dispatch) * 1000, 1)})
            if t_first_event is None:
                t_first_event = now
            if etype == "response.created":
                if t_created is None:
                    t_created = now
            elif etype == TEXT_DELTA:
                delta = getattr(event, "delta", "")
                if isinstance(delta, str) and delta:
                    if t_first_text is None:
                        t_first_text = now
                    t_last_text = now
                    n_text_events += 1
                    text_chars += len(delta)
            elif etype in TERMINAL_EVENTS:
                t_terminal = now
                terminal_type = etype
                usage = extract_usage(getattr(event, "response", None))
                if etype != "response.completed":
                    outcome = etype.rsplit(".", 1)[-1]
            elif etype == "error":
                outcome = "stream_error"
    except Exception as exc:
        outcome = classify_error(exc)
        error_type = type(exc).__name__
    finally:
        t_end = time.perf_counter()
        if gauge:
            gauge.leave()
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if outcome == "ok" and t_first_text is None:
        outcome = "no_text"

    def ms(t: float | None) -> float | None:
        return None if t is None else round((t - t_dispatch) * 1000, 1)

    decode_s = (t_last_text - t_first_text) if (t_first_text and t_last_text) else None
    out = usage["output_tokens"]
    tps = tps_excl = None
    if decode_s and decode_s > 0 and out is not None:
        tps = round(out / decode_s, 1)
        if out > 1:
            tps_excl = round((out - 1) / decode_s, 1)

    record = {
        "ts": round(wall_ts, 3),
        "outcome": outcome,
        "error_type": error_type,
        "in_flight_at_start": in_flight,
        "ttft_ms": ms(t_first_text),
        "first_event_ms": ms(t_first_event),
        "created_ms": ms(t_created),
        "decode_ms": None if decode_s is None else round(decode_s * 1000, 1),
        "total_ms": ms(t_terminal if t_terminal is not None else t_end),
        "n_text_events": n_text_events,
        "text_chars": text_chars,
        "terminal_event": terminal_type,
        "output_tps_decode": tps,
        "output_tps_decode_excl_first": tps_excl,
        **usage,
    }
    if keep_events:
        record["events"] = events
    return record
