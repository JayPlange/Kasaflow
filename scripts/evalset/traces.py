"""
Reading KASAFLOW_TURN_TRACE / KASAFLOW_TURN_METRICS lines out of logs.

Logs arrive in whatever shape the server's output was saved in: plain
JSON lines, docker compose output, or a PowerShell Tee-Object file
(UTF-16, every line hard-wrapped at the console width). The wrapping
breaks each JSON object across lines at spaces, so line breaks are
removed before decoding; the JSON itself never contains a raw newline
(json.dumps escapes them).

One customer message can produce several TURN_TRACE lines (a message
with several asks runs several tools) but exactly one TURN_METRICS line,
logged after them. Turns are rebuilt from that: consecutive traces for
the same session and message belong together, and the next metrics line
for that session closes the turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_MARKER_RE = re.compile(r"KASAFLOW_TURN_(TRACE|METRICS) ")


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text


def extract_records(text: str) -> tuple[list[dict], int]:
    """Return (records in log order, number of markers that failed to parse).

    Each record is {"kind": "trace"|"metrics", "data": {...}}.
    """
    joined = text.replace("\r", "").replace("\n", "")
    decoder = json.JSONDecoder()
    records: list[dict] = []
    failures = 0
    for match in _MARKER_RE.finditer(joined):
        try:
            data, _ = decoder.raw_decode(joined, match.end())
        except ValueError:
            failures += 1
            continue
        if isinstance(data, dict):
            records.append({"kind": match.group(1).lower(), "data": data})
    return records, failures


@dataclass
class Turn:
    session_id: str
    customer_message: str
    traces: list[dict] = field(default_factory=list)
    metrics: dict | None = None


def group_turns(records: list[dict]) -> list[Turn]:
    """Rebuild customer turns from trace and metrics records, in order."""
    turns: list[Turn] = []
    open_by_session: dict[str, Turn] = {}

    for record in records:
        data = record["data"]
        session = str(data.get("session_id", ""))
        if record["kind"] == "trace":
            message = data.get("customer_message") or ""
            current = open_by_session.get(session)
            if current is not None and current.customer_message == message and current.metrics is None:
                current.traces.append(data)
                continue
            turn = Turn(session_id=session, customer_message=message, traces=[data])
            turns.append(turn)
            open_by_session[session] = turn
        else:
            current = open_by_session.get(session)
            if current is not None and current.metrics is None:
                current.metrics = data
                open_by_session.pop(session, None)
    return turns


def group_conversations(turns: list[Turn]) -> dict[str, list[Turn]]:
    conversations: dict[str, list[Turn]] = {}
    for turn in turns:
        conversations.setdefault(turn.session_id, []).append(turn)
    return conversations


def model_output_of(trace: dict) -> dict | None:
    """The model's own tool choice for a trace, or None if the model wasn't called.

    Turns resolved by router.py's deterministic fast path or by a
    guardrail carry a "_source" or "guardrail:" marker instead.
    """
    output = trace.get("llm_structured_output")
    if not isinstance(output, dict) or "error" in output:
        return None
    if "_source" in output or str(output.get("tool", "")).startswith("guardrail:"):
        return None
    return output
