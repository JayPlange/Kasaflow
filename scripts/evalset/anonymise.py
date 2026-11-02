"""
Turn raw turn-trace logs into anonymised eval conversations.

    python -m scripts.evalset.anonymise LOGFILE [LOGFILE ...] --source pilot

Writes:
  evalset/conversations.jsonl       one conversation per line, anonymised
  evalset/anonymisation_review.md   every redaction made, plus every word
                                    left in customer text that looks like
                                    a name or place, for a person to check

What is removed or replaced:
  - session ids (the customer's WhatsApp number) -> a salted hash
  - phone numbers, email addresses, Ghana Post GPS addresses
  - house, plot and flat numbers
  - names introduced as "my name is X" / "this is X" / "I'm X"
  - in delivery addresses: every word that isn't a known Ghanaian place
    name from services/delivery_tool.py. "No 12 Ofankor Rd, East Legon"
    becomes "[ADDRESS] East Legon", so replays still resolve the same
    delivery zone

What is kept: product names, karats, quantities, prices and order
numbers, which describe the shop's catalogue and orders, not the
customer. The review file exists because pattern-based redaction will
miss things (a name given without an introduction, a landmark); nothing
should be committed until someone has read it.

The salt for session hashing comes from EVALSET_SALT (or a random one,
printed, if unset). Keep it private: with the salt, a known phone number
can be checked against the hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
from pathlib import Path

from scripts.evalset.traces import (
    extract_records,
    group_conversations,
    group_turns,
    model_output_of,
    read_log_text,
)
from services.delivery_tool import _ACCRA_PLACE_NAMES, _GHANA_PLACE_NAMES, _KUMASI_PLACE_NAMES

ROOT = Path(__file__).resolve().parents[2]
EVALSET_DIR = ROOT / "evalset"

KNOWN_PLACES = {p.lower() for p in (_GHANA_PLACE_NAMES | _ACCRA_PLACE_NAMES | _KUMASI_PLACE_NAMES)}

_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    ("gps_address", re.compile(r"\b[A-Z]{2}-?\d{3,4}-?\d{3,4}\b"), "[GPS_ADDRESS]"),
    ("phone", re.compile(r"(?<![\w.])(\+?233|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}\b"), "[PHONE]"),
    ("long_number", re.compile(r"(?<![\w.])\d{9,}\b"), "[NUMBER]"),
    # "no." only with the full stop: "no 5 please" is a quantity, not an address.
    ("house_number", re.compile(r"\b(house|hse|plot|flat|apt|apartment)\s*(no\.?\s*)?#?\s*\d+[a-z]?\b|\bno\.\s*\d+[a-z]?\b",
                                re.IGNORECASE), "[HOUSE_NO]"),
    ("introduced_name", re.compile(
        r"\b(my name is|my name's|this is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"),
     None),  # replacement keeps the introduction, see _scrub_text
)

# Fields whose values are free-text addresses anywhere in the trace.
_ADDRESS_KEYS = {"delivery_address", "address", "shipping_address"}
# Fields that hold customer-facing or customer-written free text.
_TEXT_KEYS = {"customer_message", "reply", "conversation_reply", "detail", "message"}
# Words that look like names in the review scan but are catalogue or chat vocabulary.
_REVIEW_STOPWORDS = {
    "i", "hi", "hey", "yes", "no", "ok", "okay", "confirm", "please", "thanks", "gold", "silver",
    "ring", "rings", "chain", "chains", "necklace", "earrings", "bracelet", "gye", "nyame", "white",
    "big", "crown", "stone", "custom", "leaf", "rider", "delivery", "the", "a", "an", "what", "how",
    "can", "do", "is", "actually", "deliver", "to", "and", "with", "in", "of",
}


_CATALOGUE_WORDS: set[str] | None = None


def _catalogue_words() -> set[str]:
    """Words used in product names ("this is Gye Nyame" is a product, not a name)."""
    global _CATALOGUE_WORDS
    if _CATALOGUE_WORDS is None:
        try:
            from app.config import settings

            products = json.loads(Path(settings.products_path).read_text(encoding="utf-8"))
            _CATALOGUE_WORDS = {w.lower() for p in products for w in re.findall(r"[A-Za-z]+", p.get("product", ""))}
        except Exception:
            _CATALOGUE_WORDS = set()
    return _CATALOGUE_WORDS


class Redactions:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.review_words: dict[str, int] = {}

    def add(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1


def _scrub_text(text: str, log: Redactions) -> str:
    for kind, pattern, replacement in _PATTERNS:
        if kind == "introduced_name":
            def _name(match: re.Match) -> str:
                first_word = match.group(2).split()[0].lower()
                if first_word in KNOWN_PLACES or first_word in _catalogue_words():
                    return match.group(0)
                log.add(kind)
                return f"{match.group(1)} [NAME]"
            text = pattern.sub(_name, text)
            continue
        text, n = pattern.subn(replacement, text)
        for _ in range(n):
            log.add(kind)
    return text


def _scrub_address(text: str, log: Redactions) -> str:
    """Keep known place names only; collapse everything else to [ADDRESS]."""
    if not text or text.strip().lower() == "unknown":
        return text
    lowered = text.lower()
    kept = []
    for place in sorted(KNOWN_PLACES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(place)}\b", lowered) and not any(place in k for k in kept):
            kept.append(place)
    # Present places in the order they appear in the original text.
    kept.sort(key=lambda p: lowered.find(p))
    places = " ".join(p.title() for p in kept)
    residue = lowered
    for place in kept:
        residue = residue.replace(place, " ")
    # Filler the model sometimes copies into the address field
    # ("deliver to Kasoa") identifies nobody.
    residue = re.sub(r"\b(deliver|delivery|send|ship|it|to|at|in|around|near|please|pls|the|area|town|city|ghana)\b",
                     " ", residue)
    residue = re.sub(r"[\s,.;:-]+", "", residue)
    if residue:
        log.add("address_detail")
        return f"[ADDRESS] {places}".strip()
    return places


def _scrub(value, log: Redactions, key: str | None = None):
    if isinstance(value, dict):
        return {k: _scrub(v, log, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, log, key) for v in value]
    if isinstance(value, str):
        if key in _ADDRESS_KEYS:
            return _scrub_address(value, log)
        return _scrub_text(value, log)
    return value


def _review_scan(text: str, log: Redactions) -> None:
    """Collect capitalised words left in customer text that aren't known
    vocabulary: candidates for names and landmarks a person should check."""
    for word in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        lowered = word.lower()
        if lowered in KNOWN_PLACES or lowered in _REVIEW_STOPWORDS:
            continue
        log.review_words[word] = log.review_words.get(word, 0) + 1


def session_hash(session_id: str, salt: str) -> str:
    return "conv-" + hashlib.sha256(f"{salt}:{session_id}".encode()).hexdigest()[:10]


def build_conversations(paths: list[Path], salt: str, source: str) -> tuple[list[dict], Redactions, dict]:
    log = Redactions()
    stats = {"files": len(paths), "parse_failures": 0, "turns": 0}
    all_records: list[dict] = []
    for path in paths:
        records, failures = extract_records(read_log_text(path))
        all_records.extend(records)
        stats["parse_failures"] += failures

    conversations = []
    for session_id, turns in group_conversations(group_turns(all_records)).items():
        out_turns = []
        for turn in turns:
            outputs = [o for o in (model_output_of(t) for t in turn.traces) if o is not None]
            if len(outputs) > 1:
                recorded_model_output = {"requests": outputs}
            elif outputs:
                recorded_model_output = outputs[0]
            else:
                recorded_model_output = None
            final = turn.traces[-1].get("final_result")
            out_turn = {
                "customer_message": _scrub(turn.customer_message, log, "customer_message"),
                "recorded": {
                    "model_called": recorded_model_output is not None,
                    "model_output": _scrub(recorded_model_output, log),
                    "result_kind": next(iter(final), None) if isinstance(final, dict) else None,
                    "final_result": _scrub(final, log),
                    "metrics": _scrub(turn.metrics, log) if turn.metrics else None,
                },
            }
            out_turn["recorded"]["metrics"] = _drop_session(out_turn["recorded"]["metrics"])
            _review_scan(out_turn["customer_message"], log)
            out_turns.append(out_turn)
            stats["turns"] += 1
        conversations.append({
            "id": session_hash(session_id, salt),
            "source": source,
            "turns": out_turns,
        })
    return conversations, log, stats


def _drop_session(metrics: dict | None) -> dict | None:
    if not metrics:
        return metrics
    return {k: v for k, v in metrics.items() if k != "session_id"}


def merge_into(existing_path: Path, conversations: list[dict]) -> tuple[int, int]:
    """Add new conversations to the JSONL file; never overwrite existing ids."""
    existing: dict[str, dict] = {}
    if existing_path.exists():
        for line in existing_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                conv = json.loads(line)
                existing[conv["id"]] = conv
    added = 0
    for conv in conversations:
        if conv["id"] not in existing:
            existing[conv["id"]] = conv
            added += 1
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in existing.values()), encoding="utf-8",
    )
    return added, len(existing)


def write_review(path: Path, log: Redactions, stats: dict, source: str) -> None:
    lines = [
        "# Anonymisation review",
        "",
        f"Source label: `{source}`. Files: {stats['files']}. Turns: {stats['turns']}. "
        f"Trace lines that failed to parse: {stats['parse_failures']}.",
        "",
        "## Redactions made",
        "",
    ]
    if log.counts:
        lines += [f"- {kind}: {n}" for kind, n in sorted(log.counts.items())]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Check by hand before committing",
        "",
        "Capitalised words left in customer messages that are not known places or shop vocabulary. "
        "Any that are a person's name, a landmark near someone's home or a business name should be "
        "edited out of evalset/conversations.jsonl by hand.",
        "",
    ]
    if log.review_words:
        lines += [f"- {w} ({n})" for w, n in sorted(log.review_words.items(), key=lambda x: (-x[1], x[0]))]
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--source", default="pilot",
                        help="Where these conversations came from, e.g. pilot or operator_test.")
    parser.add_argument("--out-dir", type=Path, default=EVALSET_DIR)
    args = parser.parse_args(argv)

    salt = os.getenv("EVALSET_SALT")
    if not salt:
        salt = secrets.token_hex(16)
        print(f"EVALSET_SALT not set; using a random salt for this run: {salt}", file=sys.stderr)
        print("Set EVALSET_SALT to this value to keep conversation ids stable across runs.", file=sys.stderr)

    conversations, log, stats = build_conversations(args.logs, salt, args.source)
    added, total = merge_into(args.out_dir / "conversations.jsonl", conversations)
    write_review(args.out_dir / "anonymisation_review.md", log, stats, args.source)
    print(f"{stats['turns']} turns in {len(conversations)} conversations; {added} new, {total} in the eval set.")
    print(f"Review {args.out_dir / 'anonymisation_review.md'} before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
