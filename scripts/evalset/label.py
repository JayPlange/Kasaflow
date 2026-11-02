"""
Create or extend evalset/labels.json, the ground truth for order accuracy.

    python -m scripts.evalset.label

For every conversation without a label, adds an entry with a SUGGESTED
label read from what happened in the original conversation, marked
"reviewed": false. Suggestions are only a starting point: a conversation
where the assistant got the order wrong would suggest the wrong order.
Scoring ignores unreviewed labels, so a person has to read each
conversation, correct the label and set "reviewed": true.

A label says what a correct assistant would have ended with:

    "expected_order": {"product": "...", "material": "18k", "quantity": 5}
        the customer clearly asked for this order and confirmed it
    "expected_order": null
        no order should be placed (browsing, abandoned, or not confirmed)
    "expect_handoff": true | false | null
        whether the conversation should end with a person taking over;
        null means don't score it

Existing labels are never changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALSET_DIR = ROOT / "evalset"


def suggest(conv: dict) -> dict:
    last_proposal = None
    confirmed = False
    handoff = False
    for turn in conv["turns"]:
        final = turn["recorded"].get("final_result") or {}
        if "proposal" in final:
            last_proposal = final["proposal"]
        if "order_confirmation" in final:
            confirmed = True
        if "human_handoff" in final or "order_needs_review" in final:
            handoff = True

    if confirmed and last_proposal:
        note = "Suggested: the original conversation placed this order."
        expected = {k: last_proposal.get(k) for k in ("product", "material", "quantity")}
    elif last_proposal:
        note = ("Suggested: a proposal was shown but never confirmed. Decide whether the customer "
                "meant to order; if so, fill in expected_order.")
        expected = None
    else:
        note = "Suggested: no proposal was reached. Decide whether the customer was trying to order."
        expected = None
    return {"reviewed": False, "expected_order": expected, "expect_handoff": handoff or None, "notes": note}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conversations", type=Path, default=EVALSET_DIR / "conversations.jsonl")
    parser.add_argument("--labels", type=Path, default=EVALSET_DIR / "labels.json")
    args = parser.parse_args(argv)

    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.exists() else {}
    added = 0
    for line in args.conversations.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        conv = json.loads(line)
        if conv["id"] not in labels:
            labels[conv["id"]] = suggest(conv)
            added += 1
    args.labels.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    unreviewed = sum(1 for v in labels.values() if not v.get("reviewed"))
    print(f"{added} new labels suggested; {unreviewed} of {len(labels)} still need review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
