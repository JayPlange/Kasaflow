"""
Replay the eval set through the real router.

    python -m scripts.evalset.run --mode recorded            # no model calls, free, deterministic
    python -m scripts.evalset.run --mode live --repeat 3     # real model, costs money

recorded: every turn where the model was called in the original
  conversation gets the model's original answer back. This isolates the
  application: if a code change breaks an order flow, it shows up here
  with no model variance and no cost. Latency in this mode is
  application time only, and cost is taken from the original turn's
  logged token counts when they exist.

live: the model is called for real, through the same code path as
  production. This is where order accuracy, cost per conversation and
  latency are actually measured. --repeat runs each conversation several
  times with fresh sessions so model variance shows up as disagreement
  between runs instead of hiding inside one lucky or unlucky sample.

Both modes run inside scripts/evalset/sandbox.py: orders go to an
in-memory store, staff messages are captured, geocoding is off and
embeddings are off unless allowed.

Writes eval_reports/run-<timestamp>-<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALSET_DIR = ROOT / "evalset"
REPORTS_DIR = ROOT / "eval_reports"


def _prepare_env(mode: str) -> None:
    # app.config refuses to start without these. In recorded mode no
    # model is called, so placeholders are enough; in live mode the real
    # key must come from the environment or .env.
    if mode == "recorded":
        os.environ.setdefault("OPENAI_API_KEY", "recorded-replay-makes-no-model-calls")
    os.environ.setdefault("APP_API_KEY", "evalset-runner")


def load_conversations(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RecordedModel:
    """Stands in for llm.understand_customer() during a recorded replay."""

    def __init__(self):
        self.turn: dict | None = None
        self.calls_this_turn = 0
        self.missing = False

    def start_turn(self, turn: dict) -> None:
        self.turn = turn
        self.calls_this_turn = 0
        self.missing = False

    def __call__(self, message, **kwargs):
        self.calls_this_turn += 1
        recorded = (self.turn or {}).get("recorded", {})
        output = recorded.get("model_output")
        if output is None:
            # The original turn never reached the model (fast path or
            # guardrail), but the current code wants a model answer.
            self.missing = True
            return {"tool": "converse", "arguments": {"reply": "[replay: no recorded model answer for this turn]"}}
        return json.loads(json.dumps(output))  # copy: the router mutates requests


def run_conversation(conv: dict, run_index: int, mode: str, sandbox, recorded_model) -> dict:
    from services import router
    from services.escalation import get_escalation
    from services.response_formatter import format_for_customer

    session_id = f"eval-{conv['id']}-{run_index}"
    orders_before = set(sandbox.store.orders)
    staff_before = len(sandbox.staff_messages)
    turns_out = []

    for index, turn in enumerate(conv["turns"]):
        if recorded_model is not None:
            recorded_model.start_turn(turn)
        embeddings_before = sandbox.embedding_attempts
        started = time.perf_counter()
        error = None
        try:
            result = router.route_customer(turn["customer_message"], session_id)
            reply = format_for_customer(result)
        except Exception as e:  # a crash is a result worth recording, not a reason to stop
            result, reply, error = None, None, f"{type(e).__name__}: {e}"
        wall_ms = round((time.perf_counter() - started) * 1000, 1)
        metrics = router.get_last_turn_metrics() or {}

        record = {
            "index": index,
            "customer_message": turn["customer_message"],
            "reply": reply,
            "result_kind": next(iter(result), None) if isinstance(result, dict) else None,
            "error": error,
            "turn_latency_ms": metrics.get("turn_latency_ms", wall_ms),
            "llm_calls": metrics.get("llm_calls", []),
            "guardrail_events": metrics.get("guardrail_events", []),
            "embeddings_blocked": sandbox.embedding_attempts > embeddings_before,
        }
        if recorded_model is not None:
            recorded = turn.get("recorded", {})
            record["replay"] = {
                "model_called_originally": recorded.get("model_called", False),
                "model_called_now": recorded_model.calls_this_turn > 0,
                "missing_model_answer": recorded_model.missing,
                "original_result_kind": recorded.get("result_kind"),
                "original_llm_calls": (recorded.get("metrics") or {}).get("llm_calls", []),
            }
        turns_out.append(record)

    created = [sandbox.store.orders[i] for i in sorted(set(sandbox.store.orders) - orders_before)]
    escalation_state = get_escalation(session_id)
    return {
        "conversation_id": conv["id"],
        "source": conv.get("source"),
        "run": run_index,
        "turns": turns_out,
        "orders": [
            {
                "order_id": o["id"],
                "product_id": item.get("product_id"),
                "variation_id": item.get("variation_id"),
                "product": item.get("name"),
                "quantity": item.get("quantity"),
                "material": next((m["value"] for m in o.get("meta_data", []) if m.get("key") == "kasaflow_material"), None),
                "store_total": o.get("total"),
            }
            for o in created for item in o.get("line_items", [])
        ],
        "escalation": {"reason": escalation_state["reason"]} if escalation_state else None,
        "staff_messages": len(sandbox.staff_messages) - staff_before,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["recorded", "live"], required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--conversations", type=Path, default=EVALSET_DIR / "conversations.jsonl")
    parser.add_argument("--only", nargs="*", help="Conversation ids to run (default: all).")
    parser.add_argument("--allow-embeddings", action="store_true",
                        help="Allow OpenAI embedding calls (product search fallback, policy answers). "
                             "Always on in live mode.")
    parser.add_argument("--allow-geocoding", action="store_true", help="Allow Google geocoding calls.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "live" and args.repeat < 1:
        parser.error("--repeat must be at least 1")

    _prepare_env(args.mode)
    sys.path.insert(0, str(ROOT))

    from app.config import settings
    from services import router
    from scripts.evalset.sandbox import Sandbox

    conversations = load_conversations(args.conversations)
    if args.only:
        conversations = [c for c in conversations if c["id"] in set(args.only)]
    if not conversations:
        print("No conversations to run.", file=sys.stderr)
        return 1

    recorded_model = None
    original_understand = router.understand_customer
    if args.mode == "recorded":
        recorded_model = RecordedModel()
        router.understand_customer = recorded_model

    allow_embeddings = args.allow_embeddings or args.mode == "live"
    runs = []
    try:
        with Sandbox(allow_embeddings=allow_embeddings, allow_geocoding=args.allow_geocoding) as sandbox:
            for conv in conversations:
                for run_index in range(args.repeat):
                    runs.append(run_conversation(conv, run_index, args.mode, sandbox, recorded_model))
                    print(f"{conv['id']} run {run_index + 1}/{args.repeat} done", file=sys.stderr)
    finally:
        router.understand_customer = original_understand

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or REPORTS_DIR / f"run-{stamp}-{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mode": args.mode,
        "model": settings.openai_model if args.mode == "live" else None,
        "repeat": args.repeat,
        "started_utc": stamp,
        "allow_embeddings": allow_embeddings,
        "allow_geocoding": args.allow_geocoding,
        "conversation_count": len(conversations),
        "runs": runs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
