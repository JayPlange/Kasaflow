#!/usr/bin/env python3
"""
Run the KasaFlow behavioural evaluator (scripts/evaluator/) against the
real system and report the results.

Requires a real OPENAI_API_KEY (and APP_API_KEY, same as running the
app itself -- see app/config.py) -- this drives actual LLM tool
selection for every turn, exactly like a real customer message. There
is no mock mode for a real scored run; that's the whole point.

Usage (from the repo root -- or set OPENAI_API_KEY/APP_API_KEY in a
.env file there, same as running the app itself; load_dotenv() picks
it up automatically):
    python3 scripts/run_evaluator.py
    python3 scripts/run_evaluator.py --category "REFERENCES"
    python3 scripts/run_evaluator.py --id references-04-golden-path-full-journey
    python3 scripts/run_evaluator.py --json report.json
    python3 scripts/run_evaluator.py --id ambiguity-01-bare-category-two-matches --repeat 6

--repeat N runs each selected scenario N times (fresh session each
time) and reports a pass count per scenario, to tell a genuine,
repeatable failure apart from one unlucky LLM call. Only sensible
alongside --id or --category, not against the whole corpus.

Exits non-zero if any scenario failed, so this is CI/pre-demo-gate
friendly if Webb ever wants it wired into GitHub Actions -- not done
here, since that decision (and what the pass bar should be) is his
call, not something to assume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as `python scripts/run_evaluator.py`, which only puts this folder
# on sys.path, not the repo root -- add it explicitly so `from
# scripts.evaluator...`/`from services...`/`from app...` all resolve,
# same pattern as this folder's existing manual_*_check.py scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from scripts.evaluator.report import print_repeat_summary, print_summary, to_json
from scripts.evaluator.runner import run_all, run_scenario
from scripts.evaluator.scenarios import SCENARIOS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="Only run scenarios in this category (exact match).")
    parser.add_argument("--id", dest="scenario_id", help="Only run the scenario with this id.")
    parser.add_argument("--json", dest="json_path", help="Write the full report as JSON to this path.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each selected scenario this many times (fresh session per run) and report a "
        "pass count, to check whether a failure is consistent or a one-off. Use with --id or "
        "--category, not the whole corpus.",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    if args.scenario_id:
        scenarios = [s for s in scenarios if s.id == args.scenario_id]

    if not scenarios:
        print("No scenarios matched --category/--id.", file=sys.stderr)
        return 2

    if args.repeat > 1:
        results_by_id = {s.id: [run_scenario(s) for _ in range(args.repeat)] for s in scenarios}
        print_repeat_summary(results_by_id)
        all_results = [r for results in results_by_id.values() for r in results]
        if args.json_path:
            with open(args.json_path, "w", encoding="utf-8") as f:
                f.write(to_json(all_results))
            print(f"Full report (every run, not just failures) written to {args.json_path}")
        return 0 if all(r.passed for r in all_results) else 1

    results = run_all(scenarios)
    print_summary(results)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            f.write(to_json(results))
        print(f"Full report written to {args.json_path}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
