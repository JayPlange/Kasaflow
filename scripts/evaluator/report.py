"""
Turns a list of ScenarioResult into the two things Webb actually asked
for: a scannable console summary (pass/fail by category), and a full
JSON export carrying every field his spec named (customer message,
expected intent/product/fields/tool/business result, actual LLM output,
actual tool, actual final response, pass/fail) plus each turn's
manner_note for a human reviewer to score -- this file does not invent
an automated manner verdict, see schema.py's Turn docstring for why.
"""

from __future__ import annotations

import json
from collections import defaultdict

from scripts.evaluator.schema import ScenarioResult


def to_json(results: list[ScenarioResult]) -> str:
    payload = []
    for result in results:
        scenario_entry = {
            "id": result.scenario.id,
            "category": result.scenario.category,
            "description": result.scenario.description,
            "source": result.scenario.source,
            "passed": result.passed,
            "error": result.error,
            "turns": [],
        }
        for turn_result in result.turn_results:
            llm_output = turn_result.trace.get("llm_structured_output") or {}
            scenario_entry["turns"].append({
                "customer_message": turn_result.turn.message,
                "expected_tool": turn_result.turn.expected_tool,
                "expected_fields": turn_result.turn.expected_fields,
                "expected_contains": turn_result.turn.expected_contains,
                "expected_not_contains": turn_result.turn.expected_not_contains,
                "manner_note": turn_result.turn.manner_note,
                "actual_tool": llm_output.get("tool"),
                "actual_llm_output": llm_output,
                "actual_resolved_arguments": turn_result.trace.get("resolved_arguments"),
                "actual_final_response": turn_result.final_text,
                "passed": turn_result.passed,
                "checks": [
                    {"label": c.label, "passed": c.passed, "detail": c.detail} for c in turn_result.checks
                ],
                # Left blank on purpose -- correct/wrong x natural/robotic
                # is a human (or future LLM-judge) call, not something
                # this evaluator scores itself. See schema.py.
                "manner_rating": None,
            })
        payload.append(scenario_entry)
    return json.dumps(payload, indent=2, default=str)


def print_summary(results: list[ScenarioResult]) -> None:
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        by_category[result.scenario.category].append(result)

    total = len(results)
    total_passed = sum(1 for r in results if r.passed)
    print(f"\nKasaFlow behavioural evaluator: {total_passed}/{total} scenarios passed\n")

    for category in sorted(by_category):
        cat_results = by_category[category]
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"{category}: {cat_passed}/{len(cat_results)}")
        for result in cat_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.scenario.id} -- {result.scenario.description}")
            if result.error:
                print(f"         error: {result.error}")
                continue
            for turn_result in result.turn_results:
                if turn_result.passed:
                    continue
                print(f"         turn {turn_result.turn.message!r}:")
                for check in turn_result.checks:
                    if not check.passed:
                        print(f"           - {check.label}: {check.detail}")
        print()
