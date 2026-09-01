"""
Scenario and result data types for the KasaFlow behavioural evaluator.

Deliberately separate from runner.py and scenarios.py: this file has no
dependency on route_customer() or the OpenAI client, so it can be
imported (and unit tested) anywhere without needing OPENAI_API_KEY set,
same reasoning as this codebase's own services modules staying
independently importable.

Categories are Webb's own list (2026-09-01), kept as a plain tuple
rather than an enum so a new category never needs a code change to add
-- scenarios.py just uses a new string and report.py groups by whatever
strings actually appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field


CATEGORIES = (
    "PRODUCT DISCOVERY",
    "PRICE",
    "WEIGHT",
    "KARAT",
    "DELIVERY",
    "PHOTO",
    "ORDERS",
    "CORRECTIONS",
    "REFERENCES",
    "AMBIGUITY",
    "CONFIRMATION",
    "POST-CONFIRMATION",
    "GLOBAL DELIVERY",
    "GHANAIAN LANGUAGE",
    "SOCIAL/REACTIVE",
)


@dataclass
class Turn:
    """One customer message within a scenario, and what's expected of
    KasaFlow's response to it. Every expectation field is optional --
    a turn only checks what it actually specifies, so a scenario whose
    point is "does this resolve the right product" doesn't also have to
    spell out exact response wording, and vice versa.

    expected_tool: the tool llm.py should have selected. A single
    string, or a tuple of acceptable alternatives when more than one
    tool choice is legitimately correct (e.g. generate_quote vs
    get_product_price both being defensible for a bare price ask).

    expected_fields: a subset of resolved_arguments (router.py's
    post-fill_missing_context() values, i.e. after memory resolution,
    not just what the LLM guessed) that must match exactly. Only the
    keys listed are checked -- this is a subset match, not an equality
    check against the whole dict, so a scenario can assert
    "product_name resolved to X" without also having to predict every
    other argument.

    expected_contains / expected_not_contains: substrings (case
    insensitive) the final customer-facing text (format_for_customer()
    output) must or must not contain. This is the main way a scenario
    checks manner as well as correctness -- e.g. asserting a photo-only
    ask's reply does NOT contain "Want to know about delivery" is
    exactly the caption-mismatch class of failure the intent/
    presentation audit flagged (2026-09-01).

    manner_note: not machine-checked. A short description of what a
    natural response SHOULD read like, carried through into the report
    for a human reviewer (or a future LLM-judge pass, not built yet) to
    score against -- see Webb's four-way correct/wrong x natural/robotic
    split. Leaving this unscored rather than faking an automated pass/
    fail for something this evaluator cannot actually judge yet."""

    message: str
    expected_tool: str | tuple[str, ...] | None = None
    expected_fields: dict | None = None
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    manner_note: str = ""
    notes: str = ""


@dataclass
class Scenario:
    """A full conversation (one or more turns, sharing one session) and
    what it's meant to prove. `source` is deliberately explicit about
    provenance -- "confirmed live, 2026-08-20" carries different weight
    than "designed from category spec, not yet live-tested", and this
    evaluator should never blur that distinction (see this repo's own
    standing rule against presenting an inference as a verified fact)."""

    id: str
    category: str
    description: str
    turns: list[Turn]
    source: str = "designed from category spec, not yet live-tested"

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown category {self.category!r} for scenario {self.id!r}")
        if not self.turns:
            raise ValueError(f"Scenario {self.id!r} has no turns")


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class TurnResult:
    turn: Turn
    trace: dict
    final_text: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class ScenarioResult:
    scenario: Scenario
    turn_results: list[TurnResult]
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        return all(tr.passed for tr in self.turn_results)
