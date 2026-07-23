"""
Prompt regression tests for services/llm.py

Different from every other test in this project: these call the REAL
OpenAI API. No mocks. That's the whole point -- unit tests prove your
CODE is correct, these prove the MODEL's behaviour hasn't drifted.
Your code can be bug-free and these can still start failing, because
the thing under test here is a decision made by a model you don't
control the version of.

Because of that:
- These cost real money per run (small, but real).
- These are opt-in only: `pytest --run-regression`. They never run as
  part of a normal `pytest` invocation, so they can't slow down or add
  cost to your everyday test loop.
- The model isn't perfectly deterministic. A single case flipping is a
  signal to look closer, not automatically a broken build -- that's
  why each case is its own test (via parametrize) rather than one
  giant assert, so a failure tells you exactly which customer message
  stopped routing correctly, not just "something, somewhere, failed".

Run it with:
    pytest tests/test_prompt_regression.py --run-regression -v

Requires a real OPENAI_API_KEY in your .env.
"""

import pytest

from services.llm import ToolSelectionError, understand_customer

# (customer message, expected tool). Keep this list short and
# representative rather than exhaustive -- each one costs a real call.
CASES = [
    ("how much is a gold ring", "get_product_price"),
    ("what's the price of a silver ring", "get_product_price"),
    ("how much does a gold chain cost", "get_product_price"),
    ("what are your delivery times", "get_delivery_information"),
    ("how much is shipping", "get_delivery_information"),
    ("can I get a full quote for a gold ring including delivery", "generate_quote"),
    ("give me a complete quote for a silver ring, price and shipping", "generate_quote"),
    ("what do you have in gold", "recommend_products"),
    ("show me your silver jewellery", "recommend_products"),
    ("what's available in diamond", "recommend_products"),
]


@pytest.mark.regression
@pytest.mark.parametrize("message,expected_tool", CASES)
def test_tool_selection_matches_expected(message, expected_tool):
    # Arrange: nothing to mock, this is the real thing

    # Act
    result = understand_customer(message)

    # Assert
    assert result["tool"] == expected_tool, (
        f"Message {message!r} routed to {result['tool']!r}, expected {expected_tool!r}. "
        "A single mismatch here is a drift signal, not necessarily a code bug -- "
        "check whether the model's behaviour actually changed or the prompt needs tightening."
    )


@pytest.mark.regression
def test_ambiguous_message_fails_gracefully_not_silently_wrong():
    # Arrange: a message with no real intent behind it at all

    # Act / Assert: we don't assert a *specific* tool here, because there
    # isn't a correct one. We only assert the system does one of two
    # acceptable things: pick some registered tool, or raise a clean
    # ToolSelectionError. What it must NOT do is crash with something
    # unhandled, or return malformed output.
    try:
        result = understand_customer("hi")
        assert "tool" in result and "arguments" in result
    except ToolSelectionError:
        pass  # acceptable: model declined to guess, our code handled it cleanly
