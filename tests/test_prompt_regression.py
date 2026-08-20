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
    ("what's your returns policy", "answer_policy_question"),
    ("how do I find my ring size", "answer_policy_question"),
    ("can I get a ring engraved", "answer_policy_question"),
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
@pytest.mark.parametrize(
    "message,expected_product_name",
    [
        # "sika kyɛn no bo yɛ sɛn" -- "how much is the gold chain".
        # "kyɛn" is Twi for "chain" -- the catalogue is English-only, so
        # this must come back translated, not passed through as "kyɛn".
        # Confirmed failing before the llm.py prompt fix (2026-08-07):
        # the model returned product_name="kyɛn" verbatim.
        ("sika kyɛn no bo yɛ sɛn", "chain"),
    ],
)
def test_non_english_product_terms_are_translated(message, expected_product_name):
    # Arrange: nothing to mock, this is the real thing

    # Act
    result = understand_customer(message)

    # Assert: the catalogue only has English product names, so a Twi
    # word passed straight through would never match anything in it.
    assert result["arguments"].get("product_name", "").strip().lower() == expected_product_name, (
        f"Expected product_name={expected_product_name!r} translated from Twi, "
        f"got {result['arguments'].get('product_name')!r}. A non-English value here "
        "means llm.py's translation instruction stopped working, not necessarily "
        "a code bug -- verify against the prompt before assuming drift."
    )


@pytest.mark.regression
def test_message_with_two_distinct_products_returns_both_as_requests():
    # Arrange: a message that genuinely asks about two different products.
    # Before the router.py/llm.py fix, only one of these would ever come
    # back -- the other was silently dropped, not flagged as ambiguous.
    message = "how much is a gold ring and a silver chain"

    # Act
    result = understand_customer(message)

    # Assert: both asks present, nothing silently dropped
    assert "requests" in result, (
        f"Expected the multi-request shape for a message with two distinct asks, "
        f"got a single {result.get('tool')!r} instead -- one of the two products "
        "would be silently dropped downstream."
    )
    assert len(result["requests"]) == 2
    product_names = {r["arguments"].get("product_name", "").lower() for r in result["requests"]}
    assert product_names == {"ring", "chain"}


@pytest.mark.regression
def test_single_product_message_is_not_split():
    # Arrange: guard against the model over-splitting a plain single-ask
    # message into an unnecessary "requests" list.
    message = "how much is a gold ring"

    # Act
    result = understand_customer(message)

    # Assert
    assert "requests" not in result, (
        f"A single, unambiguous ask was split into {result.get('requests')!r} -- "
        "should have returned the normal single-tool shape."
    )
    assert result["tool"] == "get_product_price"


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


# ---------------------------------------------------------------------
# Order-continuation and correction cases -- real terse WhatsApp
# replies relying on order_draft state, not full sentences. Everything
# above this line calls understand_customer() with no state at all;
# these exercise the same "short message + remembered context" path
# Webb's real live transcript (2026-08-19) surfaced two bugs in (see
# services/router.py's _describe_order_corrections() and
# services/geocoding_tool.py's infer_delivery_option(), and
# tests/test_order_conversations.py's mocked, deterministic coverage of
# those two fixes). These are the one thing that suite can't cover: is
# the MODEL itself reliably resolving language like this in the first
# place -- unit tests can only prove the code does the right thing with
# whatever arguments the model returns, not that the model returns the
# right arguments from messy input like "14" or "7 pieces, actually
# make that 5" on its own.
# ---------------------------------------------------------------------

@pytest.mark.regression
def test_bare_karat_digit_answers_the_missing_material():
    # order_draft's product/quantity already known, material is the one
    # thing still missing -- a bare "14" here means material="14k", not
    # a quantity or anything else (see llm.py's _order_draft_state_line()
    # bare-number disambiguation rule).
    order_draft = {
        "product_name": "Gye Nyame White Necklace with Earrings, 30g",
        "material": None, "quantity": 7, "delivery_address": None, "delivery_option": None,
    }

    result = understand_customer("14", order_draft=order_draft)

    assert result["tool"] == "propose_order"
    assert "14" in str(result["arguments"].get("material", ""))


@pytest.mark.regression
@pytest.mark.parametrize(
    "message,expected_material",
    [
        # Confirmed live, 2026-08-19 (Webb): "wait i want to order the
        # 14k rather" after material was already "12k" -- must resolve
        # to the NEW value, not the old one.
        ("wait i want to order the 14k rather", "14"),
        # Sequential correction within the SAME message -- only the
        # final value is the real answer.
        ("14k, no wait 18k", "18"),
    ],
)
def test_material_correction_resolves_to_the_final_stated_value(message, expected_material):
    order_draft = {
        "product_name": "Gye Nyame White Necklace with Earrings, 30g",
        "material": "12k", "quantity": 7, "delivery_address": None, "delivery_option": None,
    }

    result = understand_customer(message, order_draft=order_draft)

    assert result["tool"] == "propose_order"
    assert expected_material in str(result["arguments"].get("material", "")), (
        f"Expected material containing {expected_material!r}, got "
        f"{result['arguments'].get('material')!r} for message {message!r} -- a correction "
        "or a sequential in-message correction didn't resolve to the final stated value."
    )


@pytest.mark.regression
def test_sequential_quantity_correction_within_one_message_keeps_only_the_final_value():
    # "7 pieces, actually make that 5" -- the customer stated two
    # quantities in one message; only 5 is the real answer.
    order_draft = {
        "product_name": "Gye Nyame White Necklace with Earrings, 30g",
        "material": "14k", "quantity": None, "delivery_address": None, "delivery_option": None,
    }

    result = understand_customer("7 pieces, actually make that 5", order_draft=order_draft)

    assert result["tool"] == "propose_order"
    assert int(result["arguments"].get("quantity")) == 5


@pytest.mark.regression
def test_bare_confirmation_with_a_pending_order_uses_confirm_order():
    pending_order = {"product": "Ring", "material": "18k", "quantity": 2, "total": 2400.0}

    result = understand_customer("yh", pending_order=pending_order)

    assert result["tool"] == "confirm_order"


@pytest.mark.regression
def test_a_disputed_policy_claim_is_re_grounded_not_just_apologised_for():
    # A customer pushing back on a policy claim must re-run the real
    # lookup, not just capitulate via converse -- the customer disputing
    # something isn't proof the system was wrong (see llm.py's dispute
    # rule). answer_policy_question is retrieval-grounded against
    # data/policies.json, so this is what actually decides who's right,
    # not the model agreeing because it was pushed back on.
    result = understand_customer("no, I'm sure returns are 30 days, not 14")

    assert result["tool"] == "answer_policy_question"


@pytest.mark.regression
def test_ambiguous_product_reference_with_no_context_is_left_unknown_not_guessed():
    # "that one" with nothing in this message or any state passed in to
    # resolve it against -- the model must say "unknown" and let the
    # system's own memory resolution handle it (see llm.py's rules),
    # not hallucinate a specific catalogue item.
    result = understand_customer("I'll take that one")

    assert result["tool"] == "get_product_price"
    assert str(result["arguments"].get("product_name", "")).strip().lower() == "unknown"
