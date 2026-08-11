"""
Tests for services/response_formatter.py

Pure function, no external dependencies -- every shape router.py's
tools can produce gets exercised directly against the real formatter,
no mocking needed.
"""

from services.response_formatter import format_for_customer


def test_formats_no_match():
    assert "couldn't find" in format_for_customer(None).lower()


def test_formats_error_shape():
    assert format_for_customer({"error": "Something went wrong."}) == "Something went wrong."


def test_formats_policy_answer_shape():
    result = {"answer": "Returns are accepted within 14 days."}
    assert format_for_customer(result) == "Returns are accepted within 14 days."


def test_formats_price_only_shape():
    result = {"product": "Ring", "material": "gold", "price": 1200.0}
    formatted = format_for_customer(result)
    assert "Ring" in formatted and "gold" in formatted and "1,200.00" in formatted


def test_formats_price_and_delivery_shape():
    result = {
        "product": "Ring",
        "material": "gold",
        "price": 1200.0,
        "delivery": {"delivery_time": "2-3 days", "shipping_cost": 20},
    }
    formatted = format_for_customer(result)
    assert "2-3 days" in formatted and "1,200.00" in formatted


def test_formats_empty_recommendations():
    formatted = format_for_customer({"recommendations": []})
    assert "don't have anything" in formatted.lower()


def test_formats_recommendations_capped_at_five():
    items = [{"product": f"Ring {i}", "material": "18k", "price": 100.0 * i} for i in range(1, 8)]
    formatted = format_for_customer({"recommendations": items})
    # 7 items in, only the first 5 should actually appear in the reply
    assert formatted.count("Ring") == 5


# ---------------------------------------------------------------------
# The additive "results" (plural) shape -- router.py's answer to a
# message that contained more than one distinct ask
# ---------------------------------------------------------------------

def test_formats_single_entry_results_list_same_as_a_bare_result():
    single = {"product": "Ring", "material": "gold", "price": 1200.0}
    assert format_for_customer({"results": [single]}) == format_for_customer(single)


def test_formats_multi_entry_results_list_as_numbered_replies():
    result = {
        "results": [
            {"product": "Ring", "material": "gold", "price": 1200.0},
            {"product": "Chain", "material": "silver", "price": 350.0},
        ]
    }

    formatted = format_for_customer(result)

    # Both answers present, in order, clearly separated -- nothing silently dropped
    assert formatted.startswith("1. ")
    assert "\n\n2. " in formatted
    assert "Ring" in formatted and "1,200.00" in formatted
    assert "Chain" in formatted and "350.00" in formatted


def test_formats_order_proposal_shape():
    result = {
        "proposal": {
            "quantity": 2,
            "material": "18k",
            "product": "Ring",
            "subtotal": 2400.0,
            "delivery_address": "12 Cantonments Road, Accra",
            "delivery": {"delivery_time": "2-5 business days", "shipping_cost": 25},
            "total": 2425.0,
        }
    }
    formatted = format_for_customer(result)
    assert "2 x 18k Ring" in formatted
    assert "2,400.00" in formatted
    assert "2,425.00" in formatted
    assert "CONFIRM" in formatted


def test_formats_order_confirmation_shape():
    result = {
        "order_confirmation": {
            "order_id": 555,
            "total": 2425.0,
            "delivery_address": "12 Cantonments Road, Accra",
        }
    }
    formatted = format_for_customer(result)
    assert "555" in formatted
    assert "2,425.00" in formatted
    assert "Cantonments" in formatted


def test_formats_multi_entry_results_list_when_one_entry_errored():
    result = {
        "results": [
            {"product": "Ring", "material": "gold", "price": 1200.0},
            {"error": "Something went wrong while processing your request."},
        ]
    }

    formatted = format_for_customer(result)

    # The failure of one part doesn't erase the part that worked
    assert "Ring" in formatted
    assert "Something went wrong" in formatted
