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
    # The headline fact (the price) is bolded WhatsApp-style; the
    # trailing question is supporting text, left plain.
    assert "*GH₵1,200.00*" in formatted
    assert "Want to know about delivery too?" in formatted
    assert "*Want to know about delivery too?*" not in formatted


def test_formats_price_and_delivery_options_shape():
    # generate_quote's shape -- real delivery choices, not an invented
    # cost/time (see delivery_tool.py's module docstring)
    result = {
        "product": "Ring",
        "material": "gold",
        "price": 1200.0,
        "delivery_options": [
            {"key": "accra_rider", "label": "rider delivery within Accra"},
            {"key": "kumasi_rider", "label": "rider delivery within Kumasi"},
            {"key": "international", "label": "shipping outside Ghana"},
        ],
    }
    formatted = format_for_customer(result)
    assert "1,200.00" in formatted
    assert "*GH₵1,200.00*" in formatted
    assert "rider delivery within Accra" in formatted
    assert "rider delivery within Kumasi" in formatted
    assert "shipping outside Ghana" in formatted


def test_formats_bare_delivery_options_shape():
    # get_delivery_information()'s shape -- a customer asking generically
    # about delivery, with no product/price involved
    result = {
        "delivery_options": [
            {"key": "accra_rider", "label": "rider delivery within Accra"},
            {"key": "international", "label": "shipping outside Ghana"},
        ]
    }
    formatted = format_for_customer(result)
    assert "rider delivery within Accra" in formatted
    assert "shipping outside Ghana" in formatted


def test_formats_empty_recommendations():
    formatted = format_for_customer({"recommendations": []})
    assert "don't have anything" in formatted.lower()


def test_formats_empty_recommendations_with_available_categories():
    # recommendation_service.py sends this shape when the customer's
    # category genuinely isn't stocked (e.g. bracelets) -- should offer
    # what IS available rather than a flat dead end.
    result = {
        "recommendations": [],
        "requested_category": "Bracelets",
        "available_categories": ["Necklaces", "Rings"],
    }
    formatted = format_for_customer(result)
    assert "Bracelets" in formatted
    assert "Necklaces" in formatted and "Rings" in formatted
    assert "don't have anything" not in formatted.lower()


def test_formats_recommendations_products_are_visually_separated():
    # Distinct products must read as distinct points, not run together --
    # a blank line between each is what makes that scannable on WhatsApp.
    items = [
        {"product": "Ring A", "material": "18k", "price": 100.0},
        {"product": "Ring B", "material": "18k", "price": 200.0},
    ]
    formatted = format_for_customer({"recommendations": items})
    assert "Ring A" in formatted and "Ring B" in formatted
    a_end = formatted.index("Ring A")
    b_start = formatted.index("Ring B")
    assert "\n\n" in formatted[a_end:b_start]


def test_formats_recommendations_capped_at_four_products():
    items = [{"product": f"Ring {i}", "material": "18k", "price": 100.0 * i} for i in range(1, 8)]
    formatted = format_for_customer({"recommendations": items})
    # 7 distinct products in, only the first 4 should actually appear
    assert formatted.count("Ring") == 4


def test_formats_recommendations_lists_a_small_same_price_variant_set():
    # Three sizes of the same ring, same price -- small enough to list
    # every option directly rather than collapsing to a range
    items = [
        {"product": "Minimal White Stone Gold Ring, 1g", "material": f"US {size}", "price": 20628.0}
        for size in ["8", "8.5", "9"]
    ]
    formatted = format_for_customer({"recommendations": items})
    assert formatted.count("Minimal White Stone Gold Ring, 1g") == 1
    for size in ["US 8", "US 8.5", "US 9"]:
        assert size in formatted
    assert formatted.count("20,628.00") == 1
    # Product name and price are the headline, bolded; the "available
    # in ..." sizes are supporting detail, left plain
    assert "*Minimal White Stone Gold Ring, 1g*" in formatted
    assert "*GH₵20,628.00*" in formatted
    assert "*available in" not in formatted


def test_formats_recommendations_summarises_a_large_variant_set_as_a_range():
    # A real "what rings do you have" query can match 30+ karat/size rows
    # for one product name -- listing every one is a wall of text, not a
    # readable reply, so a large set collapses to a price range instead
    items = [
        {"product": "Minimal White Stone Gold Ring, 1g", "material": f"US {size}", "price": price}
        for size, price in [("8", 15127.20), ("8.5", 15127.20), ("9", 17877.60), ("9.5", 20628.00), ("13", 20628.00)]
    ]
    formatted = format_for_customer({"recommendations": items})
    assert formatted.count("Minimal White Stone Gold Ring, 1g") == 1
    # The individual per-size lines must NOT all be listed out
    assert "US 8.5" not in formatted
    assert "15,127.20" in formatted and "20,628.00" in formatted
    assert "5 options" in formatted


def test_formats_recommendations_diversifies_across_categories():
    # data/products.json lists every Necklaces row before any Rings row --
    # an unfiltered browse must not silently show 4 necklaces and zero
    # rings just because of catalogue file order.
    necklaces = [
        {"product": f"Necklace {i}", "material": "18k", "price": 100.0, "category": "Necklaces"}
        for i in range(1, 10)
    ]
    rings = [
        {"product": f"Ring {i}", "material": "18k", "price": 100.0, "category": "Rings"}
        for i in range(1, 10)
    ]
    formatted = format_for_customer({"recommendations": necklaces + rings})
    assert "Ring 1" in formatted, "an unfiltered browse must surface rings too, not just necklaces"
    assert "Necklace 1" in formatted


def test_formats_recommendations_omits_karat_wording_when_karat_already_fixed():
    # Customer already asked for 18k specifically -- recommend_products
    # filtered to only 18k rows before this ever runs, so every remaining
    # variant shares that karat. Saying "sizes/karats" here would be
    # inaccurate, not just imprecise.
    items = [
        {"product": "Minimal Ring", "material": f"18 / Women US {size} (21.4 mm)", "price": 100.0}
        for size in ["8", "9", "10", "11", "12"]
    ]
    formatted = format_for_customer({"recommendations": items})
    # The product line itself must not claim karat variance that doesn't
    # exist -- the generic closing "or karat?" invite is a separate,
    # always-present footer and isn't what's under test here.
    product_line = formatted.split("Want me to tell you more")[0]
    assert "karat" not in product_line.lower()
    assert "sizes" in product_line.lower()


def test_formats_recommendations_keeps_karat_wording_when_karat_varies():
    items = [
        {"product": "Minimal Ring", "material": f"{karat} / Women US 12 (21.4 mm)", "price": 100.0}
        for karat in ["14", "18", "22", "12", "9"]
    ]
    formatted = format_for_customer({"recommendations": items})
    assert "karat" in formatted.lower()


def test_formats_recommendations_keeps_variants_with_different_prices_separate():
    items = [
        {"product": "Custom Worded Gold Ring", "material": "5g", "price": 6303.0},
        {"product": "Custom Worded Gold Ring", "material": "8g", "price": 8595.0},
    ]
    formatted = format_for_customer({"recommendations": items})
    # Both real prices must survive -- collapsing to one price would hide
    # a genuine price difference from the customer
    assert "6,303.00" in formatted
    assert "8,595.00" in formatted


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
    # total is product cost only -- no invented delivery cost, see
    # order_tool.propose_order()'s docstring
    result = {
        "proposal": {
            "quantity": 2,
            "material": "18k",
            "product": "Ring",
            "subtotal": 2400.0,
            "delivery_address": "12 Cantonments Road, Accra",
            "delivery_option": "accra_rider",
            "delivery_option_label": "rider delivery within Accra",
            "total": 2400.0,
        }
    }
    formatted = format_for_customer(result)
    assert "2 x 18k Ring" in formatted
    assert "2,400.00" in formatted
    assert "CONFIRM" in formatted
    assert "*2 x 18k Ring*" in formatted
    assert "*GH₵2,400.00*" in formatted
    assert "rider delivery within Accra" in formatted


def test_formats_order_confirmation_shape():
    result = {
        "order_confirmation": {
            "order_id": 555,
            "total": 2400.0,
            "delivery_address": "12 Cantonments Road, Accra",
            "delivery_option_label": "rider delivery within Accra",
        }
    }
    formatted = format_for_customer(result)
    assert "555" in formatted
    assert "2,400.00" in formatted
    assert "Cantonments" in formatted
    assert "*order #555*" in formatted
    assert "*GH₵2,400.00*" in formatted
    assert "rider delivery within Accra" in formatted


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
