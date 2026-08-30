"""
Tests for services/response_formatter.py

Pure function, no external dependencies -- every shape router.py's
tools can produce gets exercised directly against the real formatter,
no mocking needed.
"""

from services.response_formatter import format_for_customer, select_presented_groups


def test_formats_no_match():
    assert "couldn't find" in format_for_customer(None).lower()


# ---------------------------------------------------------------------
# select_presented_groups -- the same selection router.py persists via
# memory.set_last_presented_products(), so it must match exactly what
# format_for_customer() renders for a recommendations reply.
# ---------------------------------------------------------------------

def test_select_presented_groups_groups_variants_of_the_same_product():
    items = [
        {"product": "Ring", "category": "Rings", "material": "14k"},
        {"product": "Ring", "category": "Rings", "material": "18k"},
        {"product": "Necklace", "category": "Necklaces", "material": "14k"},
    ]

    groups = select_presented_groups(items)

    assert [name for name, _ in groups] == ["Ring", "Necklace"]
    assert len(groups[0][1]) == 2


def test_select_presented_groups_caps_at_max_groups_with_round_robin():
    items = [
        {"product": f"Necklace {i}", "category": "Necklaces"} for i in range(5)
    ] + [
        {"product": "Ring 1", "category": "Rings"},
    ]

    groups = select_presented_groups(items, max_groups=4)

    categories = [variants[0]["category"] for _, variants in groups]
    assert len(groups) == 4
    assert "Rings" in categories  # round-robin, not a plain [:4] slice


def test_select_presented_groups_preserves_first_seen_order_when_under_cap():
    items = [
        {"product": "B", "category": "Rings"},
        {"product": "A", "category": "Necklaces"},
    ]

    groups = select_presented_groups(items)

    assert [name for name, _ in groups] == ["B", "A"]


def test_formats_error_shape():
    assert format_for_customer({"error": "Something went wrong."}) == "Something went wrong."


def test_correction_note_is_prepended_to_an_error_shape():
    # router.py's propose_order correction acknowledgement (see
    # _describe_order_corrections()) -- prepended to whatever reply the
    # underlying result would already produce, an error/question here.
    result = {"error": "What address should this be delivered to?", "correction_note": "Got it, I've updated the karat to 14k."}
    assert format_for_customer(result) == "Got it, I've updated the karat to 14k. What address should this be delivered to?"


def test_correction_note_is_prepended_to_a_proposal_shape():
    result = {
        "correction_note": "Got it, I've updated the karat to 14k.",
        "proposal": {
            "quantity": 7, "material": "14k", "product": "Ring", "total": 45000.0,
            "delivery_address": "East Legon", "delivery_option_label": "rider delivery within Accra",
        },
    }
    formatted = format_for_customer(result)
    assert formatted.startswith("Got it, I've updated the karat to 14k. ")
    assert "*7 x 14k Ring*" in formatted


def test_formats_conversation_reply_shape():
    # router.py's converse shape -- the LLM already wrote the reply
    # itself, passed straight through untouched, not reformatted.
    result = {"conversation_reply": "Hey! How can I help you today?"}
    assert format_for_customer(result) == "Hey! How can I help you today?"


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


_DELIVERY_OPTIONS = [
    {"key": "accra_rider", "label": "rider delivery within Accra"},
    {"key": "kumasi_rider", "label": "rider delivery within Kumasi"},
    {"key": "international", "label": "shipping outside Ghana"},
]


def test_formats_matched_zone_shape_for_a_rider_zone():
    # get_delivery_information(address=...)'s shape when a named place
    # confidently matches one of the two rider zones. Leads with a
    # direct "Yes" -- the customer asked "does delivery cover my
    # location?", not "what are your delivery options?" (Webb/GPT
    # review, 2026-08-22).
    result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": "accra_rider", "queried_address": "East Legon"}
    formatted = format_for_customer(result)
    assert formatted.startswith("Yes,")
    assert "rider delivery within Accra" in formatted
    assert "East Legon" in formatted


def test_formats_matched_zone_shape_for_kumasi():
    result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": "kumasi_rider", "queried_address": "Kumasi"}
    formatted = format_for_customer(result)
    assert formatted.startswith("Yes,")
    assert "rider delivery within Kumasi" in formatted


def test_formats_matched_zone_shape_for_ghana_other():
    # The exact live case: Bolgatanga is real, but not a rider zone, and
    # not "international" either -- must say so honestly, not repeat
    # the generic three-way list as if the place name wasn't understood.
    # Leads with "Yes" (delivery IS possible) and names cost AND timing
    # explicitly, not a vague "we'll sort it" (Webb/GPT review, 2026-08-22).
    result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": "ghana_other", "queried_address": "Bolgatanga"}
    formatted = format_for_customer(result)
    assert formatted.startswith("Yes,")
    assert "Bolgatanga" in formatted
    assert "rider zone" in formatted.lower()
    assert "cost" in formatted.lower()
    assert "timing" in formatted.lower()
    # Must not silently claim a zone that isn't real for this address.
    assert "covered by our rider delivery within accra" not in formatted.lower()
    assert "covered by our rider delivery within kumasi" not in formatted.lower()


def test_formats_matched_zone_shape_for_ghana_other_cape_coast_and_tamale():
    # Two more real Ghanaian places outside both rider zones -- same
    # honest ghana_other handling, not just Bolgatanga specifically.
    for address in ("Cape Coast", "Tamale"):
        result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": "ghana_other", "queried_address": address}
        formatted = format_for_customer(result)
        assert formatted.startswith("Yes,")
        assert address in formatted


def test_formats_matched_zone_shape_for_international():
    result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": "international", "queried_address": "London"}
    formatted = format_for_customer(result)
    assert formatted.startswith("Yes,")
    assert "shipping outside Ghana" in formatted
    assert "London" in formatted


def test_formats_matched_zone_shape_when_unclassifiable():
    # No real signal either way -- falls back to the generic list rather
    # than guessing, same conservative bias as everywhere else. This is
    # also, today, what a genuinely international address like "London"
    # gets in practice when Google Maps geocoding isn't configured (see
    # test_delivery_tool.py's offline-vs-geocoding coverage) -- the
    # offline classifier never asserts "international" on its own, only
    # this formatter shape's "international" branch does, and only once
    # something upstream actually resolved that zone with real evidence.
    result = {"delivery_options": _DELIVERY_OPTIONS, "matched_zone": None, "queried_address": "456 Workshop Lane"}
    formatted = format_for_customer(result)
    assert "456 Workshop Lane" in formatted
    assert "rider delivery within Accra" in formatted
    assert "rider delivery within Kumasi" in formatted
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


def test_formats_recommendations_never_leaves_a_dangling_asterisk():
    # Broad regression guard across every recommendation-group shape --
    # single variant, small same-price set, small different-price set
    # (the sub-bullet branch that actually broke), and a large
    # summarised set. Every bold marker must be a real pair.
    scenarios = [
        [{"product": "Solo Ring", "material": "18k", "price": 1000.0}],
        [
            {"product": "Same Price Ring", "material": f"US {s}", "price": 500.0}
            for s in ["8", "8.5", "9"]
        ],
        [
            {"product": "Diff Price Ring", "material": f"{s}g", "price": p}
            for s, p in [("5", 100.0), ("8", 200.0)]
        ],
        [
            {"product": "Big Set Ring", "material": f"18 / Women US {s} (21.4 mm)", "price": p}
            for s, p in [("8", 100.0), ("8.5", 150.0), ("9", 200.0), ("9.5", 250.0), ("13", 300.0)]
        ],
    ]
    for items in scenarios:
        formatted = format_for_customer({"recommendations": items})
        assert formatted.count("*") % 2 == 0, f"dangling asterisk for {items[0]['product']!r}: {formatted!r}"


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


def test_formats_karat_options_lists_every_variant_price():
    # product_tool.list_karat_options()'s shape -- "what karat does that
    # come in" (llm.py's tool 10), distinct from a single price quote.
    result = {
        "product": "Custom Leaf White Gold Necklace, 20g",
        "karat_options": [
            {"product": "Custom Leaf White Gold Necklace, 20g", "material": "18k", "price": 34000.0},
            {"product": "Custom Leaf White Gold Necklace, 20g", "material": "14k", "price": 30000.0},
            {"product": "Custom Leaf White Gold Necklace, 20g", "material": "12k", "price": 26000.0},
        ],
    }
    formatted = format_for_customer(result)
    assert "Custom Leaf White Gold Necklace, 20g" in formatted
    assert "18k: *GH₵34,000.00*" in formatted
    assert "14k: *GH₵30,000.00*" in formatted
    assert "12k: *GH₵26,000.00*" in formatted
    assert "Which karat would you like?" in formatted


def test_formats_karat_options_with_no_matches_gives_the_same_no_match_message_as_price():
    result = {"product": "Nonexistent Ring", "karat_options": []}
    assert "couldn't find that one" in format_for_customer(result).lower()


def test_formats_weight_shape():
    result = {"product": "Set Multi Stone Golf Ring, 7g", "weight": "7g"}
    assert format_for_customer(result) == "The *Set Multi Stone Golf Ring, 7g* weighs 7g."


def test_formats_weight_shape_honestly_when_no_weight_is_on_file():
    # A real, matched product with nothing parseable in its name -- says
    # so honestly rather than inventing a number or falling back to the
    # generic no-match message (the product WAS found).
    result = {"product": "Custom Butterfly Gold Ring", "weight": None}
    formatted = format_for_customer(result)
    assert "don't have the weight on file" in formatted
    assert "Custom Butterfly Gold Ring" in formatted


# ---------------------------------------------------------------------
# weight phrasing variants -- task #186, live evidence 2026-08-30
# (Webb): "that's the weight?", "is that really 1g?", and "how many
# grams is that?" asked back to back about the same product all came
# back character-for-character identical. weight_ask_count (threaded
# on by router.py) selects a different phrasing each time instead.
# ---------------------------------------------------------------------

def test_weight_variant_defaults_to_the_original_wording_when_count_is_absent():
    # Every pre-existing caller/test builds a bare {"product", "weight"}
    # dict with no weight_ask_count key -- must render exactly as before.
    result = {"product": "Set Multi Stone Golf Ring, 7g", "weight": "7g"}
    assert format_for_customer(result) == "The *Set Multi Stone Golf Ring, 7g* weighs 7g."


def test_weight_variant_changes_on_repeated_asks_of_the_same_fact():
    product = "Minimal White Stone Gold Ring, 1g"
    replies = [
        format_for_customer({"product": product, "weight": "1g", "weight_ask_count": count})
        for count in (1, 2, 3, 4)
    ]
    # The specific live failure: three (here, all four) replies must not
    # all be the same sentence.
    assert len(set(replies)) > 1
    # First ask keeps today's original wording -- no behaviour change
    # for the common, single-ask case.
    assert replies[0] == f"The *{product}* weighs 1g."
    # Every variant still states the actual weight -- varying phrasing
    # must never drop or alter the fact itself.
    for reply in replies:
        assert "1g" in reply


def test_weight_variant_cycles_rather_than_erroring_past_the_pool_length():
    product = "Minimal White Stone Gold Ring, 1g"
    # 4 known-weight variants exist -- the 5th ask must reuse variant 0,
    # not crash or run off the end of the list.
    reply = format_for_customer({"product": product, "weight": "1g", "weight_ask_count": 5})
    assert reply == f"The *{product}* weighs 1g."


def test_weight_variant_also_applies_when_no_weight_is_on_file():
    product = "Custom Butterfly Gold Ring"
    first = format_for_customer({"product": product, "weight": None, "weight_ask_count": 1})
    second = format_for_customer({"product": product, "weight": None, "weight_ask_count": 2})
    assert first != second
    assert "don't have the weight on file" in first or "don't have that one's weight" in first


def test_formats_identified_product_shows_every_karat_price():
    # photo_match_tool.py's confident-match shape -- unlike the
    # recommendations list (which collapses to a price range above 3
    # variants), this is exactly one identified product, so every karat
    # is shown, not a range.
    result = {
        "identified_product": {
            "product": "Custom Adinkra Chains Gold Necklace",
            "image_url": "https://example.com/necklace.jpg",
            "variants": [
                {"material": "18k", "price": 45000.0},
                {"material": "14k", "price": 39000.0},
                {"material": "12k", "price": 33000.0},
            ],
        }
    }
    formatted = format_for_customer(result)
    assert "Custom Adinkra Chains Gold Necklace" in formatted
    assert "*GH₵45,000.00*" in formatted
    assert "*GH₵39,000.00*" in formatted
    assert "*GH₵33,000.00*" in formatted
    assert "18k" in formatted and "14k" in formatted and "12k" in formatted
    # Delivery is offered, never priced -- same rule as every other shape
    assert "rider delivery within Accra" in formatted


def test_formats_identified_product_invites_correction_rather_than_asserting_certainty():
    # Confirmed live, 2026-08-18: a photo match confidently picked the
    # wrong one of two similar cross necklace-and-earring sets, despite
    # the temperature=0/prompt-conservatism fix made after an earlier,
    # similar mismatch. The wording must invite a correction rather than
    # assert "that's the X" as settled fact, since the system can be
    # confidently wrong here.
    result = {
        "identified_product": {
            "product": "Custom Cross Gold Necklace with Earrings",
            "image_url": "https://example.com/cross.jpg",
            "variants": [{"material": "18k", "price": 18000.0}],
        }
    }
    formatted = format_for_customer(result)
    assert "that's the" not in formatted.lower()
    assert "looks like" in formatted.lower()
    assert "let me know if that's not" in formatted.lower()


def test_formats_identified_product_with_no_variants_gives_an_honest_fallback():
    # The product was visually identified, but its price rows couldn't
    # be pulled up (e.g. products.json changed underneath us) -- say so
    # rather than showing an empty price list.
    result = {"identified_product": {"product": "Some Necklace", "image_url": None, "variants": []}}
    formatted = format_for_customer(result)
    assert "Some Necklace" in formatted
    assert "couldn't pull up current pricing" in formatted


def test_formats_recommendations_drops_the_redundant_per_item_cta():
    # Each multi-variant product used to end its own line with "tell me
    # your size and karat and I'll get you the exact price" -- fine for
    # one item, noisy and robotic once several items in the same list
    # each repeated it (confirmed live, 2026-08-16). The one closing
    # question at the end of the whole message already covers this.
    items = [
        {"product": f"Ring {i}", "material": f"18 / Women US {size} (21.4 mm)", "price": p}
        for i in (1, 2)
        for size, p in [("8", 100.0 + i), ("9", 200.0 + i), ("10", 300.0 + i), ("11", 400.0 + i)]
    ]
    formatted = format_for_customer({"recommendations": items})
    assert formatted.count("tell me your") == 0
    assert formatted.count("get you the exact price") == 0
    # The closing footer still invites narrowing down, just once
    assert formatted.count("narrow it down by size or karat") == 1
    # Price ranges are still shown per item
    assert "options)" in formatted


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
    # The sub-bullet marker must never be "*" -- WhatsApp only has one
    # meaning for that character (bold), so a "*" bullet on the same
    # line as a bolded *GH₵...* price causes the parser to pair the
    # bullet with the price's opening asterisk, leaking a stray
    # unbolded "*" at the end of the line (confirmed live, 2026-08-12).
    # An even count of "*" is a cheap, general guard against exactly
    # this class of bug recurring in any branch.
    assert formatted.count("*") % 2 == 0
    assert "\n   * " not in formatted


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


def test_order_confirmation_says_placed_not_confirmed():
    # "confirmed" implies a finished, settled transaction -- nothing has
    # been paid or scheduled yet at this point (see order_tool.py's
    # confirm_order(), which hands delivery/payment to staff after
    # creating the order). Confirmed live, 2026-08-14, that "confirmed"
    # reads as overpromising once you look at what state the order is
    # actually in.
    result = {
        "order_confirmation": {
            "order_id": 555,
            "total": 2400.0,
            "delivery_address": "12 Cantonments Road, Accra",
            "delivery_option_label": "rider delivery within Accra",
        }
    }
    formatted = format_for_customer(result)
    assert "has been placed" in formatted
    assert "confirmed" not in formatted.lower()


def test_order_confirmation_falls_back_when_delivery_label_missing():
    # A proposal built before delivery_option_matches_address() existed,
    # or any other path that never set a label -- still a real order, so
    # this must not blow up with a KeyError/None formatting into the text.
    result = {
        "order_confirmation": {
            "order_id": 555,
            "total": 2400.0,
            "delivery_address": "12 Cantonments Road, Accra",
            "delivery_option_label": None,
        }
    }
    formatted = format_for_customer(result)
    assert "your chosen delivery option" in formatted


def test_formats_order_cancellation_shape():
    result = {"order_cancellation": {"order_id": 6846}}
    formatted = format_for_customer(result)
    assert "*order #6846*" in formatted
    assert "cancelled" in formatted.lower()


def test_formats_order_already_cancelled_shape():
    result = {"order_already_cancelled": {"order_id": 6846}}
    formatted = format_for_customer(result)
    assert "*#6846*" in formatted
    assert "already cancelled" in formatted.lower()


def test_formats_order_escalation_shape():
    result = {"order_escalation": {"order_id": 6846, "status": "completed"}}
    formatted = format_for_customer(result)
    assert "*order #6846*" in formatted
    assert "completed" in formatted
    assert "team" in formatted.lower()


def test_formats_order_status_shape_with_item_and_total():
    result = {
        "order_status": {
            "order_id": 6846,
            "status": "processing",
            "status_label": "confirmed and being prepared",
            "item_summary": "1 x Custom Leaf White Gold Necklace, 20g",
            "total": "34000.00",
        }
    }
    formatted = format_for_customer(result)
    assert "*#6846*" in formatted
    assert "confirmed and being prepared" in formatted
    assert "1 x Custom Leaf White Gold Necklace, 20g" in formatted
    assert "34,000.00" in formatted


def test_formats_order_status_shape_without_item_or_total():
    result = {
        "order_status": {
            "order_id": 6846,
            "status": "pending",
            "status_label": "received and awaiting confirmation",
            "item_summary": None,
            "total": None,
        }
    }
    formatted = format_for_customer(result)
    assert "*#6846*" in formatted
    assert "received and awaiting confirmation" in formatted
    assert "Total" not in formatted


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
