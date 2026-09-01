"""
The KasaFlow behavioural evaluator's scenario corpus, organized under
Webb's fifteen categories (2026-09-01).

This is a starting corpus, not a finished one -- built to be extended
scenario by scenario as real live failures turn up, the same way
llm.py's own prompt already carries dated, confirmed-live examples
rather than invented ones. Every scenario's `source` says plainly
where it came from: an exact wording lifted from an existing test file
or design doc in this repo is marked "confirmed live" with its date;
anything constructed fresh for this corpus is marked "designed from
category spec, not yet live-tested" -- see schema.py's Scenario
docstring for why that distinction has to stay visible rather than
being flattened into one undifferentiated list.

Two real, confirmed-in-stock catalogue items recur across scenarios
(same two KasaFlow_Scenario_Test_Script.md already used, 2026-08-2x):
Big White Crown Stone Gold Ring, 14g (id 5892) and Custom Leaf White
Gold Necklace, 20g (id 6810). Where a scenario needs a product this
corpus hasn't independently confirmed against the live catalogue, that
is noted in `source` rather than assumed.
"""

from __future__ import annotations

from scripts.evaluator.schema import Scenario, Turn

RING = "Big White Crown Stone Gold Ring, 14g"
NECKLACE = "Custom Leaf White Gold Necklace, 20g"

SCENARIOS: list[Scenario] = []


def _add(scenario: Scenario) -> None:
    SCENARIOS.append(scenario)


# ---------------------------------------------------------------------
# PRODUCT DISCOVERY
# ---------------------------------------------------------------------

_add(Scenario(
    id="discovery-01-browse-category",
    category="PRODUCT DISCOVERY",
    description="A category browse with no karat stated should recommend, not ask for one product name.",
    turns=[
        Turn(
            message="what rings do you have",
            expected_tool="recommend_products",
            expected_fields={"category": "Rings"},
        ),
    ],
    source="designed from category spec, not yet live-tested",
))

_add(Scenario(
    id="discovery-02-browse-with-karat",
    category="PRODUCT DISCOVERY",
    description="A category browse that also states a karat should narrow by it, not drop it.",
    turns=[
        Turn(
            message="show me your necklaces in 18k",
            expected_tool="recommend_products",
            expected_fields={"category": "Necklaces", "material": "18k"},
        ),
    ],
    source="designed from category spec, not yet live-tested",
))

_add(Scenario(
    id="discovery-03-unstocked-category",
    category="PRODUCT DISCOVERY",
    description="Asking for a category this store doesn't stock (bracelets/earrings) should say what IS available, not a flat no.",
    turns=[
        Turn(
            message="do you have any bracelets",
            expected_tool="recommend_products",
            expected_not_contains=["I don't have anything matching that right now -- want me to show you something similar"],
        ),
    ],
    source="designed from category spec, not yet live-tested; response_formatter.py's available_categories branch is confirmed to exist in code, exact live wording not independently re-verified here",
))


# ---------------------------------------------------------------------
# PRICE
# ---------------------------------------------------------------------

_add(Scenario(
    id="price-01-named-product-no-karat",
    category="PRICE",
    description="A price ask for a real, named product with no karat stated must not fall through to a no-match message (regression: confirmed live 2026-08-24 that this used to say 'couldn't find that one').",
    turns=[
        Turn(
            message=f"how much is the {RING}",
            expected_tool="get_product_price",
            expected_not_contains=["couldn't find that one"],
        ),
    ],
    source="regression case confirmed live 2026-08-24 (Webb/GPT 50-turn test), see product_tool.py's get_product_price() docstring",
))

_add(Scenario(
    id="price-02-named-product-with-karat",
    category="PRICE",
    description="A price ask naming both product and karat should quote that exact karat's price.",
    turns=[
        Turn(
            message=f"{RING}, in 18k",
            expected_tool="get_product_price",
            expected_fields={"product_name": RING, "material": "18k"},
            expected_contains=["18", "GH₵"],
        ),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turn 1",
))

_add(Scenario(
    id="price-03-karat-change-follow-up",
    category="PRICE",
    description="A bare karat follow-up right after a price quote should re-quote the SAME product at the new karat, not misroute to a browse.",
    turns=[
        Turn(message=f"{NECKLACE}, in 14k", expected_tool="get_product_price"),
        Turn(
            message="what about in 18k",
            expected_tool="get_product_price",
            expected_fields={"product_name": NECKLACE, "material": "18k"},
        ),
    ],
    source="task #72 (\"price after karat change\"), designed to reproduce that fixed class of bug",
))


# ---------------------------------------------------------------------
# WEIGHT
# ---------------------------------------------------------------------

_add(Scenario(
    id="weight-01-direct-question",
    category="WEIGHT",
    description="A direct weight question is a factual answer, not a price or karat response.",
    turns=[
        Turn(message=f"how heavy is the {RING}", expected_tool="get_product_weight"),
    ],
    source="designed from category spec; get_product_weight tool built task #180",
))

_add(Scenario(
    id="weight-02-confirmation-phrasing",
    category="WEIGHT",
    description="'That's the weight?' after being told a weight is a confirmation, not a fresh factual question -- same underlying fact, different expected register.",
    turns=[
        Turn(message=f"how many grams is the {RING}", expected_tool="get_product_weight"),
        Turn(
            message="is that really 14g?",
            expected_tool="get_product_weight",
            manner_note="Should read as a grounded reassurance referencing the catalogue fact already given, not a bare re-statement or an apology.",
        ),
    ],
    source="designed from Webb's 2026-09-01 message (\"Is that really 1g?\" example); task #190 hardened get_product_weight's prompt for skeptical phrasings but this exact two-turn case is not independently confirmed live here",
))

_add(Scenario(
    id="weight-03-weight-after-list",
    category="WEIGHT",
    description="A weight question referring to a just-shown list position should resolve against last_presented_products, not ask which product.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(message="how heavy is the second one", expected_tool="get_product_weight"),
    ],
    source="designed from category spec, combining task #181 (last_presented_products) with task #180 (get_product_weight); not yet live-tested together",
))


# ---------------------------------------------------------------------
# KARAT
# ---------------------------------------------------------------------

_add(Scenario(
    id="karat-01-options-question",
    category="KARAT",
    description="'What karat does it come in' asks for the LIST of options, not one price.",
    turns=[
        Turn(message=f"what karats do you have the {NECKLACE} in", expected_tool="get_product_karat_options"),
    ],
    source="llm.py tool 10's own worked example",
))

_add(Scenario(
    id="karat-02-specific-karat-after-options",
    category="KARAT",
    description="Naming one specific karat right after an options list is pricing that option, not re-asking for the list.",
    turns=[
        Turn(message=f"what karats do you have the {NECKLACE} in", expected_tool="get_product_karat_options"),
        Turn(message="and in 18k?", expected_tool="get_product_price", expected_fields={"material": "18k"}),
    ],
    source="llm.py tool 10's own worked distinction (\"what about 12\", \"and in 18k?\" is get_product_price, not this tool)",
))

_add(Scenario(
    id="karat-03-no-default-assumption",
    category="KARAT",
    description="propose_order must never assume 18k (or any karat) when the customer hasn't stated one.",
    turns=[
        Turn(
            message=f"I want to order the {RING}",
            expected_tool="propose_order",
            expected_fields={"material": "unknown"},
        ),
    ],
    source="llm.py tool 6's explicit rule + task #73 (\"karat defaulting to 18k without asking\")",
))


# ---------------------------------------------------------------------
# DELIVERY
# ---------------------------------------------------------------------

_add(Scenario(
    id="delivery-01-generic-question",
    category="DELIVERY",
    description="A generic delivery question with no place named lists the three real arrangements, not a cost.",
    turns=[
        Turn(message="how does delivery work", expected_tool="get_delivery_information", expected_fields={"address": "unknown"}),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turn 2",
))

_add(Scenario(
    id="delivery-02-named-place-in-zone",
    category="DELIVERY",
    description="Naming a specific covered place should get a direct yes, not the generic three-way menu repeated.",
    turns=[
        Turn(message="do you deliver to Kumasi", expected_tool="get_delivery_information", expected_fields={"address": "Kumasi"}),
    ],
    source="response_formatter.py's matched_zone branch, built per Webb/GPT review 2026-08-22",
))

_add(Scenario(
    id="delivery-03-price-then-delivery-continuation",
    category="DELIVERY",
    description="Confirming interest in delivery right after a price quote should pick up the delivery question, not restart.",
    turns=[
        Turn(message=f"{RING}, in 18k", expected_tool="get_product_price"),
        Turn(message="yh", expected_tool=("get_delivery_information", "generate_quote")),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turns 1-2",
))


# ---------------------------------------------------------------------
# PHOTO
# ---------------------------------------------------------------------

_add(Scenario(
    id="photo-01-direct-photo-request",
    category="PHOTO",
    description="A direct photo request for a named product should carry an image, and read as answering 'can I see it', not as an unsolicited price quote.",
    turns=[
        Turn(
            message=f"can I see a photo of the {NECKLACE}",
            expected_tool="get_product_price",
            manner_note="Right tool, image correctly attached per the intent/presentation audit (2026-09-01) -- but the caption text still defaults to a price template ('...Want to know about delivery too?') regardless of the ask being purely visual. Flag as a manner failure even if tool/data checks pass.",
        ),
    ],
    source="intent/presentation resolution audit, 2026-09-01 -- the one concrete gap it found",
))

_add(Scenario(
    id="photo-02-show-me-ordinal",
    category="PHOTO",
    description="'Show me the second one' after a list should resolve the ordinal and effectively show that product, not restart the browse.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(message="show me the second one", expected_tool="get_product_price"),
    ],
    source="Webb's 2026-09-01 golden-path example, combined with task #181's ordinal resolution",
))

_add(Scenario(
    id="photo-03-browse-sends-one-image-per-product",
    category="PHOTO",
    description="A genuine category browse should present visually -- multiple products, each with its own recommendation entry -- not a text-only wall.",
    turns=[
        Turn(
            message="show me some rings",
            expected_tool="recommend_products",
            manner_note="Tool-level check only: the evaluator cannot see WhatsApp's actual sent messages from route_customer()'s return value alone (task #214 sends photos at the transport layer, not in the returned result dict). A full check of 'one image message per product, capped at 4' needs a transport-level test or a live WhatsApp run, not this harness as built.",
        ),
    ],
    source="task #214 (visual recommend_products browse), transport-layer behaviour not fully observable through this evaluator's route_customer()-only capture -- see manner_note",
))


# ---------------------------------------------------------------------
# ORDERS
# ---------------------------------------------------------------------

_add(Scenario(
    id="orders-01-fully-specified-single-message",
    category="ORDERS",
    description="A fully-specified order in one message should produce one complete proposal, not a missing-field question.",
    turns=[
        Turn(
            message=f"I want the {RING}, in 18k, 1 of them, deliver to East Legon, rider delivery within Accra",
            expected_tool="propose_order",
            expected_contains=["CONFIRM"],
        ),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turn 5",
))

_add(Scenario(
    id="orders-02-quantity-phrasing",
    category="ORDERS",
    description="'I'll take two' should be read as quantity 2 for the active product, not a new, separate ask.",
    turns=[
        Turn(message=f"{NECKLACE}, in 14k", expected_tool="get_product_price"),
        Turn(message="I'll take two", expected_tool="propose_order", expected_fields={"quantity": 2}),
    ],
    source="Webb's 2026-09-01 golden-path example",
))

_add(Scenario(
    id="orders-03-change-product-mid-proposal",
    category="ORDERS",
    description="Naming a different product mid-proposal should switch the order to it cleanly, not blend fields from both.",
    turns=[
        Turn(message=f"I want to order the {NECKLACE}, in 14k, deliver to Osu", expected_tool="propose_order"),
        Turn(
            message=f"actually make it the {RING} instead",
            expected_tool="propose_order",
            expected_fields={"product_name": RING},
        ),
    ],
    source="task #64 (\"change product mid-proposal\") -- turn 1 rewritten 2026-09-01: the original \"I want the necklace in 14k\" phrasing is genuinely ambiguous between a price ask and an order ask (it has no quantity or delivery detail, unlike orders-01/02's phrasing, which both passed live), and that ambiguity is what the live run's failure (get_product_price instead of propose_order) most likely reflects -- not a bug in the product-switch logic this scenario actually means to test. \"I want to order... deliver to Osu\" removes that ambiguity.",
))


# ---------------------------------------------------------------------
# CORRECTIONS
# ---------------------------------------------------------------------

_add(Scenario(
    id="corrections-01-karat-correction-acknowledged",
    category="CORRECTIONS",
    description="Correcting a karat mid-proposal should be acknowledged, not silently applied or ignored.",
    turns=[
        Turn(message=f"I want the {RING} in 12k, deliver to Osu", expected_tool="propose_order"),
        Turn(
            message="wait, 14k rather",
            expected_tool="propose_order",
            expected_fields={"material": "14k"},
            manner_note="Should acknowledge the change (a correction_note), not just silently re-quote as if nothing was said.",
        ),
    ],
    source="response_formatter.py's correction_note handling, task #90",
))

_add(Scenario(
    id="corrections-02-pushback-not-just-repeated",
    category="CORRECTIONS",
    description="Pushback on a system decision ('why did you choose 18k for me') must route to the order/product question it's actually about, not answer_policy_question.",
    turns=[
        Turn(message=f"I want the {RING}, deliver to Osu", expected_tool="propose_order"),
        Turn(
            message="I didn't choose the karat so why did you choose 18k for me?",
            expected_tool=("propose_order", "get_product_price"),
            expected_not_contains=["warranty", "return"],
        ),
    ],
    source="confirmed live, 2026-08-20 -- llm.py's own documented failure case this rule exists to prevent",
))

_add(Scenario(
    id="corrections-03-address-change-updates-zone",
    category="CORRECTIONS",
    description="A genuinely different delivery address must clear the old delivery_option, not leave a Kumasi address paired with an Accra rider.",
    turns=[
        Turn(message=f"I want the {NECKLACE}, in 14k", expected_tool="get_product_price"),
        Turn(message="I'll take 1, deliver to East Legon", expected_tool="propose_order"),
        Turn(
            message="actually deliver to Kumasi instead",
            expected_tool="propose_order",
            expected_not_contains=["rider delivery within Accra"],
        ),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 11, fixed task #136",
))


# ---------------------------------------------------------------------
# REFERENCES
# ---------------------------------------------------------------------

_add(Scenario(
    id="references-01-ordinal-into-list",
    category="REFERENCES",
    description="'The second one' after a list must resolve by POSITION against last_presented_products, not a guess.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(message="tell me more about the second one", expected_tool=("get_product_price", "get_product_karat_options")),
    ],
    source="task #181 (last_presented_products)",
))

_add(Scenario(
    id="references-02-bare-demonstrative-after-multi-item-list",
    category="REFERENCES",
    description="A bare 'I'll take that one' with no ordinal, after MULTIPLE items shown, must ask which one -- not silently guess (this is the exact bug in the real 2026-08-30 trace that motivated task #197).",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(
            message="I'll take that one",
            expected_tool="converse",
            manner_note="Must ask which item, not silently resolve to a position and jump into propose_order.",
        ),
    ],
    source="real live trace, session b5591185-3f3f-4d3a-ac94-6cfe8b675f87, 2026-08-30 (pre-#197) -- this is the reproduction of that exact failure, now expected to be fixed by task #197's commits a20b313/62665df",
))

_add(Scenario(
    id="references-04-golden-path-full-journey",
    category="REFERENCES",
    description="Webb's own end-to-end journey (2026-09-01): browse, refer, inspect, price, order, correct delivery twice, confirm -- without the customer ever having to repeat a product name. This is the corpus's single highest-value scenario: if this passes end to end, KasaFlow is doing what Webb defined as done ('a real customer can shop naturally for five minutes without having to repeat themselves or correct the assistant').",
    turns=[
        Turn(message="Show me some rings.", expected_tool="recommend_products"),
        Turn(message="Second one.", expected_tool=("get_product_price", "get_product_karat_options", "propose_order")),
        Turn(message="Can I see it?", expected_tool="get_product_price"),
        Turn(message="How heavy?", expected_tool="get_product_weight"),
        Turn(message="What about 18k?", expected_tool="get_product_price", expected_fields={"material": "18k"}),
        Turn(message="I'll take two.", expected_tool="propose_order", expected_fields={"quantity": 2}),
        Turn(message="Actually send them to Kumasi.", expected_tool="propose_order", expected_not_contains=["rider delivery within Accra"]),
        Turn(message="No, Cape Coast.", expected_tool="propose_order", expected_not_contains=["Kumasi"]),
        Turn(message="Okay confirm.", expected_tool="confirm_order"),
    ],
    source="Webb, 2026-09-01, given verbatim as the target experience -- not yet live-tested as one continuous run",
))

_add(Scenario(
    id="references-03-details-preserved-through-clarification",
    category="REFERENCES",
    description="Details stated alongside an ambiguous reference ('that one, in 14k, two pieces') must survive the clarification round-trip, not be dropped.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(message="that one, in 14k, two pieces", expected_tool="converse"),
        Turn(
            message="the second one",
            expected_tool=("propose_order", "get_product_price"),
            expected_fields={"material": "14k", "quantity": 2},
            manner_note="material/quantity from the earlier ambiguous turn must reappear here without being restated.",
        ),
    ],
    source="Webb's exact test case, 2026-08-31 (\"that one, in 14k\") -- task #197 follow-up commit 62665df's clarification_details mechanism",
))


# ---------------------------------------------------------------------
# AMBIGUITY
# ---------------------------------------------------------------------

_add(Scenario(
    id="ambiguity-01-bare-category-two-matches",
    category="AMBIGUITY",
    description="A bare category name that matches more than one item just shown must be flagged ambiguous, not silently resolved to the first match.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(message="the ring", expected_tool="converse"),
    ],
    source="task #197's core case (bare category list reference)",
))

_add(Scenario(
    id="ambiguity-02-category-not-in-list-not-flagged",
    category="AMBIGUITY",
    description="A bare category reference for something that was NOT in the list just shown (no necklace among the rings) must not be forced through the ambiguity clarification either -- there is nothing to disambiguate.",
    turns=[
        Turn(message="show me the rings", expected_tool="recommend_products"),
        Turn(
            message="the necklace",
            expected_tool=("recommend_products", "get_product_price", "get_product_karat_options", "propose_order"),
            manner_note=(
                "If this comes back converse instead, check by hand whether it is the #197 "
                "list-ambiguity guard misfiring (a real regression -- 'Necklaces' was never "
                "even a category in the list just shown, so it should never appear in "
                "ambiguous_categories) or an ordinary 'I don't have enough to go on' "
                "clarification unrelated to #197. The evaluator cannot tell these apart from "
                "the tool name alone, so expected_tool deliberately excludes converse here "
                "rather than guessing which case a converse result would be."
            ),
        ),
    ],
    source="regression guard for task #197's category-position logic, rewritten 2026-09-01 -- the original version's setup turn named two specific products directly rather than browsing a category, so last_presented_products was never populated the way the scenario assumed, and the live run's failure (recommend_products instead of a price/karat/order tool) was this scenario's own design flaw, not a KasaFlow bug",
))

_add(Scenario(
    id="ambiguity-03-genuinely-unclear-product-name",
    category="AMBIGUITY",
    description="An unstocked/made-up product name should say so honestly, not guess at a similar-sounding real item.",
    turns=[
        Turn(message="how much is the unicorn pendant", expected_tool="get_product_price", expected_contains=["couldn't find"]),
    ],
    source="confirmed live, 2026-08-12 -- product_tool.py's get_product_price() min_keyword_overlap=1 guard exists specifically for this case",
))


# ---------------------------------------------------------------------
# CONFIRMATION
# ---------------------------------------------------------------------

_add(Scenario(
    id="confirmation-01-plain-yes",
    category="CONFIRMATION",
    description="A plain 'yes' right after a proposal should confirm it.",
    turns=[
        Turn(message=f"I want the {RING}, in 18k, 1 of them, deliver to East Legon, rider delivery within Accra", expected_tool="propose_order"),
        Turn(message="confirm", expected_tool="confirm_order"),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turns 5-6",
))

_add(Scenario(
    id="confirmation-02-stale-yes-not-confirmed",
    category="CONFIRMATION",
    description="A bare 'yes' is NOT a confirmation once something else has been asked/offered since the proposal (awaiting_confirmation must be False by then).",
    turns=[
        Turn(message=f"I want the {RING}, in 18k, 1 of them, deliver to East Legon, rider delivery within Accra", expected_tool="propose_order"),
        Turn(message="do you deliver internationally too", expected_tool="get_delivery_information"),
        Turn(message="yes", expected_tool=("confirm_order",), manner_note="This is the one genuinely ambiguous case in the corpus -- whether a bare 'yes' here should re-confirm the ORIGINAL order or be treated as answering the delivery question is a real product judgement call, not obviously wrong either way. Flag for Webb's own read rather than a hard pass/fail."),
    ],
    source="designed from memory.py's is_awaiting_confirmation() docstring and task #99 (awaiting_field/awaiting_action state machine); the expected_tool here is a placeholder, not a confirmed-correct answer -- see manner_note",
))

_add(Scenario(
    id="confirmation-03-bare-agreement-not-address",
    category="CONFIRMATION",
    description="A bare agreement word ('yh', 'okay') must never be captured as a delivery_address.",
    turns=[
        Turn(message=f"{RING}, in 18k", expected_tool="get_product_price"),
        Turn(message="yh", expected_tool=("get_delivery_information", "generate_quote"), expected_not_contains=["deliver to Yh", "deliver to yh"]),
    ],
    source="task #101 (\"bare-agreement word wrongly resolved as delivery address\")",
))


# ---------------------------------------------------------------------
# POST-CONFIRMATION
# ---------------------------------------------------------------------

_add(Scenario(
    id="post-confirmation-01-order-status-by-memory",
    category="POST-CONFIRMATION",
    description="'Where is my order' right after confirming should resolve to that same order without the customer repeating the number.",
    turns=[
        Turn(message=f"I want the {RING}, in 18k, 1 of them, deliver to East Legon, rider delivery within Accra", expected_tool="propose_order"),
        Turn(message="confirm", expected_tool="confirm_order"),
        Turn(message="where is my order", expected_tool="get_order_status", expected_fields={"order_id": "unknown"}),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turns 6-7",
))

_add(Scenario(
    id="post-confirmation-02-cancel-nonexistent",
    category="POST-CONFIRMATION",
    description="Cancelling an order number that doesn't exist should say so honestly, not silently succeed.",
    turns=[
        Turn(message="cancel order 99999", expected_tool="cancel_order", expected_contains=["couldn't find"]),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 9",
))

_add(Scenario(
    id="post-confirmation-03-order-details-from-woocommerce-not-conversation",
    category="POST-CONFIRMATION",
    description="A question about what a CONFIRMED order actually contains must be answered from get_order_status's real WooCommerce read, not from what the conversation said a few turns ago.",
    turns=[
        Turn(message=f"I want the {RING}, in 12k, deliver to Osu, rider delivery within Accra", expected_tool="propose_order"),
        Turn(message="wait, 14k rather", expected_tool="propose_order"),
        Turn(message="confirm", expected_tool="confirm_order"),
        Turn(message="what karat did I order", expected_tool="get_order_status"),
    ],
    source="task #175/#176 (grounding confirmed-order facts, anti-hallucination guardrail) -- llm.py's own documented rule that a mid-conversation correction is not reliable evidence of the final confirmed karat",
))


# ---------------------------------------------------------------------
# GLOBAL DELIVERY
# ---------------------------------------------------------------------

_add(Scenario(
    id="global-delivery-01-non-rider-ghana-zone",
    category="GLOBAL DELIVERY",
    description="A real Ghanaian place outside the rider zones should say yes with an honest caveat, not force it into Accra/Kumasi rider or refuse it.",
    turns=[
        Turn(
            message="what of bolgatanga, I'm in the northern region",
            expected_tool="get_delivery_information",
            expected_contains=["Bolgatanga"],
        ),
    ],
    source="confirmed live, KasaFlow_Scenario_Test_Script.md scenario 8 turn 3",
))

_add(Scenario(
    id="global-delivery-02-international",
    category="GLOBAL DELIVERY",
    description="A non-Ghana destination should map to the international shipping arrangement or the honest generic fallback, not a Ghana-only zone.",
    turns=[
        Turn(message="what about London", expected_tool="get_delivery_information"),
    ],
    source="confirmed live (structurally), KasaFlow_Scenario_Test_Script.md scenario 8 turn 4 -- exact response depends on whether the Google Maps key is live, script names both valid outcomes explicitly",
))

_add(Scenario(
    id="global-delivery-03-wrong-region-not-assumed",
    category="GLOBAL DELIVERY",
    description="A Tamale address must never be silently priced under the Kumasi rider zone.",
    turns=[
        Turn(
            message=f"I want the {RING}, in 18k, deliver to Tamale",
            expected_tool="propose_order",
            expected_not_contains=["kumasi_rider", "rider delivery within Kumasi"],
        ),
    ],
    source="task #61 (\"wrong region: Kumasi rider chosen, Tamale address\")",
))


# ---------------------------------------------------------------------
# GHANAIAN LANGUAGE
# ---------------------------------------------------------------------

_add(Scenario(
    id="ghanaian-language-01-twi-greeting",
    category="GHANAIAN LANGUAGE",
    description="A Twi greeting should get a warm, natural converse reply in kind, not an English-only stock phrase.",
    turns=[
        Turn(message="Maakye", expected_tool="converse"),
    ],
    source="designed from llm.py's own converse guidance (\"match the customer's language\"), not yet live-tested with this exact greeting",
))

_add(Scenario(
    id="ghanaian-language-02-medaase-thanks",
    category="GHANAIAN LANGUAGE",
    description="'Medaase' (thank you) should be recognised as thanks, not routed as an unclear/unknown message.",
    turns=[
        Turn(message="medaase", expected_tool="converse"),
    ],
    source="llm.py's own worked converse example",
))

_add(Scenario(
    id="ghanaian-language-03-mixed-twi-english-product-ask",
    category="GHANAIAN LANGUAGE",
    description="A natural mixed Twi/English product question should still route to the right business tool, not fall back to converse just because it isn't pure English.",
    turns=[
        Turn(message=f"3ho b3n na {RING} y3", expected_tool="get_product_price"),
    ],
    source="designed from category spec, not yet live-tested -- flagged as the corpus's weakest-confidence scenario, since exact live Twi phrasing accuracy has not been independently confirmed here",
))


# ---------------------------------------------------------------------
# SOCIAL/REACTIVE
# ---------------------------------------------------------------------

_add(Scenario(
    id="social-01-greeting",
    category="SOCIAL/REACTIVE",
    description="A bare greeting is converse, not routed to any business tool.",
    turns=[
        Turn(message="hey", expected_tool="converse"),
    ],
    source="llm.py's own worked converse example",
))

_add(Scenario(
    id="social-02-reaction-does-not-derail-order",
    category="SOCIAL/REACTIVE",
    description="A reaction/emoji mid-order should not clear or restart the in-progress proposal.",
    turns=[
        Turn(message=f"I want the {RING}, in 18k, deliver to Osu", expected_tool="propose_order"),
        Turn(message="😍", expected_tool="converse"),
        Turn(message="confirm", expected_tool="confirm_order"),
    ],
    source="designed from task #66 (\"unrelated question mid-order\") applied to a reaction instead of a question; not independently live-tested with an emoji specifically",
))

_add(Scenario(
    id="social-03-unrelated-question-mid-order-does-not-lose-state",
    category="SOCIAL/REACTIVE",
    description="An unrelated question mid-order should be answered without discarding the pending order.",
    turns=[
        Turn(message=f"I want the {RING}, in 18k, deliver to Osu", expected_tool="propose_order"),
        Turn(message="do you have a physical shop", expected_tool=("answer_policy_question", "converse")),
        Turn(message="confirm", expected_tool="confirm_order"),
    ],
    source="confirmed live (as a class of bug), task #66 (\"Scenario 7: unrelated question mid-order\")",
))
