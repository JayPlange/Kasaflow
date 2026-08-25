"""
Turns a tool's raw result dict into a sentence a customer can actually
read over WhatsApp.

Why this exists and router.py doesn't just do it: route_customer()'s
JSON contract is already a real, presumably-already-integrated-against
API response shape (the README is explicit that this contract has
stayed stable through an internal rewrite). Reshaping that into prose
for WhatsApp specifically, rather than changing what /process returns
for everyone, keeps that promise intact.

Written as duck-typed shape matching (which keys are present) rather
than router.py telling it which tool ran, on purpose -- it means this
file can be deleted and rebuilt without touching router.py or the tools
themselves at all.

Tone, deliberately: reads like a warm boutique salesperson, not a
form-letter bot -- but every word still traces back to a real field in
the tool's result. Warmth is in phrasing and structure only, never in
inventing a claim (a size, a price, a delivery estimate) the tool
didn't actually return.

Bold markers use WhatsApp's own formatting syntax -- a single asterisk
pair, e.g. `*GH₵1,200.00*` -- not markdown's `**double asterisk**`,
which WhatsApp displays as literal asterisks rather than bold text.
Only ever wrapped around the headline fact of a line (a product name, a
price, an order number): the point a customer scanning quickly needs,
kept visually separate from the advisory clause that follows it (see
_format_recommendation_group -- this is what "the responses should
bolden the first points" was asking for).
"""


import re

from services.delivery_tool import delivery_option_label, delivery_options_phrase

# Matches the karat digits at the start of any of the catalogue's real
# material formats -- same pattern as recommendation_service.py's own
# _extract_karat, duplicated rather than imported since this file is
# meant to stay a standalone, deletable formatting layer (see module
# docstring) with no dependency on how a tool produced its result.
# (delivery_options_phrase() above is the one exception -- it's the
# single source of truth for the exact delivery-option wording, shared
# with order_tool.py's clarifying question, so the two can't drift out
# of sync with each other.)
_KARAT_RE = re.compile(r"^\s*(\d+)\s*k?\b", re.IGNORECASE)


def _extract_karat(value: str | None) -> str | None:
    if not value:
        return None
    match = _KARAT_RE.match(value)
    return match.group(1) if match else None


def _group_by_product(items: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group catalogue entries sharing a product name, preserving first-seen
    order. Exists because the same product often comes back as several
    entries that only differ in their "material" field -- which sometimes
    really is the material, but just as often encodes a size/variant
    (see woocommerce_sync.py's _variation_label()). Grouping first means a
    5-size ring reads as one product with size options, not five
    look-alike lines a customer has to puzzle over."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for item in items:
        name = item["product"]
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(item)
    return [(name, groups[name]) for name in order]


def _select_diverse_groups(
    groups: list[tuple[str, list[dict]]], max_groups: int
) -> list[tuple[str, list[dict]]]:
    """Pick up to max_groups groups, round-robinning across categories
    rather than just taking the first N in catalogue order.

    Exists because data/products.json lists every Necklaces row before
    any Rings row -- a plain [:4] slice on an unfiltered "what do you
    have" browse (no category stated) always returned 4 necklaces and
    never a single ring, hiding 96% of the real catalogue behind pure
    file order. Round-robinning by category means a broad browse shows
    a representative mix instead."""
    if len(groups) <= max_groups:
        return groups

    buckets: dict[str, list[tuple[str, list[dict]]]] = {}
    category_order: list[str] = []
    for name, variants in groups:
        category = variants[0].get("category") or ""
        if category not in buckets:
            buckets[category] = []
            category_order.append(category)
        buckets[category].append((name, variants))

    selected: list[tuple[str, list[dict]]] = []
    i = 0
    while len(selected) < max_groups and any(buckets[c] for c in category_order):
        category = category_order[i % len(category_order)]
        if buckets[category]:
            selected.append(buckets[category].pop(0))
        i += 1
    return selected


def select_presented_groups(items: list[dict], max_groups: int = 4) -> list[tuple[str, list[dict]]]:
    """Public wrapper around _group_by_product() + _select_diverse_groups()
    above -- the exact selection that decides what a recommend_products
    reply actually shows a customer, in the order it's shown.

    Exists as its own function, not just inlined in format_for_customer()
    below, so router.py can call the SAME selection when persisting
    memory.set_last_presented_products() (see that function's docstring)
    -- one function computing the list, two callers (render it, remember
    it), rather than two places that could quietly drift out of sync.
    This codebase has already been bitten by exactly that shape of bug
    twice (the karat-representation mismatch and the delivery-option
    staleness bug both came from two code paths silently disagreeing
    about the same fact) -- not repeating it a third time for "what did
    we just show this customer" was the whole point of designing it this
    way (see KasaFlow_Conversation_Context_Design.md, piece 1)."""
    return _select_diverse_groups(_group_by_product(items), max_groups=max_groups)


# Above this many variants of one product, listing every size/karat is a
# wall of text, not a readable answer -- a price range plus an invitation
# to narrow down reads the way an actual salesperson would answer "what
# rings do you have" (a broad "Rings" query with no karat/size stated can
# genuinely match 3000+ raw catalogue rows for a handful of product names).
_MAX_VARIANTS_LISTED = 3


def _format_recommendation_group(name: str, variants: list[dict]) -> str:
    if len(variants) == 1:
        v = variants[0]
        return f"- *{name}* ({v['material']}): *GH₵{v['price']:,.2f}*"

    prices = [v["price"] for v in variants]
    low, high = min(prices), max(prices)

    if len(variants) <= _MAX_VARIANTS_LISTED:
        if low == high:
            options = ", ".join(v["material"] for v in variants)
            return f"- *{name}*: *GH₵{low:,.2f}* -- available in {options}"
        # Same small handful of variants, but not all the same price --
        # can't collapse to one line without hiding a real price difference.
        # Sub-bullet uses "-", never "*" -- WhatsApp only has one meaning
        # for "*" (bold), no separate bullet-list markdown, so a "*"
        # bullet marker on the same line as a bolded *GH₵...* price
        # collides: the parser pairs the bullet's "*" with the price's
        # opening "*" instead of the price's own closing one, leaving a
        # stray unbolded "*" dangling at the end of the line (confirmed
        # live, 2026-08-12).
        sub_lines = "\n".join(f"   - {v['material']}: *GH₵{v['price']:,.2f}*" for v in variants)
        return f"- *{name}*:\n{sub_lines}"

    # Only call it "karat" variance if it actually varies within this set
    # -- if the customer already stated a karat (recommend_products
    # filtered to it before this ever runs), every remaining variant
    # shares that karat, and telling them it "comes in different karats"
    # would be inaccurate, not just imprecise.
    distinct_karats = {k for k in (_extract_karat(v["material"]) for v in variants) if k}
    varies_by_karat = len(distinct_karats) > 1
    unit = "sizes/karats" if varies_by_karat else "sizes"

    # Deliberately no per-item "tell me your size/karat" call-to-action
    # here any more -- format_for_customer()'s single closing question
    # ("Want me to tell you more about any of these... or narrow it down
    # by size or karat?") already covers every line in the list. Repeating
    # the same ask on every multi-variant item read as noisy and robotic
    # once more than one item in the same list needed it (confirmed live,
    # 2026-08-16).
    if low == high:
        return f"- *{name}*: *GH₵{low:,.2f}* -- comes in {len(variants)} {unit}"
    qualifier = "karat/size" if varies_by_karat else "size"
    return (
        f"- *{name}*: *GH₵{low:,.2f}-GH₵{high:,.2f}* depending on {qualifier} "
        f"({len(variants)} options)"
    )


def format_for_customer(result: dict | None) -> str:
    if result is not None and "correction_note" in result:
        # router.py's propose_order correction acknowledgement (see
        # _describe_order_corrections()) -- prepended to whatever reply
        # this same result would otherwise produce (the next
        # missing-field question, or a full proposal), so a correction
        # like "wait, 14k rather" gets acknowledged instead of silently
        # applied. Formatted via a plain recursive call on the same
        # result with the note key stripped, rather than a special case
        # per shape below, so this works no matter what propose_order's
        # underlying reply shape is.
        note = result["correction_note"]
        remainder = {k: v for k, v in result.items() if k != "correction_note"}
        return f"{note} {format_for_customer(remainder)}"

    if result is not None and "results" in result:
        # router.py's multi-request shape -- a message that contained
        # more than one distinct ask. Format each sub-result exactly
        # the same way a single-result reply would be, then present
        # them as a numbered list so the customer can tell which
        # answer belongs to which part of what they asked.
        parts = [format_for_customer(sub_result) for sub_result in result["results"]]
        if len(parts) == 1:
            return parts[0]
        return "\n\n".join(f"{i}. {part}" for i, part in enumerate(parts, start=1))

    if result is None:
        # get_product_price returns None directly (not an exception) when
        # nothing matches, including the no-match case after the semantic
        # search fallback comes back empty. generate_quote already wraps
        # that into a message for its own callers, but a bare
        # get_product_price call reaches this function with nothing in
        # front of it -- same customer-facing message either way, rather
        # than crashing on `"error" in None`.
        return "Hmm, I couldn't find that one -- could you tell me a bit more about what you're after?"

    if "error" in result:
        return result["error"]

    if "conversation_reply" in result:
        # router.py's converse shape -- the LLM already wrote the actual
        # customer-facing reply itself (see llm.py's tool 8 description),
        # since there's no business fact here for a deterministic template
        # to ground. Passed straight through, not reformatted.
        return result["conversation_reply"]

    if "proposal" in result:
        # order_tool.propose_order()'s shape -- a priced, not-yet-placed
        # order awaiting the customer's explicit confirmation. total is
        # product cost only -- delivery isn't priced automatically, see
        # delivery_tool.py's module docstring, so this deliberately
        # doesn't claim a delivery cost/time it doesn't actually have.
        p = result["proposal"]
        delivery_label = p.get("delivery_option_label") or "the delivery option you chose"
        return (
            f"Lovely choice -- *{p['quantity']} x {p['material']} {p['product']}* comes to "
            f"*GH₵{p['total']:,.2f}*. Delivery to {p['delivery_address']} via {delivery_label} -- "
            f"our team will confirm the exact delivery cost and timing with you directly. "
            f"Just reply CONFIRM and I'll get that placed for you."
        )

    if "order_confirmation" in result:
        # order_tool.confirm_order()'s shape -- the order now exists in
        # WooCommerce (status "on-hold": created, payment not yet
        # collected -- see order_tool.py's module docstring). The order
        # itself has also been handed to a human to arrange delivery
        # (see confirm_order()'s staff notification) -- say so, rather
        # than implying delivery is already sorted.
        #
        # "placed", not "confirmed": "confirmed" implies a finished,
        # settled transaction, but nothing has been paid or scheduled
        # yet -- that's still ahead, via staff. Confirmed live,
        # 2026-08-14, that the stronger word reads as overpromising once
        # you're actually looking at what state the order is really in.
        c = result["order_confirmation"]
        delivery_label = c.get("delivery_option_label") or "your chosen delivery option"
        return (
            f"All set -- *order #{c['order_id']}* has been placed for *GH₵{c['total']:,.2f}*, "
            f"delivering to {c['delivery_address']} via {delivery_label}. Our team will be in "
            f"touch shortly to arrange payment and delivery."
        )

    if "order_cancellation" in result:
        # order_tool.cancel_order()'s success shape.
        c = result["order_cancellation"]
        return (
            f"Done -- *order #{c['order_id']}* has been cancelled. Let me know if you'd like "
            f"to place a new one."
        )

    if "order_already_cancelled" in result:
        # order_tool.cancel_order()'s idempotent-repeat shape -- the
        # order's live WooCommerce status was already "cancelled" before
        # this request, most likely a duplicated customer message
        # ("cancel" sent twice) or a repeat after staff already
        # actioned it. Say so plainly rather than trying to cancel it
        # again.
        c = result["order_already_cancelled"]
        return f"Order *#{c['order_id']}* is already cancelled -- nothing further needed there."

    if "order_escalation" in result:
        # order_tool.cancel_order()'s shape when the order exists but
        # its live WooCommerce status isn't one this tool will touch
        # automatically (already shipped, completed, refunded, etc.) --
        # see order_tool.py's _CANCELLABLE_STATUSES. Say what's true
        # (found it, can't act on it directly) rather than either
        # silently failing or claiming a cancellation that didn't
        # happen.
        e = result["order_escalation"]
        return (
            f"I found *order #{e['order_id']}*, but it's already {e['status']} and can't be "
            f"cancelled automatically from here -- I've let our team know so they can help "
            f"directly."
        )

    if "order_status" in result:
        # order_tool.get_order_status()'s shape -- always a fresh
        # WooCommerce lookup, never assumed from what this session last
        # knew (see that function's docstring).
        s = result["order_status"]
        item_line = f" ({s['item_summary']})" if s.get("item_summary") else ""
        total_value = s.get("total")
        total_line = ""
        if total_value not in (None, ""):
            try:
                total_line = f" Total: GH₵{float(total_value):,.2f}."
            except (TypeError, ValueError):
                total_line = f" Total: GH₵{total_value}."
        return f"Order *#{s['order_id']}* is {s['status_label']}{item_line}.{total_line}"

    if "answer" in result:
        # answer_policy_question's shape -- already a sentence.
        return result["answer"]

    if "message" in result and "product" not in result:
        # generate_quote's "couldn't find that product" case.
        return result["message"]

    if "identified_product" in result:
        # A photo match (services/photo_match_tool.py's
        # identify_product_from_photo) -- the exact product is already
        # known here, not just a category, so this shows every karat
        # price rather than the recommendations list's 3-variant cap
        # (_MAX_VARIANTS_LISTED above): there's exactly one product to
        # describe, not a browse list to keep short.
        #
        # Deliberately hedged wording ("looks like", not "that's the"),
        # even though this is presented as a single confident match, not
        # a browse list. Confirmed live, 2026-08-18: this matched
        # confidently to the wrong one of two similar cross
        # necklace-and-earring sets in the catalogue, despite the
        # temperature=0 and conservative-prompt changes made after an
        # earlier, similar mismatch the day before -- those reduced but
        # did not eliminate the risk of a wrong-but-confident visual
        # match. The product's own photo is always shown alongside this
        # reply so a human can catch a wrong match by eye; the wording
        # itself should invite that check rather than assert certainty
        # the system doesn't reliably have.
        p = result["identified_product"]
        variants = p.get("variants") or []
        if not variants:
            return (
                f"This looks like our *{p['product']}* -- but I couldn't pull up current "
                f"pricing for it just now. Let me get someone to check for you."
            )
        sub_lines = "\n".join(f"- {v['material']}: *GH₵{v['price']:,.2f}*" for v in variants)
        options_phrase = delivery_options_phrase()
        return (
            f"This looks like our *{p['product']}* -- it's in stock. Prices by karat:\n{sub_lines}\n\n"
            f"We deliver via {options_phrase}. Let me know if that's not quite the right one and "
            f"I'll help you find the correct piece -- otherwise, want me to get an order started?"
        )

    if "karat_options" in result:
        # product_tool.list_karat_options()'s shape -- the customer asked
        # which karats a specific, already-identified product comes in,
        # not to price one specific karat (see llm.py's tool 10). Empty
        # list means the product_name didn't match anything real; same
        # wording as get_product_price's own no-match message, since
        # that's exactly the same underlying failure from the customer's
        # point of view.
        variants = result["karat_options"]
        if not variants:
            return "Hmm, I couldn't find that one -- could you tell me a bit more about what you're after?"
        sub_lines = "\n".join(f"- {v['material']}: *GH₵{v['price']:,.2f}*" for v in variants)
        return f"The *{result['product']}* comes in:\n{sub_lines}\n\nWhich karat would you like?"

    if "weight" in result:
        # product_tool.get_product_weight()'s shape -- "how heavy is
        # it"/"how many grams" (llm.py's tool 11), distinct from a price
        # question. weight is None (not the key's absence) for the small
        # handful of real catalogue products with no parseable weight in
        # their name -- say so honestly rather than a generic no-match
        # message, since the product itself WAS found.
        if result["weight"] is None:
            return f"I don't have the weight on file for the *{result['product']}*, sorry."
        return f"The *{result['product']}* weighs {result['weight']}."

    if "recommendations" in result:
        items = result["recommendations"]
        if not items:
            # recommendation_service.py sends available_categories along
            # when the customer's category genuinely isn't stocked (e.g.
            # bracelets/earrings) -- a real salesperson says what they DO
            # have instead of just "no", so do the same here rather than
            # a flat dead end.
            available = result.get("available_categories")
            if available:
                if len(available) == 1:
                    listed = available[0]
                else:
                    listed = ", ".join(available[:-1]) + f" and {available[-1]}"
                asked = result.get("requested_category", "that")
                return (
                    f"Hmm, I don't have any {asked} right now -- but I do have {listed}. "
                    f"Want me to show you what's there?"
                )
            return "Hmm, I don't have anything matching that right now -- want me to show you something similar instead?"
        groups = select_presented_groups(items, max_groups=4)
        lines = [_format_recommendation_group(name, variants) for name, variants in groups]
        return (
            "Here's what I found for you:\n\n"
            + "\n\n".join(lines)
            + "\n\nWant me to tell you more about any of these, show you a photo, or narrow it down by size or karat?"
        )

    if "price" in result and "delivery_options" in result:
        # generate_quote's shape -- price plus the real delivery choices
        # (not a cost/time, see delivery_tool.py's module docstring).
        options_phrase = delivery_options_phrase(result["delivery_options"])
        return (
            f"Good news -- the {result['material']} {result['product']} is "
            f"*GH₵{result['price']:,.2f}*. For delivery, would you like {options_phrase}?"
        )

    if "price" in result:
        return f"The {result['material']} {result['product']} is *GH₵{result['price']:,.2f}*. Want to know about delivery too?"

    if "matched_zone" in result:
        # get_delivery_information(address=...)'s location-aware shape --
        # a customer named a specific place ("what of Bolgatanga"), not
        # a generic delivery question. See that function's docstring.
        #
        # Every branch here leads with a direct yes/no, deliberately --
        # the customer asked "does your delivery service cover my
        # location?", not "what are your delivery options?" (Webb/GPT
        # review, 2026-08-22, of the live Bolgatanga case). Opening with
        # the generic three-way menu answers a different, more generic
        # question than the one actually asked, even when the menu
        # technically contains the right answer somewhere in it.
        zone = result["matched_zone"]
        address = result.get("queried_address") or "that address"
        if zone in ("accra_rider", "kumasi_rider", "international"):
            label = delivery_option_label(zone)
            return f"Yes, we deliver to {address} -- that's covered by our {label}."
        if zone == "ghana_other":
            # A real Ghanaian place, just not a rider zone -- same
            # "let our team confirm" handling propose_order() already
            # gives this exact classification, not a guessed-at
            # arrangement dressed up as a real one. Names cost AND
            # timing explicitly (not just "the details") -- those are
            # the two concrete things this business can't compute itself
            # for a non-rider-zone delivery (see delivery_tool.py's
            # module docstring), so naming them is what actually answers
            # the question rather than a vague "we'll sort it".
            return (
                f"Yes, we can arrange delivery to {address}, although it isn't one of our usual "
                f"rider zones -- our team will confirm the delivery cost and timing with you."
            )
        # Genuinely unclear (couldn't classify the address at all) --
        # same conservative "ask/list rather than guess" fallback as
        # everywhere else in this codebase.
        options_phrase = delivery_options_phrase(result["delivery_options"])
        return (
            f"I'm not totally sure how delivery works for {address} -- we deliver a couple of "
            f"ways: {options_phrase}. Go ahead and place your order and our team will confirm the "
            f"best option for you."
        )

    if "delivery_options" in result:
        # get_delivery_information()'s bare shape -- a customer asking
        # generically "what are your delivery options", no specific
        # place named.
        options_phrase = delivery_options_phrase(result["delivery_options"])
        return f"We deliver a couple of ways: {options_phrase}. Which works for you?"

    # Unknown shape -- surface something rather than send nothing, but
    # this branch being hit means a new tool was added without updating
    # this file, worth noticing in logs, not just in a customer's chat.
    return "Hmm, let me get back to you on that one."
