"""
Top-level orchestration: customer message -> tool selection -> tool
execution -> result. This is the only place that decides what the
customer sees when something goes wrong upstream.
"""

import json
import logging
import re

from services.llm import ToolSelectionError, understand_customer
from services.memory import (
    fill_missing_context,
    get_awaiting_field,
    get_just_confirmed_order,
    get_last_action_outcome,
    get_last_presented_products,
    get_last_priced_product,
    get_order_draft,
    get_pending_clarification_details,
    get_pending_intent,
    get_session_store,
    increment_weight_ask_count,
    is_awaiting_confirmation,
    remember_context,
    set_awaiting_confirmation,
    set_awaiting_field,
    set_just_confirmed_order,
    set_pending_clarification_details,
    set_last_action_outcome,
    set_last_presented_products,
    set_last_priced_product,
    set_pending_intent,
)
from services.order_tool import get_pending_order_summary
from services.response_formatter import select_presented_groups
from services.tool_executor import ToolExecutionError, execute_tool

logger = logging.getLogger(__name__)

# propose_order/confirm_order/cancel_order (services/order_tool.py) all
# need to know which customer's session they're acting on, but that's
# never something the LLM decides or the customer states -- it comes
# from the channel (WhatsApp) this message arrived on. Every other
# registered tool's **kwargs contract is exactly whatever the LLM
# returned, nothing more, so session_id is injected only for these three
# by name rather than added to every tool call: passing an unexpected
# session_id kwarg to any of the other five would raise inside
# tool_executor.py's TypeError handler.
_SESSION_AWARE_TOOLS = {"propose_order", "confirm_order", "cancel_order", "get_order_status"}

# converse (services/llm.py) isn't a real registered tool -- there's no
# deterministic business logic behind it, no lookup, nothing to execute.
# The LLM writes the actual customer-facing reply itself as part of the
# same tool-selection call, precisely because there's no business fact
# involved that a tool would need to ground it in. Handled entirely here,
# before execute_tool()/tool_registry.py ever see it: registering it as a
# "tool" that just echoes back its own argument would be a pointless
# round-trip through machinery built for looking things up, and would
# wrongly make it eligible for fill_missing_context()/remember_context()
# below, which exist to resolve/save business arguments (product,
# material, delivery address, ...) -- a converse reply has none of those,
# and a stray "reply" key must never leak into or overwrite that state.
_CONVERSATION_TOOL = "converse"
_CONVERSATION_FALLBACK_REPLY = "Hey! How can I help you today?"

# The two tools whose only failure mode relevant here is "customer wants
# this, but hasn't named a product yet" -- see memory.set_pending_intent()
# and llm.py's _pending_intent_state_line() for why that specific gap
# needs to be tracked across turns.
_PENDING_INTENT_TOOLS = {"get_product_price", "generate_quote"}

# Both return {"product": <resolved catalogue name>, ...} on a genuine
# match (see product_tool.get_product_price() and
# quote_service.generate_quote()) -- the same two tools tracked by
# _PENDING_INTENT_TOOLS above, but for the opposite case: a lookup that
# DID resolve. See memory.set_last_priced_product() and llm.py's
# _last_priced_product_state_line() for why a bare karat-only follow-up
# needs this remembered.
# get_product_karat_options joins this set too: it also resolves to one
# specific product (see product_tool.list_karat_options()'s "product" key,
# same shape as the other two), and a bare karat-only follow-up right
# after seeing the options list ("okay what about 12") needs the same
# continuation handling as a follow-up right after a price quote.
# get_product_weight() joins for the identical reason -- "how heavy is
# it" resolves to one specific product too (product_tool.
# get_product_weight()'s "product" key), and a follow-up right after a
# weight answer should get the same active-product continuation.
_PRICING_TOOLS = {"get_product_price", "generate_quote", "get_product_karat_options", "get_product_weight"}

# A genuine category browse means the topic has moved on from one
# specific priced item -- see memory.set_last_priced_product()'s
# docstring for why this specifically clears rather than leaves the
# old value in place.
_RECOMMEND_TOOL = "recommend_products"

# propose_order is the only tool whose arguments can genuinely "correct"
# something the customer already gave earlier in the same order -- see
# _describe_order_corrections() below.
_ORDER_TOOL = "propose_order"
_CONFIRM_TOOL = "confirm_order"
_PRICE_TOOL = "get_product_price"
_WEIGHT_TOOL = "get_product_weight"

_ORDER_CORRECTION_FIELDS = {
    "product_name": "the item",
    "material": "the karat",
    "quantity": "the quantity",
    "delivery_address": "the delivery address",
}


def _found_nothing(result: dict | None) -> bool:
    """True when a tool ran successfully but didn't actually find what
    the customer asked about: get_product_price's bare None, an empty
    recommend_products category, or generate_quote's "couldn't find that
    product" message.

    Exists because execute_tool() only raises on a genuine failure --
    an empty/no-match result is a normal, successful return, so without
    this check remember_context() below would happily save a category
    or product that produced nothing. A customer asking for something
    genuinely unstocked (say "bracelets") would then have that dead
    category silently remembered, trapping every vague follow-up
    ("show me something else", "yeah lemme see") in the same no-match
    state until they explicitly named a real category again -- exactly
    the "never learn an invalid value" promise below is meant to rule
    out, but previously didn't for this specific case."""
    if result is None:
        return True
    if "recommendations" in result and not result["recommendations"]:
        return True
    if "karat_options" in result and not result["karat_options"]:
        # product_tool.list_karat_options()'s empty-list case -- a
        # product_name that matched nothing in the catalogue, same
        # "found the shape but not the substance" case as an empty
        # recommendations list immediately above.
        return True
    if "message" in result and "product" not in result:
        return True
    return False


def _order_draft_matches_pending_order(order_draft: dict | None, pending_order: dict | None) -> bool:
    """True when order_draft and pending_order genuinely describe the
    SAME order, not two different ones -- see route_customer()'s
    order_draft computation above for why this matters. get_order_draft()
    keys its product under "product_name"; get_pending_order_summary()
    keys the same fact under "product" (its own return shape, unrelated
    to this check) -- compared case-insensitively, same as every other
    product-identity check in this codebase (see memory.py's
    _names_a_different_product())."""
    if not order_draft or not pending_order:
        return False
    draft_product = order_draft.get("product_name")
    pending_product = pending_order.get("product")
    if draft_product is None or pending_product is None:
        return False
    return str(draft_product).strip().lower() == str(pending_product).strip().lower()


# Matches a message that is JUST a karat, nothing else -- "12", "12k",
# "18K", with or without surrounding whitespace. Deliberately strict
# (the WHOLE message must match, via fullmatch): "12k, 2 of them" is a
# compound message that still needs the LLM's own, already-working
# compound-correction handling (see llm.py's _order_draft_state_line()),
# not this deterministic path -- this only ever fires for the single-
# fact reply it's actually safe to resolve without a model call at all.
_BARE_KARAT_RE = re.compile(r"^\s*(\d{1,3})\s*k?\s*$", re.IGNORECASE)

# Same strictness rule as karat above -- a bare count, optionally with
# one of a few common trailing words ("2 pieces", "2 please"), and
# nothing else.
_BARE_QUANTITY_RE = re.compile(r"^\s*(\d{1,4})\s*(pieces?|pcs?|of them|please)?\s*$", re.IGNORECASE)

# Canonical bare-agreement replies -- deliberately the same small,
# unambiguous set llm.py's own prompt already names as examples
# ("yes", "confirm", "go ahead", "yh", "ok place it") plus a few
# obvious variants. Anything not in this exact set (a longer sentence,
# a hedge, a question) falls through to the normal LLM path exactly as
# before -- this never tries to guess a borderline case.
_BARE_CONFIRMATION_PHRASES = {
    "yes", "yeah", "yh", "yep", "yup", "sure", "ok", "okay", "confirm",
    "confirmed", "go ahead", "goahead", "please", "place it", "place the order",
    "ok place it", "yes please", "confirm it", "go for it", "sounds good",
    "yes confirm", "ok confirm", "proceed",
}

# Tools whose first real argument is a specific product -- exactly the
# set the ambiguous-reference guard below cares about. Deliberately
# _PRICING_TOOLS unioned with _ORDER_TOOL rather than a separate literal
# set, so this can never silently drift out of sync with those two if a
# new product-specific tool is added later.
_PRODUCT_SPECIFIC_TOOLS = _PRICING_TOOLS | {_ORDER_TOOL}

# Which non-product_name fields each product-specific tool's underlying
# function actually accepts as a **kwarg -- see product_tool.py's
# get_product_price()/get_product_weight()/list_karat_options() and
# quote_service.generate_quote() and order_tool.propose_order()'s own
# signatures. Live bug, 2026-09-02 (Webb, references-03 repeat run,
# 6/6): _apply_pending_clarification_details() below used to merge
# EVERY stashed detail into whatever tool the customer's clarification
# answer resolved to, with no regard for whether that tool's function
# signature actually has a parameter for it -- "that one, in 14k, two
# pieces" stashes {"material": "14k", "quantity": 2}, and when the
# resolution turn's own tool was get_product_price (which takes only
# product_name and material, no quantity at all), the merged quantity
# reached execute_tool() and blew up as
# `get_product_price() got an unexpected keyword argument 'quantity'`
# -- a TypeError inside tool_executor.py, caught there and turned into
# a generic "Something went wrong" reply to the customer. The evaluator
# scenario still reported PASS 6/6 because its own check only compares
# the FINAL conversational outcome, not whether execute_tool() actually
# ran clean -- a real gap Webb caught by reading the evaluator's raw
# log output, not something the scenario's own pass/fail alone would
# ever have surfaced. Fixing this at the merge site (below), rather
# than papering over it with a broader except in tool_executor.py,
# keeps tool_executor.py's "a bad-argument TypeError means something
# upstream got the contract wrong" invariant intact -- see its own
# comment at the TypeError branch.
_TOOL_ACCEPTED_DETAIL_FIELDS = {
    "get_product_price": {"material"},
    "generate_quote": {"material"},
    "get_product_karat_options": set(),
    "get_product_weight": set(),
    _ORDER_TOOL: {"material", "quantity", "delivery_address", "delivery_option"},
}

# Anything that looks like the customer actually pointed at a specific
# position -- an ordinal word, an ordinal number ("2nd"), or a bare
# digit anywhere in the message. Deliberately broad (matches inside a
# longer sentence, not just a whole-message fullmatch like the karat/
# quantity regexes above) and deliberately over-inclusive: this only
# gates whether the ambiguous-reference guard below is ALLOWED to fire,
# never forces a read on its own, so a false "yes, there's a number
# here" just means the guard stays out of the way and today's existing
# LLM-driven behaviour decides, same as if this guard didn't exist.
_ORDINAL_OR_NUMBER_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)

# Words in ordinal order, used to build a proportional clarification
# ("the first or second one" / "the first, second, third or fourth") --
# see _build_ambiguity_clarification_reply()'s docstring.
_ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
]

# A bare reference to "it" or "this/that (one)" with nothing else
# identifying which item -- the second of the two shapes the ambiguity
# guard below treats as "plausibly talking about the list just shown"
# (the first shape is a bare mention of one of the list's own
# categories, checked directly against last_presented_products in
# _looks_like_ambiguous_list_reference() since it depends on what was
# actually shown, not a fixed word list). Deliberately narrow: matches
# the exact regression cases ("that one", "this one", "I'll take that
# one", bare "it"), not "this"/"that" used as a determiner in front of a
# noun ("this order", "that address"), which already carry their own
# meaning and aren't what this guard is for.
_DEMONSTRATIVE_RE = re.compile(
    r"\b(this one|that one|the same one|it)\b", re.IGNORECASE,
)


def _normalize_for_confirmation_check(stripped_message: str) -> str:
    """Strips punctuation and collapses whitespace so "yes, confirm."
    and "Yes Confirm" both compare equal to the canonical, lowercase,
    space-separated entries in _BARE_CONFIRMATION_PHRASES. Shared by the
    "confirmation" branch (does this reply confirm?) and the
    "delivery_address" branch (this is NOT a literal address, no matter
    which field happens to be awaited) below."""
    return " ".join(re.sub(r"[.,!]", " ", stripped_message.lower()).split())


# A reply that starts with one of these words almost certainly is NOT a
# bare answer to "what address should this be delivered to?" -- it's
# the customer asking something else, or pushing back, and belongs with
# the normal LLM path (which already has the full order_draft context
# to work out what it actually is), not a guessed-at address.
# Deliberately conservative: this only widens what falls through to the
# LLM, never what the deterministic path accepts. Matched with \b (a
# word boundary), not a literal trailing space, so punctuation right
# after the word ("no, I gave you the address already") still matches
# -- caught by this file's own test suite: a plain
# `.startswith("no ")` missed exactly that case.
_NOT_AN_ADDRESS_STARTS_RE = re.compile(
    r"^\s*(why|what|how|when|who|can i|can you|do you|does|is there|are there|no|not|wait)\b",
    re.IGNORECASE,
)

# A message CONTAINING (anywhere, not just at the start) one of these is
# a correction or topic change, not a bare address -- caught live while
# building this: "actually 14k, make it 6 instead" (a compound
# karat+quantity correction, sent while delivery_address was the
# awaited field from an earlier, since-superseded turn in the test
# fixture) does not start with any _NOT_AN_ADDRESS_STARTS prefix, so
# without this second check it would have been wrongly stored as a
# literal delivery address instead of falling through to the LLM, which
# already handles compound corrections correctly (see llm.py's
# _order_draft_state_line()). A real Ghanaian address is not going to
# contain any of these words.
_ADDRESS_DISQUALIFYING_WORDS = ("actually", "instead", "rather", "wait", "make it", "change", "sorry", "correct")

# An embedded karat mention ("14k" appearing anywhere in the message,
# not just as the whole message) is never part of a real address either
# -- same live case as above. _BARE_KARAT_RE alone only catches the
# message being JUST a karat; this catches one showing up inside a
# longer sentence.
_EMBEDDED_KARAT_RE = re.compile(r"\b\d{1,3}\s*k\b", re.IGNORECASE)

# A direct answer to "What address should this be delivered to?" often
# restates the question's own verb ("deliver to East Legon", "send it to
# Kumasi") rather than naming just the place. The LLM path already
# strips this correctly when IT extracts delivery_address (semantic
# extraction is its job) -- this deterministic bypass does no extraction
# at all, it just stores `stripped` verbatim, so without this it stores
# the whole phrase. Caught live, 2026-08-24 (Webb): "deliver to East
# Legon" answering that exact question produced "Delivery to deliver to
# East Legon" in the customer-facing proposal -- the preamble duplicated
# rather than stripped, because propose_order's own phrasing already
# prepends "Delivery to ". Only strips a leading match (mirrors
# _NOT_AN_ADDRESS_STARTS_RE's conservative, start-anchored style) --
# never touches a preamble appearing mid-message, which would risk
# mangling a genuine address.
_ADDRESS_PREAMBLE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:deliver|delivery|send|ship)(?:\s+it)?\s+to\s*[:\-]?\s*",
    re.IGNORECASE,
)

# A direct answer to "What address should this be delivered to?" can
# also name the delivery arrangement itself in the same reply ("East
# Legon, accra_rider") -- caught live, 2026-08-24 (Webb): stored
# verbatim as delivery_address (nothing here does semantic splitting),
# producing "Delivery to East Legon, accra_rider" in the customer-facing
# proposal, and delivery_option left "unknown" for propose_order to
# re-infer from the address separately -- inference happened to still
# get the right zone (the address text also contains "east legon"), but
# the raw key stayed pasted into the address either way. Only matches
# the three literal option keys (not natural phrasing like "rider
# delivery within Accra"), since that's the confirmed live shape -- the
# LLM path already handles natural phrasing correctly when the message
# goes through it instead.
_DELIVERY_OPTION_TOKEN_RE = re.compile(
    r"\b(accra_rider|kumasi_rider|international)\b", re.IGNORECASE,
)


def _try_resolve_awaiting_field(awaiting_field: str | None, message: str) -> dict | None:
    """The deterministic short-circuit Webb asked for (2026-08-21,
    P0.4): when this session's last action was propose_order asking for
    ONE specific, still-missing detail (or a full proposal asking for
    confirmation), and this message is unambiguously just a direct
    answer to that exact question, build the tool_request straight from
    the pattern match and never call the LLM at all for this turn.

    This exists because the LLM-driven path already had extensive,
    specific prompt guidance for exactly this case (see llm.py's
    _order_draft_state_line(), the paragraph covering bare-number
    disambiguation) and it still misrouted live, three separate times,
    for the exact scenario this function targets: a bare "14k"
    answering propose_order's own "What karat would you like that in?"
    went to recommend_products instead of continuing the order. More
    prompt text was not a strong enough fix for a case this narrow and
    this pattern-matchable -- removing the LLM from the decision
    entirely, for this one specific class of reply, is.

    Returns None (meaning: fall through to the normal understand_customer()
    call, exactly as before this existed) whenever the message doesn't
    unambiguously match the awaited field's pattern -- a compound
    message, a correction naming a different value for something else,
    a genuine change of topic, or awaiting_field being None/"confirmation"-
    but-the-message-not-matching all fall through here, deliberately.
    This function only ever narrows what gets bypassed, never forces a
    wrong read: on any doubt, it returns None and the existing,
    already-tested LLM path (with its own order_draft/pending_order
    context) decides exactly as it always has.

    The returned tool_request only ever fills in the ONE field that was
    actually answered, "unknown" for everything else -- fill_missing_
    context() (called downstream in _execute_single(), exactly as for
    any LLM-produced request) backfills the rest from session memory,
    so this produces the identical shape the LLM was already expected
    to return for these replies."""
    if not awaiting_field:
        return None
    stripped = message.strip()
    if not stripped:
        return None

    if awaiting_field == "material":
        match = _BARE_KARAT_RE.match(stripped)
        if not match:
            return None
        return {
            "tool": _ORDER_TOOL,
            "arguments": {
                "product_name": "unknown", "material": f"{match.group(1)}k",
                "quantity": "unknown", "delivery_address": "unknown", "delivery_option": "unknown",
            },
            "_source": "awaiting_field:material",
        }

    if awaiting_field == "quantity":
        match = _BARE_QUANTITY_RE.match(stripped)
        if not match:
            return None
        return {
            "tool": _ORDER_TOOL,
            "arguments": {
                "product_name": "unknown", "material": "unknown",
                "quantity": int(match.group(1)), "delivery_address": "unknown", "delivery_option": "unknown",
            },
            "_source": "awaiting_field:quantity",
        }

    if awaiting_field == "confirmation":
        # Strip trailing punctuation and collapse internal punctuation/
        # whitespace ("yes, confirm." -> "yes confirm") before comparing
        # to the canonical set -- a comma joining two unambiguous
        # agreement words together is still unambiguous, it shouldn't
        # need its own separate entry in the set for every possible
        # punctuation variant.
        if _normalize_for_confirmation_check(stripped) not in _BARE_CONFIRMATION_PHRASES:
            return None
        return {"tool": _CONFIRM_TOOL, "arguments": {}, "_source": "awaiting_field:confirmation"}

    if awaiting_field == "delivery_interest":
        # get_product_price's bare-price reply always ends with "Want to
        # know about delivery too?" (see response_formatter.py's plain
        # "price" shape) -- a bare affirmative here is unambiguously
        # answering THAT question, not a fresh, undirected agreement.
        # Confirmed live, 2026-08-22: "yes" fell through to converse
        # ("Great! Let me know how I can assist you further.") instead
        # of showing delivery options, because nothing tracked that a
        # delivery question had just been asked. Reuses the same
        # canonical affirmative set confirm_order's own bare "yes"
        # check uses -- this is the identical shape of question (a
        # yes/no CTA the assistant itself just asked). Anything that
        # isn't in that set (a "no", a new product, a question) is
        # deliberately left to the LLM path rather than guessed here.
        if _normalize_for_confirmation_check(stripped) not in _BARE_CONFIRMATION_PHRASES:
            return None
        return {
            "tool": "get_delivery_information",
            "arguments": {},
            "_source": "awaiting_field:delivery_interest",
        }

    if awaiting_field in ("delivery_address", "delivery_option"):
        # Both fields share this one branch on purpose: propose_order()
        # re-derives whichever of the two is missing from the other
        # (infer_delivery_option() from an address, resolve_delivery_match()
        # from an option), so a reply here can equally be a place name
        # ("Nima, Accra") or an explicit arrangement ("rider delivery
        # within Accra") regardless of which specific question was
        # asked -- there's no reason to duplicate the same extraction
        # twice for a distinction the tool itself doesn't need. Added for
        # "delivery_option" 2026-08-24 (Webb): propose_order's own
        # ambiguous-option fallback previously never tagged awaiting_field
        # at all (unlike material/quantity/delivery_address above), so a
        # bare reply restating the missing zone had no deterministic
        # bypass to catch it -- see order_tool.propose_order()'s comment
        # at the same fallback for the live bug this closes.
        #
        # A karat- or quantity-shaped reply here is never actually an
        # address ("12k" while an address is awaited is far more likely
        # a stray correction to a field asked about earlier) -- leave
        # those, and anything that looks like a question or pushback, to
        # the LLM path rather than guessing.
        if _BARE_KARAT_RE.match(stripped) or _BARE_QUANTITY_RE.match(stripped):
            return None
        # A bare agreement word ("confirm", "yes", "ok", ...) is never a
        # literal address either -- caught live, 2026-08-21 (Webb's own
        # first trace run): a stray "confirm" sent while delivery_address
        # was still awaited (the previous turn's address extraction had
        # failed, so the question was still open) was stored as the
        # literal delivery_address "confirm", which then satisfied
        # propose_order's address check and moved straight on to asking
        # about delivery_option ("Would you like rider delivery within
        # Accra..."), silently discarding the customer's actual attempt
        # to confirm. None of this branch's other exclusions (question
        # marks, correction words, karat/quantity shapes) catch a bare
        # agreement word, because there is nothing address-specific
        # about it -- it just isn't descriptive content for ANY field,
        # so it's checked against the same canonical set confirmation
        # replies use, regardless of which field happens to be awaited.
        if _normalize_for_confirmation_check(stripped) in _BARE_CONFIRMATION_PHRASES:
            return None
        lowered = stripped.lower()
        if "?" in stripped or _NOT_AN_ADDRESS_STARTS_RE.match(stripped):
            return None
        if _EMBEDDED_KARAT_RE.search(stripped) or any(w in lowered for w in _ADDRESS_DISQUALIFYING_WORDS):
            return None
        option_match = _DELIVERY_OPTION_TOKEN_RE.search(stripped)
        delivery_option = option_match.group(1).lower() if option_match else "unknown"
        address_source = _DELIVERY_OPTION_TOKEN_RE.sub("", stripped, count=1) if option_match else stripped
        address = _ADDRESS_PREAMBLE_RE.sub("", address_source).strip().strip(",;- ").strip()
        if not address:
            # The whole message WAS the preamble/option token ("deliver
            # to" or "accra_rider" on their own, nothing else) -- there's
            # no actual place name left to store. Fall through to the LLM
            # path rather than storing an empty string as a "valid" address.
            return None
        return {
            "tool": _ORDER_TOOL,
            "arguments": {
                "product_name": "unknown", "material": "unknown", "quantity": "unknown",
                "delivery_address": address, "delivery_option": delivery_option,
            },
            "_source": f"awaiting_field:{awaiting_field}",
        }

    return None


def _looks_like_ambiguous_list_reference(message: str, items: list[dict]) -> bool:
    """Webb, 2026-09-02, extending the ambiguity guard: the live
    evaluator run showed "the ring" doesn't always come back from the
    LLM as a guessed catalogue name -- it correctly returned
    product_name "unknown" (per llm.py's own instruction to do exactly
    that when it can't tell), and get_product_price then safely replied
    "couldn't find that one, tell me more". Safe, but poor: the system
    already knows there's an active, ambiguous list and threw that
    context away instead of using it.

    The naive fix, "product_name unknown -> always clarify", is too
    broad (Webb, explicitly): a customer asking about something
    genuinely unrelated to the list just shown (a stale
    last_presented_products from many turns ago, a typo'd product name
    the model also can't resolve) would get hijacked into "did you mean
    one of these rings?" when they meant neither. This function is the
    narrower gate: true only when the message itself plausibly refers
    BACK to the list, not just whenever resolution happens to fail.

    Two shapes count, matching the actual regression cases: a bare
    demonstrative ("that one", "this one", "it", see _DEMONSTRATIVE_RE),
    or a bare mention of one of the list's own categories ("the ring",
    "rings", "a necklace" when Rings/Necklaces items are what's shown).
    Anything else (a specific but wrong/misspelled product name, an
    unrelated category, plain small talk) returns False and the
    existing, unrelated "couldn't find" behaviour is left untouched."""
    if _DEMONSTRATIVE_RE.search(message):
        return True
    categories = {str(item.get("category") or "").strip().lower() for item in items}
    categories.discard("")
    lowered = message.lower()
    for category in categories:
        singular = category[:-1] if category.endswith("s") else category
        if not singular:
            continue
        if re.search(rf"\b{re.escape(singular)}s?\b", lowered):
            return True
    return False


def _build_ambiguity_clarification_reply(items: list[dict]) -> str:
    """Webb, 2026-09-02: "the clarification should be proportional to
    the ambiguity", not always the same generic sentence. Two items,
    same category: "do you mean the first or second one?". More than
    two, same category: name the category and list ordinals up to
    _ORDINAL_WORDS' length, past that (an unusual case, ten-plus items
    shown at once) fall back to listing the actual catalogue names
    rather than run out of ordinal words. A genuinely mixed-category
    list ("piece" is the only honest generic noun for "a ring and a
    necklace together") skips ordinals/category naming entirely and
    just asks which piece, matching Webb's own example verbatim rather
    than guessing at a mixed-list wording he didn't ask for."""
    categories = {str(item.get("category") or "").strip() for item in items}
    categories.discard("")
    if len(categories) != 1:
        return "Sure. Which piece do you mean?"
    category = next(iter(categories))
    noun = category[:-1].lower() if category.lower().endswith("s") else category.lower()
    count = len(items)
    if count == 2:
        return "Sure. Do you mean the first or second one?"
    if count <= len(_ORDINAL_WORDS):
        ordinals = _ORDINAL_WORDS[:count]
        choices = ", ".join(ordinals[:-1]) + " or " + ordinals[-1]
        return f"Sure. Which {noun} do you mean, the {choices}?"
    listed = "\n".join(f"- {item.get('product_name', '')}" for item in items)
    return f"Sure. Which {noun} do you mean?\n{listed}"


def _override_unresolved_ambiguous_reference(
    tool_request: dict, message: str, last_presented_products: dict | None,
) -> dict | None:
    """Evaluator finding, 2026-09-01 (Webb): three phrasings of the same
    underlying failure -- a bare category name matching two shown items
    ("the ring"), a bare demonstrative after a multi-item list ("I'll
    take that one"), and a demonstrative-plus-details opener ("that one,
    in 14k, two pieces") -- all reproduced at 0/6 across --repeat 6 runs
    each, fully consistent, not a flaky one-off. The model reliably picks
    up a strong, correct-looking TOOL signal ("I'll take that one" ->
    propose_order) and just as reliably fails to then stop and check
    whether it actually knows WHICH product from the list that tool call
    should apply to -- it guesses, usually the first or most recent item,
    instead of asking. A fully reproducible 0/18 across three distinct
    phrasings is "this basically never happens", not "this sometimes
    gets missed" -- not a case prompt reinforcement alone is safe to rely
    on, the same reasoning already applied to _try_resolve_awaiting_field()
    above for a different reliability-critical decision.

    So: after the LLM has already responded, this checks whether the
    session was actually showing more than one product (nothing to be
    ambiguous about otherwise), and the customer's raw message has no
    ordinal or number anywhere in it (a real position reference is
    always let through untouched -- see _ORDINAL_OR_NUMBER_RE). Then
    either of two things counts as "the LLM guessed rather than knew":
    it returned a product-specific tool call whose product_name matches
    one of the items from that list (a confident wrong guess), or it
    honestly returned product_name "unknown" AND the message itself
    looks like it was talking about that list anyway (see
    _looks_like_ambiguous_list_reference() -- live evaluator finding,
    2026-09-02: "the ring" produces this second shape, not the first,
    the LLM correctly abstained and get_product_price's own "couldn't
    find" reply covered the danger, but threw away context KasaFlow
    already had). Either way, override with a converse reply asking
    which item, proportional to how many are actually ambiguous (see
    _build_ambiguity_clarification_reply()).

    Any OTHER details the LLM did manage to extract alongside the
    unresolved product_name (material, quantity, delivery info from
    "that one, in 14k, two pieces") are stashed via
    set_pending_clarification_details() so the customer doesn't have to
    repeat them once they answer -- see that function's docstring and
    _apply_pending_clarification_details() below for the other half of
    this round trip.

    Deliberately narrow and one-directional, same posture as
    _try_resolve_awaiting_field(): on any doubt (fewer than two items
    shown, an ordinal/number present, a product_name that doesn't match
    the list AND doesn't look like a list reference either, or a tool
    outside _PRODUCT_SPECIFIC_TOOLS) this returns None and changes
    nothing. Worst case if a heuristic here is ever wrong is an
    unnecessary clarifying question on a message that was actually
    clear -- a mild annoyance, never worse than today's confirmed 0/6
    silent-wrong-guess behaviour this exists to replace."""
    if "_source" in tool_request:
        return None
    if tool_request.get("tool") not in _PRODUCT_SPECIFIC_TOOLS:
        return None
    if not last_presented_products:
        return None
    items = last_presented_products.get("items") or []
    names = [item["product_name"] for item in items if item.get("product_name")]
    if len(names) < 2:
        return None
    if _ORDINAL_OR_NUMBER_RE.search(message):
        return None
    arguments = tool_request.get("arguments") or {}
    product_name = arguments.get("product_name")
    if not isinstance(product_name, str):
        return None
    normalized = product_name.strip().lower()
    matches_shown_item = normalized in {n.lower() for n in names}
    if not matches_shown_item:
        if normalized != "unknown" or not _looks_like_ambiguous_list_reference(message, items):
            return None
    reply = _build_ambiguity_clarification_reply(items)
    other_details = {
        key: value for key, value in arguments.items()
        if key != "product_name" and value not in (None, "unknown", "")
    }
    stash = (
        {"generation": last_presented_products.get("generation"), "details": other_details}
        if other_details else None
    )
    return {
        "tool": _CONVERSATION_TOOL,
        "arguments": {"reply": reply},
        "_source": "ambiguous_reference_guard",
        "_clarification_stash": stash,
    }


def _apply_pending_clarification_details(
    tool_request: dict, session_id: str, last_presented_products: dict | None,
) -> dict:
    """The other half of the round trip _override_unresolved_ambiguous_
    reference() above starts: once the customer answers a forced
    clarification by naming or positioning a specific item from the
    SAME list ("the second one"), merge back in whatever other details
    they already stated in the original ambiguous turn ("that one, in
    14k, two pieces") rather than making them repeat themselves.

    Only ever fills in a field this turn's own resolution left
    "unknown"/missing -- if the customer restated or changed a detail
    while answering the clarification, their fresh answer always wins,
    this never overwrites it. Only consumes the stash when its
    generation matches the CURRENT last_presented_products exactly, so
    a stash from an old, superseded list can never leak into a new
    one -- see set_pending_clarification_details()'s docstring.

    Returns tool_request unchanged (same object) whenever there is
    nothing to apply, so callers can use the return value unconditionally
    without a None check."""
    if tool_request.get("tool") not in _PRODUCT_SPECIFIC_TOOLS:
        return tool_request
    stash = get_pending_clarification_details(session_id)
    if not stash or not last_presented_products:
        return tool_request
    if stash.get("generation") != last_presented_products.get("generation"):
        return tool_request
    items = last_presented_products.get("items") or []
    names = {str(item.get("product_name") or "").strip().lower() for item in items}
    arguments = tool_request.get("arguments") or {}
    product_name = arguments.get("product_name")
    if not isinstance(product_name, str) or product_name.strip().lower() not in names:
        return tool_request
    # Only ever merge a field the RESOLVED tool's own function actually
    # takes -- see _TOOL_ACCEPTED_DETAIL_FIELDS's docstring for the live
    # bug this closes (a stashed "quantity" reaching get_product_price(),
    # which has no such parameter at all). The stash can carry details
    # that belonged to a different tool than the one this clarification
    # answer happens to resolve to (the customer could, in principle,
    # ask a plain price question -- get_product_price -- to answer a
    # clarification that was originally raised by a propose_order call),
    # so this is a real filter, not a defensive no-op.
    accepted_fields = _TOOL_ACCEPTED_DETAIL_FIELDS.get(tool_request.get("tool"), set())
    merged = dict(arguments)
    for key, value in (stash.get("details") or {}).items():
        if key not in accepted_fields:
            continue
        if merged.get(key) in (None, "unknown", ""):
            merged[key] = value
    return {**tool_request, "arguments": merged}


def route_customer(message: str, session_id: str) -> dict:
    """Public entry point. Holds this session's turn lock for the
    ENTIRE sequence below (read state -> call the LLM -> execute a tool
    -> write state), not just around individual memory.py calls -- see
    SessionStore.session_lock()'s docstring for the concrete corruption
    this closes: two messages from the same customer arriving close
    together, on separate threads, interleaving reads/writes so a
    later write from the slower request lands after an earlier one
    meant to come first."""
    with get_session_store().session_lock(session_id):
        return _route_customer_locked(message, session_id)


def _route_customer_locked(message: str, session_id: str) -> dict:
    try:
        # Tells the LLM whether this session actually has anything
        # pending to confirm -- see llm.py's _pending_order_state_line()
        # for why a bare "yh"/"yeah" is unresolvable without it.
        pending_order = get_pending_order_summary(session_id)
        # One step earlier in the same problem: an order that's been
        # started (propose_order asked "how many?") but isn't priced
        # yet. Only suppressed while there's a full pending_order for
        # the SAME product -- once a proposal exists, that's the active
        # state, and showing both would be redundant. But a pending_order
        # can go stale: the customer can start a genuinely different,
        # not-yet-priced order without ever confirming or cancelling the
        # old proposal, and that old pending_order previously suppressed
        # order_draft unconditionally, leaving the NEW order's own
        # in-progress draft (product known, karat/quantity missing) with
        # zero representation in the prompt. Confirmed live, 2026-08-20:
        # a bare "14k" answering propose_order's own "What karat would
        # you like?" misrouted to recommend_products three separate
        # times, every time with an old, unconfirmed proposal for a
        # DIFFERENT product still sitting there -- with no order_draft
        # state line at all, the model had nothing telling it a bare
        # karat reply continues that in-progress order rather than being
        # a fresh, ambiguous browse request. See llm.py's
        # _order_draft_state_line().
        order_draft = get_order_draft(session_id)
        if pending_order and _order_draft_matches_pending_order(order_draft, pending_order):
            order_draft = None
        # A product lookup the customer asked for but hadn't named a
        # product for yet ("yeah i wanna see pictures") -- see
        # memory.get_pending_intent() and llm.py's
        # _pending_intent_state_line() for why a follow-up naming the
        # product needs this to avoid asking the customer to repeat
        # themselves a second time.
        pending_intent = get_pending_intent(session_id)
        # A real business action (usually propose_order/confirm_order)
        # that was fully specified and still hit a genuine, unrecoverable
        # failure -- different axis from all three above, which are about
        # missing information. See memory.get_last_action_outcome() and
        # llm.py's _last_action_outcome_state_line().
        last_action_outcome = get_last_action_outcome(session_id)
        # The specific product a get_product_price/generate_quote call
        # most recently resolved to, so a bare karat-only follow-up
        # ("what about in 18k") can re-quote the same item instead of
        # falling through to recommend_products. See
        # memory.get_last_priced_product() and llm.py's
        # _last_priced_product_state_line().
        last_priced_product = get_last_priced_product(session_id)
        # The exact numbered/bulleted list the most recent successful
        # recommend_products reply showed this customer, so "the second
        # one"/"the first ring" can resolve deterministically against a
        # position instead of the model guessing. See
        # memory.get_last_presented_products() and llm.py's
        # _last_presented_products_state_line().
        last_presented_products = get_last_presented_products(session_id)
        # Only meaningful when a pending_order actually exists -- see
        # memory.set_awaiting_confirmation()'s docstring. True only when
        # proposing THIS order was the last thing that happened in the
        # session; False once anything else has been asked/offered
        # since, so a bare "yes" doesn't wrongly confirm a stale order.
        awaiting_confirmation = is_awaiting_confirmation(session_id) if pending_order else False
        # Non-None for exactly one turn: the one right after confirm_order
        # succeeded, nothing else since. See
        # memory.set_just_confirmed_order()'s docstring.
        just_confirmed_order = get_just_confirmed_order(session_id)
        # Deterministic short-circuit (Webb, 2026-08-21, P0.4): if this
        # session's own last action was asking for exactly one specific
        # detail, and this message unambiguously answers it, resolve it
        # here and skip the LLM tool-selection call entirely for this
        # turn -- see _try_resolve_awaiting_field()'s docstring for why.
        # Falls through to the normal understand_customer() call, byte
        # for byte as before this existed, whenever it can't confidently
        # resolve the message itself.
        awaiting_field = get_awaiting_field(session_id)
        tool_request = _try_resolve_awaiting_field(awaiting_field, message)
        if tool_request is None:
            tool_request = understand_customer(
                message,
                pending_order=pending_order,
                awaiting_confirmation=awaiting_confirmation,
                order_draft=order_draft,
                pending_intent=pending_intent,
                last_action_outcome=last_action_outcome,
                last_priced_product=last_priced_product,
                just_confirmed_order=just_confirmed_order,
                last_presented_products=last_presented_products,
                # Only reaches here at all when the fast-path check just
                # above declined to resolve it -- for product_name that's
                # every time, since it has no fast-path branch (see
                # llm._awaiting_field_state_line()'s docstring). Task #60
                # follow-up, 2026-08-30 (Webb): tagging awaiting_field
                # alone didn't change behaviour live, because nothing was
                # passing it into the prompt at all.
                awaiting_field=awaiting_field,
            )
            # Post-hoc guard, not a fast path: the LLM has already run
            # here, this only checks its answer against a list it can't
            # see the ambiguity of reliably. See
            # _override_unresolved_ambiguous_reference()'s docstring for
            # the live evaluator evidence this exists to fix (2026-09-01,
            # ambiguity-01/references-02, 0/6 each, fully reproducible).
            override = _override_unresolved_ambiguous_reference(
                tool_request, message, last_presented_products,
            )
            if override is not None:
                tool_request = override
            else:
                # The other half of the round trip: this call did NOT get
                # overridden, meaning the LLM returned (or fill_missing_
                # context() will shortly resolve) a real, specific
                # product_name -- exactly the shape a customer's answer to
                # a forced clarification takes ("the second one" resolves
                # against last_presented_products' own position context
                # already passed into understand_customer()'s prompt, same
                # as any other ordinal reference; see _ORDINAL_OR_NUMBER_RE
                # above). If a clarification is still pending from an
                # earlier turn in THIS same list, merge its stashed details
                # back in now. A no-op (returns tool_request unchanged)
                # whenever there's nothing stashed, the tool isn't
                # product-specific, or the stash belongs to a different,
                # superseded list -- see
                # _apply_pending_clarification_details()'s docstring.
                tool_request = _apply_pending_clarification_details(
                    tool_request, session_id, last_presented_products,
                )
    except ValueError as e:
        error_result = {"error": str(e)}
        _log_turn_trace(
            session_id, message, _snapshot_session_state(session_id), {"error": str(e)},
            resolved_arguments=None, tool_result=None, final_result=error_result, post_state=None,
        )
        return error_result
    except ToolSelectionError as e:
        logger.error("Tool selection failed: %s", e)
        error_result = {"error": "I couldn't understand that request. Could you rephrase it?"}
        _log_turn_trace(
            session_id, message, _snapshot_session_state(session_id), {"error": str(e)},
            resolved_arguments=None, tool_result=None, final_result=error_result, post_state=None,
        )
        return error_result

    # understand_customer() only returns "requests" (plural) when the
    # message genuinely contained more than one distinct ask -- see
    # llm.py. The single-request path below is completely unchanged
    # from before that existed, so route_customer()'s original,
    # documented contract-stable shape is untouched for every message
    # that doesn't need splitting (still the overwhelming majority).
    if "requests" in tool_request:
        results = [_execute_single(req, session_id, message) for req in tool_request["requests"]]
        return {"results": results}

    return _execute_single(tool_request, session_id, message)


def _snapshot_session_state(session_id: str) -> dict:
    """Everything route_customer() reads/writes to decide a turn,
    captured in one shot -- used both before and after a tool call so
    _log_turn_trace() below can show exactly what changed. Deliberately
    the same seven fields _route_customer_locked() already reads for
    understand_customer()'s prompt, re-read here rather than threaded
    through as parameters: these are cheap in-memory reads, and the
    session lock is held for the whole turn, so nothing else can have
    changed them in between."""
    pending_order = get_pending_order_summary(session_id)
    return {
        "pending_order": pending_order,
        "order_draft": None if pending_order else get_order_draft(session_id),
        "pending_intent": get_pending_intent(session_id),
        "last_action_outcome": get_last_action_outcome(session_id),
        "last_priced_product": get_last_priced_product(session_id),
        "last_presented_products": get_last_presented_products(session_id),
        "awaiting_confirmation": is_awaiting_confirmation(session_id) if pending_order else False,
        "just_confirmed_order": get_just_confirmed_order(session_id),
        "awaiting_field": get_awaiting_field(session_id),
    }


def _log_turn_trace(
    session_id: str,
    message: str,
    pre_state: dict,
    llm_structured_output: dict,
    resolved_arguments: dict | None,
    tool_result: dict | None,
    final_result: dict | None,
    post_state: dict | None,
) -> None:
    """One structured log line per turn, capturing the full path from
    customer message to final response -- the exact instrumentation
    Webb asked for, 2026-08-21, after a live transcript exposed several
    failures (a bare "14k" answering "what karat" misrouting to
    recommend_products, three times; a stale product resurrecting
    itself) that no deterministic test caught, and that "the model
    probably just missed it" cannot honestly be concluded from without
    seeing what the model actually returned.

    The point isn't cosmetic logging -- it's making Webb's own failure
    taxonomy checkable from evidence instead of guesswork: did the LLM
    return the wrong tool/arguments (a prompt/context problem), or did
    it return the RIGHT tool/arguments and the application still did
    the wrong thing (an application bug)? `llm_structured_output` vs
    `resolved_arguments`/`tool_result` is exactly that split -- compare
    them and the category falls out, rather than being assumed.

    This is a new logging pattern for this codebase (no other module
    logs structured JSON per turn) -- deliberately additive: it changes
    nothing about what any endpoint returns to a customer or a test,
    only what appears in the server's own logs. NOT a replacement for
    real conversation-turn history (recent_turns) -- that's a genuinely
    separate, not-yet-built layer (see the architecture discussion this
    was written alongside); this only ever shows ONE turn's own
    before/after state, not the literal conversation that led to it."""
    try:
        trace = {
            "session_id": session_id,
            "customer_message": message,
            "pre_tool_state": pre_state,
            "llm_structured_output": llm_structured_output,
            "resolved_arguments": resolved_arguments,
            "tool_result": tool_result,
            "final_result": final_result,
            "post_tool_state": post_state,
        }
        logger.info("KASAFLOW_TURN_TRACE %s", json.dumps(trace, default=str))
    except Exception:
        # A logging bug must never take down the actual customer-facing
        # turn -- see this file's general "diagnostics are best-effort,
        # never load-bearing" posture (same reasoning as
        # order_tool.py's staff-notification failures).
        logger.exception("Failed to log turn trace for session %s", session_id)


def _execute_single(tool_request: dict, session_id: str, message: str = "") -> dict:
    pre_state = _snapshot_session_state(session_id)
    # Confirmation invariant (Webb, 2026-08-21): captured here, before
    # the unconditional reset two lines down clears it for this turn --
    # this is the one place that still reflects "was proposing the
    # CURRENT pending order actually the last thing that happened in
    # this session". Passed into confirm_order() below as
    # confirmation_allowed so a bare "yes" can never confirm a stale
    # proposal left over from an abandoned or incomplete product switch,
    # or from an unrelated question asked since. See
    # order_tool.confirm_order()'s docstring for the real gap this
    # closes (found via test_active_product_never_resurrects_after_an_
    # explicit_product_switch).
    was_awaiting_confirmation = pre_state["awaiting_confirmation"]

    # Default for every turn: this call does NOT represent confirming
    # whatever order is pending, and no order was JUST confirmed this
    # turn. Only a genuinely successful propose_order/confirm_order call
    # below re-asserts either (see memory.set_awaiting_confirmation() and
    # memory.set_just_confirmed_order()). Set unconditionally up front,
    # before the converse early-return, since a converse reply is
    # exactly the "asked something else" case this exists to catch.
    set_awaiting_confirmation(session_id, False)
    set_just_confirmed_order(session_id, None)
    # Same unconditional-reset-then-selectively-reassert pattern as
    # awaiting_confirmation above, and for the same reason: awaiting_field
    # must only ever reflect what THIS turn's own propose_order call
    # asked for, never something left over from an earlier turn that a
    # converse reply or a different tool call has since moved past. See
    # memory.set_awaiting_field()'s docstring (Webb, 2026-08-21, P0.4).
    set_awaiting_field(session_id, None)
    # Identical discipline, identical reason, for the ambiguity guard's
    # own stash: only ever reflects a clarification THIS turn's own
    # override just raised (reasserted a few lines below, in the
    # CONVERSATION_TOOL branch, only when tool_request actually carries a
    # "_clarification_stash"). Any other turn -- a normal tool call, a
    # converse reply that wasn't this guard, or simply the turn where the
    # customer answers the clarification and _apply_pending_clarification_
    # details() (called from _route_customer_locked()) consumes it -- must
    # not leave a stale stash sitting there to leak into a later,
    # unrelated resolution. This is also what makes consumption
    # single-use: the very turn that reads it also resets it, before
    # anything below has a chance to reassert it.
    set_pending_clarification_details(session_id, None)

    if tool_request["tool"] == _CONVERSATION_TOOL:
        stash = tool_request.get("_clarification_stash")
        if stash is not None:
            # This converse reply IS the ambiguity guard's own
            # clarification question (see
            # _override_unresolved_ambiguous_reference()) -- reassert the
            # stash the reset above just cleared, so the customer's next
            # reply naming which item they meant can still recover the
            # other details ("14k, two pieces") they already gave in the
            # same breath. Any OTHER converse reply (a genuine question,
            # small talk) never carries this key at all, so this branch
            # is a no-op for every converse call except this guard's own.
            set_pending_clarification_details(session_id, stash)
        result = _handle_conversation(tool_request["arguments"])
        _log_turn_trace(
            session_id, message, pre_state, tool_request,
            resolved_arguments=None, tool_result=None, final_result=result,
            post_state=_snapshot_session_state(session_id),
        )
        return result

    # Resolve any "this" / "that one" reference the model couldn't
    # answer from the message alone against what this session last
    # talked about, before the arguments ever reach a tool.
    arguments = fill_missing_context(session_id, tool_request["arguments"])

    if tool_request["tool"] in _SESSION_AWARE_TOOLS:
        arguments = {**arguments, "session_id": session_id}
        if tool_request["tool"] == _CONFIRM_TOOL:
            arguments["confirmation_allowed"] = was_awaiting_confirmation

    # Read BEFORE execute_tool()/remember_context() below overwrite it --
    # this is deliberately the state as it was going into this call, so
    # it can be compared against `arguments` (this call's resolved
    # values) to detect a genuine correction. See
    # _describe_order_corrections()'s docstring for why this is safe to
    # compute unconditionally (whether or not a pending_order already
    # exists) -- an earlier version of this gated the whole computation
    # on "no pending_order", which fixed the one bug it was written for
    # (a stale draft wrongly diffed against an unrelated NEW order) but
    # broke the much more common case: correcting a field of an order
    # that's already been proposed and is sitting there pending
    # confirmation (confirmed live in testing, 2026-08-20 -- "actually
    # make it 18k" after a full 14k proposal stopped being acknowledged
    # at all). The real fix belongs inside _describe_order_corrections()
    # itself, which now recognises a genuinely different product and
    # declines to call that a "correction" regardless of pending state.
    correction_note = None
    if tool_request["tool"] == _ORDER_TOOL:
        correction_note = _describe_order_corrections(get_order_draft(session_id), arguments)

    try:
        result = execute_tool(tool_request["tool"], **arguments)
    except ToolExecutionError as e:
        logger.error("Tool execution failed: %s", e)
        error_result = {"error": "Something went wrong while processing your request."}
        _log_turn_trace(
            session_id, message, pre_state, tool_request,
            resolved_arguments=arguments, tool_result={"exception": str(e)}, final_result=error_result,
            post_state=_snapshot_session_state(session_id),
        )
        return error_result

    if tool_request["tool"] == _ORDER_TOOL and isinstance(result, dict) and "proposal" in result:
        # This turn's propose_order call produced a real, full proposal --
        # the most recent thing that happened in this session IS this
        # order, so a bare "yes" next turn is safe to read as confirming
        # it. Overrides the False set at the top of this function.
        set_awaiting_confirmation(session_id, True)
        # And the specific thing this proposal now asks the customer for
        # is confirmation itself -- see _try_resolve_awaiting_field()'s
        # "confirmation" branch.
        set_awaiting_field(session_id, "confirmation")
    elif tool_request["tool"] == _ORDER_TOOL and isinstance(result, dict) and "awaiting_field" in result:
        # propose_order's own missing-detail error tags exactly which
        # single field it's asking for (see order_tool.propose_order()) --
        # carry that forward so the customer's very next reply can be
        # checked deterministically before the LLM ever runs again.
        set_awaiting_field(session_id, result["awaiting_field"])
    elif (
        tool_request["tool"] == _PRICE_TOOL
        and isinstance(result, dict)
        and "price" in result
        and "delivery_options" not in result
    ):
        # get_product_price's bare-price shape (as opposed to
        # generate_quote's combined price+delivery shape, which already
        # answers the delivery question up front and needs no follow-up
        # tracking) always ends with "Want to know about delivery too?"
        # -- see response_formatter.py. Track that this specific
        # question is now open, the same way propose_order's own
        # missing-field questions are tracked above, so a bare "yes"
        # next turn resolves deterministically instead of falling
        # through to converse. See _try_resolve_awaiting_field()'s
        # "delivery_interest" branch for the other half of this.
        set_awaiting_field(session_id, "delivery_interest")

    if tool_request["tool"] == _CONFIRM_TOOL and isinstance(result, dict) and "order_confirmation" in result:
        # This turn's confirm_order call actually placed the order --
        # tell the next turn's prompt so an unrelated next message
        # doesn't get read against an order that's already done, and so
        # a genuine "what's my order number?" gets a naturally different
        # answer than a fresh "what would you like to order?".
        set_just_confirmed_order(session_id, result["order_confirmation"])

    # Only remember context once the tool has actually run against it
    # AND found something -- see _found_nothing()'s docstring for the
    # concrete failure mode this avoids.
    if not _found_nothing(result):
        context_to_remember = arguments
        if tool_request["tool"] == _ORDER_TOOL and isinstance(result, dict) and "proposal" in result:
            # propose_order can resolve delivery_address/delivery_option to
            # something DIFFERENT from what was passed in -- inferring a
            # zone from the address when delivery_option came in "unknown",
            # or overriding a stale restated address entirely (see the
            # stale-address-override comment in order_tool.propose_order()).
            # Whatever it decided is the truth for this order going
            # forward; `arguments` is only what was true BEFORE this call
            # ran. Remembering `arguments` unconditionally meant the
            # override's fix never actually persisted -- confirmed live,
            # 2026-08-24 (Webb): an East Legon -> Kumasi correction
            # displayed cleanly (the override caught it), then a
            # following, unrelated quantity-only correction ("actually
            # make the quantity 3") silently reverted delivery_address
            # back to the stale East Legon value, because fill_missing_
            # context() on that next turn backfilled from what was
            # actually remembered -- the pre-override value, since only
            # the proposal shown to the customer had been corrected, not
            # what gets remembered for next time. Reading the correction
            # from the proposal itself, not the pre-call arguments, is
            # what makes the fix actually stick across turns.
            # material has the identical problem, a different shape: the
            # customer's raw wording ("18", "18 karat", ...) can differ
            # textually from the catalogue's canonical form ("18k") even
            # when propose_order() successfully matched and priced it.
            # Remembering the raw form meant a LATER turn that restates
            # the SAME karat using the canonical form (which is exactly
            # what the pending-order prompt context shows the model, in
            # prose like "5 x 18k Custom Leaf...") compared unequal
            # against the raw remembered value and produced a false "Got
            # it, I've updated the karat to 18k" correction_note for a
            # karat that never actually changed. Confirmed live, 2026-08-24
            # (Webb): "actually deliver to Kumasi instead" -- a pure
            # delivery correction -- came back acknowledging a karat
            # change instead. Reproduced in isolation: material passed in
            # as "18", remembered raw, then restated as "18k" on a later
            # turn reproduces the exact false note. Same fix shape as the
            # delivery override above -- remember what the tool actually
            # resolved, not the raw wording that produced it.
            proposal = result["proposal"]
            context_to_remember = {
                **arguments,
                "material": proposal.get("material", arguments.get("material")),
                "delivery_address": proposal.get("delivery_address", arguments.get("delivery_address")),
                "delivery_option": proposal.get("delivery_option", arguments.get("delivery_option")),
            }
        remember_context(session_id, context_to_remember)

    _update_pending_intent(session_id, tool_request["tool"], arguments, result)
    _update_last_priced_product(session_id, tool_request["tool"], result)
    _update_last_presented_products(session_id, tool_request["tool"], result)
    weight_ask_count = _update_weight_ask_count(session_id, tool_request["tool"], result)

    if _tool_succeeded(result):
        # A genuine success means whatever failed before is no longer
        # the active topic -- see memory.set_last_action_outcome()'s
        # docstring. Deliberately stricter than "not _found_nothing()":
        # that treats any {"error": ...} shape (including the very
        # failure this session just recorded, e.g. propose_order's own
        # no-id return) as "found something", which would wipe out the
        # outcome on the exact same call that just set it.
        set_last_action_outcome(session_id, None)

    raw_tool_result = result

    if isinstance(result, dict) and "awaiting_field" in result:
        # Internal routing state, not something a customer-facing reply
        # (or this endpoint's API contract) should ever surface --
        # already consumed above to update the session's own
        # awaiting_field. raw_tool_result (traced separately) keeps the
        # untouched original for diagnosis.
        result = {k: v for k, v in result.items() if k != "awaiting_field"}

    if correction_note and isinstance(result, dict):
        result = {**result, "correction_note": correction_note}

    if weight_ask_count is not None and isinstance(result, dict):
        result = {**result, "weight_ask_count": weight_ask_count}

    _log_turn_trace(
        session_id, message, pre_state, tool_request,
        resolved_arguments=arguments, tool_result=raw_tool_result, final_result=result,
        post_state=_snapshot_session_state(session_id),
    )
    return result


def _describe_order_corrections(old_draft: dict | None, arguments: dict) -> str | None:
    """Builds a short acknowledgement sentence when this propose_order
    call changes a field the customer had already given earlier in the
    same order (e.g. "wait, 14k rather" after material was already
    "12k") -- confirmed live, 2026-08-19 (Webb): the correction was
    applied to session memory correctly, but the very next reply just
    asked for the next missing field with no acknowledgement anything
    had changed, which read as the assistant not having registered the
    change at all.

    Deliberately doesn't force a "please confirm this change" round
    trip -- an extra yes/no turn for an unambiguous correction is
    friction a real assistant wouldn't add (Webb and a second AI's
    review of the same transcript both flagged this independently,
    2026-08-19). This just states what changed; response_formatter.py
    prepends it to whatever reply would already be sent (the next
    missing-field question, or the full proposal if everything's now
    known), so the conversation continues normally afterwards.

    Returns None when there's nothing to acknowledge: no prior draft at
    all (a fresh order, not a correction -- see get_order_draft()'s
    None case), the product itself is different (see below), or none of
    the fields this call resolved actually differ from what was already
    known.

    The product-identity check exists because old_draft can reflect an
    order that's already fully proposed and pending confirmation, not
    just one still being built up field by field -- and diffing every
    field against it regardless of whether this call is even about the
    SAME item is exactly how a customer's complete new, unrelated order
    got a fabricated correction_note referencing a product they never
    mentioned (confirmed live, 2026-08-20). A product that's explicitly
    named and differs from old_draft's means this message isn't
    correcting anything -- it's describing something else entirely, and
    every other field "changing" alongside a different product is a
    symptom of that, not a real correction to acknowledge."""
    if not old_draft:
        return None

    old_product = old_draft.get("product_name")
    new_product = arguments.get("product_name")
    if (
        old_product is not None
        and new_product is not None
        and not (isinstance(new_product, str) and new_product.strip().lower() == "unknown")
        and str(old_product).strip().lower() != str(new_product).strip().lower()
    ):
        return None

    changed = []
    for key, label in _ORDER_CORRECTION_FIELDS.items():
        old_value = old_draft.get(key)
        new_value = arguments.get(key)
        if old_value is None or new_value is None:
            continue
        if isinstance(new_value, str) and new_value.strip().lower() == "unknown":
            continue
        if str(old_value).strip().lower() == str(new_value).strip().lower():
            continue
        changed.append(f"{label} to {new_value}")

    if not changed:
        return None
    if len(changed) == 1:
        return f"Got it, I've updated {changed[0]}."
    return f"Got it, I've updated {' and '.join(changed)}."


def _tool_succeeded(result: dict | None) -> bool:
    if result is None or "error" in result:
        return False
    return not _found_nothing(result)


def _update_pending_intent(session_id: str, tool_name: str, arguments: dict, result: dict | None) -> None:
    if tool_name in _PENDING_INTENT_TOOLS and _found_nothing(result):
        product_name = arguments.get("product_name")
        if product_name is None or str(product_name).strip().lower() == "unknown":
            # Missing the product itself, specifically -- not a real name
            # that just didn't match anything (a made-up item, a typo).
            # Only this case means "ask again once they tell you which
            # one", so only this case is worth remembering.
            set_pending_intent(session_id, tool_name)
            return

    if not _found_nothing(result):
        # Whatever was pending (if anything) is now resolved or has been
        # superseded by a successful, different request -- either way,
        # stale intent left behind would risk misreading an unrelated
        # later message as still answering it.
        set_pending_intent(session_id, None)


def _update_last_priced_product(session_id: str, tool_name: str, result: dict | None) -> None:
    if tool_name in _PRICING_TOOLS and _tool_succeeded(result):
        # get_product_price/generate_quote's success shape always
        # includes "product" -- see _found_nothing()'s "message" without
        # "product" check above, which is exactly what rules the failure
        # case out here.
        set_last_priced_product(session_id, result.get("product"))
    elif tool_name == _RECOMMEND_TOOL and _tool_succeeded(result):
        set_last_priced_product(session_id, None)


def _update_last_presented_products(session_id: str, tool_name: str, result: dict | None) -> None:
    """A successful recommend_products call remembers exactly what it
    just showed, via the SAME selection response_formatter.py uses to
    render it -- see response_formatter.select_presented_groups()'s own
    docstring for why this matters, and memory.set_last_presented_
    products()'s docstring for the shape/lifecycle. Deliberately never
    clears this on any other tool -- see that docstring for why a
    single-item follow-up shouldn't erase the ability to reference the
    list it came from.

    Reads recommendations via .get(), not result["recommendations"],
    deliberately: _tool_succeeded()/_found_nothing() only guarantee "not
    an empty recommendations list when that key IS present" -- they say
    nothing about a result shape that lacks the key altogether (found via
    real test failures against pre-existing tests that mock execute_tool()
    with an unrelated shape while understand_customer() still names
    recommend_products, e.g. a turn-trace test using {"products": []}).
    That mismatch is only ever a test-mock artefact, never real
    recommend_products output, but this function has no way to tell the
    difference -- so it fails safe (remembers nothing) rather than crash,
    the same duck-typed defensiveness every other shape check in this
    codebase already uses."""
    if tool_name != _RECOMMEND_TOOL or not _tool_succeeded(result):
        return
    recommendations = result.get("recommendations")
    if not recommendations:
        return
    groups = select_presented_groups(recommendations)
    set_last_presented_products(session_id, groups)


def _update_weight_ask_count(session_id: str, tool_name: str, result: dict | None) -> int | None:
    """Returns an incremented weight-ask count for response_formatter.py
    to key its phrasing variant off, or None when this turn isn't a
    weight question at all -- see memory.increment_weight_ask_count()'s
    docstring for the live failure this closes (Webb, 2026-08-30: three
    follow-ups about the same product's weight came back
    character-for-character identical).

    Checks "weight" in result, not _tool_succeeded(result) -- unlike
    every other _update_* helper here, a genuinely resolved "no weight
    on file for this product" (result["weight"] is None) is not a
    failure to vary phrasing on; get_product_weight's own shape (see
    product_tool.py) always includes the "weight" key once a product
    was actually found, whether or not a value was parseable. A real
    failure (product not found at all) has no "weight" key and is
    correctly excluded here, same as every other tool's error shape."""
    if tool_name != _WEIGHT_TOOL or not isinstance(result, dict) or "weight" not in result:
        return None
    return increment_weight_ask_count(session_id)


def _handle_conversation(arguments: dict) -> dict:
    # Defensive only: the LLM is instructed to always write a real reply
    # for converse (see llm.py's tool 8 description) since there's no
    # deterministic fallback that could construct one -- an empty/missing
    # reply here means the model didn't follow that, not a real business
    # state to recover from, so a generic greeting is the only sane default.
    reply = arguments.get("reply") if isinstance(arguments, dict) else None
    reply = str(reply).strip() if reply else ""
    return {"conversation_reply": reply or _CONVERSATION_FALLBACK_REPLY}
