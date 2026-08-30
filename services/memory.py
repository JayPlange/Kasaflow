"""
Per-session conversation memory.

Wired in now. The previous version of this file kept a single dict
shared by every request the process handled, which meant two different
customers' in-flight requests could silently overwrite each other's
context -- unsafe under any real concurrency, so it was deliberately
left unregistered as a tool.

This version keys memory by a session_id the caller provides (or that
main.py generates for them on their first request and hands back), so
one customer's context can never leak into another's. Sessions expire
after a period of inactivity so a long-running process doesn't
accumulate state forever.
"""

import threading
import time

_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity


class SessionStore:
    """Thread-safe, per-session key/value store with TTL-based expiry."""

    def __init__(self, ttl_seconds: float = _SESSION_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}  # session_id -> {"data": {...}, "expires_at": float}
        # Per-session locks for session_lock() below -- deliberately
        # separate from self._lock, which only ever protects a single
        # get()/set() dict operation. self._lock never spanned the real
        # risk window: read state -> call the LLM -> execute a tool ->
        # write state back, all as one logical turn. Two WhatsApp
        # messages from the same customer arriving close together (the
        # second sent before the first's slow LLM call has returned) can
        # otherwise interleave across that whole sequence on separate
        # threads and silently corrupt state -- e.g. a later write
        # landing after an earlier one meant to come first, clobbering
        # the customer's actual last-stated value. See router.py's
        # route_customer(), which holds session_lock() for the entire
        # turn, not just individual memory calls.
        self._session_locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def _is_expired(self, entry: dict, now: float) -> bool:
        return entry["expires_at"] < now

    def get(self, session_id: str, key: str, default=None):
        now = time.monotonic()
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or self._is_expired(entry, now):
                return default
            return entry["data"].get(key, default)

    def set(self, session_id: str, key: str, value) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or self._is_expired(entry, now):
                entry = {"data": {}, "expires_at": now + self._ttl}
                self._sessions[session_id] = entry
            entry["data"][key] = value
            entry["expires_at"] = now + self._ttl

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_session_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for entry in self._sessions.values() if not self._is_expired(entry, now))

    def session_lock(self, session_id: str) -> threading.Lock:
        """A lock dedicated to this session_id, created on first use.

        Callers must hold this for the FULL duration of a turn (read
        state -> call the LLM -> execute a tool -> write state), not
        just around individual get()/set() calls -- those already have
        their own, separate protection (self._lock above), which is
        correct for what it does but was never meant to, and doesn't,
        cover a whole turn. See the docstring on self._session_locks_guard
        in __init__ for the concrete failure this closes.

        Locks are never removed once created -- a small, bounded amount
        of long-lived memory per distinct session_id (a WhatsApp phone
        number, in production), not worth TTL-based cleanup machinery
        for a single small business's real customer volume."""
        with self._session_locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock


# Module-level store shared by the app process. Safe now because every
# access is keyed by session_id and guarded by a lock, unlike the old
# shared dict this replaced.
_store = SessionStore()

# Arguments worth remembering across turns within the same session, so
# "how much is shipping for that one" can resolve without the customer
# repeating themselves. Not every tool takes all of these, callers only
# fill and save whichever keys are actually present in that tool's
# arguments. "category" lets a follow-up like "what about in 14k?"
# after "what rings do you have" remember "Rings" without repeating it.
# "delivery_option" means a customer who already said "deliver to Accra"
# earlier in the conversation doesn't have to repeat it when they get to
# actually placing the order. "quantity" and "delivery_address" close a
# real gap propose_order's own clarifying questions ("How many would
# you like?", "What address?") otherwise fell into: without these two
# remembered too, a customer's bare "2" or bare address in reply never
# stuck, and the next turn re-asked the same question forever (confirmed
# live, 2026-08-12) -- see get_order_draft() below for the other half of
# this fix.
_REMEMBERED_KEYS = ("product_name", "material", "category", "delivery_option", "quantity", "delivery_address")

# Which of the remembered keys matter for an in-progress order, and the
# words used to describe each to the LLM (see get_order_draft() and
# llm.py's _order_draft_state_line()).
_ORDER_DRAFT_KEYS = ("product_name", "material", "quantity", "delivery_address", "delivery_option")

# Fields that describe a specific ITEM and shouldn't be assumed to
# still apply once the customer names a different product -- see
# fill_missing_context()'s and remember_context()'s product-switch
# guards below. A ring's karat and a necklace's karat are genuinely
# different facts, so these are always required to be restated (or
# explicitly re-given in the same message) once the product changes.
_PRODUCT_SPECIFIC_KEYS = ("material", "quantity")

# Fields that describe the CUSTOMER's order as a whole, not the
# specific item -- "deliver to Accra, rider delivery" doesn't stop
# being true because the customer swapped which ring they want. These
# are deliberately NOT cleared or blocked from backfilling on a product
# switch, unlike _PRODUCT_SPECIFIC_KEYS above.
#
# Webb, 2026-08-20 (check #6 follow-up): the first pass of this fix put
# delivery_address/delivery_option in _PRODUCT_SPECIFIC_KEYS too, on the
# conservative "ask rather than guess" reasoning used elsewhere in this
# codebase. On review, Webb concluded that reasoning doesn't actually
# apply here -- unlike karat or quantity, an address isn't a fact that
# could plausibly differ per product in the same conversation, so
# treating it as product-specific meant re-asking a question the
# customer had already answered, for no real safety benefit. Corrected
# the classification rather than carrying the original choice forward
# by default.
_ORDER_LEVEL_KEYS = ("delivery_address", "delivery_option")


def _is_unknown(value) -> bool:
    return isinstance(value, str) and value.strip().lower() == "unknown"


def _names_a_different_product(session_id: str, arguments: dict) -> bool:
    """True when `arguments` explicitly names a product other than the
    one currently remembered for this session. Shared by
    fill_missing_context() (don't backfill the old product's details
    onto the new one) and remember_context() (don't leave the old
    product's details sitting there once the new one's own turn writes
    its product_name) -- see both docstrings."""
    new_product = arguments.get("product_name")
    remembered_product = _store.get(session_id, "product_name")
    return (
        remembered_product is not None
        and new_product is not None
        and not _is_unknown(new_product)
        and str(new_product).strip().lower() != str(remembered_product).strip().lower()
    )


def _names_a_different_address(session_id: str, arguments: dict) -> bool:
    """True when `arguments` explicitly states a delivery_address other
    than the one currently remembered for this session.

    delivery_option is DERIVED from delivery_address (see
    geocoding_tool.infer_delivery_option()), not an independent fact --
    "rider delivery within Kumasi" only means anything in relation to
    some specific address. Confirmed live, 2026-08-20: once
    delivery_option started carrying forward across turns (see
    _ORDER_LEVEL_KEYS above), a customer who gave a NEW address later in
    the same conversation (Accra, then Kasoa) kept getting the OLD
    address's stale delivery_option ("kumasi_rider") silently reused --
    order_tool.propose_order() only re-infers delivery_option from the
    address when the current value isn't already a valid key, so a
    stale-but-still-valid one is never revisited. Every reply came back
    "this address doesn't match our usual rider delivery within Kumasi
    zone", for an address nowhere near Kumasi. Used by
    fill_missing_context() and remember_context() below to stop
    delivery_option being backfilled/kept once the address itself has
    genuinely moved on, forcing propose_order() to re-derive it fresh
    for wherever the customer actually just said.

    Also True when NO address has ever actually been remembered yet, but
    a delivery_option somehow already has been. Webb's own first live
    trace run against the awaiting_field instrumentation, 2026-08-21,
    surfaced exactly this: a compound order message ("1 ... in 12k,
    deliver to Kumasi, rider delivery within Kumasi") had delivery_option
    ("kumasi_rider") extracted correctly but delivery_address silently
    failed to extract at all that turn (still "unknown"), so nothing was
    ever remembered for it. Several turns later, an address FINALLY
    arrived ("deliver to Kasoa") -- but because remembered_address was
    None rather than some earlier, genuinely different value, the
    original "same address, nothing to re-derive" case below never
    fired, and the stale "kumasi_rider" (which was never actually
    derived from any address at all) carried straight through to a real
    Kasoa order, producing the same "doesn't match our usual ... Kumasi
    zone" message this function exists to prevent. A delivery_option
    that was never grounded in ANY address is exactly as stale as one
    grounded in a now-superseded address -- both need to be re-derived
    the moment a real address actually lands."""
    new_address = arguments.get("delivery_address")
    if new_address is None or _is_unknown(new_address):
        return False
    remembered_address = _store.get(session_id, "delivery_address")
    if remembered_address is None:
        return _store.get(session_id, "delivery_option") is not None
    return str(new_address).strip().lower() != str(remembered_address).strip().lower()


def fill_missing_context(session_id: str, arguments: dict) -> dict:
    """Resolve any argument the model marked "unknown" using this session's last-known values.

    The prompt in services/llm.py tells the model to return "unknown"
    rather than guess when it cannot determine a product or material
    from the message alone -- this is what turns that sentinel into an
    actual resolved value from earlier in the same conversation.

    Guards against a real, reachable contamination path: if THIS call
    explicitly names a DIFFERENT product than the one remembered, its
    other "unknown" ITEM-specific fields (material, quantity -- see
    _PRODUCT_SPECIFIC_KEYS) are NOT backfilled from the old product's
    memory. Traced while designing a live check for the router-level
    product-identity correction guard, 2026-08-20: "order Product A in
    14k, 6 of them... actually I'll take Product B" -- without this
    guard, Product B would be silently, fully priced using Product A's
    exact karat/quantity, neither of which the customer ever stated for
    B, because fill_missing_context() had no concept of "these fields
    belonged to a different item". product_name and category are
    unaffected -- those describe what's being asked about, not a detail
    of a specific item, so they still resolve normally. Nor are
    delivery_address/delivery_option (see _ORDER_LEVEL_KEYS): those
    describe the customer's order as a whole, not the item, so they
    keep resolving from memory even across a product switch.

    Separately, and for the same "don't backfill a fact that's actually
    tied to something that just changed" reason: if THIS call states a
    genuinely different delivery_address than the one remembered, and
    doesn't ALSO explicitly restate delivery_option, delivery_option is
    NOT backfilled from the old address's memory either -- see
    _names_a_different_address()'s docstring for the live bug this
    closes. It's deliberately left unresolved ("unknown") rather than
    silently carrying the old address's arrangement, so
    order_tool.propose_order()'s own infer_delivery_option() call runs
    fresh against the new address instead of being skipped because a
    stale-but-still-valid key was already sitting there.

    Returns a new dict rather than mutating the one passed in. The
    caller's arguments dict may be referenced elsewhere (tests and
    callers that snapshot the LLM's raw output are the obvious cases),
    so mutating it in place risks a session's resolved value silently
    leaking into code that still expects the original "unknown".
    """
    resolved = dict(arguments)
    product_changed = _names_a_different_product(session_id, resolved)
    address_changed = _names_a_different_address(session_id, resolved)

    for key in _REMEMBERED_KEYS:
        if key not in resolved or not _is_unknown(resolved[key]):
            continue
        if product_changed and key in _PRODUCT_SPECIFIC_KEYS:
            continue
        if address_changed and key == "delivery_option":
            continue
        remembered = _store.get(session_id, key)
        if remembered is not None:
            resolved[key] = remembered
    return resolved


def remember_context(session_id: str, arguments: dict) -> None:
    """Persist any resolved (non-"unknown") arguments for later turns in this session.

    Deliberately not gated on `isinstance(value, str)` -- quantity comes
    back from the LLM as a real JSON number (an int), not a string, and
    the old `isinstance(value, str)` check silently dropped it every
    time, which is exactly why quantity never used to survive to the
    next turn. Only skips a key that's genuinely absent from this call's
    arguments (a different tool that doesn't take it) or explicitly
    "unknown" -- never overwrites a real remembered value with a
    stale/missing one.

    Write-side counterpart to fill_missing_context()'s read-side guard
    above: when THIS call explicitly names a different product than the
    one remembered, the OLD product's _PRODUCT_SPECIFIC_KEYS (material,
    quantity) are cleared before writing this call's own values. Without
    this, get_order_draft() (used by _describe_order_corrections() for
    the correction-note diff) would still see the old product's stale
    material/quantity once the new product_name lands, and misreport
    fields the customer is stating for the first time on the new item as
    "corrections". Traced while building the live check for the
    router-level product-identity correction guard, 2026-08-20 --
    product_changed must be computed BEFORE product_name itself is
    overwritten below, since _names_a_different_product() compares
    against whatever is currently remembered.

    _ORDER_LEVEL_KEYS (delivery_address, delivery_option) are NOT cleared
    on a product switch alone -- a product switch doesn't change where
    the customer lives or how they want it delivered, so those keep
    carrying forward exactly as they did before that fix.

    delivery_option specifically IS cleared here when delivery_address
    itself changes to something genuinely different (see
    _names_a_different_address()'s docstring for the live bug this
    closes: a stale "kumasi_rider" surviving long after the conversation
    moved to Accra/Kasoa addresses, because nothing ever re-derived it).
    delivery_option is a property OF an address, not an independent fact
    -- once the address moves on, the old arrangement shouldn't just sit
    there as if it still applies.
    """
    product_changed = _names_a_different_product(session_id, arguments)
    address_changed = _names_a_different_address(session_id, arguments)
    if product_changed:
        for key in _PRODUCT_SPECIFIC_KEYS:
            _store.set(session_id, key, None)
    if address_changed:
        _store.set(session_id, "delivery_option", None)

    for key in _REMEMBERED_KEYS:
        if key not in arguments:
            continue
        value = arguments[key]
        if not _is_unknown(value):
            _store.set(session_id, key, value)


def get_order_draft(session_id: str) -> dict | None:
    """Snapshot of this session's remembered order-relevant slots.

    Exists for llm.py's prompt (_order_draft_state_line()): the model
    otherwise sees only the customer's current message, no conversation
    history, so once propose_order has asked "How many would you like?"
    a bare "2" in reply is unresolvable on its own -- the model has no
    way to know that's an answer to a question it can't see. Handing it
    this snapshot lets it recognise a short reply as continuing an order
    already in progress, rather than guessing at a different tool
    entirely (confirmed live, 2026-08-12: it did guess wrong).

    Returns None when nothing order-relevant has been given yet, so the
    prompt can skip this section entirely for a fresh conversation.
    """
    draft = {key: _store.get(session_id, key) for key in _ORDER_DRAFT_KEYS}
    if not any(value is not None for value in draft.values()):
        return None
    return draft


_PENDING_INTENT_KEY = "pending_intent"


def set_pending_intent(session_id: str, tool_name: str | None) -> None:
    """Marks (or clears, with None) that this session asked for a
    product's price/photo/quote but the product itself couldn't be
    resolved yet -- get_product_price/generate_quote called with
    product_name genuinely unknown, and found nothing.

    Exists for the same reason as get_order_draft() above, one layer
    earlier: without it, a customer who says "yeah i wanna see pictures"
    (no product named -- genuinely ambiguous with several items just
    shown) then names the product on their VERY NEXT message gets asked
    yet another clarifying question instead of the system just doing
    what they already asked for (confirmed live, 2026-08-13: the
    customer had to name the product twice and was still told "I
    couldn't find that one"). See llm.py's _pending_intent_state_line()."""
    _store.set(session_id, _PENDING_INTENT_KEY, tool_name)


def get_pending_intent(session_id: str) -> str | None:
    return _store.get(session_id, _PENDING_INTENT_KEY)


_LAST_ACTION_OUTCOME_KEY = "last_action_outcome"


def set_last_action_outcome(session_id: str, outcome: dict | None) -> None:
    """Records (or clears, with None) a genuine, unrecoverable failure of
    a real business action -- e.g. propose_order finding a product with
    no WooCommerce id behind it, or confirm_order's write to WooCommerce
    itself failing. Deliberately NOT used for ordinary "still missing a
    detail" prompts (how many, what address, ...) -- those are already
    self-explanatory and get_order_draft() already covers them; this is
    for the case where the customer did everything right and it still
    didn't work, for a reason they have no way to guess.

    Exists because a customer's very next message after a failure like
    that is often "why?" -- and without this, understand_customer() has
    no way to know a failure even happened a moment ago, let alone why
    (confirmed live, 2026-08-13: "why?" got "could you clarify what you
    mean", and the next message got told "I haven't seen any order from
    you yet" -- an outright false claim, not just a vague one). Expected
    shape: {"action": "propose_order", "customer_safe_explanation": "..."}
    -- a ready-made, already-safe-to-say sentence, not raw internal detail,
    so the model explains the real reason instead of guessing at one or
    denying anything happened. See llm.py's _last_action_outcome_state_line()."""
    _store.set(session_id, _LAST_ACTION_OUTCOME_KEY, outcome)


def get_last_action_outcome(session_id: str) -> dict | None:
    return _store.get(session_id, _LAST_ACTION_OUTCOME_KEY)


_LAST_PRICED_PRODUCT_KEY = "last_priced_product"


def set_last_priced_product(session_id: str, product_name: str | None) -> None:
    """Records (or clears, with None) the exact catalogue product name a
    get_product_price/generate_quote call most recently resolved to,
    independent of _REMEMBERED_KEYS's "product_name" -- that one only
    fills an argument the LLM already decided to pass as "unknown"; this
    is the thing a NEW context line can offer the model proactively, the
    same pattern _pending_intent_state_line() already uses.

    Exists because a bare karat-only follow-up ("what about in 18k")
    after a specific product was just priced has no product name of its
    own for the LLM to work with, and previously had nothing telling it
    one was still the active topic -- confirmed live, 2026-08-18: exactly
    that message returned an unrelated 4-item recommendations list
    instead of re-quoting the same product at 18k. See llm.py's
    _last_priced_product_state_line().

    Deliberately cleared on a successful recommend_products call (see
    router.py's _execute_single()) -- a genuine category browse means
    the topic has moved on from one specific item, so a later bare
    karat message should mean "this category at that karat", not
    silently re-attach itself to whatever was priced several turns
    ago."""
    _store.set(session_id, _LAST_PRICED_PRODUCT_KEY, product_name)


def get_last_priced_product(session_id: str) -> str | None:
    return _store.get(session_id, _LAST_PRICED_PRODUCT_KEY)


_LAST_PRESENTED_PRODUCTS_KEY = "last_presented_products"
_LAST_PRESENTED_PRODUCTS_GENERATION_KEY = "last_presented_products_generation"


def set_last_presented_products(session_id: str, groups: list[tuple[str, list[dict]]]) -> None:
    """Records the exact numbered/bulleted list a recommend_products
    reply just showed this customer, so "the second one"/"the first
    ring" can resolve deterministically against it -- see llm.py's
    _last_presented_products_state_line().

    `groups` is response_formatter.select_presented_groups()'s own
    return shape (`[(product_name, [catalogue_row, ...]), ...]`, already
    in shown order) -- passed straight through from router.py's call to
    that SAME function, not recomputed here, so what gets remembered can
    never drift from what was actually rendered (see that function's own
    docstring for why this matters: this codebase has already been
    bitten twice by two code paths silently disagreeing about the same
    fact). Only `product_name` and `category` are kept per entry --
    deliberately NOT price or material (Webb, 2026-08-25): the list
    tells the model WHICH product a position refers to, never what it
    costs or what karat it was shown at, since a fresh get_product_price
    call determines the current, authoritative price once the referent
    is resolved. A list rendered at 14k does not mean "the second one"
    is committed to 14k if the customer then asks its price in 18k.

    Overwritten wholesale by the next successful recommend_products call
    (last write wins, same as every other single-slot memory value in
    this file) -- deliberately NOT cleared by a single-item follow-up
    (pricing item #2, asking its weight, ...), so a customer can still
    say "actually show me the first one instead" after drilling into a
    later item. Also deliberately NOT touched by clear_order_state() --
    browsing context and order context are different things, same
    reasoning memory.py already applies to `category`/`pending_intent`
    there (see that function's own docstring).

    A monotonically increasing `generation` is stored alongside the
    items themselves, requested by Webb, 2026-08-25, specifically so a
    later consumer of this state can tell "this is the list currently on
    screen" apart from "this is some earlier, already-superseded list"
    if this state is ever read from somewhere other than one single
    locked turn. Belt-and-suspenders today: router.py's
    session_lock() already serialises an entire turn end to end (see
    that lock's own docstring), so there is no live race this actually
    closes yet -- but it costs one integer to have ready before it's
    needed, rather than retrofitting it once something new (a
    background check-in process, say) reads this state outside that
    lock."""
    generation = (_store.get(session_id, _LAST_PRESENTED_PRODUCTS_GENERATION_KEY) or 0) + 1
    items = [
        {"position": position, "product_name": name, "category": variants[0].get("category")}
        for position, (name, variants) in enumerate(groups, start=1)
    ]
    _store.set(session_id, _LAST_PRESENTED_PRODUCTS_GENERATION_KEY, generation)
    _store.set(session_id, _LAST_PRESENTED_PRODUCTS_KEY, {"generation": generation, "items": items})


def get_last_presented_products(session_id: str) -> dict | None:
    return _store.get(session_id, _LAST_PRESENTED_PRODUCTS_KEY)


_WEIGHT_ASK_COUNT_KEY = "weight_ask_count"


def increment_weight_ask_count(session_id: str) -> int:
    """Increments and returns this session's running count of
    get_product_weight calls that actually reached a weight-bearing
    result (found a value or confirmed none is on file -- see
    router.py's _update_weight_ask_count(), the only caller). response_
    formatter.py uses the returned count to select a phrasing variant,
    so a customer questioning or re-asking about the same fact
    ("that's the weight?", "is that really 1g?", "how many grams is
    that?") doesn't get the identical canned sentence every time.

    Confirmed live, 2026-08-30 (Webb): those three follow-ups, asked
    back to back about the same product, all came back character-for-
    character identical -- "The Minimal White Stone Gold Ring, 1g
    weighs 1g." three times. The routing underneath was correct each
    time (a fresh get_product_weight call, not an answer pulled from
    conversation memory -- see llm.py's disputed-weight guardrail), the
    repetition was purely in response_formatter.py's single fixed
    template. This counter is the fix's other half.

    Deliberately a single per-session count, not per-product: this
    exists to fix "the same question sounds robotic asked twice in a
    row", not to track which specific product's weight has been
    discussed how many times. Deliberately never reset by anything else
    in this file (not clear_order_state, not a product switch) --
    unlike last_priced_product, phrasing variety has no business-state
    correctness requirement to protect, so there's no failure mode from
    letting it keep climbing for the rest of the session."""
    count = (_store.get(session_id, _WEIGHT_ASK_COUNT_KEY) or 0) + 1
    _store.set(session_id, _WEIGHT_ASK_COUNT_KEY, count)
    return count


def clear_order_state(session_id: str) -> None:
    """Clears the remembered order-relevant slots and last_priced_product.

    Call this the moment an order is CONFIRMED, not just when a proposal
    is superseded. Previously nothing did: _finalize_confirmation() in
    order_tool.py only ever cleared its own pending-order key, so
    product_name, material, quantity, delivery_address, delivery_option,
    and last_priced_product all sat untouched in session memory after a
    real, paid-for order -- ready to silently bleed their stale values
    into whatever the customer asks for next (2026-08-20 architecture
    audit, failure #3: a second, unrelated order placed shortly after
    could inherit the just-confirmed order's product/address, or receive
    a fabricated "I've updated..." correction note for a field the new
    order never even mentioned).

    Deliberately does NOT clear "category" or pending_intent -- a
    customer browsing "what other rings do you have" right after
    confirming an order is a normal continuation, not contamination."""
    for key in _ORDER_DRAFT_KEYS:
        _store.set(session_id, key, None)
    _store.set(session_id, _LAST_PRICED_PRODUCT_KEY, None)


_JUST_CONFIRMED_ORDER_KEY = "just_confirmed_order"


def set_just_confirmed_order(session_id: str, confirmation: dict | None) -> None:
    """Marks (or clears, with None) that an order was confirmed as the
    LAST thing that happened in this session, nothing since.

    Deliberately ephemeral, unlike order_tool.py's own long-lived
    "most recently confirmed order" record (used there to resolve a
    bare "cancel my order" days later) -- this one exists only to tell
    the model, for exactly one turn, that a confirmation just happened,
    so a genuinely unrelated next message doesn't get its details read
    against a stale, already-completed order, and so the model can give
    a naturally different reply to "what's my order number?" than to a
    fresh "what would you like to order?". router.py clears this back to
    None at the start of handling every turn, same pattern as
    set_awaiting_confirmation() above, and only sets it non-None again
    when THIS turn's action was confirm_order succeeding."""
    _store.set(session_id, _JUST_CONFIRMED_ORDER_KEY, confirmation)


def get_just_confirmed_order(session_id: str) -> dict | None:
    return _store.get(session_id, _JUST_CONFIRMED_ORDER_KEY)


_AWAITING_CONFIRMATION_KEY = "awaiting_confirmation"


def set_awaiting_confirmation(session_id: str, value: bool) -> None:
    """Marks whether a bare agreement ("yes", "yeah", "ok") right now
    should be read as confirming the session's pending order.

    Exists because a pending order can sit unconfirmed for the rest of
    the session's 30-minute TTL, and in that time the assistant can go
    on to offer or ask the customer something completely unrelated
    ("want to see a few cheaper options?"). Without this flag, a bare
    "yeah" answering THAT question reads exactly like a bare "yeah"
    confirming the stale order -- llm.py's pending_order guidance
    previously had no way to tell the two apart, which is a real
    commerce-integrity risk (an order gets placed the customer never
    meant to place), not just a UX rough edge.

    router.py sets this to True only immediately after a propose_order
    call that actually produced a full proposal, and back to False at
    the start of handling every other turn -- so it's True only when
    the LAST thing that happened in this session was proposing this
    exact order, nothing since. See llm.py's _pending_order_state_line()
    for how this changes the confirm_order guidance."""
    _store.set(session_id, _AWAITING_CONFIRMATION_KEY, value)


def is_awaiting_confirmation(session_id: str) -> bool:
    return bool(_store.get(session_id, _AWAITING_CONFIRMATION_KEY, False))


_AWAITING_FIELD_KEY = "awaiting_field"

# The only values this ever holds -- one specific thing this assistant
# itself just asked the customer for, deterministically, as its very
# last action. Deliberately a small, closed set: only the fields
# propose_order's own missing-detail questions ask for one at a time,
# plus "confirmation" for the question a full proposal itself poses
# ("would you like to go ahead?"). Not every gap in an order needs an
# entry here -- e.g. "which product" has no single deterministic
# pattern a reply could be checked against the way a bare karat or a
# bare number does, so it's left to the existing LLM-driven path
# entirely, same as before this existed.
AWAITING_FIELDS = ("material", "quantity", "delivery_address", "delivery_option", "confirmation", "delivery_interest")


def set_awaiting_field(session_id: str, field: str | None) -> None:
    """Marks (or clears, with None) the one specific thing this
    assistant's own last message asked the customer for, so their very
    next reply can be checked against a deterministic pattern for that
    field BEFORE the general LLM tool-selection call ever runs -- see
    router.py's _try_resolve_awaiting_field().

    Requested by Webb, 2026-08-21 (P0.4), specifically to close the
    repeated live failure where a bare "14k" answering propose_order's
    own "What karat would you like that in?" misrouted to
    recommend_products three separate times, despite order_draft's
    prompt guidance already covering this exact case in detail (see
    llm.py's _order_draft_state_line()) -- more prompt text was not
    enough; deciding correctly needs to not depend on the model at all
    for this specific, narrow, pattern-matchable class of reply.

    Same "last thing that happened, nothing since" lifetime as
    set_awaiting_confirmation() above: router.py resets this to None at
    the very start of handling every turn, and only sets it again based
    on what THIS turn's own propose_order/confirm_order call actually
    produced. It does not persist across an unrelated turn in between
    (a converse reply, a browse) -- Webb's own phrasing was "direct
    answers to the assistant's IMMEDIATELY PRECEDING question", not
    "at any point later in the conversation"; get_order_draft() already
    covers the longer-lived version of this for the LLM's own prompt."""
    if field is not None and field not in AWAITING_FIELDS:
        raise ValueError(f"Unknown awaiting_field: {field!r} (must be one of {AWAITING_FIELDS} or None)")
    _store.set(session_id, _AWAITING_FIELD_KEY, field)


def get_awaiting_field(session_id: str) -> str | None:
    return _store.get(session_id, _AWAITING_FIELD_KEY)


def get_session_store() -> SessionStore:
    """Exposed so tests (and, later, a healthcheck or admin endpoint) can
    inspect the store without reaching into the module-level `_store`
    directly."""
    return _store
