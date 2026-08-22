# KasaFlow architecture audit — 2026-08-20

Scope: full code-level audit requested after a live transcript surfaced a karat-defaulting bug and a policy-misroute bug on the same day. Both were already patched (see `services/llm.py` diff, commits pending). This audit asks the bigger question raised on review of the repository: is KasaFlow a tool router with memory slots, or a conversational agent, and what does the code actually prove either way.

No code changes in this document. Five sub-audits were run in parallel against the actual source (not the transcript alone): state isolation, conversation continuity/response generation, turn concurrency, tool routing/order safety, and voice/delivery. Every claim below is traceable to a specific file and line; nothing here is inferred from the transcript without independent code confirmation.

## Where the review's diagnosis was right, and where it needs correcting

The central claim — that KasaFlow behaves like a tool router with memory slots rather than a conversational agent because the model never sees what it or the customer actually said in prior turns — is confirmed without qualification. `_build_prompt()` in `services/llm.py` has exactly six interpolation points: five computed state-summary lines (`pending_order`, `order_draft`, `pending_intent`, `last_action_outcome`, `last_priced_product`) and the current message. None of them, nor anything in `services/memory.py`'s `SessionStore`, ever carries a literal prior reply or a literal prior customer message. The raw text of every turn is discarded the moment the HTTP request that produced it returns — there is no transcript, log, or history structure anywhere in the codebase to draw on even if someone wanted to wire it in later. The fix already applied to the karat-dispute bug is a hardcoded example sentence in the prompt's rules section, not a structural fix — it will not generalise to a differently-worded dispute or reaction.

One piece of the review's framing needs correcting: `response_formatter.py` is not, on inspection, a major contributor to the "robotic" feel. It was read in full. Nearly every templated phrase correctly isolates a safety-critical fact (price, order ID, delivery arrangement, confirmation status) from surrounding tone-only boilerplate ("Lovely choice", "Good news"), exactly as its own docstring claims. The boilerplate could be varied freely without correctness risk, but that's a minor polish job, not an architectural rewrite. The actual failure the transcript showed — "wow" answered with "Wow indeed!", a dispute answered with an unrelated warranty paragraph — happened upstream of the formatter, in tool selection, because the model had no grounding for what just happened. Fixing the formatter's wording would not have prevented either bug.

The audit also surfaced two things the original review didn't flag, and one of them is more severe than anything in the original transcript: a stale, still-pending, unconfirmed order can be accidentally confirmed by an unrelated "yeah" to something else entirely, creating a real order the customer never meant to place. That's ranked #1 below.

## Ranked failure taxonomy

### 1. A stale pending order can be confirmed by an unrelated "yeah"

**Evidence.** `services/llm.py`'s pending-order prompt line instructs: if the message "clearly confirms this ('yes', 'yh', 'ok'...)", call `confirm_order`. The session's `pending_order` stays live for the full 30-minute session TTL (`services/memory.py`, `_SESSION_TTL_SECONDS`). `converse`'s own worked example in the prompt is "ei that's expensive oo" → "want me to show you a few more affordable options?" — a customer replying "yeah" to *that* offer, while an earlier proposal from the same session is still sitting unconfirmed, reads as a confirmation under the rule as written.

**Example.** Customer proposes an order, doesn't confirm, browses something else, gets asked "want to see a few more affordable options?", says "yeah" meaning *show me options*. The system places the original order instead.

**Impact.** Real WooCommerce order created without genuine confirmation intent. This is the one failure mode with direct financial/inventory consequences, not just a bad reply.

**Current protection.** `confirm_order()` requires a `pending_order` to exist at all — it can't fabricate an order from nothing. That's the only guard.

**Missing protection.** Nothing checks that the current message is actually about the *specific* pending proposal, as opposed to any pending proposal existing somewhere in session memory.

**Fix.** Tighten `confirm_order` tool selection to require the message plausibly reference the specific pending item (not just an isolated "yes"/"yeah" with a competing offer more recently on the table), or track "what was most recently offered" separately from "what order is pending" so the two can't be conflated. Small, contained change — does not require the full conversation-buffer work.

### 2. Two near-simultaneous messages can corrupt session state

**Evidence.** `services/memory.py`'s `SessionStore` has one `threading.Lock`, held only inside individual `get()`/`set()` calls — never across a read→LLM-call→execute→write sequence. `app/main.py`'s `/process` is a sync `def`, run on Starlette's thread pool; `app/whatsapp_routes.py`'s webhook handler dispatches each incoming message as a separate `BackgroundTask`, also thread-pool-executed. The LLM call in the middle of a turn is a blocking network call, typically 1-3s, up to 15-30s+ on retries, with no lock held across it. Deployment is single-worker (`Dockerfile` has no `--workers` flag, `fly.toml` is a single shared-CPU VM), which is good news — an in-process lock would actually be sufficient, no distributed-lock complexity needed.

**Example.** Customer sends "I want 14k" then "make it 6" a second later. Both requests read the same starting state; if the first is still waiting on its LLM call when the second's finishes and writes, the first's write can land afterward and silently overwrite the customer's actual last-stated value.

**Impact.** Wrong quantity, karat, address, or delivery choice silently recorded, with no error visible to anyone.

**Current protection.** None beyond the per-field lock, which prevents dict corruption but not sequence interleaving.

**Missing protection.** A per-session lock spanning the full turn.

**Fix.** A `threading.Lock` per `session_id`, held for the duration of `route_customer()`, not just individual memory calls. Contained, mechanical, and — per the deployment audit — sufficient given the current single-worker setup.

### 3. Memory isn't cleared after a CONFIRMED order, only after a pending one

**Evidence.** `order_tool.py`'s `_finalize_confirmation()` clears the pending-order key but never touches `services/memory.py`'s `_REMEMBERED_KEYS` (`product_name`, `material`, `quantity`, `delivery_address`, `delivery_option`) or `last_priced_product`. The guard added earlier today (`get_pending_order_summary(session_id) is None`) is also true immediately *after* a successful confirmation, since confirmation clears that same key — so it doesn't distinguish "nothing pending because never proposed" from "nothing pending because it just got placed." `_describe_order_corrections()` then diffs the new order's arguments against the old, now-confirmed order's stale fields, and the LLM prompt has no signal a confirmation ever happened at all (there's a well-built, tested prompt line for the pending case; none exists for the confirmed case).

**Example.** Customer confirms an order for necklace A, then a few minutes later orders ring B for a different address. The proposal for ring B can arrive carrying a fabricated "I've updated your order" note referencing necklace A's stale details, and any field the customer didn't restate can get silently backfilled from the already-placed order.

**Impact.** Same class of customer-visible contamination as the pending-order bug fixed today, on the more common path (most sessions that reach a pending order go on to confirm it).

**Current protection.** None specific to this case — the fix applied today only covers the unconfirmed branch.

**Missing protection.** A clear/reset of `_REMEMBERED_KEYS` and `last_priced_product` on confirmation, and a prompt signal for "an order was just placed" mirroring the one that already exists for "an order is pending."

**Fix.** Mirror the pending-order fix: clear the remembered slots in `_finalize_confirmation()`, and extend the guard to also check the immediately-preceding action, not just current pending-order presence.

### 4. No conversation history — the root cause behind items 1, 5, and most "feels robotic" complaints

**Evidence.** Confirmed in full above. The model reconstructs "what just happened" entirely from five computed summary lines; it never sees literal prior text.

**Example.** Any reactive message not already hardcoded as a worked example in the prompt — "wow", "same one", "no that's not it" — has no ground truth to resolve against.

**Impact.** This is the mechanism, not a single bug — it's why each new transcript surfaces a *new* misfire even after the previous one gets a hardcoded fix. Patches accumulate; the underlying gap doesn't close.

**Current protection.** None structural. Two hardcoded example sentences exist for the two cases found live so far.

**Missing protection.** Real (even minimal) recent-turn grounding.

**Fix.** Add the last 2-4 turns (literal customer message + literal assistant reply text, not just structured fields) to the prompt. This is architectural work, not a prompt tweak, and is the single change most directly responsible for closing this entire class rather than one instance of it at a time.

### 5. `converse` is the one tool that writes customer-facing prose with zero grounding or downstream check

**Evidence.** Every other tool returns structured data that `response_formatter.py` turns into text through a template that ties every claim to a real field. `converse`'s `reply` argument is written directly by the same tool-selection LLM call that has no conversation history (item 4), and `response_formatter.py` does zero templating on it — it passes straight through.

**Example.** Same as item 4's example, specifically for the branch that lands in `converse` rather than a misrouted business tool.

**Impact.** Any future reactive/social message not covered by the business-tool rules gets an ungrounded, unverified reply with no safety net.

**Current protection.** None beyond the (currently thin) rules telling the model when *not* to use `converse`.

**Missing protection.** Either grounding (item 4) or a downstream check on `converse` output, or both.

**Fix.** Depends on item 4 landing first; longer-term this is the natural candidate for folding into a constrained response composer, per the sequencing already agreed — build that only after the evaluator exists to catch regressions.

### 6. The policy knowledge base has no topic-relevance cross-check

**Evidence.** `knowledge_base.py`'s `retrieve()` returns the top-scoring document above a `min_score=0.15` cosine-similarity threshold with no check that the returned document's topic actually matches the question. With only ~6 short policy documents, a coincidental above-threshold match against the wrong document is a structural risk, not a one-off — it's exactly what happened with the karat dispute (scored above 0.15 against the warranty doc).

**Example.** Any future message that (a) the LLM doesn't recognise as an order/product dispute — so it still gets routed to `answer_policy_question` — and (b) coincidentally scores above 0.15 against an unrelated policy doc, reproduces today's exact failure with no code-level backstop.

**Impact.** Confidently wrong-sounding answers to genuinely off-topic questions.

**Current protection.** Only the LLM-side routing rule patched today, which prevents the LLM from calling the tool for one specific worked example — it does nothing to the retrieval mechanism itself.

**Missing protection.** A confidence floor high enough to actually distinguish real matches from noise (0.15 may be too permissive for a 6-document corpus), or a secondary check that the retrieved document's title/category plausibly matches the question's apparent topic.

**Fix.** Raise `min_score` and validate against real query/document pairs (both true positives and known-bad queries like the karat one) before picking a new threshold; consider tagging documents with topic keywords for a cheap secondary check.

### 7. Some real Ghanaian towns fall through delivery classification entirely; two entries risk substring false positives

**Evidence.** Kasoa — a real, populous Greater Accra town — is absent from all three of `_ACCRA_NEIGHBOURHOODS`, `_KUMASI_NEIGHBOURHOODS`, and `_GHANA_PLACE_NAMES`, so it resolves to `None` and forces the generic three-way delivery question, rather than the `ghana_other`/team-confirm treatment Tema and Madina (also Greater Accra satellite towns) correctly get. Separately, `"ho"` and `"wa"` are two-character substring matches in `_GHANA_PLACE_NAMES`, which risk matching unrelated words containing those letters.

**Impact.** Kasoa customers get an unnecessary clarifying question the neighbourhood-matching feature was specifically built to avoid. The substring risk is currently untriggered in testing but is a latent false-positive source.

**Current protection.** None for Kasoa specifically; no test exists for either issue.

**Missing protection.** Kasoa (and likely other populous Greater Accra towns) added to the list; word-boundary matching instead of bare substring for two-letter place names.

**Fix.** Small, additive data/logic change to `services/delivery_tool.py`.

### 8. Cross-product field bleed in price/quote lookups is plausible and untested

**Evidence.** `fill_missing_context()` backfills `material` (and other remembered fields) with no per-product scoping. If a customer asks about product A in 14k, then asks about product B without restating a karat, the system will happily reuse 14k for B.

**Impact.** A quoted price for the wrong karat on an unrelated product, without any correction mechanism (that exists for orders, not for plain price lookups).

**Current protection.** None specific; category-level contamination (a dead/empty category sticking) is proven safe and tested, but the karat-carryover case across two named products is not.

**Missing protection.** Either scope `material`/`category` memory to "same product" more strictly, or extend the same correction-acknowledgment pattern already built for orders to price/quote lookups.

**Fix.** Needs a concrete test first to confirm this actually reproduces against the real model before prioritising a code change — currently a plausible risk, not a confirmed live bug.

### 9. Stored `delivery_option` and displayed label can disagree in one specific mismatch case

**Evidence.** When a customer explicitly claims "ship internationally" for an address that's actually in Ghana, `propose_order()` softens the customer-facing label ("a delivery arrangement to be confirmed by our team") but the raw `delivery_option` field stored in the proposal remains `"international"` verbatim.

**Impact.** Minor data-integrity inconsistency; the customer sees the correct message, but the stored record doesn't match what's displayed.

**Current protection.** The label-level fix already prevents the customer-visible harm.

**Missing protection.** The stored key isn't corrected to match.

**Fix.** Low priority; align the stored value with the corrected label the next time this code is touched.

### 10. No sanity-check layer between tool selection and tool execution

**Evidence.** `services/tool_executor.py` is a 36-line bare dispatcher: look up the tool by name, call it, wrap exceptions. No check anywhere that the chosen tool is appropriate given session state — the LLM's tool choice is trusted unconditionally at the dispatch layer.

**Impact.** This is the general mechanism underlying items 1, 5, and 6 — each of those is a specific case of "the model picked a tool that didn't fit the situation, and nothing caught it before it ran."

**Current protection.** Only tool-specific internal guards (e.g. `confirm_order`'s "is anything pending at all" check).

**Missing protection.** A lightweight state-aware sanity check at the dispatch layer, not a full rewrite.

**Fix.** Lower priority than fixing the specific instances (1, 5, 6) directly, but worth keeping in mind as those get addressed — a shared, general fix might end up cheaper than three separate ones.

## A. Already correct — don't redesign

`response_formatter.py`'s deterministic fact-injection design is sound and is not the main source of the robotic feel; every safety-critical fact is correctly isolated from tone-only wording. Order-safety invariants are proven for four of five `propose_order` fields (product, material, quantity, address are rejected-and-asked, never defaulted) and pricing is always backend-computed — the LLM cannot inject a price. `cancel_order` re-verifies live against WooCommerce and is fully isolated from the remembered-context bag. The delivery-zone design principle ("LLM extracts the address, backend determines the zone") is correctly enforced in code, not just prompt text — confirmed the LLM cannot override it. Single-worker deployment means the concurrency fix (item 2) can be a simple in-process lock; no distributed-lock complexity needed.

## B. Genuinely broken

Items 1, 2, 3, 7, and 9 above. Also the two bugs already patched today (karat defaulting, policy misroute) — noting that item 6 shows the policy misroute's underlying mechanism is only partially closed.

## C. UX imperfection only, not a correctness bug

Voice replies read the WhatsApp-text formatter output verbatim through TTS, with no speech-tuned phrasing — confirmed exactly as flagged, this is a product decision about voice quality, not a bug. `response_formatter.py`'s boilerplate tone phrasing could be varied for warmth without any correctness risk. "Wow" → "Wow indeed!" is flat, not wrong — it stops being awkward once item 4 exists, since the reply can then reference what actually just happened.

## D. Requires architectural work, not a prompt change

Item 4 (recent-turn conversation buffer), item 5 (folding `converse` into a grounded/checked path — already sequenced by Webb to come after the evaluator), item 2 (per-session turn lock), item 3 (task-scoped memory clearing on confirmation, not just on pending-order supersession), and item 10 (a general tool-selection sanity layer, lower priority).

## E. Recommended sequence for the next round (not this round — this round is audit + evaluator only, per agreed scope)

1. Per-session turn lock (item 2) — contained, mechanical, no conversational behaviour change, closes the highest-likelihood silent-corruption path.
2. Clear remembered state on confirmation (item 3) — small, follows the exact pattern already proven for the pending-order case.
3. Tighten `confirm_order` to require the message actually reference the specific pending proposal (item 1) — closes the highest-*impact* item without needing the full conversation buffer.
4. Add a minimal recent-turn buffer to the prompt (item 4) — the structural change most directly tied to this review's diagnosis and to most of today's live bugs; closes item 5 as a side effect for the cases the buffer covers.
5. Only after 1-4 are in and evaluator-verified: build the constrained response composer for `converse`/natural phrasing, exactly as already agreed — sequenced last so the evaluator can catch any regression it introduces.

## What this round is actually delivering

Per the scope agreed before this audit started: no code changes from this document. Next: build the ~20-30 scenario conversation evaluator (mocked-first, small real-LLM tier, critical scenarios repeated for stability) so that whichever of the above gets built next has a real regression net under it before it ships.
