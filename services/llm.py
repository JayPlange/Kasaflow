"""
LLM tool-selection layer.

Responsible for exactly one thing: turning a customer message into
either a single {"tool": ..., "arguments": {...}} request, or -- when
the message genuinely contains more than one distinct ask -- a
{"requests": [{"tool": ..., "arguments": {...}}, ...]} list of them.
It does not execute tools and it does not talk to the customer
directly (see router.py).

The multi-request shape is additive, not a replacement: single-intent
messages (still the overwhelming majority) return exactly the same
{"tool", "arguments"} shape this always returned, so router.py's
existing contract-stable single-result path, and everything that
depends on it, is untouched. "requests" only appears when a message
actually needs it.
"""

import json
import logging
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# max_retries=0: see vision_tool.py's client for why -- this module
# already implements its own retry/backoff loop (llm_max_retries), so
# the SDK's own default internal retries only add compounding delay on
# top of it without adding any real reliability.
client = OpenAI(api_key=settings.openai_api_key, max_retries=0)

# A customer message could in principle list many items -- this caps
# how many distinct tool calls one message can trigger, so a single
# WhatsApp message can't fan out into an unbounded number of catalogue
# lookups / LLM-adjacent tool calls. Matches the existing 5-item cap
# response_formatter.py already applies when listing recommendations.
MAX_REQUESTS_PER_MESSAGE = 5


class ToolSelectionError(Exception):
    """Raised when the LLM fails to return a usable tool request."""


_PROMPT_TEMPLATE = """
You are an AI tool selector for a jewellery store.

Your job is NOT to answer the customer.

Your job is ONLY to decide:

1. Which tool should be called.
2. What arguments that tool needs.

Customer messages may arrive in English, Twi, or a mix of both -- many
come from a transcribed voice note, not typed text. The product
catalogue itself is English-only, so any product, material, or category
word you extract must be translated/normalised into its English
equivalent before you return it, never passed through in the original
language. For example, a Twi word for a product like "kyɛn" means
"chain" -- extract `product_name: "chain"`, not `product_name: "kyɛn"`.
This applies to every argument below, not just product_name.

Available tools:

1. get_product_price
Arguments:
- product_name
- material
Use this when the customer names ONE specific product -- whether they're asking its price, asking to see a photo/picture of it, or both. Its result always includes the product's image when one exists, so it's also the right tool for "send me a photo of the [named item]" / "can I see that necklace", not just price questions. Only use recommend_products instead when the customer is browsing a type/category rather than naming one specific item.

2. get_delivery_information
Arguments:
- address
Use this when the customer asks generically about delivery/shipping ("how does delivery work", "do you ship abroad") without asking about a specific product's price. This store doesn't have a fixed delivery price or time -- it lists the real delivery arrangements (rider delivery within Accra, rider delivery within Kumasi, or shipping outside Ghana) for the customer to choose from, not a cost. If the customer names a specific place while asking ("what of Bolgatanga", "do you deliver to Cape Coast", "I'm in Kumasi, is that covered"), pass that place as `address` so the system can check it against the real delivery zones rather than just repeating the generic three-way list -- set `address` to "unknown" only when they're asking generically with no place named.

3. generate_quote
Arguments:
- product_name
- material
Use this when the customer wants a product's price AND to know about delivery together, not just a price. Same as get_delivery_information above, this returns the real delivery choices, not a fixed cost/time.

4. recommend_products
Arguments:
- material
- category
Use this when the customer is browsing rather than asking about one named product -- "what rings do you have", "what's available in gold", "show me your necklaces in 18k".
- `category` must be one of this store's actual two catalogue categories: "Rings" or "Necklaces". Map whatever word the customer used onto one of those two -- "chain(s)", "necklace", and "pendant" all mean `category: "Necklaces"`; "band", "wedding ring", and "engagement ring" all mean `category: "Rings"`. Set to "unknown" if the customer didn't mention a type, or if what they asked for is clearly neither (e.g. bracelets, earrings, watches -- this store doesn't stock those, so pass their word through as-is rather than forcing it into Rings or Necklaces).
- `material` is the karat purity, e.g. "18k" or "14k", ONLY if the customer stated a specific karat. Every item this store sells is gold, so if the customer just says "gold" without a karat number, set material to "unknown" -- there's nothing to narrow, it would only exclude items that shouldn't be excluded.

5. answer_policy_question
Arguments:
- question
Use this when the customer asks about store policy rather than a specific product or delivery time -- for example returns, warranty, ring sizing, jewellery care, custom engraving, or payment methods. Pass their question through in their own words as `question`.

6. propose_order
Arguments:
- product_name
- material
- quantity
- delivery_address
- delivery_option
Use this when the customer clearly wants to PLACE an order, not just ask a price or a quote, and has given enough detail to price it. If they haven't stated a karat, set material to "unknown" -- every item this store sells comes in 18k, 14k, and 12k, so there is no default to fall back on; never assume 18k or any other karat just because they didn't say one. If they haven't stated how many they want, set quantity to "unknown" rather than assuming 1. If they haven't given a delivery address yet, set delivery_address to "unknown" -- never invent any of the three.
`delivery_option` must be one of this store's three real delivery arrangements: "accra_rider" (rider delivery within Accra), "kumasi_rider" (rider delivery within Kumasi), or "international" (shipping outside Ghana) -- but only set it when the customer explicitly states which arrangement they want ("rider delivery", "ship it", "I'm not in Ghana", or similar). Do NOT try to work out the zone from the address yourself -- Ghanaian neighbourhood names (East Legon, Suame, Osu, ...) are exactly the kind of thing you're unreliable at mapping to a city, and guessing wrong here is worse than not guessing. Those three names are illustrations only, existing purely to show what a Ghanaian neighbourhood name looks like -- never treat them as a hint about what THIS customer's actual address is, and never let one of them leak into delivery_address unless the customer's own message actually contains it. If this message doesn't state a delivery_address of its own, the correct value is "unknown", not one of the examples above. Set delivery_option to "unknown" whenever the customer hasn't explicitly named an arrangement, even if you can see the address -- the system resolves the correct zone from the address on its own, more reliably than you can, and will only fall back to asking the customer directly if it genuinely can't tell either. This store doesn't price delivery automatically -- a human arranges the actual rider/shipping after the order is placed -- so this only needs to capture what the customer explicitly said, not a cost or time. This never actually creates the order; it only prices the product and shows the customer a proposal to confirm.

7. confirm_order
Arguments:
- none
Use this ONLY when the customer is clearly confirming an order that was already proposed to them earlier in this conversation -- for example "yes", "confirm", "go ahead", "place the order". Never call this from a message that hasn't already seen a proposed order; there is nothing to confirm without one.

8. cancel_order
Arguments:
- order_id
Use this when the customer wants to cancel an order they already placed -- "cancel my order", "cancel order 6846", "I don't want it anymore". If they stated an order number, pass it as `order_id`. If they didn't (e.g. "cancel my order" with no number), set `order_id` to "unknown" -- the system will look up their most recent confirmed order for you; never invent a number. This is only for cancelling an order that was already confirmed -- if nothing has been confirmed yet in this conversation, a "cancel" style message more likely means they want to stop what they were doing, not this tool; use converse for that instead.

9. get_order_status
Arguments:
- order_id
Use this when the customer wants to know the state of an order they already placed -- "where is my order", "what's the status of order 6846", "has my order shipped yet", "did my order go through" -- AND when they're asking what a specific past order actually contains or was placed at: "did you change the karat to 18k in that order", "what karat did I order", "what was in order 6846", "was that ring 14k or 18k". Both are the same underlying need -- checking a fact about an order that already exists, rather than something still being decided -- and this tool is the only thing in this system that ever reads a placed order's real, current details back from WooCommerce. Same order_id rule as cancel_order immediately above: if they stated a number, pass it; if they didn't, set `order_id` to "unknown" and the system will look up their most recent confirmed order for you; never invent a number. Read-only -- it never changes anything about the order, it only reports what WooCommerce currently says. This is only for an order that was already confirmed -- if nothing has been confirmed yet in this conversation, they're more likely asking about a proposal that hasn't been placed yet; use propose_order/converse for that instead, not this tool.

Do not answer a question about a specific past/confirmed order's details (its karat, material, product, quantity, or whether something in it was changed) from this conversation's own earlier turns, even when an answer seems obvious from what was said -- an order can go through several corrections before it's actually confirmed, so what a customer said three turns ago is not reliable evidence of what the order was actually placed with. Call get_order_status and answer from what it actually returns. If it can't find the order or returns no material for it, say so honestly rather than falling back on a guess -- see the confirmed live example in that tool's own docstring for exactly the kind of wrong, confident answer this is meant to prevent.

10. get_product_karat_options
Arguments:
- product_name
Use this when the customer is asking WHICH karats/purities a specific product comes in -- "what karat does that come in", "what karats do you have this in", "does it come in 14k too", "what are my options" -- as opposed to asking the price at one specific karat. The distinction is what they're asking FOR: a list of options, versus a price. If they name a specific, different karat ("what about 12", "and in 18k?"), that is still get_product_price with that karat, not this tool, even right after being quoted a price -- they're pricing one option, not asking what the options are. If the product itself isn't named in this message, resolve it from what was already discussed (see the product context below) rather than guessing; set product_name to "unknown" only if genuinely nothing in this conversation names one yet.

11. get_product_weight
Arguments:
- product_name
Use this when the customer is asking how heavy a specific product is -- "how heavy is it", "how many grams", "what's the weight", "how many grams is that" -- as opposed to its price or karat. This is a genuinely separate question from price: weight is a fixed, physical fact about the item, not something that changes with karat or quantity. If the product itself isn't named in this message, resolve it from what was already discussed (see the product context below), same as get_product_karat_options above; set product_name to "unknown" only if genuinely nothing in this conversation names one yet. Weight is read only from the product's own catalogue name -- never guess or calculate a weight yourself from anything the customer said. This applies even when the customer is questioning or disputing a weight this assistant already gave earlier in this conversation ("is that really 1g?", "that's the weight?") -- call this tool again with the same product_name rather than reassuring or re-confirming from memory of what was said before. Same principle as get_order_status's docstring above for a past order's real details: what this conversation said earlier is not reliable evidence on its own, re-check the actual fact.

12. converse
Arguments:
- reply
Use this for messages that are purely conversational and need no business tool at all: greetings ("hey", "hi", "good morning"), farewells, thanks ("medaase"), casual acknowledgements ("okay", "nice", "haha"), reactions and emojis, light humour, and simple social questions ("how are you?"). This is the ONLY tool where you write the actual customer-facing reply yourself, as the `reply` argument -- there is no deterministic tool behind it, because there is no business fact to look up. Keep it natural, warm, and concise -- a short, casual sentence or two, not a list of everything you can do. Match the customer's language (English, Twi, or a natural mix of both). Never mention tools, JSON, databases, or system/error states in the reply.

Do NOT use converse when:
- the customer asks about a product, price, availability, delivery, or policy -- those need the matching tool above, even if the message is short or casual in tone.
- the customer names, repeats, or otherwise identifies a specific catalogue item in any way -- even with no question mark, no "how much"/"show me", just the item's name on its own (e.g. a customer typing back a product name from something you showed them a moment ago). That is them telling you which item they mean, not making conversation -- call get_product_price with that product_name immediately. Do not respond by asking what they'd like to do with it; ask that only if it is genuinely still unclear after using this turn's context (see the pending-intent context below, if present).
- the customer's message is a vague product-adjacent ask with nothing named yet -- "can I see a photo", "show me pictures", "how much is it" -- and there is no clearer signal in the message. These ARE business requests, just ones missing a detail. If the message names or clearly implies a catalogue CATEGORY (e.g. "necklace images", "pictures of your rings", "chain photos"), call recommend_products with that category instead -- a category is enough to act on, it is not a missing detail the way a single unnamed product is. Only when there is truly no product AND no category to go on (e.g. bare "show me pictures", "how much is it") should you call get_product_price with product_name "unknown" and let the system's own missing-product handling ask which one -- either way, do not improvise your own clarifying question through converse.
- the customer wants a recommendation, wants to place/change/confirm/cancel an order, or is answering a question this assistant itself just asked while an order was being collected (see any pending-order / partial-order context below, if present) -- those must go to the matching business tool, not converse, no matter how short the reply looks.
- answering would require inventing or implying any catalogue, pricing, stock, delivery, or order information converse itself has no access to. If in doubt whether something is a real business question, prefer the business tool over converse.
- the customer asks what a specific past/confirmed order actually contains or was placed at ("did you change the karat", "what karat did I order", "was that 14k or 18k") -- even if this conversation's own earlier turns seem to already answer it. That is a fact about a real order, not about what was said in the chat; use get_order_status, never converse, and never your own reconstruction of the conversation so far.

Exception: if the last-action-outcome context further below tells you to use converse to explain a recent failure (a customer asking "why?" or reacting to something that just went wrong), do that -- that instruction overrides the business-question rules above for that one reply, since explaining what already happened is not the same as answering a new business question.

Examples:

Customer: "hey"
{{"tool": "converse", "arguments": {{"reply": "Hey! 👋 How can I help you today?"}}}}

Customer: "how are you?"
{{"tool": "converse", "arguments": {{"reply": "I'm doing great, thanks for asking! How about you?"}}}}

Customer: "medaase"
{{"tool": "converse", "arguments": {{"reply": "You're very welcome!"}}}}

Customer: "ei that's expensive oo"
{{"tool": "converse", "arguments": {{"reply": "Haha I hear you! Want me to show you a few more affordable options?"}}}}

Customer: "urm what do you do?"
{{"tool": "converse", "arguments": {{"reply": "I help with pretty much everything here 😊 Ask me about our jewellery, prices, photos, delivery, or place an order any time."}}}}

Customer: "how much is the gold chain?"
-> NOT converse. Use get_product_price.

Customer: "what karat does that come in" (just quoted the price of the Custom Leaf White Gold Necklace, 20g)
{{"tool": "get_product_karat_options", "arguments": {{"product_name": "Custom Leaf White Gold Necklace, 20g"}}}}
-> Asking what the options ARE, not pricing one. Use get_product_karat_options, not get_product_price, even though a product and a price were just discussed.

Customer: "okay what about 12" (right after being quoted a price, or after being shown karat options)
{{"tool": "get_product_price", "arguments": {{"product_name": "Custom Leaf White Gold Necklace, 20g", "material": "12k"}}}}
-> A specific karat named -- this is pricing one option, not asking what the options are. get_product_price, not get_product_karat_options.

Customer: "how heavy is it" (just quoted the price of the Custom Leaf White Gold Necklace, 20g)
{{"tool": "get_product_weight", "arguments": {{"product_name": "Custom Leaf White Gold Necklace, 20g"}}}}
-> A weight question, not a price question -- get_product_weight, not get_product_price, even though a price was just discussed.

Customer: "is that really 1g?" (this assistant already told them the Set Multi Stone Golf Ring, 7g weighs 7g)
{{"tool": "get_product_weight", "arguments": {{"product_name": "Set Multi Stone Golf Ring, 7g"}}}}
-> Still a weight question, even phrased as disbelief and even though a different number ("1g") was thrown in -- do not answer from what was already said in this conversation (right or wrong), and do not use converse to reassure them. Call the tool again and let the real catalogue figure answer it.

Customer: "how many grams does the second one have?" (just shown a numbered list: 1. Ring A, 2. Necklace B, 3. Ring C)
{{"tool": "get_product_weight", "arguments": {{"product_name": "Necklace B"}}}}
-> Two instructions compose here, not one: resolve "the second one" against the numbered list above first (see that context below), then apply the weight question to whatever product that position resolves to.

Customer: "do you have bracelets?"
-> NOT converse. Use recommend_products.

Customer: "this Set Multi Stone Golf Ring, 7g" (just naming a product, no question)
-> NOT converse. Use get_product_price with product_name "Set Multi Stone Golf Ring, 7g".

Customer: "yeah i wanna see pictures" (nothing specific named yet)
-> NOT converse. Use get_product_price with product_name "unknown" -- let the system's own missing-product handling ask which one, don't improvise that question yourself.

Customer: "necklace images" (a category, not a specific product)
-> NOT converse, and NOT get_product_price. Use recommend_products with category "Necklaces" -- a category is enough to act on immediately.

Customer: "why?" (right after being told an order/action failed)
-> converse, using the last-action-outcome context below to explain the real reason -- do not say you haven't seen anything, and do not guess a different reason.

Customer: "cancel my order" (no number given)
{{"tool": "cancel_order", "arguments": {{"order_id": "unknown"}}}}

Customer: "please cancel order 6846"
{{"tool": "cancel_order", "arguments": {{"order_id": "6846"}}}}

Customer: "where is my order"
{{"tool": "get_order_status", "arguments": {{"order_id": "unknown"}}}}

Customer: "what's the status of order 6846"
{{"tool": "get_order_status", "arguments": {{"order_id": "6846"}}}}

Customer: "did you change the karat to 18k in that order?" (an order was confirmed earlier this conversation; the customer asked about 18k in a later turn, but was never told it was applied)
{{"tool": "get_order_status", "arguments": {{"order_id": "unknown"}}}}
-> NOT converse, even though the conversation looks like it already answers this. What the order was actually placed with is a fact about a real WooCommerce order, not about what was said in the chat -- get_order_status is the only way to check it. Answering from this conversation's earlier turns risks asserting something confidently that was never actually true.

Customer: "I already told you Kumasi, why do you keep asking" (delivery_address in this order is already "Kumasi" -- nothing about it is actually wrong)
{{"tool": "propose_order", "arguments": {{"product_name": "...", "material": "...", "quantity": "...", "delivery_address": "Kumasi", "delivery_option": "kumasi_rider"}}, "customer_tone": "pushback"}}
-> Pushback: a complaint, with no new value in it at all. Tag customer_tone.

Customer: "no, I said 14k" (material was "18k")
{{"tool": "propose_order", "arguments": {{"product_name": "...", "material": "14k", "quantity": "...", "delivery_address": "...", "delivery_option": "..."}}, "customer_tone": "pushback"}}
-> ALSO pushback, even though this one also states a new value -- the two are not mutually exclusive. "No, I said X" carries the same complaint ("you already had this wrong") as the Kumasi example, it just happens to restate the value alongside it. Tag customer_tone here too; propose_order still gets the corrected material either way.

Customer: "actually make it 14k" (material was "18k")
{{"tool": "propose_order", "arguments": {{"product_name": "...", "material": "14k", "quantity": "...", "delivery_address": "...", "delivery_option": "..."}}}}
-> NOT pushback -- an ordinary correction with no complaint in it. "Actually" signals a change of mind, not a dispute about something the system got wrong. No customer_tone field.

Customer: "still 14k and 6 pieces" (material and quantity already "14k"/6 -- reaffirming, not disputing)
{{"tool": "propose_order", "arguments": {{"product_name": "...", "material": "14k", "quantity": 6, "delivery_address": "...", "delivery_option": "..."}}}}
-> NOT pushback -- nothing is being disputed or complained about. No customer_tone field.

Rules:

- If the customer asks only for a price, use get_product_price.
- If the customer asks to see a photo/picture of one specific, named product, use get_product_price (its result carries the image) -- even if they never mention price and even if that product was only just recommended to them a moment ago. Do not use recommend_products for this; that returns a browse-style list, not one item's photo.
- If the customer asks for delivery/shipping info only, use get_delivery_information.
- If the customer wants a full quote (price + delivery), use generate_quote.
- If the customer is browsing by product type and/or karat rather than asking about one item, use recommend_products.
- If the customer is asking what karats/purities a specific, already-identified product comes in, use get_product_karat_options. If they instead name one specific karat to price, that's get_product_price with that karat, even for the same product.
- If the customer is asking how heavy a specific, already-identified product is, use get_product_weight -- a separate question from price or karat.
- If the customer is asking about returns, warranty, sizing, care, engraving, or payment methods, use answer_policy_question.
- If the customer wants to actually place an order and has given enough detail, use propose_order.
- If the customer is confirming an order proposed earlier in this conversation, use confirm_order.
- If the customer wants to cancel an order they already placed, use cancel_order. If they gave an order number, pass it; otherwise set order_id to "unknown" and let the system find their most recent order.
- If the customer wants to know an order's status ("where is my order", "has it shipped", "is my order confirmed") OR what a past/confirmed order actually contains ("did you change the karat", "what did I order", "was that 14k or 18k"), use get_order_status. Same order_id rule as cancel_order: pass it if given, otherwise "unknown". Never answer either kind of question from this conversation's own earlier turns alone.
- If the customer's message is purely social (a greeting, thanks, a reaction, small talk) with no business question in it, use converse. If there's ANY pending order or order-in-progress context below and the message plausibly answers it (a bare number, an address, a delivery choice), that takes priority over converse -- continue the order instead, exactly as instructed in that context.
- If the customer mentions "this", "that one", or similar references, infer the product, material, or category from earlier in THIS message if possible.
- If a tool needs product_name, material, or category and you genuinely cannot determine it from this message alone, set that argument to the literal string "unknown" rather than guessing. The system remembers what the customer discussed earlier in the conversation and will fill "unknown" in for you -- inventing a value yourself would override that and risk quoting the wrong product.
- If the customer refers to a photo/image they already sent earlier in THIS conversation ("the image I sent", "that photo", "order the one I sent a picture of") to identify which product they mean, do not say you can't view images -- a photo sent earlier in this conversation may already have been matched to a specific catalogue item and remembered. Treat it exactly like naming that product: use whichever tool their request actually needs (get_product_price, propose_order, generate_quote, ...) with product_name "unknown", and let the system's own memory resolve it. Only fall back to converse and explain you can't view images if the customer is sending or describing an image right now, in this message, with no earlier photo anywhere in the conversation to refer back to.
- If the customer is disagreeing or pushing back on something ("that's wrong", "no, that's not right", "Cape Coast is in Ghana", "that's not what I said") rather than stating a new preference of their own, do not just repeat the same claim or brush past it -- but also do not simply agree with them either, you have no way to see what you said a moment ago or to verify their claim yourself. Work out what the pushback is actually ABOUT before picking a tool: a dispute about store policy (returns, warranty, sizing, engraving, payment terms) is the only case that means answer_policy_question -- do not send it there just because the message contains a "why" or sounds like a complaint. A dispute about THIS order or THIS product -- a price, a karat, a quantity, an address, a delivery arrangement, or a decision the system itself made while building the order ("I didn't choose the karat so why did you choose 18k for me?", "that's not the address I gave you", "I never said 5") -- means propose_order/get_product_price/generate_quote, whichever this conversation's order or product question actually needs, using their current message as the input. Confirmed live, 2026-08-20: "I didn't choose the karat so why did you choose 18k for me?" was sent to answer_policy_question and came back with an unrelated warranty answer -- that message was disputing an order decision, not asking about policy, and belonged with propose_order instead. Let the matching tool's real, grounded answer -- not your own agreement, apology, or a policy lookup that has nothing to do with what they actually said -- decide what's true. If a customer states a specific replacement value while disputing something ("no, I said 14k"), that's an ordinary correction as far as which TOOL to call and what arguments to pass (see the order-draft correction rule above) -- this rule doesn't change that part. It can still be worth tagging as pushback for tone purposes, though; see the customer_tone instruction below, which is a separate question from tool selection.
- When you route a message because of the pushback rule immediately above -- the customer is disputing a decision or a past statement, with some real complaint or frustration in how they said it -- also include a top-level "customer_tone": "pushback" field in your JSON response, alongside "tool" and "arguments", not inside "arguments". This exists only so the system can acknowledge that complaint warmly, whatever the tool call itself ends up doing. Pushback and an ordinary correction are NOT mutually exclusive: "no, I said 14k" is both a complaint (you already had this wrong) AND a new value (14k) in the same breath -- tag customer_tone here, the same as a pure complaint with no value in it at all ("I already told you Kumasi, why do you keep asking"). What actually distinguishes pushback from an ordinary correction is tone, not whether a value is present: "actually make it 14k" or "sorry, 14k rather" carry no complaint, just a change of mind -- do NOT tag those, or a neutral reaffirmation of something already true ("still 14k and 6 pieces"), or routine questions, browsing, or small talk. The only valid value is the literal string "pushback" -- never a price, address, order detail, or anything else, and never guess at one when you're not sure real frustration is present; when in doubt, omit the field.

Multiple requests in one message:

A single customer message can genuinely contain more than one distinct
ask -- two different products ("how much is a gold ring and a silver
chain"), the same product in more than one karat ("do you have this
ring in 14k and 18k"), or a product question plus an unrelated policy
question ("how much is a gold chain, and what's your returns policy").
When that happens, do NOT pick only one and drop the rest -- return
EVERY distinct ask as its own entry in a "requests" list instead of a
single "tool"/"arguments" pair:

{{
  "requests": [
    {{"tool": "...", "arguments": {{...}}}},
    {{"tool": "...", "arguments": {{...}}}}
  ]
}}

Only do this when the message actually contains more than one distinct
ask. A single product with a single price question is still just one
request in the normal shape -- don't split something that doesn't need
splitting.

Return ONLY valid JSON. Do not include markdown formatting or commentary.

Example:

{{
  "tool": "get_product_price",
  "arguments": {{
    "product_name": "ring",
    "material": "gold"
  }}
}}

Example (Twi transcript, translated into English before being returned):

Customer said: "sika kyɛn no bo yɛ sɛn" ("how much is the gold chain")

{{
  "tool": "get_product_price",
  "arguments": {{
    "product_name": "chain",
    "material": "gold"
  }}
}}

Example (message contains two distinct asks):

Customer said: "how much is a gold ring and a silver chain"

{{
  "requests": [
    {{
      "tool": "get_product_price",
      "arguments": {{
        "product_name": "ring",
        "material": "gold"
      }}
    }},
    {{
      "tool": "get_product_price",
      "arguments": {{
        "product_name": "chain",
        "material": "silver"
      }}
    }}
  ]
}}

{pending_order_state}

{order_draft_state}

{pending_intent_state}

{last_action_outcome_state}

{last_priced_product_state}

{just_confirmed_order_state}

{last_presented_products_state}

Customer:
{message}
"""


def _pending_order_state_line(pending_order: dict | None, awaiting_confirmation: bool = False) -> str:
    """Describes whether this session currently has an order awaiting
    confirmation, for the confirm_order guidance above.

    Exists because understand_customer() otherwise only ever sees the
    customer's current message in isolation -- no conversation history,
    no memory of what this assistant itself last asked. A bare "yh"/
    "yeah" is genuinely ambiguous with zero context: it could confirm a
    real pending order, or it could just as easily be agreeing to "want
    me to show you a photo?" from a browsing reply that never proposed
    an order at all. Without this line the model has no way to tell
    those apart and would sometimes guess confirm_order regardless of
    whether anything was actually pending (confirmed live -- see
    services/order_tool.py's confirm_order() decline message, which
    exists specifically to handle that guess coming back wrong).

    awaiting_confirmation narrows this further: a pending order can sit
    unconfirmed for the rest of the session's 30-minute TTL, and in that
    time the assistant can go on to offer or ask something completely
    unrelated ("want to see a few cheaper options?"). A bare "yeah"
    answering THAT reads exactly like a bare "yeah" confirming the stale
    order -- this is a commerce-integrity risk (an order gets placed the
    customer never meant to place), not just a UX rough edge, so it is
    not left to the model's judgement alone: router.py computes this
    deterministically (True only when proposing this exact order was the
    last thing that happened in the session) and hands it over as a
    fact, not an inference."""
    if pending_order and awaiting_confirmation:
        return (
            f"This customer currently has a pending order awaiting confirmation, proposed just "
            f"now with nothing else asked or offered since: "
            f"{pending_order['quantity']} x {pending_order['material']} {pending_order['product']}, "
            f"total GH₵{pending_order['total']:,.2f}. If their message clearly confirms this "
            f"(\"yes\", \"confirm\", \"go ahead\", \"yh\", \"ok place it\"), use confirm_order. "
            f"If their message is NOT a confirmation, but instead states new order details of its "
            f"own -- a product, quantity, address, or delivery choice, especially one that names a "
            f"DIFFERENT product than the one above -- treat it as a completely fresh propose_order "
            f"request, not an update to the order above: extract every detail explicitly stated in "
            f"THIS message exactly as given, even where it differs from the pending order shown "
            f"here. Do not leave a field \"unknown\" just because you're unsure whether this is the "
            f"same order continuing -- only use \"unknown\" for a field this message genuinely "
            f"doesn't state at all. Confirmed live, 2026-08-20: a customer with an unconfirmed "
            f"Tamale order still pending then gave a complete new order (different product, 14k, "
            f"deliver to Accra, rider delivery within Accra) -- the resulting proposal wrongly kept "
            f"the old product and the old Tamale address, because part of the new message's own "
            f"detail was treated as unknown instead of being read from the message actually sent. "
            f"The reverse mistake is just as real and must be avoided just as strictly: for any "
            f"field this message does NOT explicitly state, the correct value is \"unknown\", full "
            f"stop -- never invent or infer a value for it, even one that seems like a reasonable "
            f"guess. \"Treat it as a completely fresh request\" above means fresh in the sense of "
            f"not being pre-filled from the old proposal, not an instruction to reproduce the old "
            f"proposal's other fields from memory or from a plausible-sounding guess. Example: a "
            f"pending order awaiting confirmation, then \"actually deliver to Kumasi instead\" -- "
            f"correctly update delivery_option to \"kumasi_rider\", and set delivery_address to "
            f"\"unknown\" (this message names a delivery arrangement, not a place), never a "
            f"remembered or guessed address of your own. The system resolves delivery_address from "
            f"session memory correctly on its own once you return \"unknown\" -- guessing at a "
            f"value that happens to look plausible is exactly what produces a proposal that "
            f"contradicts itself."
        )
    if pending_order and not awaiting_confirmation:
        return (
            f"This customer has an EARLIER order still sitting unconfirmed: "
            f"{pending_order['quantity']} x {pending_order['material']} {pending_order['product']}, "
            f"total GH₵{pending_order['total']:,.2f}. Since it was proposed, you have asked or offered "
            f"them something else in this conversation, so a bare agreement right now (\"yes\", "
            f"\"yeah\", \"ok\", \"sure\") most likely answers THAT more recent thing, not this order -- "
            f"do NOT use confirm_order for a bare agreement alone. Only use confirm_order if the "
            f"message unambiguously names the order itself (\"yes place the order\", \"confirm my "
            f"order\", \"go ahead and order it\", \"place it\"). This is a deliberate safeguard: a "
            f"stale pending order must never get confirmed by a customer agreeing to something "
            f"unrelated, because that places a real order they never meant to place."
        )
    return (
        "This customer does NOT currently have any pending order awaiting confirmation. "
        "Do not use confirm_order for this message, even if it sounds like a bare "
        "agreement (\"yh\", \"yeah\", \"ok\", \"sure\") -- there is nothing for them to be "
        "confirming right now, so treat a bare agreement as being about whatever was "
        "actually offered (seeing a photo, narrowing down a choice, or similar), not "
        "as placing an order."
    )


_ORDER_DRAFT_LABELS = {
    "product_name": "product",
    "material": "material/karat",
    "quantity": "quantity",
    "delivery_address": "delivery address",
    "delivery_option": "delivery option (accra_rider / kumasi_rider / international)",
}


def _order_draft_state_line(order_draft: dict | None) -> str:
    """Describes an order already in progress for this session, so a
    short reply like a bare number or a bare address can be recognised
    as continuing it.

    Exists for the same reason as _pending_order_state_line() above, one
    step earlier in the flow: once propose_order has asked "How many
    would you like?", the customer's next message is often just "2" --
    with no conversation history, the model has no way to know that's
    an answer to a question it can't see, and would sometimes route it
    to a completely different tool instead of continuing the order
    (confirmed live, 2026-08-12). get_order_draft() in memory.py builds
    this from whatever's already been remembered."""
    if not order_draft:
        return ""

    known = [f"{_ORDER_DRAFT_LABELS[key]}={value}" for key, value in order_draft.items() if value]
    missing = [_ORDER_DRAFT_LABELS[key] for key, value in order_draft.items() if not value]
    if not missing:
        # Everything's already known -- nothing left for a short reply
        # to be answering, so there's nothing useful to say here.
        return ""

    known_text = ", ".join(known) if known else "nothing yet"
    missing_text = ", ".join(missing)
    return (
        f"This customer already has an order in progress. Known so far: {known_text}. "
        f"Still missing: {missing_text}. If their current message is short and only makes "
        f"sense as answering one of the missing pieces, treat it as continuing this order: "
        f"use propose_order, fill in the new piece from their message, and keep every OTHER "
        f"already-known value exactly as given above. "
        f"A bare number on its own, WITH or WITHOUT a trailing \"k\" (\"12\", \"18\", \"12k\", "
        f"\"18k\") almost always means the karat if material is still missing -- e.g. \"12\" or "
        f"\"12k\" -> material \"12k\" -- NOT the quantity, and NOT a request to browse other "
        f"products (do not use recommend_products here just because the reply only contains a "
        f"karat -- this customer is answering a specific question this conversation already "
        f"asked, not browsing). Only read a bare number as quantity if material is already known, "
        f"or if the message clearly states a separate count (\"2 please\", \"I want 3\", "
        f"\"12k, 2 of them\"). Never use the "
        f"same number from the message to fill two different fields -- if a digit already "
        f"answered the karat, do not also copy that same digit into quantity unless the "
        f"customer separately gave a count; leave quantity as \"unknown\" (never assume 1) "
        f"if they didn't. A bare address, or \"Accra\"/\"Kumasi\"/\"ship it\"/\"outside "
        f"Ghana\", answers delivery option. This still applies when the message is longer "
        f"and pushes back or complains, as long as it also names or confirms a place or "
        f"zone -- \"nima is obviously in Accra\" or \"it's in Accra, why would you ask\" "
        f"still answers delivery option (call propose_order with delivery_address \"Accra\" "
        f"or the place actually named), it is not a generic delivery question for "
        f"get_delivery_information. Confirmed live, 2026-08-24: a message exactly this shape "
        f"was sent to get_delivery_information instead, which broke out of the order and "
        f"needed an extra turn to get back to the proposal that was already one field away "
        f"from complete. "
        f"Exception: if their current message clearly states a different value for one of "
        f"the already-known fields instead -- a correction, e.g. \"sorry, Kumasi not Accra\", "
        f"\"wait, 14k rather\", \"actually make that 5\", \"I'll take the white gold instead\", "
        f"\"use my other address\" -- use their new value for that field, not the old one; do "
        f"not silently keep a value they just corrected. If the SAME message states two values "
        f"for the same field in sequence (\"7 pieces, actually make that 5\", \"14k, no wait "
        f"18k\"), only the final one is their answer -- resolve it yourself before returning "
        f"your arguments; never return the earlier, superseded value. Don't ask them to repeat "
        f"anything they haven't changed, and don't restart with a different tool."
    )


_PENDING_INTENT_TOOL_DESCRIPTIONS = {
    "get_product_price": "see its price and photo",
    "generate_quote": "get a full quote (price and delivery options)",
}


def _pending_intent_state_line(pending_intent: str | None) -> str:
    """Describes an unresolved product-lookup attempt from a previous
    turn, so a customer who then names/identifies the product on their
    very next message gets it resolved immediately instead of being
    asked yet another clarifying question.

    Exists because a message like "yeah i wanna see pictures" or "how
    much is it" with no product named is a genuine business ask, just
    missing one detail -- get_product_price runs with product_name
    "unknown", finds nothing, and this session is left with an
    unresolved intent. Without this line, the customer's next message
    naming the product (e.g. "this Set Multi Stone Golf Ring, 7g") looks,
    in isolation, like they're just repeating a name with no clear ask --
    and converse (see llm.py's tool 9) or another clarifying question can
    swallow it instead of actually completing the lookup the customer
    already asked for (confirmed live, 2026-08-13: the customer had to
    name the product twice, then got told "I couldn't find that one",
    then had to explain that it had literally just been shown to them).
    router.py's set_pending_intent()/get_pending_intent() track this."""
    if not pending_intent:
        return ""
    action = _PENDING_INTENT_TOOL_DESCRIPTIONS.get(pending_intent, "look something up")
    return (
        f"This customer recently tried to {action} but hadn't named a specific product, "
        f"so nothing could be found yet. If their current message now names, repeats, or "
        f"otherwise identifies a specific product -- even just the name on its own, with no "
        f"question -- treat that as completing this request: call {pending_intent} with that "
        f"product name right away. Do not ask them what they'd like to do with it and do not "
        f"use converse; they already told you what they wanted, they're now just telling you "
        f"which item."
    )


def _last_action_outcome_state_line(last_action_outcome: dict | None) -> str:
    """Describes a genuine, unrecoverable failure of a real business
    action the customer just tried -- not a "still missing a detail"
    prompt (those are self-explanatory and covered by
    _order_draft_state_line() above), but something that failed for a
    real reason with nothing the customer could have done differently.

    Exists because a customer's very next message after a failure like
    that is very often "why?", or confusion/frustration expressed some
    other way -- and without this, understand_customer() has no idea a
    failure even happened a moment ago (confirmed live, 2026-08-13: "why?"
    got "could you clarify what you mean", and the very next message got
    told "I haven't seen any order from you yet" -- an outright false
    claim, since an order attempt had just failed, not never happened).
    See memory.set_last_action_outcome()/get_last_action_outcome()."""
    if not last_action_outcome:
        return ""
    explanation = last_action_outcome.get("customer_safe_explanation")
    if not explanation:
        return ""
    return (
        f"The last thing this customer tried just failed, for a real reason: {explanation} "
        f"If their current message is asking why, expressing confusion, or reacting to that "
        f"failure in any way (\"why?\", \"why not?\", \"oh wow\", \"sigh\", or similar), use "
        f"converse and explain using that exact reason. Do not claim nothing happened, do not "
        f"say you haven't seen anything from them, and do not invent a different reason. If "
        f"their current message is clearly about something else entirely, ignore this and "
        f"proceed normally."
    )


def _last_priced_product_state_line(last_priced_product: str | None) -> str:
    """Describes the specific product a get_product_price/generate_quote
    call most recently resolved to, so a bare karat-only follow-up
    ("what about in 18k") re-quotes the SAME product instead of falling
    through to a category browse.

    Exists because a message with a karat but no product name of its
    own previously had no signal telling the model a specific product
    was still the active topic -- confirmed live, 2026-08-18: after
    being quoted the Big White Crown Stone Gold Ring at 14k, "what about
    in 18k" returned four unrelated products instead of that same ring's
    18k price. recommend_products's own canonical usage example
    ("necklaces in 18k") is itself a bare-karat phrase, which pulled
    ambiguous follow-ups like this toward a browse instead of a
    continuation. See memory.set_last_priced_product()."""
    if not last_priced_product:
        return ""
    return (
        f"The last specific product this customer asked about was "
        f"\"{last_priced_product}\". If their current message states only a "
        f"material/karat with no product name of its own, and does not read as a "
        f"fresh browse request for a whole category (\"what rings do you have\", "
        f"\"show me your necklaces\"), treat it as asking about "
        f"\"{last_priced_product}\" at that karat -- call get_product_price with "
        f"product_name \"{last_priced_product}\" and the newly stated material, "
        f"rather than recommend_products."
    )


def _just_confirmed_order_state_line(just_confirmed_order: dict | None) -> str:
    """Describes an order that was confirmed as the LAST thing that
    happened in this session, nothing since.

    Exists because confirm_order clears the pending-order state and
    clears the remembered product/material/quantity/address/delivery
    fields the moment an order is placed (see memory.clear_order_state()
    -- 2026-08-20 architecture audit, failure #3), but the model still
    has no way to know an order was JUST successfully placed rather than
    never having existed at all. Without this, a genuinely unrelated
    next message risks being read against nothing, or a customer asking
    "what's my order number?" right after confirming gets treated like a
    fresh, contextless question instead of an obvious follow-up. Only
    ever non-empty for the one turn immediately after confirm_order
    succeeded -- see memory.set_just_confirmed_order()'s docstring for
    why this doesn't linger."""
    if not just_confirmed_order:
        return ""
    order_id = just_confirmed_order.get("order_id")
    total = just_confirmed_order.get("total")
    total_text = f", total GH₵{total:,.2f}" if isinstance(total, (int, float)) else ""
    return (
        f"This customer's order (#{order_id}{total_text}) was JUST confirmed and placed -- "
        f"this is now a completed action, not something still in progress. If their current "
        f"message asks about this order's number or total, answer using this information "
        f"directly via converse -- that's genuinely all that's known here. If they ask about "
        f"its karat, material, product, or quantity, or whether something in it was changed, "
        f"do NOT answer from this information or from earlier turns in this conversation -- "
        f"this line deliberately doesn't carry that, use get_order_status instead (order_id "
        f"\"unknown\" resolves to this same order). If their current message describes "
        f"something new and unrelated -- a different product, a fresh question -- treat it as "
        f"exactly that: a completely new, independent request. It genuinely has no "
        f"product/karat/quantity/address to inherit from the order that was just placed, so "
        f"use \"unknown\" for whatever this new message doesn't itself state, same as any "
        f"other fresh request."
    )


def _last_presented_products_state_line(last_presented_products: dict | None) -> str:
    """Describes the exact numbered list a recommend_products reply most
    recently showed this customer, so an ordinal/positional reference
    ("show me the second one", "I want the first ring", "number 3") can
    resolve to the actual product_name at that position instead of the
    model guessing or falling back to whatever it remembers as the
    single active product.

    Exists because the customer-visible list is grouped/diversity-
    selected (see response_formatter.select_presented_groups() and
    memory.set_last_presented_products()'s docstrings), so its order
    is not the same as data/products.json's raw file order, and nothing
    before this told the model what that rendered order actually was --
    confirmed against the Webb/GPT 50-turn live test, 2026-08-24: "show
    me the second one" (after 4 necklaces were listed) and "I want the
    first ring" (after 4 rings were listed) both resolved to an
    unrelated, earlier-discussed product instead of the actual 2nd/1st
    item shown.

    `category` is carried per entry specifically so "the first ring" can
    be resolved against a category-filtered position within a MIXED
    list (rings and necklaces together), not just raw list position 1
    -- see the instruction text below.

    Only ever describes a genuinely ORDINAL/positional reference to an
    item in this specific list -- it does not change how "this"/"that
    one"/"the same one" are resolved (see the existing "this"/"that one"
    instruction elsewhere in this prompt, and llm.py's
    _last_priced_product_state_line() -- both already handle a single
    active product with no list involved)."""
    if not last_presented_products:
        return ""
    items = last_presented_products.get("items") or []
    if not items:
        return ""
    lines = "\n".join(
        f"{item['position']}. {item['product_name']}" + (f" ({item['category']})" if item.get("category") else "")
        for item in items
    )
    return (
        f"This customer was just shown this numbered list:\n{lines}\n\n"
        f"If their current message refers to one of these items by position -- an ordinal "
        f"(\"the second one\", \"the first ring\", \"the last one\") or a number (\"number 3\", "
        f"just \"2\") -- resolve product_name to that exact entry's name above, rather than "
        f"treating the message as still-ambiguous or reusing a different, earlier-discussed "
        f"product. When the reference names a category (\"the first RING\") and this list mixes "
        f"categories, match position within that category only (the first entry above whose "
        f"category is Rings), not raw position in the whole list. If the position referenced "
        f"doesn't exist in this list (e.g. \"the fifth one\" when only {len(items)} were shown), "
        f"do not guess -- set product_name to \"unknown\" as usual. This only resolves WHICH "
        f"product a position means; it does not by itself decide which tool to call -- use "
        f"whatever the rest of the message asks for (price, karat options, weight, an order, a "
        f"photo, ...) with this resolved product_name."
    )


def _build_prompt(
    message: str,
    pending_order: dict | None,
    order_draft: dict | None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
    last_priced_product: str | None = None,
    awaiting_confirmation: bool = False,
    just_confirmed_order: dict | None = None,
    last_presented_products: dict | None = None,
) -> str:
    return _PROMPT_TEMPLATE.format(
        message=message,
        pending_order_state=_pending_order_state_line(pending_order, awaiting_confirmation),
        order_draft_state=_order_draft_state_line(order_draft),
        pending_intent_state=_pending_intent_state_line(pending_intent),
        last_action_outcome_state=_last_action_outcome_state_line(last_action_outcome),
        last_priced_product_state=_last_priced_product_state_line(last_priced_product),
        just_confirmed_order_state=_just_confirmed_order_state_line(just_confirmed_order),
        last_presented_products_state=_last_presented_products_state_line(last_presented_products),
    )


def _call_llm(
    message: str,
    pending_order: dict | None,
    order_draft: dict | None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
    last_priced_product: str | None = None,
    awaiting_confirmation: bool = False,
    just_confirmed_order: dict | None = None,
    last_presented_products: dict | None = None,
) -> str:
    """Call the model with retries on transient failures only.

    Auth errors, bad requests, etc. (APIError) are not retried — retrying
    those just burns time and money for a call that will never succeed.
    """
    last_error: Exception | None = None
    total_attempts = settings.llm_max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=_build_prompt(
                    message, pending_order, order_draft, pending_intent, last_action_outcome,
                    last_priced_product, awaiting_confirmation, just_confirmed_order,
                    last_presented_products,
                ),
                timeout=settings.llm_timeout_seconds,
            )
            return response.output_text

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning(
                "LLM call failed (attempt %s/%s): %s", attempt, total_attempts, e
            )
            if attempt < total_attempts:
                time.sleep(min(2**attempt, 8))  # 2s, 4s, 8s...

        except APIError as e:
            logger.error("Non-retryable LLM API error: %s", e)
            raise ToolSelectionError(f"LLM request failed: {e}") from e

    raise ToolSelectionError(f"LLM unreachable after {total_attempts} attempts: {last_error}")


def _parse_tool_request(raw_text: str) -> dict:
    cleaned = raw_text.strip()

    # Defensive: models occasionally wrap JSON in markdown fences even
    # when told not to. Strip them rather than crashing on it.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("LLM output was not valid JSON: %s | raw=%r", e, raw_text)
        raise ToolSelectionError("LLM did not return valid JSON") from e

    if "requests" in data:
        return _validate_multi_request(data)

    if "tool" not in data or "arguments" not in data:
        logger.error("LLM output missing required keys: %s", data)
        raise ToolSelectionError(f"LLM response missing 'tool' or 'arguments': {data}")

    return data


def _validate_multi_request(data: dict) -> dict:
    requests = data["requests"]

    if not isinstance(requests, list) or not requests:
        logger.error("LLM 'requests' was not a non-empty list: %s", data)
        raise ToolSelectionError(f"LLM response's 'requests' must be a non-empty list: {data}")

    for entry in requests:
        if not isinstance(entry, dict) or "tool" not in entry or "arguments" not in entry:
            logger.error("LLM 'requests' entry missing 'tool' or 'arguments': %s", entry)
            raise ToolSelectionError(f"LLM 'requests' entry missing 'tool' or 'arguments': {entry}")

    if len(requests) > MAX_REQUESTS_PER_MESSAGE:
        logger.warning(
            "LLM returned %d requests for one message -- truncating to %d",
            len(requests), MAX_REQUESTS_PER_MESSAGE,
        )
        requests = requests[:MAX_REQUESTS_PER_MESSAGE]

    return {"requests": requests}


def understand_customer(
    message: str,
    pending_order: dict | None = None,
    awaiting_confirmation: bool = False,
    order_draft: dict | None = None,
    pending_intent: str | None = None,
    last_action_outcome: dict | None = None,
    last_priced_product: str | None = None,
    just_confirmed_order: dict | None = None,
    last_presented_products: dict | None = None,
) -> dict:
    """pending_order, when provided, is router.py's read of this
    session's pending proposal (see order_tool.get_pending_order_summary)
    -- see _pending_order_state_line()'s docstring for why this needs to
    be passed in explicitly rather than left for the model to infer.

    awaiting_confirmation is only meaningful when pending_order is set --
    True only when proposing that exact order was the last thing that
    happened in this session, False once anything else has been asked or
    offered since (see memory.set_awaiting_confirmation()). Governs
    whether a bare "yes"/"yeah" is safe to read as confirming it.

    order_draft is the equivalent one step earlier -- an order that's
    been started but not yet fully priced (see memory.get_order_draft()
    and _order_draft_state_line()).

    pending_intent is one step earlier again -- a product lookup the
    customer asked for but hadn't named a product for yet (see
    memory.get_pending_intent() and _pending_intent_state_line()).

    last_action_outcome is a different axis entirely -- not "still
    missing something", but a real business action that was already
    fully specified and still failed (see memory.get_last_action_outcome()
    and _last_action_outcome_state_line()).

    last_priced_product is the specific product a recent
    get_product_price/generate_quote call resolved to, so a bare
    karat-only follow-up re-quotes that same product instead of falling
    through to a category browse (see memory.get_last_priced_product()
    and _last_priced_product_state_line()).

    just_confirmed_order is non-None for exactly one turn: the one right
    after confirm_order succeeded, nothing since (see
    memory.get_just_confirmed_order() and
    _just_confirmed_order_state_line()).

    last_presented_products is the numbered list of products this session
    was last shown by recommend_products (see
    memory.get_last_presented_products() and
    _last_presented_products_state_line()), so an ordinal/positional
    follow-up ("the second one", "the ring in the middle") can be resolved
    to a specific product_name before the model picks whichever tool the
    rest of the message actually asks for."""
    if not message or not message.strip():
        raise ValueError("message must not be empty")

    raw_text = _call_llm(
        message, pending_order, order_draft, pending_intent, last_action_outcome,
        last_priced_product, awaiting_confirmation, just_confirmed_order,
        last_presented_products,
    )
    logger.info("Raw LLM output: %s", raw_text)

    return _parse_tool_request(raw_text)
