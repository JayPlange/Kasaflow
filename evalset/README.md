# Conversation eval set

Real conversations, anonymised, replayed through the current code and
scored on three things: **order accuracy**, **cost per conversation** and
**latency**. Guardrail events (handoffs, blocked writes, price changes) are
recorded alongside.

## What's in here

| File | What it is |
|---|---|
| `conversations.jsonl` | One anonymised conversation per line: each customer message, plus what the model chose and what the assistant did at the time |
| `labels.json` | Ground truth per conversation: the order a correct assistant would have placed (or none), and whether it should have ended with a person |
| `anonymisation_review.md` | What the last anonymisation run redacted, and words left in that a person should check |

**Current contents:** 2 conversations (10 turns) from operator test sessions on
21 August 2026, labelled `source: operator_test`. KasaFlow has not run with real
customers yet, so there are no customer conversations to add. Both labels were
pre-filled from the transcripts and still need review (`"reviewed": false`), so
scoring counts neither until they are confirmed.

## Workflow

```bash
# 1. Capture. The server already logs a KASAFLOW_TURN_TRACE line and a
#    KASAFLOW_TURN_METRICS line per customer message. Save the logs.
docker compose logs kasaflow > pilot-week1.log

# 2. Anonymise into the eval set (keep EVALSET_SALT private and constant).
EVALSET_SALT=... python -m scripts.evalset.anonymise pilot-week1.log --source pilot
#    Then read evalset/anonymisation_review.md and fix anything it lists.

# 3. Label. Adds suggested labels for new conversations, unreviewed.
python -m scripts.evalset.label
#    Read each new conversation, correct its label, set "reviewed": true.

# 4. Replay and score.
python -m scripts.evalset.run --mode recorded          # free, no model calls
python -m scripts.evalset.run --mode live --repeat 3   # real model, costs money

# 5. Dashboard (latest run in eval_reports/ by default).
python -m scripts.evalset.dashboard
```

## The two replay modes

**recorded** gives every turn the model's original answer back. It isolates the
application: a code change that breaks an order flow shows up with no model
variance and no cost. Latency here is application time only. Turns where the
current code wants a model answer the original conversation never needed are
flagged, because the replay can't invent one.

**live** calls the model through the production code path. This is where order
accuracy, cost and latency are actually measured. Use `--repeat` so variance
shows up as disagreement between runs (the "consistency" column) rather than
hiding in one sample.

Both run in a sandbox (`scripts/evalset/sandbox.py`): orders go to an in-memory
store that prices them from the catalogue the way WooCommerce does, staff
WhatsApp messages are captured, geocoding is off and embeddings are off in
recorded mode unless `--allow-embeddings`.

## What each metric means

- **Order accuracy**: reviewed labels only. Correct when no order was expected
  and none was placed, or exactly one order was placed with the labelled product,
  karat and quantity, and the store total matched the quote. Reported with a 95%
  Wilson interval because the set is small.
- **Cost per conversation**: tokens of the tool-selection model call, priced with
  `scripts/evalset/pricing.json`. Check those prices before quoting a figure.
  Embeddings, vision, voice and geocoding are not included.
- **Latency**: per customer message, inside the router, split into model time and
  everything else. Excludes WhatsApp delivery.

## Anonymisation limits

Pattern-based redaction removes phone numbers, emails, Ghana Post GPS addresses,
house and plot numbers, introduced names, session ids (hashed with a salt), and
every address word that isn't a known place name. It will miss a name given
without an introduction, or a lowercase street or landmark inside a message.
That is what `anonymisation_review.md` is for: nothing should be committed until
someone has read it.

## What the seed replay already shows

Replaying the two 21 August conversations against the current code (recorded mode):

- "actually change the karat to 18" failed on 21 August and now produces a correct
  re-quote at 18k, so the karat-correction fix holds with the same model answers.
- "no east legon" still doesn't capture the address. On 21 August the model
  returned `delivery_address: "unknown"` for it, and the address fast path doesn't
  handle a leading "no", so the order never reaches confirmation. Whether the
  current prompt fixes the model side is a live-mode question.
