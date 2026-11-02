# LLM baseline: metric definitions

Defined before the numbers exist, so the numbers cannot shape the definitions.
Code: `services/llm_metrics.py` (per-attempt records), `scripts/bench/` (benchmark).

## Latency and tokens

| Metric | Starts | Ends | Measured by |
|---|---|---|---|
| TTFT | Just before `responses.create()` is called | First `response.output_text.delta` event with non-empty text | streaming probe only |
| First event | Same start | First stream event of any kind (metadata, not text) | streaming probe only |
| Decode time | First non-empty text delta | Last text delta | streaming probe only |
| Tokens/sec | n/a | n/a | `output_tokens / decode seconds`. `_excl_first` uses `output_tokens - 1`, because the first token arrives at TTFT |
| Total LLM latency | Just before the call | Complete response (stream: terminal event) | probe and `llm_metrics` |
| End-to-end latency | `route_customer()` called | `route_customer()` returns | benchmark, from outside `router.py` |
| Input / output tokens | n/a | n/a | `response.usage` |
| Cached input tokens | n/a | n/a | `usage.input_tokens_details.cached_tokens`, the provider's own prompt cache, reported apart from everything else |

TTFT is never the first event. `first_event_ms` is reported next to it so the gap is visible.

## Success is four separate things

1. `api_ok`: the API returned a response (no timeout, connection or API error).
2. `valid_response`: the response had non-empty text.
3. `parse_ok`: the text parsed as a KasaFlow tool request (`_parse_tool_request`). Production-path phase only.
4. `route_ok`: `route_customer()` returned a result with no `error` key (end-to-end phase only). "Nothing found" counts as ok. Whether a WooCommerce call succeeded is not judged here.

A turn that never calls the model has `api_ok = None`, not `False`.

## Concurrency

The ladder is closed-loop: N worker threads, all released together by a barrier, each taking the next request
when its last one finishes. `peak` in the report is the most requests observed in flight at once. If it is
lower than the stated level, the level was not reached.

## Wording and limits

* "First request" and "later requests" describe order only. The provider's servers cannot be made cold or warm from here.
* With fewer than 20 requests per cell, p95 is the maximum. The report marks it `*`.
* Prompts are built with empty session state, so input tokens are at the low end of real sessions.
* The streaming probe and the production path are different calls. Compare them at concurrency 1 only.
* Chunks are not tokens: one stream chunk can carry several. Decode timing has the resolution of the chunks.
* Rows and reports store the scenario id, never the message text. Error text is not stored (the parser's own errors can embed model output), only the exception class name. The tool names the model chose are stored, as categories. `llm_metrics` records hold no prompts, messages or model text, only lengths and counts.
* Benchmark messages are the first turns of the evaluator scenarios. Where scenarios share a first message, it is labelled with the alphabetically first id.
