"""
Tests for services/llm_metrics.py and its use inside llm._call_llm().

No real API calls. The model client is replaced with a fake, as in
test_llm.py. What these pin down:

  * Off by default: no records, no log line, _call_llm behaves as before.
  * The request sent to the API is unchanged (same keyword arguments).
  * One record per attempt, with usage when present and None when not
    (existing tests use responses with no usage attribute).
  * Failures are classified, and the in-flight counter always comes back
    to zero, including on unexpected exceptions.
  * Records never contain the customer's message.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIError, APITimeoutError

from services import llm, llm_metrics
from services.llm import ToolSelectionError

REPLY = '{"tool": "converse", "arguments": {}}'
_real_sleep = time.sleep


def _usage(inp=1000, out=40, cached=800, reasoning=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _response(usage=None, text=REPLY):
    ns = SimpleNamespace(output_text=text)
    if usage is not None:
        ns.usage = usage
    return ns


def _request():
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


@pytest.fixture
def records(monkeypatch):
    monkeypatch.delenv(llm_metrics.ENV_FLAG, raising=False)
    collected = []
    llm_metrics.add_sink(collected.append)
    yield collected
    llm_metrics.remove_sink(collected.append)
    assert llm_metrics._in_flight == 0


def _use_client(monkeypatch, client):
    monkeypatch.setattr(llm, "client", client)


def test_off_by_default_records_nothing(monkeypatch, caplog):
    monkeypatch.delenv(llm_metrics.ENV_FLAG, raising=False)
    client = MagicMock()
    client.responses.create.return_value = _response(_usage())
    _use_client(monkeypatch, client)

    with caplog.at_level("INFO"):
        assert llm._call_llm("hi", None, None) == REPLY

    assert not llm_metrics.is_enabled()
    assert "KASAFLOW_LLM_METRICS" not in caplog.text
    assert llm_metrics._in_flight == 0


def test_request_arguments_are_unchanged(monkeypatch, records):
    client = MagicMock()
    client.responses.create.return_value = _response(_usage())
    _use_client(monkeypatch, client)

    llm._call_llm("hi", None, None)

    kwargs = client.responses.create.call_args.kwargs
    assert set(kwargs) == {"model", "input", "timeout"}
    assert "stream" not in kwargs


def test_success_record_carries_usage(monkeypatch, records):
    client = MagicMock()
    client.responses.create.return_value = _response(_usage(1000, 40, 800, 5))
    _use_client(monkeypatch, client)

    llm._call_llm("hi", None, None)

    assert len(records) == 1
    rec = records[0]
    assert rec["outcome"] == "ok"
    assert (rec["attempt"], rec["max_attempts"]) == (1, llm.settings.llm_max_retries + 1)
    assert rec["input_tokens"] == 1000
    assert rec["output_tokens"] == 40
    assert rec["cached_tokens"] == 800
    assert rec["reasoning_tokens"] == 5
    assert rec["output_chars"] == len(REPLY)
    assert rec["in_flight_at_start"] == 1
    assert rec["latency_ms"] >= 0


def test_response_without_usage_gives_none_tokens(monkeypatch, records):
    # Existing tests return SimpleNamespace(output_text=...) with no usage.
    client = MagicMock()
    client.responses.create.return_value = _response()
    _use_client(monkeypatch, client)

    assert llm._call_llm("hi", None, None) == REPLY

    rec = records[0]
    assert rec["outcome"] == "ok"
    assert rec["input_tokens"] is None
    assert rec["output_tokens"] is None
    assert rec["cached_tokens"] is None


def test_magicmock_response_never_leaks_mock_objects(monkeypatch, records):
    client = MagicMock()
    _use_client(monkeypatch, client)  # create() returns a MagicMock

    llm._call_llm("hi", None, None)

    rec = records[0]
    assert rec["input_tokens"] is None
    assert rec["output_tokens"] is None
    assert rec["output_chars"] is None


def test_retry_gives_one_record_per_attempt_and_excludes_backoff(monkeypatch, records):
    client = MagicMock()
    client.responses.create.side_effect = [APITimeoutError(request=_request()), _response(_usage())]
    _use_client(monkeypatch, client)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: _real_sleep(0.3))

    start = time.perf_counter()
    assert llm._call_llm("hi", None, None) == REPLY
    wall_ms = (time.perf_counter() - start) * 1000

    assert [r["outcome"] for r in records] == ["timeout", "ok"]
    assert [r["attempt"] for r in records] == [1, 2]
    assert records[0]["error_type"] == "APITimeoutError"
    assert wall_ms >= 300
    assert all(r["latency_ms"] < 200 for r in records)  # the back-off is not inside an attempt



def test_non_retryable_api_error_is_classified(monkeypatch, records):
    client = MagicMock()
    client.responses.create.side_effect = APIError("bad request", _request(), body=None)
    _use_client(monkeypatch, client)

    with pytest.raises(ToolSelectionError):
        llm._call_llm("hi", None, None)

    assert [r["outcome"] for r in records] == ["api_error"]


def test_unexpected_exception_propagates_and_releases_in_flight(monkeypatch, records):
    client = MagicMock()
    client.responses.create.side_effect = RuntimeError("boom")
    _use_client(monkeypatch, client)

    with pytest.raises(RuntimeError):
        llm._call_llm("hi", None, None)

    assert [r["outcome"] for r in records] == ["other_error"]
    assert llm_metrics._in_flight == 0


def test_failing_sink_does_not_break_the_call(monkeypatch):
    def bad_sink(_record):
        raise ValueError("sink bug")

    llm_metrics.add_sink(bad_sink)
    try:
        client = MagicMock()
        client.responses.create.return_value = _response(_usage())
        _use_client(monkeypatch, client)
        assert llm._call_llm("hi", None, None) == REPLY
        assert llm_metrics._in_flight == 0
    finally:
        llm_metrics.remove_sink(bad_sink)


def test_flag_logs_one_line_without_customer_text(monkeypatch, caplog):
    monkeypatch.setenv(llm_metrics.ENV_FLAG, "1")
    client = MagicMock()
    client.responses.create.return_value = _response(_usage())
    _use_client(monkeypatch, client)
    secret = "my card number is 4111"

    with caplog.at_level("INFO"):
        llm._call_llm(secret, None, None)

    lines = [m for m in caplog.messages if m.startswith("KASAFLOW_LLM_METRICS")]
    assert len(lines) == 1
    assert secret not in lines[0]
    assert REPLY not in lines[0]


def test_in_flight_counter_sees_real_parallel_calls(monkeypatch, records):
    workers = 4
    barrier = threading.Barrier(workers)

    def slow_create(**_kwargs):
        time.sleep(0.2)
        return _response(_usage())

    client = MagicMock()
    client.responses.create.side_effect = slow_create
    _use_client(monkeypatch, client)

    def worker():
        barrier.wait()
        llm._call_llm("hi", None, None)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(records) == workers
    assert max(r["in_flight_at_start"] for r in records) == workers
    assert llm_metrics._in_flight == 0
