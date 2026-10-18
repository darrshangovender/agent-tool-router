"""Retry wrapper, and the two real SDK backends driven through the real SDKs over a mock transport."""

from __future__ import annotations

import importlib
import json

import pytest

from agent_tool_router import RefusalReason, RetryingBackend
from agent_tool_router.llm_judge import AnthropicBackend, LLMJudge, OpenAIBackend
from tests.helpers import ScriptedBackend, make_router, verdict

# ---- RetryingBackend ------------------------------------------------------------------------


class Flaky:
    model = "flaky"

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures, self.calls = failures, 0
        self.error = error or ConnectionError("overloaded")

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def retrying(inner, **kw):
    sleeps: list[float] = []
    return RetryingBackend(inner, sleep=sleeps.append, **kw), sleeps


def test_retries_until_success_with_exponential_delays():
    inner = Flaky(failures=2)
    backend, sleeps = retrying(inner, retries=3, base_delay=0.5)
    assert backend.complete("s", "u") == "ok" and inner.calls == 3 and sleeps == [0.5, 1.0]


def test_after_the_last_attempt_the_original_error_propagates():
    inner = Flaky(failures=99, error=TimeoutError("slow"))
    backend, sleeps = retrying(inner, retries=2, base_delay=1.0)
    with pytest.raises(TimeoutError, match="slow"):
        backend.complete("s", "u")
    assert inner.calls == 3 and sleeps == [1.0, 2.0]


def test_success_never_sleeps_and_zero_retries_means_one_call():
    inner = Flaky(failures=0)
    backend, sleeps = retrying(inner)
    assert backend.complete("s", "u") == "ok" and sleeps == [] and inner.calls == 1
    once = Flaky(failures=1)
    zero, _ = retrying(once, retries=0)
    with pytest.raises(ConnectionError):
        zero.complete("s", "u")
    assert once.calls == 1


def test_only_listed_exception_types_are_retried():
    inner = Flaky(failures=5, error=ValueError("bad request, retrying will not help"))
    backend, sleeps = retrying(inner, retries=3, retry_on=(ConnectionError, TimeoutError))
    with pytest.raises(ValueError):
        backend.complete("s", "u")
    assert inner.calls == 1 and sleeps == []


def test_keyboard_interrupt_is_never_swallowed():
    inner = Flaky(failures=5, error=KeyboardInterrupt())
    backend, sleeps = retrying(inner, retries=3)
    with pytest.raises(KeyboardInterrupt):
        backend.complete("s", "u")
    assert inner.calls == 1


def test_arguments_and_model_are_passed_through():
    inner = ScriptedBackend(verdict("issue_refund"))
    backend, _ = retrying(inner)
    backend.complete("sys", "usr", max_tokens=77)
    assert inner.calls == [{"system": "sys", "user": "usr", "max_tokens": 77}]
    assert backend.model == "scripted"


@pytest.mark.parametrize("kw", [{"retries": -1}, {"base_delay": -0.1}])
def test_negative_settings_are_refused(kw):
    with pytest.raises(ValueError):
        RetryingBackend(Flaky(0), **kw)


def test_a_dead_judge_behind_the_router_degrades_to_a_typed_refusal():
    dead = Flaky(failures=99)
    router = make_router(RetryingBackend(dead, retries=1, sleep=lambda s: None),
                         on_judge_error="refuse")
    d = router.route("refund order 9921")
    assert d.is_refusal and d.refusal.reason is RefusalReason.LOW_CONFIDENCE
    assert d.error is not None and "ConnectionError" in d.error and dead.calls == 2


# ---- the real SDKs over a mock transport ----------------------------------------------------

def http_library(sdk):
    """The HTTP package the installed SDK is built on: recent releases use `httpx2`, older ones
    `httpx`, and each rejects the other's client objects."""
    base = importlib.import_module(f"{sdk.__name__}._base_client")
    return getattr(base, "httpx2", None) or base.httpx


def recorder(sdk, status: int, body: dict):
    lib = http_library(sdk)
    seen: list[dict] = []

    def handler(request):
        seen.append({"path": request.url.path, "json": json.loads(request.content or b"{}"),
                     "headers": dict(request.headers)})
        return lib.Response(status, json=body)

    return seen, lib.Client(transport=lib.MockTransport(handler))


def anthropic_message(*blocks: dict) -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": list(blocks), "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 4}}


def anthropic_backend(status: int, body: dict, **kw):
    anthropic = pytest.importorskip("anthropic")
    seen, http = recorder(anthropic, status, body)
    client = anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0)
    return AnthropicBackend(client=client, **kw), seen


def test_anthropic_request_shape_and_text_extraction():
    reply = json.dumps(verdict("issue_refund"))
    backend, seen = anthropic_backend(200, anthropic_message(
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": reply[:20]}, {"type": "text", "text": reply[20:]}))
    out = backend.complete("SYSTEM", "USER", max_tokens=123)
    assert out == reply                                  # thinking skipped, text blocks joined
    body = seen[0]["json"]
    assert seen[0]["path"].endswith("/v1/messages")
    assert (body["model"], body["max_tokens"], body["system"], body["temperature"]) == (
        "claude-haiku-4-5", 123, "SYSTEM", 0.0)
    assert body["messages"] == [{"role": "user", "content": "USER"}]


def test_anthropic_temperature_can_be_left_out_for_models_that_reject_it():
    backend, seen = anthropic_backend(200, anthropic_message({"type": "text", "text": "{}"}),
                                      model="claude-haiku-5-5", temperature=None)
    backend.complete("s", "u")
    assert "temperature" not in seen[0]["json"] and seen[0]["json"]["model"] == "claude-haiku-5-5"


def test_anthropic_with_no_text_block_gives_an_empty_string_and_a_refusal_downstream():
    backend, _ = anthropic_backend(200, anthropic_message(
        {"type": "thinking", "thinking": "...", "signature": "s"}))
    router = make_router(None, judge=LLMJudge(backend))
    d = router.route("refund order 9921")
    assert d.is_refusal and "Empty response" in d.reasoning


def test_a_rate_limit_from_the_real_sdk_reaches_the_router_as_a_typed_refusal():
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    backend, seen = anthropic_backend(429, body)
    router = make_router(None, judge=LLMJudge(backend), on_judge_error="refuse")
    d = router.route("refund order 9921")
    assert d.is_refusal and d.refusal.reason is RefusalReason.LOW_CONFIDENCE
    assert d.error is not None and "RateLimitError" in d.error and len(seen) == 1
    with pytest.raises(Exception, match="slow down"):
        make_router(None, judge=LLMJudge(backend)).route("refund order 9921")


def test_the_whole_router_over_the_real_anthropic_sdk():
    reply = json.dumps(verdict("issue_refund", 0.93, reasoning="refund request"))
    backend, seen = anthropic_backend(200, anthropic_message({"type": "text", "text": reply}))
    d = make_router(None, judge=LLMJudge(backend)).route("refund order 9921")
    assert d.is_route and d.tool == "issue_refund" and d.confidence == pytest.approx(0.93)
    prompt = seen[0]["json"]["messages"][0]["content"]
    assert "refund order 9921" in prompt and '"issue_refund"' in prompt


def openai_completion(content):
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def openai_backend(content, **kw):
    openai = pytest.importorskip("openai")
    seen, http = recorder(openai, 200, openai_completion(content))
    client = openai.OpenAI(api_key="test-key", http_client=http, max_retries=0)
    return OpenAIBackend(client=client, **kw), seen


def test_openai_request_shape():
    backend, seen = openai_backend('{"tool": null}')
    assert backend.complete("SYSTEM", "USER", max_tokens=55) == '{"tool": null}'
    body = seen[0]["json"]
    assert seen[0]["path"].endswith("/chat/completions")
    assert (body["model"], body["max_tokens"], body["temperature"]) == ("gpt-4o-mini", 55, 0.0)
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"},
                                {"role": "user", "content": "USER"}]


def test_openai_temperature_can_be_left_out_and_null_content_is_empty_text():
    backend, seen = openai_backend(None, temperature=None)
    assert backend.complete("s", "u") == "" and "temperature" not in seen[0]["json"]


def test_the_whole_router_over_the_real_openai_sdk():
    reply = json.dumps(verdict(None, 0.2, "no_matching_tool", "nothing fits"))
    backend, _ = openai_backend(reply)
    d = make_router(None, judge=LLMJudge(backend)).route("what is the weather in Rome")
    assert d.is_refusal and d.refusal.reason is RefusalReason.NO_MATCHING_TOOL
