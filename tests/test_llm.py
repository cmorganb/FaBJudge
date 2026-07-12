"""Tests for the model-agnostic chat client. No real model, key, or network.

The SDK layer is mocked by injecting a fake client into the constructor and by
raising the real SDK exception types (constructed locally).
"""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from fab_agent.agent import llm
from fab_agent.agent.llm import (
    GEMINI_BASE_URL,
    AnthropicChatClient,
    ChatResponse,
    LLMConnectionError,
    OpenAIChatClient,
    ToolCall,
    _coerce_json_text,
    _normalize_anthropic,
    _normalize_openai,
    _to_anthropic,
    _to_anthropic_tools,
    make_client,
)
from fab_agent.config import OLLAMA_DEFAULT_BASE_URL


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def make_completion(content, tool_calls=None, usage=(1, 1, 2), model="test-model",
                    finish="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    usage_ns = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1],
                               total_tokens=usage[2])
    return SimpleNamespace(choices=[choice], usage=usage_ns, model=model)


class Recorder:
    """Callable that records kwargs and yields queued results / raises errors."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeOpenAI:
    def __init__(self, create=None, models_list=None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create or Recorder([])))
        self.models = SimpleNamespace(list=models_list or (lambda: []))


def make_openai_client(create=None, models_list=None, provider="openai", **kw):
    return OpenAIChatClient("test-model", "key", provider=provider, backoff_base=0.0,
                            client=FakeOpenAI(create=create, models_list=models_list), **kw)


def _conn_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))


def _bad_request():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x/v1"))
    return openai.BadRequestError("response_format not supported", response=resp, body=None)


# --------------------------------------------------------------------------- #
# Factory / base URLs
# --------------------------------------------------------------------------- #
def test_make_client_base_urls_and_types():
    gem = make_client("gemini", "gemini-3.5-flash", api_key="k")
    assert isinstance(gem, OpenAIChatClient)
    assert gem.provider == "gemini" and gem._base_url == GEMINI_BASE_URL

    oai = make_client("openai", "gpt-4o-mini", api_key="k")
    assert oai._base_url is None  # SDK default

    oll = make_client("ollama", "llama3.1:8b")
    assert oll._base_url == OLLAMA_DEFAULT_BASE_URL

    with pytest.raises(ValueError):
        make_client("mistral-hosted", "m")


def test_make_client_anthropic_with_injected_client():
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: None),
                           models=SimpleNamespace(list=lambda: []))
    client = make_client("anthropic", "claude-sonnet-5", api_key="k", client=fake)
    assert isinstance(client, AnthropicChatClient)
    assert client.provider == "anthropic"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_openai_with_tool_calls():
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(
        name="search_rules", arguments='{"query": "go again", "k": 3}'))
    completion = make_completion(None, tool_calls=[tc], usage=(30, 10, 40), finish="tool_calls")
    resp = _normalize_openai(completion)
    assert resp.tool_calls == [ToolCall("call_1", "search_rules", {"query": "go again", "k": 3})]
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.total_tokens == 40


def test_normalize_openai_handles_missing_usage():
    completion = make_completion("hi")
    completion.usage = None
    assert _normalize_openai(completion).usage.total_tokens == 0


# --------------------------------------------------------------------------- #
# chat(): usage accounting
# --------------------------------------------------------------------------- #
def test_chat_accumulates_token_usage():
    rec = Recorder([make_completion("a", usage=(10, 5, 15)),
                    make_completion("b", usage=(20, 7, 27))])
    client = make_openai_client(create=rec)

    r1 = client.chat([{"role": "user", "content": "one"}])
    r2 = client.chat([{"role": "user", "content": "two"}])

    assert r1.content == "a" and r2.content == "b"
    assert client.call_count == 2
    assert client.last_usage.total_tokens == 27
    assert client.total_usage.total_tokens == 42
    assert client.total_usage.prompt_tokens == 30


# --------------------------------------------------------------------------- #
# chat(): retry with backoff
# --------------------------------------------------------------------------- #
def test_chat_retries_transient_then_succeeds():
    rec = Recorder([_conn_error(), _conn_error(), make_completion("ok")])
    client = make_openai_client(create=rec)
    resp = client.chat([{"role": "user", "content": "x"}])
    assert resp.content == "ok"
    assert len(rec.calls) == 3  # two failures + success


def test_chat_raises_after_exhausting_retries():
    rec = Recorder([_conn_error(), _conn_error(), _conn_error()])
    client = make_openai_client(create=rec)
    with pytest.raises(openai.APIConnectionError):
        client.chat([{"role": "user", "content": "x"}])
    assert len(rec.calls) == 3


def test_chat_does_not_retry_non_transient():
    rec = Recorder([ValueError("bad input")])
    client = make_openai_client(create=rec)
    with pytest.raises(ValueError):
        client.chat([{"role": "user", "content": "x"}])
    assert len(rec.calls) == 1


# --------------------------------------------------------------------------- #
# chat(): response_format + fallback
# --------------------------------------------------------------------------- #
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def test_response_format_passed_through_when_supported():
    rec = Recorder([make_completion('{"ok": true}')])
    client = make_openai_client(create=rec)
    resp = client.chat([{"role": "user", "content": "x"}], response_format=SCHEMA)
    assert rec.calls[0]["response_format"]["type"] == "json_schema"
    assert resp.json_fallback is False
    assert resp.as_json() == {"ok": True}


def test_response_format_falls_back_to_prompt_json():
    rec = Recorder([_bad_request(), make_completion('```json\n{"ok": true}\n```')])
    client = make_openai_client(create=rec)
    resp = client.chat([{"role": "user", "content": "x"}], response_format=SCHEMA)

    assert resp.json_fallback is True
    assert resp.as_json() == {"ok": True}
    # Retry omitted response_format and injected the schema as a system message.
    second = rec.calls[1]
    assert "response_format" not in second
    assert any(m["role"] == "system" and "JSON schema" in m["content"]
               for m in second["messages"])


# --------------------------------------------------------------------------- #
# Connectivity checks
# --------------------------------------------------------------------------- #
def test_check_connection_missing_key_message():
    client = make_openai_client(provider="gemini", models_list=lambda: [])
    client._api_key = None
    with pytest.raises(LLMConnectionError) as ei:
        client.check_connection()
    assert "Google AI Studio" in str(ei.value)


def test_check_connection_ollama_hint():
    def boom():
        raise _conn_error()
    client = OpenAIChatClient("llama3.1:8b", "ollama", base_url=OLLAMA_DEFAULT_BASE_URL,
                              provider="ollama", client=FakeOpenAI(models_list=boom))
    with pytest.raises(LLMConnectionError) as ei:
        client.check_connection()
    assert "ollama serve" in str(ei.value)


def test_check_connection_success():
    client = make_openai_client(provider="openai", models_list=lambda: [SimpleNamespace(id="gpt")])
    client.check_connection()  # no raise


# --------------------------------------------------------------------------- #
# Anthropic translation (pure; no SDK needed)
# --------------------------------------------------------------------------- #
def test_to_anthropic_translates_roles_tools_and_results():
    messages = [
        {"role": "system", "content": "You are a judge."},
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": "thinking",
         "tool_calls": [{"id": "t1", "function": {"name": "search_rules",
                                                  "arguments": '{"query": "x"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result text"},
    ]
    system, conv = _to_anthropic(messages)
    assert system == "You are a judge."
    assert conv[0] == {"role": "user", "content": "Question?"}

    assistant = conv[1]
    assert assistant["role"] == "assistant"
    assert {"type": "text", "text": "thinking"} in assistant["content"]
    assert any(b["type"] == "tool_use" and b["name"] == "search_rules"
               and b["input"] == {"query": "x"} for b in assistant["content"])

    assert conv[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "result text"}]}


def test_to_anthropic_tools_schema():
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    tools = [{"type": "function",
              "function": {"name": "get_card", "description": "d", "parameters": parameters}}]
    out = _to_anthropic_tools(tools)
    assert out[0]["name"] == "get_card"
    assert out[0]["input_schema"]["required"] == ["name"]


def test_normalize_anthropic():
    msg = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello"),
                 SimpleNamespace(type="tool_use", id="tu1", name="get_card",
                                 input={"name": "Sink Below"})],
        usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        stop_reason="tool_use", model="claude-sonnet-5")
    resp = _normalize_anthropic(msg)
    assert resp.content == "Hello"
    assert resp.tool_calls[0] == ToolCall("tu1", "get_card", {"name": "Sink Below"})
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.total_tokens == 20


# --------------------------------------------------------------------------- #
# JSON coercion helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here it is: {"a": 1} — hope that helps', {"a": 1}),
])
def test_coerce_json_text(raw, expected):
    assert json.loads(_coerce_json_text(raw)) == expected


def test_chat_response_as_json():
    assert ChatResponse(content='{"x": 5}').as_json() == {"x": 5}
