"""One model-agnostic chat client used everywhere in fab_agent.

The whole point of this module is the research question in PROJECT.md: run the
*identical* architecture across closed models (Gemini, GPT, Claude) and
open-weights models (Ollama), choosing the backend purely by configuration.

* Gemini / OpenAI / Ollama are reached through the **openai** package (they all
  speak the OpenAI chat-completions protocol); only the base URL differs.
* Anthropic is reached through the **anthropic** SDK, adapted behind the same
  internal interface by translating the message and tool-calling formats.

Every backend returns the same normalized :class:`ChatResponse`, retries
transient errors with exponential backoff, tracks token usage, and supports an
optional JSON-schema ``response_format`` (falling back to prompt-instructed JSON
when a backend does not support it).

The heavy SDKs are imported lazily so this module imports cheaply and unit tests
can inject a fake client without any SDK, API key, or network.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fab_agent.config import OLLAMA_DEFAULT_BASE_URL, Settings, get_settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5  # seconds; delay = base * 2**attempt (+ jitter)
_ANTHROPIC_MAX_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Normalized types
# --------------------------------------------------------------------------- #
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Provider-independent chat result."""

    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = None
    #: True when a JSON-schema response_format was emulated via prompt injection.
    json_fallback: bool = False

    def as_json(self) -> Any:
        """Parse ``content`` as JSON (tolerating markdown fences / stray prose)."""
        return json.loads(_coerce_json_text(self.content or ""))


class LLMConnectionError(RuntimeError):
    """Raised by ``check_connection`` with a friendly, actionable message."""


class ResponseFormatUnsupported(Exception):
    """Internal signal that a backend cannot honour a JSON-schema response_format."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _coerce_json_text(text: str) -> str:
    """Best-effort cleanup of model output into a parseable JSON string."""
    stripped = _FENCE_RE.sub("", text.strip()).strip()
    if stripped[:1] not in "{[":
        start = min((i for i in (stripped.find("{"), stripped.find("[")) if i != -1),
                    default=-1)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start:end + 1]
    return stripped


def _json_instruction(schema: dict) -> str:
    return (
        "You must respond with a single valid JSON object only — no prose, no "
        "explanation, no markdown code fences. The JSON must conform to this "
        "JSON schema:\n" + json.dumps(schema)
    )


def _missing_key_message(provider: str) -> str:
    return {
        "gemini": "Missing LLM_API_KEY — get a free key at Google AI Studio: "
                  "https://aistudio.google.com/app/apikey",
        "openai": "Missing LLM_API_KEY for OpenAI — create one at "
                  "https://platform.openai.com/api-keys",
        "anthropic": "Missing LLM_API_KEY for Anthropic — create one at "
                     "https://console.anthropic.com/",
    }.get(provider, f"Missing LLM_API_KEY for provider {provider!r}.")


# --------------------------------------------------------------------------- #
# Base client: retry, usage accounting, response_format fallback
# --------------------------------------------------------------------------- #
class BaseChatClient:
    """Shared retry / usage / response_format logic; subclasses talk to an SDK."""

    def __init__(self, provider: str, model: str, *,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 backoff_base: float = DEFAULT_BACKOFF_BASE):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Token counters exposed on the client.
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()
        self.call_count = 0

    # -- public API -------------------------------------------------------- #
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             response_format: Optional[dict] = None,
             temperature: float = 0.1) -> ChatResponse:
        """Send a chat request and return a normalized :class:`ChatResponse`.

        ``response_format`` is a JSON schema. If the backend rejects it, we retry
        once with the schema injected into the prompt and parse JSON from the
        reply (``json_fallback=True`` on the result).
        """
        if response_format is None:
            return self._with_retry(messages, tools, None, temperature)
        try:
            return self._with_retry(messages, tools, response_format, temperature)
        except ResponseFormatUnsupported:
            injected = list(messages) + [
                {"role": "system", "content": _json_instruction(response_format)}
            ]
            resp = self._with_retry(injected, tools, None, temperature)
            resp.json_fallback = True
            return resp

    def reset_usage(self) -> None:
        self.total_usage = TokenUsage()
        self.call_count = 0

    # -- retry / accounting ------------------------------------------------ #
    def _with_retry(self, messages, tools, response_format, temperature) -> ChatResponse:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._raw_chat(messages, tools, response_format, temperature)
            except ResponseFormatUnsupported:
                raise  # not transient; handled by chat()
            except Exception as exc:  # noqa: BLE001 - classified below
                if not self._is_transient(exc):
                    raise
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                delay = self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base)
                time.sleep(delay)
                continue
            self._record(resp.usage)
            return resp
        assert last_exc is not None
        raise last_exc

    def _record(self, usage: TokenUsage) -> None:
        self.last_usage = usage
        self.total_usage = self.total_usage + usage
        self.call_count += 1

    # -- subclass hooks ---------------------------------------------------- #
    def _raw_chat(self, messages, tools, response_format, temperature) -> ChatResponse:
        raise NotImplementedError

    def _is_transient(self, exc: Exception) -> bool:
        raise NotImplementedError

    def check_connection(self) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (Gemini, OpenAI, Ollama)
# --------------------------------------------------------------------------- #
class OpenAIChatClient(BaseChatClient):
    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, provider: str = "openai",
                 *, client: Any = None, **kwargs):
        super().__init__(provider, model, **kwargs)
        self._api_key = api_key
        self._base_url = base_url
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._client = client

    def _raw_chat(self, messages, tools, response_format, temperature) -> ChatResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_format, "strict": True},
            }
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except openai.APIStatusError as exc:
            # A backend that can't do structured output typically 400s; treat any
            # 400 (or an error naming response_format) while a schema was set as
            # "unsupported" so chat() can fall back to prompt-instructed JSON.
            if response_format is not None and (
                getattr(exc, "status_code", None) == 400 or _mentions_response_format(exc)
            ):
                raise ResponseFormatUnsupported() from exc
            raise
        return _normalize_openai(completion)

    def _is_transient(self, exc: Exception) -> bool:
        import openai

        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                            openai.APITimeoutError, openai.InternalServerError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
        return False

    def check_connection(self) -> None:
        import openai

        if self.provider != "ollama" and not self._api_key:
            raise LLMConnectionError(_missing_key_message(self.provider))
        try:
            list(self._client.models.list())  # cheap, spends no tokens
        except openai.AuthenticationError as exc:
            raise LLMConnectionError(
                f"Authentication failed for {self.provider} — check LLM_API_KEY. ({exc})"
            ) from exc
        except openai.APIConnectionError as exc:
            if self.provider == "ollama":
                raise LLMConnectionError(
                    f"Cannot reach Ollama at {self._base_url}. Is Ollama running? "
                    f"Try: ollama serve   (then: ollama pull {self.model})"
                ) from exc
            raise LLMConnectionError(
                f"Cannot reach {self.provider} at {self._base_url or 'the default endpoint'}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMConnectionError(f"{self.provider} connectivity check failed: {exc}") from exc


def _mentions_response_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(t in text for t in ("response_format", "json_schema", "response format"))


def _normalize_openai(completion: Any) -> ChatResponse:
    choice = completion.choices[0]
    message = choice.message
    tool_calls = []
    for tc in (getattr(message, "tool_calls", None) or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {"_raw_arguments": tc.function.arguments}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    usage = TokenUsage()
    if getattr(completion, "usage", None):
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens or 0,
            completion_tokens=completion.usage.completion_tokens or 0,
            total_tokens=completion.usage.total_tokens or 0,
        )
    return ChatResponse(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        model=getattr(completion, "model", ""),
        usage=usage,
        raw=completion,
    )


# --------------------------------------------------------------------------- #
# Anthropic client (translated behind the same interface)
# --------------------------------------------------------------------------- #
def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Translate OpenAI-style messages to (system_prompt, anthropic_messages)."""
    system_parts: list[str] = []
    converted: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_parts.append(msg.get("content") or "")
        elif role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": msg.get("content") or "",
                }],
            })
        elif role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": fn["name"], "input": args})
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": role, "content": msg.get("content") or ""})
    return "\n\n".join(p for p in system_parts if p), converted


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Translate OpenAI tool schemas to Anthropic tool schemas."""
    out = []
    for tool in tools:
        fn = tool["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


_ANTHROPIC_STOP_MAP = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls",
                       "stop_sequence": "stop"}


def _normalize_anthropic(message: Any) -> ChatResponse:
    text_parts, tool_calls = [], []
    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            args = block.input if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=args))

    usage = TokenUsage()
    if getattr(message, "usage", None):
        inp = message.usage.input_tokens or 0
        out = message.usage.output_tokens or 0
        usage = TokenUsage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)

    return ChatResponse(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        finish_reason=_ANTHROPIC_STOP_MAP.get(getattr(message, "stop_reason", ""), "stop"),
        model=getattr(message, "model", ""),
        usage=usage,
        raw=message,
    )


class AnthropicChatClient(BaseChatClient):
    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, provider: str = "anthropic",
                 *, client: Any = None, **kwargs):
        super().__init__(provider, model, **kwargs)
        self._api_key = api_key
        if client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
                raise LLMConnectionError(
                    "The 'anthropic' package is required for the anthropic provider. "
                    "Install it with: uv add anthropic"
                ) from exc
            client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._client = client

    def _raw_chat(self, messages, tools, response_format, temperature) -> ChatResponse:
        # The Messages API has no response_format; signal fallback to prompt JSON.
        if response_format is not None:
            raise ResponseFormatUnsupported()
        system, converted = _to_anthropic(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _ANTHROPIC_MAX_TOKENS,
            "temperature": temperature,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        message = self._client.messages.create(**kwargs)
        return _normalize_anthropic(message)

    def _is_transient(self, exc: Exception) -> bool:
        import anthropic

        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError,
                            anthropic.APITimeoutError, anthropic.InternalServerError)):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
        return False

    def check_connection(self) -> None:
        import anthropic

        if not self._api_key:
            raise LLMConnectionError(_missing_key_message("anthropic"))
        try:
            list(self._client.models.list())
        except anthropic.AuthenticationError as exc:
            raise LLMConnectionError(
                f"Authentication failed for anthropic — check LLM_API_KEY. ({exc})"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(f"Cannot reach the Anthropic API: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMConnectionError(f"anthropic connectivity check failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_client(provider: str, model: str, api_key: Optional[str] = None,
                base_url: Optional[str] = None, **kwargs) -> BaseChatClient:
    """Instantiate a chat client for one provider/model.

    Lets the Stage-5 experiment runner hold several different models in one
    process for a head-to-head comparison.
    """
    provider = provider.lower()
    if provider == "gemini":
        return OpenAIChatClient(model, api_key, base_url or GEMINI_BASE_URL, "gemini", **kwargs)
    if provider == "openai":
        return OpenAIChatClient(model, api_key, base_url, "openai", **kwargs)
    if provider == "ollama":
        return OpenAIChatClient(model, api_key or "ollama",
                                base_url or OLLAMA_DEFAULT_BASE_URL, "ollama", **kwargs)
    if provider == "anthropic":
        return AnthropicChatClient(model, api_key, base_url, "anthropic", **kwargs)
    raise ValueError(f"unknown provider: {provider!r}")


def make_client_from_settings(settings: Optional[Settings] = None, **kwargs) -> BaseChatClient:
    """Build the client described by configuration (`.env`)."""
    settings = settings or get_settings()
    return make_client(
        settings.llm_provider, settings.llm_model,
        api_key=settings.llm_api_key, base_url=settings.llm_base_url, **kwargs,
    )
