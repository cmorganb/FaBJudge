"""The ReAct rules-adjudication loop.

``RulesAgent.run`` drives an OpenAI-style tool-calling loop over the Stage-2
tool registry, gathers grounding, and produces an IRAC :class:`Verdict`. Two
guards protect faithfulness and structure:

* **Citation guard** — every ``chunk_id`` in the final verdict must have been
  returned by a tool during the run; otherwise the model is asked once to fix
  it, and any still-unsupported citations are marked ``valid=False``.
* **Schema guard** — if the final output fails validation, the model is asked
  once more with the validation error appended.

The full trace is persisted to ``data/traces/<timestamp>.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from fab_agent.agent.llm import TokenUsage, _coerce_json_text
from fab_agent.agent.schema import Verdict
from fab_agent.config import get_settings
from fab_agent.tools.registry import call_tool, openai_tool_schemas

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "traces"
_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"

_FORCE_FINAL = ("Stop calling tools now and output ONLY the final Verdict JSON "
                "object, using only chunk_ids you retrieved this conversation.")


def load_system_prompt(setting: str, rel_level: str) -> str:
    template = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{setting}", setting).replace("{rel_level}", rel_level)


class TraceStep(BaseModel):
    step: int
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result_summary: Optional[str] = None


class AgentResult(BaseModel):
    query: str
    setting: str
    verdict: Optional[Verdict] = None
    trace: list[TraceStep] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    invalid_citations: list[str] = Field(default_factory=list)
    trace_path: Optional[str] = None


class RulesAgent:
    """A tool-using ReAct agent that issues IRAC verdicts."""

    def __init__(self, client=None, *, tool_schemas: Optional[list[dict]] = None,
                 tool_caller: Callable[[str, dict], Any] = call_tool,
                 traces_dir: Optional[Path] = TRACES_DIR, temperature: float = 0.1,
                 rel_level: Optional[str] = None):
        self.client = client if client is not None else _default_client()
        self.tool_schemas = tool_schemas if tool_schemas is not None else openai_tool_schemas()
        self.call_tool = tool_caller
        self.traces_dir = traces_dir
        self.temperature = temperature
        self.rel_level = rel_level or get_settings().rel_level

    # ------------------------------------------------------------------ #
    def run(self, query: str, setting: str = "tournament", max_steps: int = 8) -> AgentResult:
        self._usage = TokenUsage()
        messages: list[dict] = [
            {"role": "system", "content": load_system_prompt(setting, self.rel_level)},
            {"role": "user", "content": f"Setting: {setting} (REL: {self.rel_level}).\n\n"
                                        f"Question: {query}"},
        ]
        trace: list[TraceStep] = []
        retrieved_ids: set[str] = set()
        final_content: Optional[str] = None

        for _ in range(max_steps):
            resp = self._chat(messages, tools=self.tool_schemas)
            if resp.tool_calls:
                messages.append(_assistant_tool_message(resp))
                clarify = self._handle_tool_calls(resp, messages, trace, retrieved_ids)
                if clarify is not None:  # ask_clarification is terminal
                    return self._finish(query, setting, clarify, trace, [])
                continue
            messages.append({"role": "assistant", "content": resp.content or ""})
            final_content = resp.content
            break

        if final_content is None:  # max_steps reached while still calling tools
            messages.append({"role": "user", "content": _FORCE_FINAL})
            resp = self._chat(messages, tools=None)
            messages.append({"role": "assistant", "content": resp.content or ""})
            final_content = resp.content

        verdict, error, invalid = self._produce_verdict(final_content, messages, retrieved_ids)
        return self._finish(query, setting, verdict, trace, invalid, error=error)

    # ------------------------------------------------------------------ #
    def _chat(self, messages, tools):
        resp = self.client.chat(messages, tools=tools, temperature=self.temperature)
        self._usage = self._usage + resp.usage
        return resp

    def _handle_tool_calls(self, resp, messages, trace, retrieved_ids) -> Optional[Verdict]:
        for tc in resp.tool_calls:
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            if tc.name == "ask_clarification":
                trace.append(TraceStep(step=len(trace) + 1, thought=resp.content,
                                       tool_name=tc.name, tool_args=args,
                                       tool_result_summary="clarification requested"))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(args)})
                return _clarification_verdict(args)

            try:
                result = self.call_tool(tc.name, args)
            except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": f"ERROR: {exc}"})
                trace.append(TraceStep(step=len(trace) + 1, thought=resp.content,
                                       tool_name=tc.name, tool_args=args,
                                       tool_result_summary=f"error: {exc}"))
                continue

            if tc.name == "search_rules":
                for hit in result:
                    retrieved_ids.add(hit.chunk_id)
            content, summary = _serialize_tool_result(tc.name, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            trace.append(TraceStep(step=len(trace) + 1, thought=resp.content,
                                   tool_name=tc.name, tool_args=args,
                                   tool_result_summary=summary))
        return None

    def _produce_verdict(self, content, messages, retrieved_ids):
        verdict, err = _parse_verdict(content)
        if verdict is None:  # schema guard: one corrective retry
            messages.append({"role": "user", "content":
                f"Your previous reply was not a valid Verdict JSON object ({err}). "
                f"Respond with ONLY a JSON object matching the Verdict schema."})
            resp = self._chat(messages, tools=None)
            messages.append({"role": "assistant", "content": resp.content or ""})
            verdict, err = _parse_verdict(resp.content)
            if verdict is None:
                return None, f"schema_validation_failed: {err}", []

        verdict, invalid = self._citation_guard(verdict, messages, retrieved_ids)
        return verdict, None, invalid

    def _citation_guard(self, verdict, messages, retrieved_ids):
        bad = [c.chunk_id for c in verdict.citations if c.chunk_id not in retrieved_ids]
        if not bad:
            return verdict, []

        valid = sorted(retrieved_ids) or ["(none retrieved)"]
        messages.append({"role": "user", "content":
            f"Citation error: these chunk_ids were never retrieved and must not be "
            f"cited: {bad}. You may only cite chunk_ids from this list: {valid}. "
            f"Reissue the full Verdict JSON citing only retrieved chunk_ids "
            f"(drop unsupported citations, or set clarification_request)."})
        resp = self._chat(messages, tools=None)
        messages.append({"role": "assistant", "content": resp.content or ""})

        fixed, _ = _parse_verdict(resp.content)
        if fixed is None:  # correction unparseable: keep original, flag bad citations
            for c in verdict.citations:
                if c.chunk_id in bad:
                    c.valid = False
            return verdict, bad

        still_bad = [c.chunk_id for c in fixed.citations if c.chunk_id not in retrieved_ids]
        for c in fixed.citations:
            if c.chunk_id in still_bad:
                c.valid = False
        return fixed, still_bad

    def _finish(self, query, setting, verdict, trace, invalid, error=None) -> AgentResult:
        result = AgentResult(
            query=query, setting=setting, verdict=verdict, trace=trace,
            token_usage={
                "prompt_tokens": self._usage.prompt_tokens,
                "completion_tokens": self._usage.completion_tokens,
                "total_tokens": self._usage.total_tokens,
            },
            error=error, invalid_citations=invalid,
        )
        result.trace_path = self._persist(result)
        return result

    def _persist(self, result: AgentResult) -> Optional[str]:
        if self.traces_dir is None:
            return None
        try:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            path = self.traces_dir / f"{stamp}.json"
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return str(path)
        except OSError:
            return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_client():
    from fab_agent.agent.llm import make_client_from_settings

    return make_client_from_settings()


def _assistant_tool_message(resp) -> dict:
    """The assistant turn to append before the tool results.

    Prefer echoing the provider's raw message verbatim — some backends (e.g.
    Gemini 3) attach a ``thought_signature`` to each function call that must be
    sent back unchanged on the next turn, and rebuilding the message from
    normalized fields would drop it. Fall back to a reconstruction (used by the
    mocked tests, where ``raw`` is absent).
    """
    raw = getattr(resp, "raw", None)
    if raw is not None:
        try:
            message = raw.choices[0].message.model_dump(exclude_none=True)
            if message.get("tool_calls"):
                message["role"] = "assistant"
                return message
        except (AttributeError, IndexError, TypeError):
            pass
    return {
        "role": "assistant",
        "content": resp.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in resp.tool_calls
        ],
    }


def _serialize_tool_result(name: str, result: Any) -> tuple[str, str]:
    if name == "search_rules":  # list[RuleHit]
        payload = [h.model_dump() for h in result]
        ids = [h.chunk_id for h in result]
        summary = f"{len(result)} passages: {', '.join(ids[:6])}"
        return json.dumps(payload), summary[:200]

    content = result.model_dump_json()
    if name == "get_card":
        summary = (f"matched={result.matched} name={result.name!r}"
                   + ("" if result.matched else f" suggestions={result.suggestions}"))
    elif name == "get_precedence_context":
        summary = f"precedence setting={result.setting} rel={result.rel_level}"
    else:
        summary = content[:150]
    return content, summary[:200]


def _parse_verdict(content: Optional[str]) -> tuple[Optional[Verdict], Optional[str]]:
    text = _coerce_json_text(content or "")
    if not text:
        return None, "empty response"
    try:
        return Verdict.model_validate_json(text), None
    except (ValidationError, ValueError) as exc:
        return None, str(exc).splitlines()[0]


def _clarification_verdict(args: dict) -> Verdict:
    question = args.get("question") or "Additional information is required to rule."
    missing = args.get("missing_info") or "unspecified"
    return Verdict(
        issue="A ruling was requested but required information is missing.",
        rule="A ruling requires facts that were not provided.",
        application="The outcome depends on the missing information below, so no "
                    "ruling can be made yet.",
        conclusion="Clarification is required before a ruling can be given.",
        citations=[],
        precedence_notes=None,
        clarification_request=f"{question} (needed because: {missing})",
        confidence="low",
    )
