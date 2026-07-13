"""Loop-mechanics tests for RulesAgent. The LLM is a scripted mock; tools are
dispatched through a fake caller. No real model, index, or network."""

import json

from fab_agent.agent.llm import ChatResponse, TokenUsage, ToolCall
from fab_agent.agent.loop import RulesAgent
from fab_agent.tools.rules import RuleHit


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class MockClient:
    provider = "mock"
    model = "mock-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, response_format=None, temperature=0.1):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


def tool_msg(content, tool_calls=None, usage=(5, 5, 10)):
    return ChatResponse(content=content, tool_calls=tool_calls or [],
                        usage=TokenUsage(*usage))


def call(name, args, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=args)


def verdict_json(citations=None, clarification=None, confidence="high", **over):
    payload = {
        "issue": "I", "rule": "R", "application": "A", "conclusion": "C",
        "citations": citations if citations is not None else
        [{"chunk_id": "CR-1.2.3", "rule_id": "1.2.3", "doc": "CR", "quoted_snippet": "snip"}],
        "precedence_notes": None, "clarification_request": clarification,
        "confidence": confidence,
    }
    payload.update(over)
    return json.dumps(payload)


def fake_search(name, args):
    if name == "search_rules":
        return [RuleHit(chunk_id="CR-1.2.3", rule_id="1.2.3", doc="CR",
                        text="A rule about blocking and paying costs.")]
    raise AssertionError(f"unexpected tool {name}")


def make_agent(responses, tool_caller=fake_search):
    return RulesAgent(client=MockClient(responses), tool_schemas=[],
                      tool_caller=tool_caller, traces_dir=None, rel_level="competitive")


# --------------------------------------------------------------------------- #
# Tool dispatch + happy path
# --------------------------------------------------------------------------- #
def test_tool_dispatch_then_verdict():
    responses = [
        tool_msg("let me search", [call("search_rules", {"query": "blocking", "doc_filter": ["CR"]})]),
        tool_msg(verdict_json()),
    ]
    agent = make_agent(responses)
    result = agent.run("Can I block then pay a cost with the same card?")

    assert result.error is None
    assert result.verdict is not None
    assert result.verdict.conclusion == "C"
    assert result.invalid_citations == []
    assert result.verdict.citations[0].chunk_id == "CR-1.2.3"
    assert result.verdict.citations[0].valid is True

    # Trace recorded the tool call with a result summary.
    assert len(result.trace) == 1
    step = result.trace[0]
    assert step.tool_name == "search_rules"
    assert step.tool_args == {"query": "blocking", "doc_filter": ["CR"]}
    assert "CR-1.2.3" in step.tool_result_summary
    # Token usage summed across both LLM calls.
    assert result.token_usage["total_tokens"] == 20


# --------------------------------------------------------------------------- #
# Citation guard
# --------------------------------------------------------------------------- #
def test_citation_guard_repairs_fake_chunk_id():
    responses = [
        tool_msg("searching", [call("search_rules", {"query": "x"})]),
        tool_msg(verdict_json(citations=[  # cites a chunk that was never retrieved
            {"chunk_id": "CR-9.9.9z", "rule_id": "9.9.9z", "doc": "CR", "quoted_snippet": "fake"}])),
        tool_msg(verdict_json()),  # corrected to the retrieved CR-1.2.3
    ]
    agent = make_agent(responses)
    result = agent.run("q")

    assert [c.chunk_id for c in result.verdict.citations] == ["CR-1.2.3"]
    assert result.invalid_citations == []
    assert all(c.valid for c in result.verdict.citations)
    # A corrective message naming the offending id was sent.
    corrective = agent.client.calls[-1]["messages"][-1]["content"]
    assert "CR-9.9.9z" in corrective and "never retrieved" in corrective


def test_citation_guard_marks_invalid_when_not_repaired():
    responses = [
        tool_msg("searching", [call("search_rules", {"query": "x"})]),
        tool_msg(verdict_json(citations=[
            {"chunk_id": "CR-9.9.9z", "rule_id": "9.9.9z", "doc": "CR", "quoted_snippet": "fake"}])),
        tool_msg(verdict_json(citations=[  # still fabricated on the retry
            {"chunk_id": "CR-9.9.9z", "rule_id": "9.9.9z", "doc": "CR", "quoted_snippet": "fake"}])),
    ]
    result = make_agent(responses).run("q")

    assert result.invalid_citations == ["CR-9.9.9z"]
    bad = result.verdict.citations[0]
    assert bad.chunk_id == "CR-9.9.9z" and bad.valid is False


# --------------------------------------------------------------------------- #
# Schema guard
# --------------------------------------------------------------------------- #
def test_schema_retry_on_invalid_json():
    responses = [
        tool_msg("here is my answer: {not valid json"),  # no tool calls -> final; unparseable
        tool_msg(verdict_json(clarification="need the card name", citations=[])),  # corrected
    ]
    agent = make_agent(responses)
    result = agent.run("q")

    assert result.error is None
    assert result.verdict is not None
    assert result.verdict.clarification_request == "need the card name"
    # The retry message carried the validation error.
    assert "not a valid Verdict" in agent.client.calls[-1]["messages"][-1]["content"]
    assert len(agent.client.calls) == 2


def test_schema_retry_failure_returns_error():
    responses = [tool_msg("garbage"), tool_msg("still garbage")]
    result = make_agent(responses).run("q")
    assert result.verdict is None
    assert result.error and result.error.startswith("schema_validation_failed")


# --------------------------------------------------------------------------- #
# ask_clarification is terminal
# --------------------------------------------------------------------------- #
def test_ask_clarification_short_circuits():
    responses = [
        tool_msg("need info", [call("ask_clarification",
                                    {"question": "Which hero is attacking?",
                                     "missing_info": "attacking hero unspecified"})]),
    ]
    agent = make_agent(responses)
    result = agent.run("Is the attack legal?")

    assert result.verdict is not None
    assert result.verdict.citations == []
    assert result.verdict.confidence == "low"
    assert "Which hero is attacking?" in result.verdict.clarification_request
    assert len(agent.client.calls) == 1  # stopped immediately
    assert result.trace[0].tool_name == "ask_clarification"


# --------------------------------------------------------------------------- #
# max_steps cutoff
# --------------------------------------------------------------------------- #
def test_max_steps_cutoff_forces_finalize():
    # Model keeps calling tools forever; loop must cut off and force a verdict.
    responses = [
        tool_msg("s1", [call("search_rules", {"query": "a"})]),
        tool_msg("s2", [call("search_rules", {"query": "b"})]),
        tool_msg(verdict_json()),  # produced by the forced finalize call
    ]
    agent = make_agent(responses)
    result = agent.run("q", max_steps=2)

    assert len(result.trace) == 2  # exactly max_steps tool turns
    assert result.verdict is not None
    assert len(agent.client.calls) == 3  # 2 tool turns + 1 forced finalize
    forced = agent.client.calls[-1]
    assert forced["tools"] is None
    assert "only the final verdict json" in forced["messages"][-1]["content"].lower()
