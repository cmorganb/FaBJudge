"""Tool: ask_clarification — the agent's signal that it must stop and ask.

This tool does not answer anything. It is a sentinel the agent emits when a
ruling genuinely cannot be determined without a missing fact, so the loop halts
and the question is put to the user.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskClarificationInput(BaseModel):
    question: str = Field(
        description="The specific question to ask the user."
    )
    missing_info: str = Field(
        description="The precise fact that is missing and why it is required to rule."
    )


class ClarificationRequest(BaseModel):
    type: Literal["clarification_request"] = "clarification_request"
    question: str
    missing_info: str


def ask_clarification(question: str, missing_info: str) -> ClarificationRequest:
    """Stop and ask the user for a missing fact needed to rule.

    Use this ONLY when the ruling genuinely cannot be determined without the
    missing fact — for example when the board state, the specific cards, the
    active player, or the tournament setting is ambiguous and the answer changes
    depending on it. Do NOT use it to hedge, to ask for confirmation, or when a
    reasonable ruling can be given from the information already provided:
    unnecessary clarification requests are penalized by the evaluation. When you
    can rule, rule.
    """
    return ClarificationRequest(question=question, missing_info=missing_info)
