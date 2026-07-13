"""IRAC verdict schema — the citation-bearing structured output of the agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Citation(BaseModel):
    """A reference to a specific retrieved corpus passage."""

    chunk_id: str = Field(description="The exact chunk_id returned by a tool this run.")
    rule_id: str = Field(description="The rule identifier or card name, e.g. '1.2.3a'.")
    doc: str = Field(description="Document code: CR / PPG / TRP / CARD.")
    quoted_snippet: str = Field(
        description="A short verbatim quote from the cited passage (<=200 chars).",
    )
    #: Set to False by the agent's citation guard when the chunk_id was never
    #: actually retrieved during the run. Models should not populate this.
    valid: bool = True

    @field_validator("quoted_snippet", mode="before")
    @classmethod
    def _truncate_snippet(cls, value: object) -> str:
        text = "" if value is None else str(value)
        return text[:200]


class Verdict(BaseModel):
    """An IRAC-structured ruling, or a clarification request instead of a ruling."""

    issue: str = Field(description="IRAC Issue: the question restated precisely.")
    rule: str = Field(description="IRAC Rule: the governing rules, in the agent's words.")
    application: str = Field(description="IRAC Application: applying the rules to these facts.")
    conclusion: str = Field(description="IRAC Conclusion: the ruling.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Supporting citations; at least one unless a clarification is requested.",
    )
    precedence_notes: Optional[str] = Field(
        default=None,
        description="Which authority prevailed and why (CR / card text / TRP / PPG), if relevant.",
    )
    clarification_request: Optional[str] = Field(
        default=None,
        description="Set INSTEAD of a ruling when a required fact is missing.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="The agent's confidence in the ruling.",
    )

    @model_validator(mode="after")
    def _citations_required_for_a_ruling(self) -> "Verdict":
        if not self.clarification_request and len(self.citations) < 1:
            raise ValueError(
                "a ruling must include at least one citation (or set clarification_request)"
            )
        return self
