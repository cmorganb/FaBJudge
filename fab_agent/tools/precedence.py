"""Tool: get_precedence_context — the defeasible-reasoning framing for a ruling.

The prose lives in ``precedence.md`` (next to this module) so it can be reviewed
and edited without touching code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

PRECEDENCE_PATH = Path(__file__).resolve().parent / "precedence.md"

_BLOCK_RE = {
    "casual": re.compile(r"<!-- CASUAL -->(.*?)<!-- /CASUAL -->", re.DOTALL),
    "tournament": re.compile(r"<!-- TOURNAMENT -->(.*?)<!-- /TOURNAMENT -->", re.DOTALL),
}
# Any HTML comment block (setting blocks + the leading editor note).
_STRIP_RE = re.compile(r"<!--.*?-->\s*", re.DOTALL)


class GetPrecedenceInput(BaseModel):
    setting: Literal["casual", "tournament"] = Field(
        description="Whether the ruling is for casual play or a sanctioned tournament."
    )
    rel_level: Optional[str] = Field(
        default=None,
        description="Rules Enforcement Level (e.g. 'casual', 'competitive', "
        "'professional'). Defaults to the configured REL if omitted.",
    )


class PrecedenceContext(BaseModel):
    setting: str
    rel_level: str
    text: str


def _render(setting: str, rel_level: str, template: str) -> str:
    block_match = _BLOCK_RE[setting].search(template)
    block = block_match.group(1).strip() if block_match else ""

    # Common text = template with BOTH setting blocks removed (markers + their
    # content), then any remaining standalone comment (the editor note) stripped.
    base = template
    for block_re in _BLOCK_RE.values():
        base = block_re.sub("", base)
    base = _STRIP_RE.sub("", base).strip()

    rendered = f"{base}\n\n{block}".strip()
    return rendered.replace("{setting}", setting).replace("{rel_level}", rel_level)


def get_precedence_context(
    setting: Literal["casual", "tournament"],
    rel_level: Optional[str] = None,
) -> PrecedenceContext:
    """Return the applicable precedence / defeasible-reasoning context for a ruling.

    Explains which authorities govern and in what order: the Comprehensive Rules
    apply by default, card text overrides general CR rules where they conflict,
    and in sanctioned tournaments the TRP and PPG prevail within their scope. The
    active Rules Enforcement Level is stated. Call this to frame *how* to weigh
    conflicting authorities before concluding a ruling.
    """
    if rel_level is None:
        from fab_agent.config import get_settings

        rel_level = get_settings().rel_level

    template = PRECEDENCE_PATH.read_text(encoding="utf-8")
    return PrecedenceContext(
        setting=setting,
        rel_level=rel_level,
        text=_render(setting, rel_level, template),
    )
