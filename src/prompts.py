"""Prompt templates that turn structured DS findings into Flan-T5 instructions."""

from __future__ import annotations

import json
from typing import Any, Mapping

TONES = (
    "Executive summary",
    "Technical report",
    "Stakeholder update",
)

_TONE_INSTRUCTIONS = {
    "Executive summary": (
        "Write a concise executive summary for leadership. "
        "Lead with the business implication, then the key numbers, then one recommended next step."
    ),
    "Technical report": (
        "Write a precise technical report for data scientists. "
        "Include metrics, statistical caveats, and methodology notes where relevant."
    ),
    "Stakeholder update": (
        "Write a stakeholder update for product and business partners. "
        "Use plain language, highlight impact, and keep jargon minimal."
    ),
}


def _findings_to_text(findings: str | Mapping[str, Any] | list[Any]) -> str:
    """Normalize free-text, dict, or bullet list findings into a single block."""
    if isinstance(findings, str):
        text = findings.strip()
        if not text:
            raise ValueError("findings text must be non-empty")
        return text

    if isinstance(findings, Mapping):
        if not findings:
            raise ValueError("findings mapping must be non-empty")
        # Prefer a stable, readable key: value layout for metrics.
        lines = []
        for key, value in findings.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                value_str = str(value)
            lines.append(f"- {key}: {value_str}")
        return "\n".join(lines)

    if isinstance(findings, list):
        if not findings:
            raise ValueError("findings list must be non-empty")
        bullets = []
        for item in findings:
            if isinstance(item, Mapping):
                bullets.append(f"- {json.dumps(item, ensure_ascii=False, sort_keys=True)}")
            else:
                bullets.append(f"- {str(item).strip()}")
        return "\n".join(bullets)

    raise TypeError(
        f"findings must be str, mapping, or list; got {type(findings).__name__}"
    )


def build_prompt(
    findings: str | Mapping[str, Any] | list[Any],
    tone: str = "Executive summary",
) -> str:
    """Build a Flan-T5 instruction prompt from structured data-science findings.

    Parameters
    ----------
    findings:
        Free-text summary, metric dict, or list of bullet findings.
    tone:
        One of ``TONES`` — controls the narrative style instruction.
    """
    if tone not in _TONE_INSTRUCTIONS:
        allowed = ", ".join(TONES)
        raise ValueError(f"tone must be one of: {allowed}")

    body = _findings_to_text(findings)
    instruction = _TONE_INSTRUCTIONS[tone]
    return (
        "Translate the following data science findings into a clear human narrative.\n"
        f"Style: {tone}. {instruction}\n\n"
        f"Findings:\n{body}\n\n"
        "Narrative:"
    )


def format_training_example(
    complex_data_science_result: str,
    human_storytelling_narrative: str | None = None,
) -> dict[str, str]:
    """Format a JSONL-style training pair (optionally with target narrative)."""
    result = (complex_data_science_result or "").strip()
    if not result:
        raise ValueError("complex_data_science_result must be non-empty")

    prompt = build_prompt(result, tone="Executive summary")
    row: dict[str, str] = {
        "complex_data_science_result": result,
        "prompt": prompt,
    }
    if human_storytelling_narrative is not None:
        narrative = human_storytelling_narrative.strip()
        if not narrative:
            raise ValueError("human_storytelling_narrative must be non-empty when provided")
        row["human_storytelling_narrative"] = narrative
    return row
