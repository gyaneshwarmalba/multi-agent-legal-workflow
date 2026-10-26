"""Summarization agent: an executive summary built only from upstream agent outputs."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.llm import LLM, TransientLLMError

logger = logging.getLogger(__name__)

SYSTEM = (
    "You write executive summaries of contract reviews for business stakeholders. "
    "Use only the facts in the JSON you are given; do not add new facts. Under 150 words, plain language, no legal advice."
)


def template_summary(review: dict[str, Any], compliance: dict[str, Any]) -> str:
    parties = " and ".join(p["name"] for p in review.get("parties", [])) or "the parties"
    lines = [
        f"{review.get('title', 'Contract')} between {parties}"
        + (f", effective {review['effective_date']}." if review.get("effective_date") else ".")
    ]
    if review.get("term"):
        lines.append(f"Term: {review['term']}")
    failures = [r for r in compliance["results"] if r["status"] == "fail"]
    lines.append(
        f"Risk level: {compliance['risk_level'].upper()} with {len(failures)} policy exception(s)"
        + (": " + "; ".join(f"{r['title']} ({r['clause'] or 'missing'})" for r in failures) + "." if failures else ".")
    )
    if review.get("deadlines"):
        lines.append("Key periods: " + "; ".join(f"{d['duration']} ({d['clause']})" for d in review["deadlines"][:5]) + ".")
    if compliance["requires_approval"]:
        lines.append(
            "Recommended action: route to legal for approval and negotiate the high-severity exceptions before signature."
        )
    elif failures:
        lines.append("Recommended action: proceed, tracking the listed exceptions.")
    else:
        lines.append("Recommended action: proceed; no policy exceptions found.")
    return "\n".join(lines)


def run_summary(review: dict[str, Any], compliance: dict[str, Any], llm: LLM | None) -> str:
    if llm is None:
        return template_summary(review, compliance)
    facts = {
        "contract": {k: review.get(k) for k in ("title", "parties", "effective_date", "term", "renewal")},
        "deadlines": review.get("deadlines", [])[:8],
        "risk_level": compliance["risk_level"],
        "exceptions": [r for r in compliance["results"] if r["status"] == "fail"],
        "requires_approval": compliance["requires_approval"],
    }
    try:
        return llm.complete(SYSTEM, json.dumps(facts, indent=2), max_tokens=400)
    except TransientLLMError:
        raise
    except Exception:
        logger.warning("summary LLM call failed; using template summary", exc_info=True)
        return template_summary(review, compliance)
