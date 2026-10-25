"""Contract review agent: extracts parties, dates, term, obligations, and deadlines.

With an LLM configured it asks for structured JSON and validates it. If the
model output is malformed, it falls back to the deterministic extractor and
records that it did, rather than failing the workflow or passing bad data on.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.contracts import Clause, clause_ref, sentences
from app.llm import LLM, extract_json

logger = logging.getLogger(__name__)

PARTIES_RE = re.compile(
    r"between\s+(?P<a>[A-Z][^()]+?)(?:\s+\(\"(?P<ra>[\w ]+)\"\))?\s+and\s+(?P<b>[A-Z][^()]+?)\s+\((?:\"(?P<rb>[\w ]+)\")?"
)
DATE_RE = re.compile(r"as of\s+([A-Z][a-z]+ \d{1,2}, \d{4})")
OBLIGATION_RE = re.compile(
    r"^(?:\d+\.\d+\s+)?(?P<party>(?:Each |Either |The )?[A-Z][A-Za-z]+(?: [A-Z][a-z]+)?) shall (?!not be liable)"
)
DURATION_RE = re.compile(r"(?:\w+(?:-\w+)?\s+)?\((\d+)\)\s+(day|hour|month|year)s?", re.IGNORECASE)
REQUIRED_KEYS = {"parties", "effective_date", "term", "renewal", "obligations", "deadlines"}

SYSTEM = (
    "You extract facts from contracts for a legal reviewer. Use only the text provided. "
    "Return a single JSON object and nothing else."
)
PROMPT = """Extract these fields from the contract below and return JSON with exactly these keys:
- parties: list of {{"name": str, "role": str | null}}
- effective_date: str | null
- term: str | null (one sentence quoted from the contract)
- renewal: str | null (one sentence quoted from the contract)
- obligations: list of {{"party": str, "text": str, "clause": str}} (clause like "§4 Fees And Payment")
- deadlines: list of {{"text": str, "duration": str, "clause": str}}

Contract:
{contract}"""


def extract_with_rules(title: str, clauses: list[Clause]) -> dict[str, Any]:
    full = "\n".join(c["text"] for c in clauses)
    parties = []
    if m := PARTIES_RE.search(full):
        parties = [{"name": m.group("a").strip(), "role": m.group("ra")}, {"name": m.group("b").strip(), "role": m.group("rb")}]

    term = renewal = None
    obligations, deadlines = [], []
    for clause in clauses:
        ref = clause_ref(clause)
        for s in sentences(clause["text"]):
            low = s.lower()
            if (
                term is None
                and "term" in clause["heading"].lower()
                and any(k in low for k in ("term of", "term is", "remains in effect", "continues for"))
            ):
                term = s
            if renewal is None and "renew" in low:
                renewal = s
            if clause["number"] != "0" and (m := OBLIGATION_RE.match(s)):
                obligations.append({"party": m.group("party"), "text": s, "clause": ref})
            if clause["number"] != "0" and (d := DURATION_RE.search(s)):
                n, unit = d.group(1), d.group(2).lower()
                deadlines.append({"text": s, "duration": f"{n} {unit}{'' if n == '1' else 's'}", "clause": ref})

    date = DATE_RE.search(full)
    return {
        "title": title,
        "parties": parties,
        "effective_date": date.group(1) if date else None,
        "term": term,
        "renewal": renewal,
        "obligations": obligations[:15],
        "deadlines": deadlines[:15],
        "method": "rules",
    }


def run_review(title: str, clauses: list[Clause], llm: LLM | None) -> dict[str, Any]:
    if llm is None:
        return extract_with_rules(title, clauses)
    contract = "\n\n".join(f"{clause_ref(c)}\n{c['text']}" for c in clauses)
    raw = llm.complete(SYSTEM, PROMPT.format(contract=contract), max_tokens=1500)
    try:
        data = extract_json(raw)
        missing = REQUIRED_KEYS - data.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("review agent LLM output invalid; using rule-based extraction", extra={"error": str(exc)})
        result = extract_with_rules(title, clauses)
        result["method"] = "rules-fallback"
        return result
    return {"title": title, **{k: data[k] for k in REQUIRED_KEYS}, "method": f"llm:{llm.name}"}
