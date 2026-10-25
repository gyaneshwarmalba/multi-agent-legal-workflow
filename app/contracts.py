"""Clause parsing shared by all agents, so every finding can cite a section number."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

HEADING_RE = re.compile(
    r"^\s*(?:(?P<num>\d{1,2})\.\s+(?P<title>[A-Z][A-Z0-9 ,&'/()\-]{2,80})"
    r"|(?:Section|SECTION)\s+(?P<snum>\d{1,2}(?:\.\d+)?)[.:]?\s+(?P<stitle>.{2,80}))\s*$"
)


class Clause(TypedDict):
    number: str
    heading: str
    text: str


def parse_contract(text: str) -> tuple[str, list[Clause]]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    title = next((ln for ln in lines if ln), "Untitled contract")
    clauses: list[Clause] = [{"number": "0", "heading": "Preamble", "text": ""}]
    for line in lines:
        if not line or line == title:
            continue
        m = HEADING_RE.match(line)
        if m:
            number = m.group("num") or m.group("snum")
            heading = (m.group("title") or m.group("stitle")).strip()
            clauses.append({"number": number, "heading": heading.title() if heading.isupper() else heading, "text": ""})
        else:
            clauses[-1]["text"] = f"{clauses[-1]['text']}\n{line}".strip()
    return title, [c for c in clauses if c["text"]]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z0-9\"(])", text) if s.strip()]


def clause_ref(clause: Clause) -> str:
    return f"§{clause['number']} {clause['heading']}"


def contract_id_for(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "contract"
