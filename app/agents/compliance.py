"""Compliance agent: checks clauses against a versioned policy.

Pass/fail decisions are deterministic on purpose: the same contract and policy
version always give the same result, which is what an approval gate and an audit
need. An LLM, when configured, only adds a plain-language rationale per failure.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.contracts import Clause, clause_ref
from app.llm import LLM, TransientLLMError

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    valid_types = {"require", "prohibit", "min_number", "allowed_values"}
    for check in policy["checks"]:
        if check["type"] not in valid_types:
            raise ValueError(f"check {check['id']} has unknown type {check['type']!r}")
        if check["severity"] not in SEVERITY_ORDER:
            raise ValueError(f"check {check['id']} has unknown severity {check['severity']!r}")
    return policy


def _scope(check: dict[str, Any], clauses: list[Clause]) -> list[Clause]:
    kw = check.get("clause_keyword", "").lower()
    if not kw:
        return clauses
    return [c for c in clauses if kw in c["heading"].lower() or kw in c["text"].lower()[:200]]


def _result(check: dict[str, Any], status: str, clause: Clause | None, evidence: str) -> dict[str, Any]:
    return {
        "id": check["id"],
        "title": check["title"],
        "severity": check["severity"],
        "status": status,
        "clause": clause_ref(clause) if clause else None,
        "evidence": evidence,
    }


def evaluate(check: dict[str, Any], clauses: list[Clause]) -> dict[str, Any]:
    scoped = _scope(check, clauses)
    kind = check["type"]

    if kind == "prohibit":
        for clause in scoped:
            for pattern in check["patterns"]:
                if m := re.search(pattern, clause["text"], re.IGNORECASE):
                    return _result(check, "fail", clause, m.group(0))
        return _result(check, "pass", None, "No prohibited language found")

    if not scoped:
        missing = check.get("if_missing", "fail" if kind == "require" else "not_applicable")
        return _result(check, missing, None, "No matching clause in contract")

    if kind == "require":
        for clause in scoped:
            for pattern in check["patterns"]:
                if m := re.search(pattern, clause["text"], re.IGNORECASE):
                    return _result(check, "pass", clause, m.group(0))
        return _result(check, "fail", scoped[0], "Required language not found")

    if kind == "min_number":
        for clause in scoped:
            if m := re.search(check["pattern"], clause["text"], re.IGNORECASE):
                value = int(m.group(1))
                status = "pass" if value >= check["minimum"] else "fail"
                return _result(check, status, clause, f"{m.group(0)} (minimum {check['minimum']})")
        return _result(check, "not_applicable", None, "No matching term found")

    for clause in scoped:  # allowed_values
        if m := re.search(check["pattern"], clause["text"], re.IGNORECASE):
            value = m.group(1).strip().lower()
            status = "pass" if value in check["allowed"] else "fail"
            return _result(check, status, clause, f"{value.title()} (allowed: {', '.join(v.title() for v in check['allowed'])})")
    return _result(check, "not_applicable", None, "No matching term found")


def risk_level(results: list[dict[str, Any]]) -> str:
    failed = [SEVERITY_ORDER[r["severity"]] for r in results if r["status"] == "fail"]
    if not failed:
        return "low"
    return {3: "high", 2: "medium", 1: "low"}[max(failed)]


def run_compliance(clauses: list[Clause], policy: dict[str, Any], llm: LLM | None) -> dict[str, Any]:
    results = [evaluate(check, clauses) for check in policy["checks"]]
    if llm is not None:
        for r in results:
            if r["status"] != "fail":
                continue
            try:
                r["rationale"] = llm.complete(
                    "You explain contract policy exceptions to business stakeholders in two sentences. No legal advice.",
                    f"Policy: {r['title']}\nClause: {r['clause']}\nEvidence: {r['evidence']}\nExplain the business risk.",
                    max_tokens=200,
                )
            except TransientLLMError:
                raise
            except Exception:
                logger.warning("compliance rationale generation failed; continuing without it", exc_info=True)
    level = risk_level(results)
    return {
        "policy_version": policy["version"],
        "risk_level": level,
        "failed": sum(r["status"] == "fail" for r in results),
        "results": results,
        "requires_approval": level in policy["approval_required_levels"],
    }
