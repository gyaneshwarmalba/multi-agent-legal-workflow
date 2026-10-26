import json

from app.agents.compliance import evaluate, load_policy, run_compliance
from app.agents.review import extract_with_rules, run_review
from app.config import Settings
from app.contracts import parse_contract
from tests.conftest import sample


def parsed(name):
    return parse_contract(sample(name))


def test_review_extracts_parties_dates_and_obligations():
    title, clauses = parsed("acme_master_services_agreement")
    review = extract_with_rules(title, clauses)
    assert review["parties"] == [
        {"name": "Acme Analytics LLC", "role": "Provider"},
        {"name": "Northwind Retail Inc.", "role": "Customer"},
    ]
    assert review["effective_date"] == "March 1, 2025"
    assert "two (2) years" in review["term"]
    assert "automatically renew" in review["renewal"]
    assert {"party": "Customer", "clause": "§4 Fees And Payment"}.items() <= next(
        o for o in review["obligations"] if "pay all invoices" in o["text"]
    ).items()
    assert any(d["duration"] == "15 days" and d["clause"] == "§4 Fees And Payment" for d in review["deadlines"])


def test_review_handles_parties_without_defined_roles():
    title, clauses = parsed("globex_mutual_nda")
    review = extract_with_rules(title, clauses)
    assert [p["name"] for p in review["parties"]] == ["Globex Corporation", "Northwind Retail Inc."]


class FakeLLM:
    name = "fake"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system, prompt, max_tokens=800):
        self.calls += 1
        return self.response


def test_review_uses_valid_llm_json():
    payload = {"parties": [], "effective_date": "x", "term": "t", "renewal": None, "obligations": [], "deadlines": []}
    title, clauses = parsed("globex_mutual_nda")
    review = run_review(title, clauses, FakeLLM("Here you go:\n```json\n" + json.dumps(payload) + "\n```"))
    assert review["method"] == "llm:fake" and review["effective_date"] == "x"


def test_review_falls_back_to_rules_on_malformed_llm_output():
    title, clauses = parsed("globex_mutual_nda")
    review = run_review(title, clauses, FakeLLM('{"parties": []}'))
    assert review["method"] == "rules-fallback"
    assert review["effective_date"] == "January 15, 2025"


def statuses(name):
    policy = load_policy(Settings().policy_path)
    return {r["id"]: r["status"] for r in run_compliance(parsed(name)[1], policy, None)["results"]}


def test_compliance_flags_one_sided_msa():
    result = statuses("acme_master_services_agreement")
    assert all(v == "fail" for v in result.values()), result


def test_compliance_passes_balanced_saas_agreement():
    result = statuses("initech_saas_subscription")
    assert set(result.values()) == {"pass"}, result


def test_missing_optional_clause_is_not_applicable():
    result = statuses("globex_mutual_nda")
    assert result["BREACH-NOTICE"] == "not_applicable"
    assert result["CAP-LIABILITY"] == "pass"


def test_check_types():
    clauses = [{"number": "4", "heading": "Fees", "text": "Invoices are payable within sixty (60) days."}]
    check = {
        "id": "P",
        "title": "t",
        "severity": "low",
        "type": "min_number",
        "clause_keyword": "payable",
        "pattern": r"within [\w-]+ \((\d+)\) days",
        "minimum": 30,
    }
    assert evaluate(check, clauses)["status"] == "pass"
    missing = {"id": "C", "title": "t", "severity": "high", "type": "require", "clause_keyword": "liabilit", "patterns": ["cap"]}
    assert evaluate(missing, clauses)["status"] == "fail"
