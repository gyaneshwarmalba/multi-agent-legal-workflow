import pytest

from app.cli import main as cli_main
from app.llm import TransientLLMError
from app.workflow import LegalWorkflow, route
from tests.conftest import sample


def agents_run(result):
    return [t["agent"] for t in result.state["trace"] if t["agent"] != "supervisor"]


def test_high_risk_contract_pauses_for_approval_then_completes(workflow):
    result = workflow.start("acme", sample("acme_master_services_agreement"))
    assert result.status == "awaiting_approval"
    assert result.pending_approval["reason"] == "high risk"
    assert len(result.pending_approval["failed_checks"]) == 8
    assert agents_run(result) == ["intake", "review_agent", "compliance_agent", "summary_agent"]

    done = workflow.resume(result.thread_id, approve=False, reviewer="alice", note="renegotiate liability")
    assert done.status == "rejected"
    assert done.state["report"]["approval"]["reviewer"] == "alice"
    assert agents_run(done)[-2:] == ["human_approval", "finalize"]

    with pytest.raises(ValueError):
        workflow.resume(result.thread_id, approve=True, reviewer="bob")


def test_low_risk_contract_is_auto_approved(workflow):
    result = workflow.start("initech", sample("initech_saas_subscription"))
    assert result.status == "auto_approved"
    assert result.pending_approval is None
    assert "human_approval" not in agents_run(result)


def test_route_is_a_pure_function_of_state():
    assert route({}) == "review_agent"
    assert route({"review": {}}) == "compliance_agent"
    assert route({"review": {}, "compliance": {"requires_approval": True}}) == "summary_agent"
    assert route({"review": {}, "compliance": {"requires_approval": True}, "summary": "s"}) == "human_approval"
    assert route({"review": {}, "compliance": {"requires_approval": True}, "summary": "s", "approval": {}}) == "finalize"
    assert route({"errors": [{"agent": "x"}]}) == "human_approval"
    assert route({"errors": [{"agent": "x"}], "approval": {}}) == "finalize"


class BrokenLLM:
    name = "broken"

    def complete(self, system, prompt, max_tokens=800):
        raise RuntimeError("model endpoint misconfigured")


def test_agent_failure_escalates_to_human_instead_of_crashing(settings):
    from langgraph.checkpoint.memory import InMemorySaver

    wf = LegalWorkflow(settings, llm=BrokenLLM(), checkpointer=InMemorySaver())
    result = wf.start("nda", sample("globex_mutual_nda"))
    assert result.status == "awaiting_approval"
    assert result.pending_approval["reason"] == "agent errors"
    assert result.state["errors"][0]["agent"] == "review_agent"
    done = wf.resume(result.thread_id, approve=True, reviewer="ops")
    assert done.status == "failed"


class FlakyLLM:
    """Throttles twice, then returns invalid JSON so the review agent falls back to rules."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt, max_tokens=800):
        self.calls += 1
        if self.calls <= 2:
            raise TransientLLMError("ThrottlingException")
        return "not json"


def test_transient_llm_errors_are_retried(settings, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import RetryPolicy

    import app.workflow as wf_module

    monkeypatch.setattr(
        wf_module, "AGENT_RETRY", RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False, retry_on=TransientLLMError)
    )
    llm = FlakyLLM()
    wf = LegalWorkflow(settings, llm=llm, checkpointer=InMemorySaver())
    result = wf.start("nda", sample("globex_mutual_nda"))
    assert result.state["review"]["method"] == "rules-fallback"
    assert llm.calls >= 3


def test_rejects_text_without_clauses(workflow):
    with pytest.raises(ValueError, match="clause headings"):
        workflow.start("x", "just some prose without headings")


def test_unknown_thread(workflow):
    with pytest.raises(KeyError):
        workflow.get("missing")


def test_cli_run_and_resume_across_processes(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cli.sqlite"))
    path = tmp_path / "acme.txt"
    path.write_text(sample("acme_master_services_agreement"))
    assert cli_main(["run", str(path)]) == 0
    started = json.loads(capsys.readouterr().out)
    assert started["status"] == "awaiting_approval"

    assert cli_main(["resume", started["thread_id"], "--approve", "--reviewer", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "approved"
    assert cli_main(["show", "nope"]) == 2
