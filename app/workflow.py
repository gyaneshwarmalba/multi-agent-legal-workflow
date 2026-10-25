"""LangGraph workflow: a supervisor routes work between specialist agents.

    START -> intake -> supervisor -> review_agent      -> supervisor
                                  -> compliance_agent  -> supervisor
                                  -> summary_agent     -> supervisor
                                  -> human_approval    -> supervisor   (interrupt; resumes with a decision)
                                  -> finalize -> END

The supervisor is deterministic: routing is a pure function of state, so every
run is reproducible and auditable. Agents may use an LLM; the control flow does
not.
"""

from __future__ import annotations

import logging
import operator
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.agents.compliance import load_policy, run_compliance
from app.agents.review import run_review
from app.agents.summary import run_summary
from app.config import Settings
from app.contracts import Clause, parse_contract
from app.llm import LLM, BedrockLLM, TransientLLMError

logger = logging.getLogger(__name__)

AGENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, retry_on=TransientLLMError)


class WorkflowState(TypedDict, total=False):
    contract_id: str
    contract_text: str
    title: str
    clauses: list[Clause]
    review: dict[str, Any]
    compliance: dict[str, Any]
    summary: str
    approval: dict[str, Any]
    next: str
    status: str
    report: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def _step(agent: str, detail: str, started: float | None = None) -> dict[str, Any]:
    return {
        "agent": agent,
        "at": datetime.now(UTC).isoformat(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1) if started is not None else 0.0,
        "detail": detail,
    }


def route(state: WorkflowState) -> str:
    """Supervisor policy. Pure function of state, unit-tested directly."""
    if state.get("errors"):
        return "finalize" if "approval" in state else "human_approval"
    if "review" not in state:
        return "review_agent"
    if "compliance" not in state:
        return "compliance_agent"
    if "summary" not in state:
        return "summary_agent"
    if state["compliance"].get("requires_approval") and "approval" not in state:
        return "human_approval"
    return "finalize"


def build_graph(llm: LLM | None, policy: dict[str, Any]) -> StateGraph:
    def guarded(agent: str, fn):
        """Record non-retryable agent failures in state instead of crashing the run."""

        def node(state: WorkflowState) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                update, detail = fn(state)
            except TransientLLMError:
                raise
            except Exception as exc:
                logger.exception("agent failed", extra={"agent": agent, "contract_id": state.get("contract_id")})
                return {
                    "errors": [{"agent": agent, "error": f"{type(exc).__name__}: {exc}"}],
                    "trace": [_step(agent, f"failed: {exc}", started)],
                }
            logger.info("agent completed", extra={"agent": agent, "contract_id": state.get("contract_id"), "detail": detail})
            return {**update, "trace": [_step(agent, detail, started)]}

        return node

    def intake(state: WorkflowState) -> dict[str, Any]:
        started = time.perf_counter()
        title, clauses = parse_contract(state["contract_text"])
        if len(clauses) < 2:
            raise ValueError("contract has no recognisable clause headings")
        return {
            "title": title,
            "clauses": clauses,
            "status": "running",
            "trace": [_step("intake", f"parsed {len(clauses)} clauses", started)],
        }

    def supervisor(state: WorkflowState) -> dict[str, Any]:
        nxt = route(state)
        return {"next": nxt, "trace": [_step("supervisor", f"-> {nxt}")]}

    def review(state: WorkflowState):
        result = run_review(state["title"], state["clauses"], llm)
        detail = f"{len(result['obligations'])} obligations, {len(result['deadlines'])} deadlines ({result['method']})"
        return {"review": result}, detail

    def compliance(state: WorkflowState):
        result = run_compliance(state["clauses"], policy, llm)
        return {"compliance": result}, f"risk {result['risk_level']}, {result['failed']} failed checks"

    def summary(state: WorkflowState):
        return {"summary": run_summary(state["review"], state["compliance"], llm)}, "summary written"

    def human_approval(state: WorkflowState) -> dict[str, Any]:
        compliance_state = state.get("compliance", {})
        decision = interrupt(
            {
                "contract_id": state["contract_id"],
                "title": state.get("title"),
                "reason": "agent errors" if state.get("errors") else f"{compliance_state.get('risk_level')} risk",
                "summary": state.get("summary"),
                "failed_checks": [r for r in compliance_state.get("results", []) if r["status"] == "fail"],
                "errors": state.get("errors", []),
            }
        )
        approval = {
            "approved": bool(decision.get("approve")),
            "reviewer": decision.get("reviewer", "unknown"),
            "note": decision.get("note"),
            "decided_at": datetime.now(UTC).isoformat(),
        }
        verdict = "approved" if approval["approved"] else "rejected"
        return {"approval": approval, "trace": [_step("human_approval", f"{verdict} by {approval['reviewer']}")]}

    def finalize(state: WorkflowState) -> dict[str, Any]:
        approval = state.get("approval")
        if state.get("errors"):
            status = "failed"
        elif approval:
            status = "approved" if approval["approved"] else "rejected"
        else:
            status = "auto_approved"
        report = {
            "contract_id": state["contract_id"],
            "title": state.get("title"),
            "status": status,
            "risk_level": state.get("compliance", {}).get("risk_level"),
            "policy_version": policy["version"],
            "summary": state.get("summary"),
            "approval": approval,
        }
        return {"status": status, "report": report, "trace": [_step("finalize", status)]}

    graph = StateGraph(WorkflowState)
    graph.add_node("intake", intake)
    graph.add_node("supervisor", supervisor)
    graph.add_node("review_agent", guarded("review_agent", review), retry_policy=AGENT_RETRY)
    graph.add_node("compliance_agent", guarded("compliance_agent", compliance), retry_policy=AGENT_RETRY)
    graph.add_node("summary_agent", guarded("summary_agent", summary), retry_policy=AGENT_RETRY)
    graph.add_node("human_approval", human_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        ["review_agent", "compliance_agent", "summary_agent", "human_approval", "finalize"],
    )
    for agent in ("review_agent", "compliance_agent", "summary_agent", "human_approval"):
        graph.add_edge(agent, "supervisor")
    graph.add_edge("finalize", END)
    return graph


@dataclass
class RunResult:
    thread_id: str
    status: str
    state: dict[str, Any]
    pending_approval: dict[str, Any] | None


class LegalWorkflow:
    def __init__(
        self, settings: Settings | None = None, *, llm: LLM | None = None, checkpointer: BaseCheckpointSaver | None = None
    ) -> None:
        self.settings = settings or Settings()
        if llm is None and self.settings.llm_provider == "bedrock":
            llm = BedrockLLM(self.settings.bedrock_model_id, self.settings.aws_region)
        elif self.settings.llm_provider not in {"local", "bedrock"}:
            raise ValueError(f"unknown LLM_PROVIDER {self.settings.llm_provider!r}")
        self.llm = llm
        self.policy = load_policy(self.settings.policy_path)
        self.checkpointer = checkpointer or self._sqlite_checkpointer()
        self.app = build_graph(llm, self.policy).compile(checkpointer=self.checkpointer)

    def _sqlite_checkpointer(self) -> BaseCheckpointSaver:
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(self.settings.checkpoint_db), check_same_thread=False))

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str) -> RunResult:
        snapshot = self.app.get_state(self._config(thread_id))
        if not snapshot.values:
            raise KeyError(thread_id)
        pending = next((i.value for task in snapshot.tasks for i in task.interrupts), None)
        status = "awaiting_approval" if pending else snapshot.values.get("status", "running")
        return RunResult(thread_id=thread_id, status=status, state=dict(snapshot.values), pending_approval=pending)

    def start(self, contract_id: str, contract_text: str, thread_id: str | None = None) -> RunResult:
        if len(parse_contract(contract_text)[1]) < 2:
            raise ValueError("contract has no recognisable clause headings (expected e.g. '1. TERM' or 'Section 2 Fees')")
        thread_id = thread_id or f"{contract_id}-{uuid.uuid4().hex[:8]}"
        logger.info("workflow started", extra={"thread_id": thread_id, "contract_id": contract_id})
        self.app.invoke({"contract_id": contract_id, "contract_text": contract_text}, self._config(thread_id))
        return self._result(thread_id)

    def resume(self, thread_id: str, approve: bool, reviewer: str, note: str | None = None) -> RunResult:
        current = self._result(thread_id)
        if current.status != "awaiting_approval":
            raise ValueError(f"thread {thread_id} is {current.status}, not awaiting approval")
        self.app.invoke(Command(resume={"approve": approve, "reviewer": reviewer, "note": note}), self._config(thread_id))
        return self._result(thread_id)

    def get(self, thread_id: str) -> RunResult:
        return self._result(thread_id)
