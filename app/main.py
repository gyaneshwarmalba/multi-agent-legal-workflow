"""Streamlit UI: `streamlit run app/main.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.contracts import contract_id_for  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.workflow import LegalWorkflow  # noqa: E402

configure_logging()
st.set_page_config(page_title="Multi-Agent Legal Workflow", layout="wide")


@st.cache_resource
def workflow() -> LegalWorkflow:
    return LegalWorkflow()


wf = workflow()
st.title("Multi-Agent Legal Workflow")
llm_label = wf.llm.name if wf.llm else "local rules"
st.caption(f"Agents: review · compliance · summary · human approval  |  LLM: `{llm_label}`  |  Policy `{wf.policy['version']}`")

samples = sorted(wf.settings.sample_contracts_dir.glob("*.txt"))
with st.sidebar:
    st.header("Start a review")
    choice = st.selectbox("Sample contract", [p.name for p in samples])
    if st.button("Run workflow", type="primary"):
        path = wf.settings.sample_contracts_dir / choice
        st.session_state.thread_id = wf.start(contract_id_for(path), path.read_text(encoding="utf-8")).thread_id
    st.divider()
    lookup = st.text_input("Open existing thread id")
    if st.button("Open") and lookup:
        st.session_state.thread_id = lookup.strip()

thread_id = st.session_state.get("thread_id")
if not thread_id:
    st.info("Pick a sample contract and run the workflow.")
    st.stop()

try:
    result = wf.get(thread_id)
except KeyError:
    st.error(f"Unknown thread {thread_id}")
    st.stop()

state = result.state
st.subheader(f"{state.get('title', '')} · `{thread_id}`")
c1, c2, c3 = st.columns(3)
c1.metric("Status", result.status.replace("_", " ").title())
c2.metric("Risk", (state.get("compliance", {}).get("risk_level") or "-").upper())
c3.metric("Failed checks", state.get("compliance", {}).get("failed", 0))

if result.pending_approval:
    st.warning(f"Awaiting human approval: {result.pending_approval['reason']}")
    reviewer = st.text_input("Reviewer", "legal.reviewer")
    note = st.text_input("Note")
    a, r = st.columns(2)
    if a.button("Approve", type="primary"):
        wf.resume(thread_id, True, reviewer, note or None)
        st.rerun()
    if r.button("Reject"):
        wf.resume(thread_id, False, reviewer, note or None)
        st.rerun()

summary_tab, compliance_tab, review_tab, trace_tab = st.tabs(["Summary", "Compliance", "Extracted terms", "Agent trace"])
with summary_tab:
    st.markdown(state.get("summary", "_not yet generated_").replace("\n", "  \n"))
with compliance_tab:
    st.dataframe(state.get("compliance", {}).get("results", []), use_container_width=True)
with review_tab:
    review = state.get("review", {})
    st.json({k: review.get(k) for k in ("parties", "effective_date", "term", "renewal", "method")})
    st.write("**Obligations**")
    st.dataframe(review.get("obligations", []), use_container_width=True)
    st.write("**Deadlines and notice periods**")
    st.dataframe(review.get("deadlines", []), use_container_width=True)
with trace_tab:
    st.dataframe(state.get("trace", []), use_container_width=True)
