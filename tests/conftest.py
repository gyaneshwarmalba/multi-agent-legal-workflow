from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.workflow import LegalWorkflow

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_contracts"


def sample(name: str) -> str:
    return (SAMPLES / f"{name}.txt").read_text(encoding="utf-8")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(llm_provider="local", checkpoint_db=tmp_path / "checkpoints.sqlite")


@pytest.fixture()
def workflow(settings) -> LegalWorkflow:
    return LegalWorkflow(settings, checkpointer=InMemorySaver())
