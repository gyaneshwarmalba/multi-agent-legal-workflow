from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parent.parent / "app" / "main.py"


def test_ui_runs_workflow_and_approves(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ui.sqlite"))
    at = AppTest.from_file(str(UI), default_timeout=30).run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "Awaiting Approval"
    next(b for b in at.button if b.label == "Approve").click().run()
    assert at.metric[0].value == "Approved"
