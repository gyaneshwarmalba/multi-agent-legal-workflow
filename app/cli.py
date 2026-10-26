"""Command-line entry point.

    python -m app.cli run data/sample_contracts/acme_master_services_agreement.txt
    python -m app.cli resume <thread_id> --approve --reviewer alice --note "escalated"
    python -m app.cli show <thread_id>

State is checkpointed to SQLite, so `resume` works from a new process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.contracts import contract_id_for
from app.logging_config import configure_logging
from app.workflow import LegalWorkflow, RunResult


def _print(result: RunResult, verbose: bool) -> None:
    out = {"thread_id": result.thread_id, "status": result.status}
    if result.pending_approval:
        out["pending_approval"] = {
            k: result.pending_approval[k] for k in ("reason", "failed_checks", "errors") if result.pending_approval.get(k)
        }
    if result.state.get("summary"):
        out["summary"] = result.state["summary"]
    if verbose:
        out["trace"] = result.state.get("trace", [])
        out["review"] = result.state.get("review")
        out["compliance"] = result.state.get("compliance")
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multi-agent-legal-workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start a workflow for a contract file")
    run.add_argument("path", type=Path)
    run.add_argument("-v", "--verbose", action="store_true")
    res = sub.add_parser("resume", help="record a human decision for a paused workflow")
    res.add_argument("thread_id")
    decision = res.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    res.add_argument("--reviewer", required=True)
    res.add_argument("--note")
    show = sub.add_parser("show", help="show the state of a workflow")
    show.add_argument("thread_id")
    show.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    wf = LegalWorkflow()
    try:
        if args.command == "run":
            result = wf.start(contract_id_for(args.path), args.path.read_text(encoding="utf-8"))
        elif args.command == "resume":
            result = wf.resume(args.thread_id, approve=args.approve, reviewer=args.reviewer, note=args.note)
        else:
            result = wf.get(args.thread_id)
    except KeyError as exc:
        print(f"error: unknown thread {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result, getattr(args, "verbose", False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
