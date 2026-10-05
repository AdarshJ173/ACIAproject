"""Executor: outcomes are persisted once per batch, emails are logged,
dry runs never write."""
from __future__ import annotations

from actions.executor import execute_batch
from agent.planner import PlannedAction
from agent.scheduler import schedule
from data.db import connect


def _plan(cid: str, action: str) -> PlannedAction:
    return PlannedAction(
        customer_id=cid, action_type=action, reason="because", priority=4,
        rule_id="CHN-002", llm_enhanced=False, confidence=0.8,
        metadata={"triggered_by": {}, "context": {}},
    )


def _reset(cid: str) -> None:
    conn = connect()
    conn.execute("DELETE FROM agent_actions WHERE customer_id=?", (cid,))
    conn.commit()
    conn.close()


def _count_email_log() -> int:
    conn = connect()
    n = conn.execute("SELECT COUNT(*) FROM email_log").fetchone()[0]
    conn.close()
    return n


def _action_row(cid: str):
    conn = connect()
    row = conn.execute(
        "SELECT status, outcome, executed_at, feedback_applied_at "
        "FROM agent_actions WHERE customer_id=?", (cid,)
    ).fetchone()
    conn.close()
    return row


def test_execute_batch_persists_outcome_and_email_log(acia_db):
    _reset("C0108")
    schedule([_plan("C0108", "send_retention_email")], {"C0108": 10}, verbose=False)
    emails_before = _count_email_log()

    results = execute_batch(limit=500, dry_run=False, verbose=False)
    mine = [r for r in results if r.customer_id == "C0108"]
    assert len(mine) == 1
    assert mine[0].outcome
    assert mine[0].status in {"success", "failure", "neutral"}

    row = _action_row("C0108")
    assert row["status"] == "executed"
    assert row["outcome"] == mine[0].outcome
    assert row["executed_at"]
    assert row["feedback_applied_at"] is None      # feedback loop hasn't run yet
    assert _count_email_log() > emails_before      # email actions are logged


def test_dry_run_never_writes(acia_db):
    _reset("C0109")
    schedule([_plan("C0109", "send_health_checkin")], {"C0109": 10}, verbose=False)

    results = execute_batch(limit=500, dry_run=True, verbose=False)
    assert any(r.customer_id == "C0109" for r in results)

    row = _action_row("C0109")
    assert row["status"] == "pending"
    assert row["outcome"] is None
    assert row["executed_at"] is None
