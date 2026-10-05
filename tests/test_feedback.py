"""Feedback loop must fold each execution into health scores exactly once."""
from __future__ import annotations

from actions.feedback import apply_feedback
from data.db import connect

YEAR_HOURS = 24 * 365


def _health(cid: str) -> float:
    conn = connect()
    row = conn.execute("SELECT health_score FROM customers WHERE customer_id=?", (cid,)).fetchone()
    conn.close()
    return row["health_score"]


def _insert_executed(action_id: str, cid: str, outcome: str) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO agent_actions (action_id, customer_id, action_type, reason, "
        "priority, status, outcome, created_at, executed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (action_id, cid, "send_retention_email", "r", 4, "executed", outcome,
         "2026-10-05 12:00:00", "2026-10-05 12:00:00"),
    )
    conn.commit()
    conn.close()


def _cleanup(action_id: str) -> None:
    conn = connect()
    conn.execute("DELETE FROM agent_actions WHERE action_id=?", (action_id,))
    conn.commit()
    conn.close()


def test_health_delta_applied_once_not_per_call(acia_db):
    cid = "C0001"
    apply_feedback(lookback_hours=YEAR_HOURS, verbose=False)   # drain backlog
    before = _health(cid)

    _insert_executed("fb-1", cid, "upgraded")      # +10 health

    first = apply_feedback(lookback_hours=YEAR_HOURS, verbose=False)
    assert first["executions_processed"] >= 1
    assert _health(cid) == min(100.0, before + 10.0)

    # Re-running the feedback loop must not double-count the same outcome
    second = apply_feedback(lookback_hours=YEAR_HOURS, verbose=False)
    assert second == {}
    assert _health(cid) == min(100.0, before + 10.0)

    _cleanup("fb-1")


def test_negative_outcome_lowers_health_and_clamps_at_zero(acia_db):
    cid = "C0002"
    apply_feedback(lookback_hours=YEAR_HOURS, verbose=False)   # drain backlog
    conn = connect()
    conn.execute("UPDATE customers SET health_score=3 WHERE customer_id=?", (cid,))
    conn.commit()
    conn.close()

    _insert_executed("fb-2", cid, "call_completed_churned")   # −10 health
    apply_feedback(lookback_hours=YEAR_HOURS, verbose=False)

    assert _health(cid) == 0.0      # clamped, never negative
    _cleanup("fb-2")
