"""Scheduler: metadata + urgency score persist, cooldown blocks re-queueing,
queue ordering follows the composite score. Uses real customer ids — the DB
enforces the foreign key to `customers`."""
from __future__ import annotations

import json

from agent.planner import PlannedAction
from agent.scheduler import _score_action, get_pending_queue, schedule
from data.db import connect


def _plan(cid="C0101", action="send_retention_email", priority=4, confidence=0.8,
          context=None) -> PlannedAction:
    return PlannedAction(
        customer_id=cid, action_type=action, reason="because", priority=priority,
        rule_id="CHN-002", llm_enhanced=False, confidence=confidence,
        metadata={"triggered_by": {"churn_score": 0.6}, "context": context or {}},
    )


def _reset(*cids: str) -> None:
    """Drop any history for these customers so cooldown state can't leak in
    from tests that ran earlier."""
    conn = connect()
    for cid in cids:
        conn.execute("DELETE FROM agent_actions WHERE customer_id=?", (cid,))
    conn.commit()
    conn.close()


def test_score_increases_with_priority_and_ltv():
    low = _score_action(_plan(priority=2), ltv=0)
    high = _score_action(_plan(priority=5), ltv=10_000)
    assert high > low


def test_schedule_persists_metadata_and_urgency_score(acia_db):
    _reset("C0101")
    plans = [_plan("C0101", context={"suggested_plan": "Pro"})]
    queued = schedule(plans, {"C0101": 1500.0}, verbose=False)

    assert len(queued) == 1
    conn = connect()
    row = conn.execute(
        "SELECT * FROM agent_actions WHERE customer_id='C0101'"
    ).fetchone()
    conn.close()

    meta = json.loads(row["metadata"])
    assert meta["context"]["suggested_plan"] == "Pro"
    assert row["urgency_score"] is not None
    assert row["urgency_score"] == queued[0]["_score"]


def test_queue_is_ordered_by_urgency_score(acia_db):
    _reset("C0102", "C0103", "C0104")
    plans = [
        _plan("C0102", priority=1, confidence=0.5),
        _plan("C0103", priority=5, confidence=0.9),
        _plan("C0104", priority=3, confidence=0.7),
    ]
    schedule(plans, {"C0102": 0, "C0103": 100, "C0104": 0}, verbose=False)

    queue = [a for a in get_pending_queue()
             if a["customer_id"] in {"C0102", "C0103", "C0104"}]
    assert [a["customer_id"] for a in queue] == ["C0103", "C0104", "C0102"]
    scores = [a["urgency_score"] for a in queue]
    assert scores == sorted(scores, reverse=True)


def test_recently_executed_action_is_cooldown_blocked(acia_db):
    cid = "C0105"
    _reset(cid)
    conn = connect()
    conn.execute(
        "INSERT INTO agent_actions (action_id, customer_id, action_type, reason, "
        "priority, status, outcome, created_at, executed_at) "
        "VALUES ('cd-1',?, 'send_retention_email','r',4,'executed','opened_only',"
        "datetime('now'), datetime('now'))",
        (cid,),
    )
    conn.commit()
    conn.close()

    plans = [
        _plan(cid, action="send_retention_email"),   # blocked: cooldown
        _plan(cid, action="send_loyalty_reward"),    # allowed: different type
    ]
    queued = schedule(plans, {cid: 100.0}, verbose=False)
    assert [a["action_type"] for a in queued] == ["send_loyalty_reward"]


def test_rescheduling_replaces_previous_pending_rows(acia_db):
    _reset("C0106", "C0107")
    schedule([_plan("C0106")], {"C0106": 0}, verbose=False)
    schedule([_plan("C0107")], {"C0107": 0}, verbose=False)

    queue = get_pending_queue()
    ids = {a["customer_id"] for a in queue}
    assert "C0106" not in ids
    assert "C0107" in ids
