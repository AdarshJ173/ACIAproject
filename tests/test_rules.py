"""Rule engine: every rule fires on its trigger, nothing fires spuriously,
and the registry priority matches the priority each rule emits."""
from __future__ import annotations

import pandas as pd

from agent.rules import RULES, _priority_mismatches, run_rules, get_enriched_df


def _row(**overrides) -> dict:
    """Row that triggers NO rule; tests override just the signals they need."""
    base = {
        "customer_id": "C9001",
        "churn_score": 0.10,
        "conversion_score": 0.10,
        "mrr": 10.0,
        "plan": "Pro",
        "plan_rank": 2,
        "segment": "Loyal",
        "health_score": 80.0,
        "nps_score": 30,
        "tenure_days": 500,
        "days_since_last_event": 5,
        "event_count_30d": 30,
        "cancellation_risk": 0,
        "avg_sentiment": 0.5,
        "ticket_count": 1,
        "open_tickets": 0,
        "engagement_index": 42.0,
        "ltv_estimate": 100.0,
    }
    base.update(overrides)
    return base


# rule_id → (expected action_type, row that triggers it)
TRIGGER_CASES = {
    "CHN-001": ("escalate_to_csm",     dict(churn_score=0.85, mrr=200)),
    "CHN-002": ("send_retention_email", dict(churn_score=0.60)),
    "CHN-003": ("send_reengagement_email", dict(days_since_last_event=40, churn_score=0.40)),
    "CHN-004": ("proactive_support_call", dict(cancellation_risk=1)),
    "CNV-001": ("send_upgrade_offer",   dict(conversion_score=0.75, plan="Free", plan_rank=0)),
    "CNV-002": ("enroll_nurture_sequence", dict(conversion_score=0.50, plan="Free", plan_rank=0)),
    "CNV-003": ("send_upgrade_offer",   dict(plan="Starter", plan_rank=1,
                                             event_count_30d=25, conversion_score=0.60)),
    "LYL-001": ("send_loyalty_reward",  dict(segment="Champion", tenure_days=200)),
    "LYL-002": ("send_health_checkin",  dict(health_score=30.0, churn_score=0.20)),
}


def test_every_registered_rule_has_a_trigger_case():
    assert {r["id"] for r in RULES} == set(TRIGGER_CASES)


def test_default_row_fires_no_rules():
    assert run_rules(pd.DataFrame([_row()])) == []


def test_each_rule_fires_expected_action_and_priority():
    rows, expected = [], {}
    for rule_id, (action_type, overrides) in TRIGGER_CASES.items():
        rows.append(_row(customer_id=f"C-{rule_id}", **overrides))
        expected[rule_id] = action_type

    candidates = run_rules(pd.DataFrame(rows))
    fired = {c.rule_id: c for c in candidates}

    for rule_id, action_type in expected.items():
        assert rule_id in fired, f"{rule_id} did not fire"
        assert fired[rule_id].action_type == action_type
        assert 1 <= fired[rule_id].priority <= 5


def test_registry_priority_matches_emitted_priority():
    """A decorator/candidate priority mismatch silently corrupts queue labels."""
    rows = [_row(customer_id=f"C-{rid}", **ov)
            for rid, (_at, ov) in TRIGGER_CASES.items()]
    candidates = run_rules(pd.DataFrame(rows))
    assert _priority_mismatches(candidates) == {}


def test_enriched_frame_is_complete_and_nan_free(acia_db):
    df = get_enriched_df()
    assert len(df) == 500
    for col in ("churn_score", "conversion_score", "ltv_estimate", "segment"):
        assert not df[col].isna().any(), f"{col} contains NaN"
    # LTV feeds the scheduler's urgency score — NaN would break queue ordering
    assert (df["ltv_estimate"] >= 0).all()
