"""Templates: planner context drives personalisation; unknown types degrade."""
from __future__ import annotations

import json

from actions.templates import NEXT_PLAN, build_template


CUSTOMER = {"customer_id": "C0042", "name": "Ada Lovelace", "plan": "Starter",
            "mrr": 49, "health_score": 60, "tenure_days": 400}


def _action(atype: str, context: dict | None = None, raw_metadata=None) -> dict:
    meta = raw_metadata if raw_metadata is not None else json.dumps(
        {"context": context or {}}
    )
    return {"action_type": atype, "metadata": meta, "reason": "triggered"}


def test_upgrade_offer_uses_suggested_plan_from_metadata():
    action = _action("send_upgrade_offer", context={"suggested_plan": "Enterprise"})
    t = build_template(action, dict(CUSTOMER, plan="Pro"), {})
    assert "Enterprise" in t.subject
    assert t.channel == "email"


def test_upgrade_offer_defaults_to_next_plan_tier():
    assert NEXT_PLAN["Free"] == "Starter"
    assert NEXT_PLAN["Starter"] == "Pro"
    assert NEXT_PLAN["Pro"] == "Enterprise"

    t = build_template(_action("send_upgrade_offer"), CUSTOMER, {})
    assert "Pro" in t.subject            # Starter → next tier up


def test_unparseable_metadata_does_not_crash():
    t = build_template(_action("send_upgrade_offer", raw_metadata="{not json"),
                       CUSTOMER, {})
    assert "Pro" in t.subject


def test_missing_metadata_key_does_not_crash():
    t = build_template({"action_type": "send_upgrade_offer"}, CUSTOMER, {})
    assert t.action_type == "send_upgrade_offer"


def test_unknown_action_type_falls_back_to_generic_task():
    t = build_template(_action("teleport_customer"), CUSTOMER, {})
    assert t.channel == "internal_task"
    assert "teleport_customer" in t.subject


def test_csm_escalation_is_internal_channel_with_scores():
    t = build_template(_action("escalate_to_csm"), CUSTOMER,
                       {"churn_score": 0.91, "ltv_estimate": 5000})
    assert t.channel == "slack_csm"
    assert "91%" in t.subject
