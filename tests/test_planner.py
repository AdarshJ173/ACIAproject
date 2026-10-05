"""LLM planner: untrusted model output must be validated before it can reach
the action queue, budget must be respected, and rules stay the fallback."""
from __future__ import annotations

import pytest

from agent import planner
from agent.planner import (
    ACTION_TYPES,
    NO_ACTION,
    PlannedAction,
    _validate_llm_decision,
    plan_actions,
)
from agent.rules import ActionCandidate


def _cand(cid="C1", action="send_retention_email", priority=4, rule="CHN-002",
          context=None) -> ActionCandidate:
    return ActionCandidate(
        customer_id=cid, action_type=action, reason=f"rule fired: {action}",
        priority=priority, rule_id=rule, triggered_by={"churn_score": 0.6},
        context=context or {},
    )


# ── Response validation ───────────────────────────────────────────────────────

def test_non_dict_response_rejected():
    assert _validate_llm_decision(["not", "a", "dict"], _cand()) is None
    assert _validate_llm_decision("just text", _cand()) is None
    assert _validate_llm_decision(None, _cand()) is None


def test_unknown_action_type_rejected():
    resp = {"action_type": "send_free_beer", "priority": 3, "confidence": 0.9}
    assert _validate_llm_decision(resp, _cand()) is None


def test_no_action_is_recognised():
    assert _validate_llm_decision({"action_type": NO_ACTION}, _cand()) == {
        "action_type": NO_ACTION
    }


def test_out_of_range_priority_falls_back_to_rule():
    fallback = _cand(priority=4)
    resp = {"action_type": "send_loyalty_reward", "priority": 99, "confidence": 0.5}
    out = _validate_llm_decision(resp, fallback)
    assert out["priority"] == 4

    resp["priority"] = "not-a-number"
    assert _validate_llm_decision(resp, fallback)["priority"] == 4


def test_confidence_clamped_and_defaulted():
    fallback = _cand()
    base = {"action_type": "send_loyalty_reward", "reason": "r"}
    assert _validate_llm_decision({**base, "confidence": 5.0}, fallback)["confidence"] == 1.0
    assert _validate_llm_decision({**base, "confidence": -2}, fallback)["confidence"] == 0.0
    assert _validate_llm_decision({**base, "confidence": "abc"}, fallback)["confidence"] == 0.7


def test_reason_defaults_to_rule_and_is_truncated():
    fallback = _cand()
    out = _validate_llm_decision(
        {"action_type": "send_loyalty_reward", "reason": "x" * 1000}, fallback
    )
    assert len(out["reason"]) == 400

    out = _validate_llm_decision({"action_type": "send_loyalty_reward", "reason": ""}, fallback)
    assert out["reason"] == fallback.reason


def test_action_registry_covers_all_planned_types():
    # every type the rules can emit must exist in the canonical registry
    assert "send_retention_email" in ACTION_TYPES
    assert "no_action" in ACTION_TYPES


# ── plan_actions orchestration ────────────────────────────────────────────────

def test_passthrough_picks_highest_priority_and_keeps_context():
    cands = [
        _cand(action="send_loyalty_reward", priority=2, rule="LYL-001"),
        _cand(action="escalate_to_csm", priority=5, rule="CHN-001",
              context={"suggested_plan": "Pro"}),
    ]
    plans = plan_actions(cands, {"C1": {"ltv_estimate": 0}}, use_llm=False,
                         verbose=False)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.action_type == "escalate_to_csm"
    assert plan.priority == 5
    assert plan.llm_enhanced is False
    assert plan.metadata["context"]["suggested_plan"] == "Pro"


def test_llm_budget_caps_model_calls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def fake_llm(prompt, retries=2):
        calls.append(prompt)
        return {"action_type": "send_retention_email", "reason": "high risk",
                "priority": 4, "confidence": 0.9, "notes": ""}

    monkeypatch.setattr(planner, "_call_llm", fake_llm)

    cands = [_cand(cid=f"C{i}") for i in range(5)]
    lookup = {f"C{i}": {"ltv_estimate": 5000, "churn_score": 0.9} for i in range(5)}
    plans = plan_actions(cands, lookup, use_llm=True, llm_budget=2, verbose=False)

    assert len(calls) == 2                       # budget respected
    assert len(plans) == 5                       # everyone still gets a plan
    assert sum(1 for p in plans if p.llm_enhanced) == 2


def test_garbage_llm_response_never_reaches_the_queue(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        planner, "_call_llm",
        lambda prompt, retries=2: {"action_type": "send_free_beer", "priority": 99},
    )

    cands = [_cand(action="escalate_to_csm", priority=5, rule="CHN-001")]
    lookup = {"C1": {"ltv_estimate": 5000, "churn_score": 0.9}}
    plans = plan_actions(cands, lookup, use_llm=True, llm_budget=5, verbose=False)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.action_type == "escalate_to_csm"   # rule fallback, not the hallucination
    assert plan.llm_enhanced is False
    assert plan.priority == 5


def test_no_action_response_drops_the_customer(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_llm(prompt, retries=2):
        # the model clears C1 as healthy, keeps an action for C2
        if "Customer: C1" in prompt:
            return {"action_type": NO_ACTION}
        return {"action_type": "escalate_to_csm", "reason": "high risk",
                "priority": 5, "confidence": 0.9, "notes": ""}

    monkeypatch.setattr(planner, "_call_llm", fake_llm)

    cands = [_cand(cid="C1"), _cand(cid="C2", action="escalate_to_csm", priority=5)]
    lookup = {cid: {"customer_id": cid, "ltv_estimate": 5000, "churn_score": 0.9}
              for cid in ("C1", "C2")}
    plans = plan_actions(cands, lookup, use_llm=True, llm_budget=10, verbose=False)

    assert [p.customer_id for p in plans] == ["C2"]
    assert plans[0].llm_enhanced is True


def test_missing_api_key_falls_back_to_rules(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)

    cands = [_cand(action="escalate_to_csm", priority=5)]
    lookup = {"C1": {"ltv_estimate": 9999, "churn_score": 0.95}}
    plans = plan_actions(cands, lookup, use_llm=True, verbose=False)
    assert len(plans) == 1
    assert plans[0].llm_enhanced is False
