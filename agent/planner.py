"""
ACIA — LLM Planner (OpenRouter)
Takes ActionCandidates from the rule engine and uses an OpenRouter free model to:
  1. Resolve conflicts when multiple rules fire for one customer
  2. Personalise action reasoning with full customer context
  3. Decide on ambiguous edge cases rules can't handle
  4. Generate the final action plan with human-readable justification

Env:
  OPENROUTER_API_KEY  — required for LLM mode
  OPENROUTER_MODEL    — optional; default openrouter/free (auto-picks a free model)
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.rules import ActionCandidate

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

# ── Canonical action set ──────────────────────────────────────────────────────
# Single source of truth: drives the prompt, response validation and downstream
# template/executor dispatch. An LLM answer outside this set is discarded.
ACTION_TYPES: dict[str, str] = {
    "escalate_to_csm":          "assign to customer success manager",
    "send_retention_email":     "personalised retention campaign",
    "send_reengagement_email":  "re-activate dormant users",
    "send_upgrade_offer":       "upsell to higher plan",
    "enroll_nurture_sequence":  "drip campaign for prospects",
    "proactive_support_call":   "schedule outbound call",
    "send_loyalty_reward":      "reward high-value loyal customers",
    "send_health_checkin":      "low-pressure check-in for at-risk",
    "no_action":                "customer is healthy, no intervention needed",
}
NO_ACTION = "no_action"
MAX_REASON_LEN = 400
MAX_NOTES_LEN = 300


# ── Planned Action (output of planner) ───────────────────────────────────────
@dataclass
class PlannedAction:
    customer_id:   str
    action_type:   str
    reason:        str          # LLM-enriched justification
    priority:      int
    rule_id:       str
    llm_enhanced:  bool         # True if OpenRouter revised the candidate
    confidence:    float        # 0–1, LLM self-reported confidence
    metadata:      dict


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ACIA, an autonomous CRM intelligence agent.
Your role is to review customer data and action candidates, then decide the optimal action plan.

You receive:
- Customer profile data (plan, MRR, tenure, health score, segment, churn/conversion scores)
- One or more action candidates triggered by business rules

You must output a single JSON object with this exact structure:
{
  "action_type": "<chosen action type>",
  "reason": "<clear 1-2 sentence justification referencing the customer data>",
  "priority": <integer 1-5>,
  "confidence": <float 0.0-1.0>,
  "notes": "<any important considerations>"
}

Action types available:
""" + "\n".join(
    f"- {name:25s} ({desc})" for name, desc in ACTION_TYPES.items()
) + """

Rules:
- If multiple actions conflict, choose the highest-impact one
- Always consider LTV: high-value customers at risk deserve higher priority
- Never escalate to CSM for low-MRR free-tier customers — use email instead
- For new customers (tenure < 30 days), prefer nurture over retention
- Output ONLY valid JSON, no preamble or explanation outside the JSON block
"""

def _build_customer_prompt(candidates: list[ActionCandidate], customer_row: dict) -> str:
    c = customer_row
    candidate_list = "\n".join([
        f"  - Rule {cand.rule_id} → {cand.action_type} (P{cand.priority}): {cand.reason}"
        for cand in candidates
    ])
    return f"""Customer: {c.get('customer_id')}
Plan: {c.get('plan')} | MRR: ${c.get('mrr', 0):.2f}/mo | Tenure: {int(c.get('tenure_days', 0))} days
Segment: {c.get('segment')} | Health: {c.get('health_score', 0):.0f}/100 | NPS: {c.get('nps_score', 0)}
Churn score: {c.get('churn_score', 0):.3f} | Conversion score: {c.get('conversion_score', 0):.3f}
LTV estimate: ${c.get('ltv_estimate', 0):.0f}
Events (30d): {int(c.get('event_count_30d', 0))} | Tickets open: {int(c.get('open_tickets', 0))} | Sentiment: {c.get('avg_sentiment', 0):.2f}
Days since last event: {int(c.get('days_since_last_event', 0))}

Action candidates fired:
{candidate_list}

Decide the single best action for this customer."""


# ── OpenRouter API call ───────────────────────────────────────────────────────

def _openrouter_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")


def _openrouter_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def _call_llm(prompt: str, retries: int = 2) -> dict | None:
    """Call OpenRouter (OpenAI-compatible) and parse JSON response."""
    api_key = _openrouter_api_key()
    if not api_key:
        print("  [LLM] OPENROUTER_API_KEY not set — skipping LLM call")
        return None

    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": _openrouter_model(),
            "max_tokens": 1000,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }).encode()

        req = urllib.request.Request(
            OPENROUTER_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/acia-crm",
                "X-Title": "ACIA CRM Agent",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not text:
            raise ValueError(f"Empty LLM response: {data}")

        # Strip markdown fences if present
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    except Exception as e:
        if retries > 0:
            time.sleep(1.5)
            return _call_llm(prompt, retries - 1)
        print(f"  [LLM] OpenRouter call failed: {e}")
        return None


# ── Planner logic ──────────────────────────────────────────────────────────────

def _passthrough_plan(candidates: list[ActionCandidate]) -> PlannedAction:
    """Fallback: pick highest-priority candidate without LLM."""
    best = max(candidates, key=lambda c: c.priority)
    return PlannedAction(
        customer_id=best.customer_id,
        action_type=best.action_type,
        reason=best.reason,
        priority=best.priority,
        rule_id=best.rule_id,
        llm_enhanced=False,
        confidence=0.7,
        metadata={"triggered_by": best.triggered_by, "context": best.context},
    )


def _validate_llm_decision(resp: object, fallback: ActionCandidate) -> dict | None:
    """
    Normalise an LLM decision against the canonical action set.
    Returns:
      - {"action_type": "no_action"}  → healthy customer, skip entirely
      - dict of validated fields      → use the LLM decision
      - None                          → unusable response, caller falls back to rules
    LLM output is untrusted: unknown action types, out-of-range priorities and
    non-numeric confidences are clamped or rejected instead of reaching the DB.
    """
    if not isinstance(resp, dict):
        return None

    action_type = resp.get("action_type")
    if not isinstance(action_type, str) or action_type.strip() not in ACTION_TYPES:
        return None
    action_type = action_type.strip()
    if action_type == NO_ACTION:
        return {"action_type": NO_ACTION}

    reason = resp.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = fallback.reason

    try:
        priority = int(resp.get("priority"))
    except (TypeError, ValueError):
        priority = fallback.priority
    if not 1 <= priority <= 5:
        priority = fallback.priority

    try:
        confidence = float(resp.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = min(max(confidence, 0.0), 1.0)

    notes = resp.get("notes")
    if not isinstance(notes, str):
        notes = ""

    return {
        "action_type": action_type,
        "reason":      reason[:MAX_REASON_LEN],
        "priority":    priority,
        "confidence":  confidence,
        "notes":       notes[:MAX_NOTES_LEN],
    }


def plan_actions(
    candidates: list[ActionCandidate],
    customer_lookup: dict,
    use_llm: bool = True,
    llm_budget: int = 30,       # max customers to send to LLM (cost control)
    verbose: bool = True,
) -> list[PlannedAction]:
    """
    Groups candidates by customer, resolves conflicts, and produces
    one PlannedAction per customer.

    use_llm=True  → complex/conflicted cases go to OpenRouter free models
    use_llm=False → rule-based passthrough only (for testing / cost control)
    llm_budget    → caps LLM calls; remaining are handled by passthrough
    """
    if use_llm and not _openrouter_api_key():
        print("  [LLM] OPENROUTER_API_KEY missing — using rule passthrough only")
        use_llm = False

    # Group candidates by customer
    by_customer: dict[str, list[ActionCandidate]] = defaultdict(list)
    for c in candidates:
        by_customer[c.customer_id].append(c)

    plans: list[PlannedAction] = []
    llm_calls = 0

    # Customers that benefit most from LLM: multi-rule conflict OR high LTV
    def needs_llm(cust_id: str, cands: list[ActionCandidate]) -> bool:
        if not use_llm:
            return False
        if llm_calls >= llm_budget:
            return False
        row = customer_lookup.get(cust_id, {})
        has_conflict = len(set(c.action_type for c in cands)) > 1
        high_ltv     = float(row.get("ltv_estimate", 0)) >= 500
        high_risk    = float(row.get("churn_score", 0)) >= 0.60
        return has_conflict or high_ltv or high_risk

    for cust_id, cands in by_customer.items():
        row = customer_lookup.get(cust_id, {})

        if needs_llm(cust_id, cands):
            prompt   = _build_customer_prompt(cands, row)
            llm_resp = _call_llm(prompt)
            llm_calls += 1
            best_rule = max(cands, key=lambda c: c.priority)
            decision  = _validate_llm_decision(llm_resp, best_rule) if llm_resp else None

            if decision is not None and decision["action_type"] == NO_ACTION:
                continue   # LLM judged this customer healthy — no action queued

            if decision is not None:
                # Merge validated LLM decision with rule context
                plans.append(PlannedAction(
                    customer_id=cust_id,
                    action_type=decision["action_type"],
                    reason=decision["reason"],
                    priority=decision["priority"],
                    rule_id=best_rule.rule_id,
                    llm_enhanced=True,
                    confidence=decision["confidence"],
                    metadata={
                        "llm_notes":   decision.get("notes", ""),
                        "candidates":  len(cands),
                        "triggered_by": best_rule.triggered_by,
                        "context":     best_rule.context,
                    },
                ))
                continue
            # Unusable LLM response → fall through to rule passthrough

        plans.append(_passthrough_plan(cands))

    if verbose:
        llm_count   = sum(1 for p in plans if p.llm_enhanced)
        rule_count  = len(plans) - llm_count
        action_dist = {}
        for p in plans:
            action_dist[p.action_type] = action_dist.get(p.action_type, 0) + 1

        print(f"\n── LLM Planner Results ───────────────────────────────")
        print(f"  Customers with actions : {len(plans)}")
        print(f"  LLM-enhanced decisions : {llm_count}")
        print(f"  Rule passthrough       : {rule_count}")
        print(f"\n  Final action plan:")
        for act, n in sorted(action_dist.items(), key=lambda x: -x[1]):
            print(f"    {act:35s}: {n}")
        print()

    return plans


if __name__ == "__main__":
    import pandas as pd
    from agent.rules import evaluate, get_enriched_df

    df         = get_enriched_df()
    candidates = evaluate(verbose=False)
    cust_lookup = df.set_index("customer_id").to_dict("index")

    print(f"Running LLM planner on {len(candidates)} candidates...")
    plans = plan_actions(candidates, cust_lookup, use_llm=True, llm_budget=10, verbose=True)

    print("Sample planned actions (LLM-enhanced):")
    for p in [x for x in plans if x.llm_enhanced][:3]:
        print(f"\n  {p.customer_id} → {p.action_type} [P{p.priority}, conf={p.confidence:.2f}]")
        print(f"  {p.reason}")
        if p.metadata.get("llm_notes"):
            print(f"  Notes: {p.metadata['llm_notes']}")
