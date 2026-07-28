"""
ACIA — Feedback Loop
Reads execution outcomes from agent_actions and feeds signals
back into the system: health score adjustments, retraining flags,
and a summary of what worked.
"""

from __future__ import annotations
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.db import DB_PATH

# Outcome → health score delta mapping
HEALTH_DELTAS: dict[str, float] = {
    "opened_clicked":            +3.0,
    "discount_redeemed":         +5.0,
    "logged_in":                 +4.0,
    "upgraded":                  +10.0,
    "reward_claimed":            +4.0,
    "replied_positive":          +5.0,
    "call_completed_retained":   +8.0,
    "issue_resolved":            +6.0,
    "enrolled_active":           +3.0,
    "scheduled_followup":        +1.0,
    "callback_scheduled":        +1.0,
    "opened_only":               +1.0,
    "clicked_not_converted":     +0.5,
    "enrolled_passive":          +0.5,
    # Neutral / negative
    "no_response":               -1.0,
    "no_answer":                 -0.5,
    "bounced":                   -0.5,
    "unsubscribed":              -5.0,
    "call_completed_churned":    -10.0,
    "replied_negative":          -3.0,
    "already_enrolled":           0.0,
    "escalated_csm":             -1.0,
}


def apply_feedback(lookback_hours: int = 24, verbose: bool = True) -> dict:
    """
    Process recent execution outcomes:
    1. Adjust health scores based on outcomes
    2. Flag customers for model re-scoring
    3. Return feedback summary
    """
    conn    = sqlite3.connect(DB_PATH)
    cutoff  = (datetime.now() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M:%S")

    executed = pd.read_sql(
        "SELECT * FROM agent_actions WHERE status='executed' AND executed_at >= ?",
        conn, params=(cutoff,)
    )

    if executed.empty:
        if verbose:
            print("No recent executions to process.")
        conn.close()
        return {}

    updates   = []
    pos_count = 0
    neg_count = 0

    for _, row in executed.iterrows():
        delta = HEALTH_DELTAS.get(row["outcome"], 0.0)
        if delta != 0.0:
            updates.append((delta, row["customer_id"]))
            if delta > 0:
                pos_count += 1
            else:
                neg_count += 1

    # Apply health score deltas (clamped to 0–100)
    for delta, cid in updates:
        conn.execute(
            """UPDATE customers
               SET health_score = MAX(0, MIN(100, health_score + ?))
               WHERE customer_id = ?""",
            (delta, cid)
        )

    conn.commit()

    # Customers needing urgent re-scoring (churned or upgraded based on outcome)
    churn_signals   = executed[executed["outcome"] == "call_completed_churned"]["customer_id"].tolist()
    upgrade_signals = executed[executed["outcome"] == "upgraded"]["customer_id"].tolist()

    summary = {
        "executions_processed": len(executed),
        "health_updates":       len(updates),
        "positive_outcomes":    pos_count,
        "negative_outcomes":    neg_count,
        "churn_confirmed":      churn_signals,
        "upgrades_confirmed":   upgrade_signals,
        "processed_at":         datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n── Feedback Loop ────────────────────────────────────")
        print(f"  Executions processed   : {len(executed)}")
        print(f"  Health score updates   : {len(updates)}")
        print(f"  Positive signal updates: {pos_count}")
        print(f"  Negative signal updates: {neg_count}")
        print(f"  Churn confirmed        : {len(churn_signals)} customers")
        print(f"  Upgrades confirmed     : {len(upgrade_signals)} customers")
        if upgrade_signals:
            print(f"    Upgraded: {', '.join(upgrade_signals[:5])}")
        if churn_signals:
            print(f"    Churned : {', '.join(churn_signals[:5])}")
        print(f"  Health scores updated in DB ✅")
        print()

    conn.close()
    return summary


if __name__ == "__main__":
    summary = apply_feedback(lookback_hours=48)
