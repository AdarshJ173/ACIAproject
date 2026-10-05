"""
ACIA — Terminal UI (TUI) Dashboard
Live terminal dashboard using standard curses (zero extra dependencies).

Run:
  python main.py --tui
  /home/aaj/Projects/ACIAproject/.venv/bin/python main.py --tui

Keys:
  q / ESC   Quit
  r         Run one agent cycle (rules -> planner -> scheduler)
  e         Execute pending queue (simulates sends + updates DB)
  f         Apply feedback loop (adjusts health scores)
  SPACE     Refresh data from DB immediately
"""

from __future__ import annotations

import curses
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.db import connect


# ── DB metrics loader ─────────────────────────────────────────────────────────

def _load_stats() -> dict:
    conn = connect()
    try:
        # Customers & financial KPIs
        tot = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        act = conn.execute("SELECT COUNT(*) FROM customers WHERE is_churned=0").fetchone()[0]
        churned = tot - act
        churn_rate = (churned / tot * 100) if tot else 0.0
        mrr = conn.execute(
            "SELECT COALESCE(SUM(mrr), 0) FROM customers WHERE is_churned=0"
        ).fetchone()[0]
        ltv = conn.execute(
            "SELECT COALESCE(SUM(ltv_estimate), 0) FROM ml_predictions"
        ).fetchone()[0]
        hi_risk = conn.execute(
            "SELECT COUNT(*) FROM ml_predictions WHERE churn_score >= 0.6"
        ).fetchone()[0]
        hot_conv = conn.execute(
            "SELECT COUNT(*) FROM ml_predictions WHERE conversion_score >= 0.7"
        ).fetchone()[0]

        # Action counts
        pending = conn.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE status='pending'"
        ).fetchone()[0]
        executed = conn.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE status='executed'"
        ).fetchone()[0]

        # Segments
        seg_rows = conn.execute(
            "SELECT segment, COUNT(*) AS n, ROUND(AVG(churn_score), 2) AS ch "
            "FROM ml_predictions GROUP BY segment ORDER BY n DESC"
        ).fetchall()

        # Top 6 churn risk active
        top_risk = conn.execute(
            "SELECT c.customer_id, c.plan, c.mrr, p.churn_score, p.segment "
            "FROM customers c JOIN ml_predictions p ON p.customer_id=c.customer_id "
            "WHERE c.is_churned=0 ORDER BY p.churn_score DESC LIMIT 6"
        ).fetchall()

        # Latest executed actions
        latest_act = conn.execute(
            "SELECT customer_id, action_type, priority, outcome "
            "FROM agent_actions WHERE status='executed' "
            "ORDER BY executed_at DESC, rowid DESC LIMIT 6"
        ).fetchall()

        # Pending queue
        queue = conn.execute(
            "SELECT customer_id, action_type, priority, urgency_score "
            "FROM agent_actions WHERE status='pending' "
            "ORDER BY urgency_score DESC, priority DESC, created_at ASC LIMIT 6"
        ).fetchall()

        return {
            "total": tot, "active": act, "churned": churned, "churn_rate": churn_rate,
            "mrr": mrr, "ltv": ltv, "hi_risk": hi_risk, "hot_conv": hot_conv,
            "pending": pending, "executed": executed,
            "segments": [dict(r) for r in seg_rows],
            "top_risk": [dict(r) for r in top_risk],
            "latest_act": [dict(r) for r in latest_act],
            "queue": [dict(r) for r in queue],
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:.0f}"


def _bar(pct: float, width: int = 12) -> str:
    filled = max(0, min(width, int(round((pct / 100.0) * width))))
    return "█" * filled + "░" * (width - filled)


# ── Curses Renderer ───────────────────────────────────────────────────────────

def _draw(stdscr, stats: dict, status_msg: str) -> None:
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    if max_y < 24 or max_x < 80:
        stdscr.addstr(0, 0, f"Terminal too small: {max_x}x{max_y}. Resize to >= 80x24.")
        stdscr.refresh()
        return

    # Color pairs (initialized in main)
    C_HEADER = curses.color_pair(1)
    C_GREEN  = curses.color_pair(2)
    C_BLUE   = curses.color_pair(3)
    C_RED    = curses.color_pair(4)
    C_AMBER  = curses.color_pair(5)
    C_DIM    = curses.color_pair(6)

    # 1. Header
    title = " ACIA — Autonomous CRM Intelligence Agent [Terminal UI] "
    stamp = datetime.now().strftime("%H:%M:%S")
    stdscr.attron(C_HEADER | curses.A_BOLD)
    stdscr.addstr(0, 0, " " * (max_x - 1))
    stdscr.addstr(0, 1, title)
    stdscr.addstr(0, max(1, max_x - len(stamp) - 3), f"[{stamp}]")
    stdscr.attroff(C_HEADER | curses.A_BOLD)

    if "error" in stats:
        stdscr.addstr(2, 2, f"DB Error: {stats['error']}", C_RED)
        stdscr.refresh()
        return

    # 2. KPI Cards (Row 1)
    kpis = [
        ("ACTIVE CUSTOMERS", f"{stats['active']:,}", f"{stats['churned']} churn ({stats['churn_rate']:.1f}%)", C_BLUE),
        ("MONTHLY MRR", _fmt_money(stats["mrr"]), "active base only", C_GREEN),
        ("ESTIMATED LTV", _fmt_money(stats["ltv"]), "whole customer base", C_BLUE),
        ("HIGH RISK (>60%)", str(stats["hi_risk"]), "urgent churn risk", C_RED),
        ("HOT CONVERSIONS", str(stats["hot_conv"]), "prob >= 70%", C_AMBER),
        ("ACTIONS", f"{stats['pending']} P / {stats['executed']} E", "queue / executed", C_GREEN),
    ]

    card_w = (max_x - 2) // len(kpis)
    for i, (label, val, sub, color) in enumerate(kpis):
        x = 1 + i * card_w
        stdscr.addstr(2, x, ("┌" + "─" * (card_w - 2) + "┐")[:card_w], C_DIM)
        stdscr.addstr(3, x, ("│ " + label[:card_w - 4] + " " * card_w)[:card_w - 1] + "│", C_DIM)
        val_line = f"│ {val}"
        stdscr.addstr(4, x, (val_line + " " * card_w)[:card_w - 1] + "│")
        stdscr.addstr(4, x + 2, val[:card_w - 4], color | curses.A_BOLD)
        stdscr.addstr(5, x, ("│ " + sub[:card_w - 4] + " " * card_w)[:card_w - 1] + "│", C_DIM)
        stdscr.addstr(6, x, ("└" + "─" * (card_w - 2) + "┘")[:card_w], C_DIM)

    # 3. Two columns: Left (Top Churn Risk), Right (Segments)
    col_w = (max_x - 3) // 2

    # Left box: Churn Risk
    stdscr.addstr(8, 1, " HIGHEST CHURN RISK — ACTIVE ACCOUNTS", C_BLUE | curses.A_BOLD)
    stdscr.addstr(9, 1, f" {'ID':<7} {'PLAN':<10} {'MRR':<8} {'CHURN RISK':<16} {'SEGMENT':<10}", C_DIM)
    stdscr.addstr(10, 1, "─" * (col_w - 1), C_DIM)
    for idx, r in enumerate(stats["top_risk"][:6]):
        y = 11 + idx
        pct = r["churn_score"] * 100
        bar = _bar(pct, 8)
        color = C_RED if pct >= 60 else (C_AMBER if pct >= 30 else C_GREEN)
        stdscr.addstr(y, 1, f" {r['customer_id']:<7} {r['plan']:<10} {_fmt_money(r['mrr']):<8} ")
        stdscr.addstr(y, 29, f"{bar} {pct:>4.0f}%", color | curses.A_BOLD)
        stdscr.addstr(y, 45, f" {r['segment']:<10}")

    # Right box: Segments
    rx = col_w + 2
    stdscr.addstr(8, rx, " CUSTOMER SEGMENTS & RFM METRICS", C_BLUE | curses.A_BOLD)
    stdscr.addstr(9, rx, f" {'SEGMENT':<12} {'COUNT':<7} {'SHARE':<16} {'AVG CHURN':<10}", C_DIM)
    stdscr.addstr(10, rx, "─" * (max_x - rx - 1), C_DIM)
    tot_cust = max(1, stats["total"])
    for idx, s in enumerate(stats["segments"][:6]):
        y = 11 + idx
        pct = (s["n"] / tot_cust) * 100
        bar = _bar(pct, 10)
        stdscr.addstr(y, rx, f" {s['segment']:<12} {s['n']:<7} ")
        stdscr.addstr(y, rx + 21, f"{bar} {pct:>4.0f}%", C_BLUE)
        stdscr.addstr(y, rx + 38, f"   {s['ch'] * 100:>3.0f}%")

    # 4. Bottom row: Latest Executed (Left) vs Pending Queue (Right)
    mid_y = 18
    stdscr.addstr(mid_y, 1, " LATEST EXECUTED ACTIONS (FEEDBACK LOOP)", C_BLUE | curses.A_BOLD)
    stdscr.addstr(mid_y + 1, 1, f" {'CUSTOMER':<10} {'ACTION TYPE':<24} {'P':<3} {'OUTCOME':<18}", C_DIM)
    stdscr.addstr(mid_y + 2, 1, "─" * (col_w - 1), C_DIM)
    for idx, a in enumerate(stats["latest_act"][:5]):
        y = mid_y + 3 + idx
        out = (a["outcome"] or "—").replace("_", " ")
        out_color = C_GREEN if any(k in out for k in ("resolved", "positive", "claimed", "upgraded", "retained")) else \
                    (C_RED if any(k in out for k in ("churned", "unsub", "bounced")) else C_DIM)
        stdscr.addstr(y, 1, f" {a['customer_id']:<10} {a['action_type'][:22]:<24} P{a['priority']:<2} ")
        stdscr.addstr(y, 40, out[:18], out_color)

    # Right bottom: Pending Queue
    stdscr.addstr(mid_y, rx, " PENDING ACTION QUEUE (URGENCY ORDERED)", C_BLUE | curses.A_BOLD)
    stdscr.addstr(mid_y + 1, rx, f" {'CUSTOMER':<10} {'ACTION TYPE':<24} {'P':<3} {'URGENCY SCORE':<12}", C_DIM)
    stdscr.addstr(mid_y + 2, rx, "─" * (max_x - rx - 1), C_DIM)
    if not stats["queue"]:
        stdscr.addstr(mid_y + 3, rx, " (Queue is empty — press 'r' to trigger an agent cycle)", C_DIM)
    else:
        for idx, q in enumerate(stats["queue"][:5]):
            y = mid_y + 3 + idx
            score = f"{q['urgency_score']:.1f}" if q.get("urgency_score") is not None else "—"
            stdscr.addstr(y, rx, f" {q['customer_id']:<10} {q['action_type'][:22]:<24} P{q['priority']:<2} ")
            stdscr.addstr(y, rx + 38, f"{score:>6}", C_AMBER | curses.A_BOLD)

    # 5. Footer: Key bindings + status line
    stdscr.attron(C_HEADER)
    footer = " [r] Run Cycle   [e] Execute Queue   [f] Feedback   [SPACE] Refresh   [q] Quit "
    stdscr.addstr(max_y - 2, 0, (footer + " " * max_x)[:max_x - 1])
    stdscr.attroff(C_HEADER)

    status_bar = f" Status: {status_msg}"
    stdscr.addstr(max_y - 1, 0, (status_bar + " " * max_x)[:max_x - 1], C_GREEN | curses.A_BOLD)

    stdscr.refresh()


# ── Event Loop ────────────────────────────────────────────────────────────────

def _run_tui(stdscr) -> None:
    curses.curs_set(0)          # Hide cursor
    curses.start_color()
    curses.use_default_colors()

    # Color definitions (foreground, background)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)    # Header/Footer
    curses.init_pair(2, curses.COLOR_GREEN, -1)                   # Green accent
    curses.init_pair(3, curses.COLOR_CYAN,  -1)                   # Blue/Cyan
    curses.init_pair(4, curses.COLOR_RED,   -1)                   # Red
    curses.init_pair(5, curses.COLOR_YELLOW,-1)                   # Amber
    curses.init_pair(6, curses.COLOR_WHITE, -1)                   # Dim

    stdscr.nodelay(True)         # Non-blocking getch
    status_msg = "Live. Auto-refreshing every 3s. Press 'r' to run agent cycle."
    stats = _load_stats()
    last_refresh = time.time()

    while True:
        _draw(stdscr, stats, status_msg)

        # Handle keyboard input (timeout 200ms)
        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1

        now = time.time()

        if ch in (ord('q'), ord('Q'), 27):   # 27 = ESC
            break

        elif ch in (ord('r'), ord('R')):
            status_msg = "Running agent cycle..."
            _draw(stdscr, stats, status_msg)
            from agent.runner import run_agent_cycle
            res = run_agent_cycle(use_llm=False, verbose=False)
            status_msg = f"Agent cycle complete: {res['candidates']} candidates, {res['queued_actions']} queued."
            stats = _load_stats()
            last_refresh = now

        elif ch in (ord('e'), ord('E')):
            status_msg = "Executing pending queue..."
            _draw(stdscr, stats, status_msg)
            from actions.executor import execute_batch
            res = execute_batch(limit=50, dry_run=False, verbose=False)
            succ = sum(1 for r in res if r.status == 'success')
            status_msg = f"Executed {len(res)} actions ({succ} successful). Ready."
            stats = _load_stats()
            last_refresh = now

        elif ch in (ord('f'), ord('F')):
            status_msg = "Applying feedback loop..."
            _draw(stdscr, stats, status_msg)
            from actions.feedback import apply_feedback
            fb = apply_feedback(lookback_hours=48, verbose=False)
            status_msg = f"Feedback loop complete: {fb.get('executions_processed', 0)} processed."
            stats = _load_stats()
            last_refresh = now

        elif ch == ord(' '):
            stats = _load_stats()
            status_msg = "Refreshed."
            last_refresh = now

        # Auto-refresh every 3 seconds
        if now - last_refresh > 3.0:
            stats = _load_stats()
            last_refresh = now

        time.sleep(0.05)


def launch_tui() -> None:
    """Entry point for python main.py --tui"""
    try:
        curses.wrapper(_run_tui)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    launch_tui()
