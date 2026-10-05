"""
ACIA — REST API (FastAPI)
Exposes the full CRM intelligence system as a REST service.

Install:  pip install fastapi uvicorn
Run:      python main.py --api   (or: uvicorn api:app --reload --port 8000)
Docs:     http://localhost:8000/docs
Dashboard: http://localhost:8000/ui
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

# FastAPI imports — install with: pip install fastapi uvicorn
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.db import connect

DASHBOARD_HTML = ROOT / "dashboard" / "index.html"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CustomerRisk(BaseModel):
    customer_id:       str
    name:              str
    plan:              str
    mrr:               float
    churn_score:       float
    churn_risk:        str          # Low / Medium / High
    conversion_score:  float
    conversion_tier:   str          # Cold / Warm / Hot
    segment:           str
    ltv_estimate:      float
    health_score:      float
    predicted_at:      str


class ActionItem(BaseModel):
    action_id:   str
    customer_id: str
    action_type: str
    reason:      str
    priority:    int
    status:      str
    outcome:     Optional[str]
    created_at:  str
    executed_at: Optional[str]


class AgentCycleResult(BaseModel):
    cycle_at:        str
    customers:       int
    candidates:      int
    plans:           int
    queued_actions:  int
    llm_enhanced:    int
    elapsed_sec:     float
    critical:        int
    high:            int


class DashboardSummary(BaseModel):
    total_customers:    int
    active_customers:   int
    churned_customers:  int
    churn_rate:         float
    total_mrr:          float
    total_ltv:          float
    high_risk_count:    int
    hot_conversions:    int
    actions_executed:   int
    actions_pending:    int
    segments:           dict
    generated_at:       str


class ExecuteRequest(BaseModel):
    limit:   int = 50
    dry_run: bool = False


class TrainRequest(BaseModel):
    retrain_models: bool = True
    run_inference:  bool = True


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    return connect()


def row_to_dict(row) -> dict:
    return dict(row) if row else {}


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> "FastAPI":
    app = FastAPI(
        title="ACIA — Autonomous CRM Intelligence Agent",
        description=(
            "AI-powered CRM system that predicts churn and conversion, "
            "then autonomously executes engagement actions.\n\n"
            "Built with: Python · SQLite · scikit-learn · OpenRouter"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    def root():
        """Health check — confirms API is running."""
        return {
            "status": "ok",
            "service": "ACIA",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health", tags=["Health"])
    def health():
        """Deep health check — verifies DB connectivity."""
        try:
            conn = get_db()
            n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            conn.close()
            return {"status": "ok", "db": "connected", "customers": n}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"DB error: {e}")

    # ── Dashboard ─────────────────────────────────────────────────────────────

    @app.get("/ui", response_class=HTMLResponse, tags=["Dashboard"])
    def dashboard_ui():
        """Live dashboard — single HTML page that renders this API's own data."""
        if not DASHBOARD_HTML.exists():
            raise HTTPException(status_code=404, detail="dashboard/index.html not found")
        return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))

    @app.get("/dashboard", response_model=DashboardSummary, tags=["Dashboard"])
    def get_dashboard():
        """Full dashboard summary — KPIs, segments, action counts."""
        conn = get_db()
        try:
            total    = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            active   = conn.execute("SELECT COUNT(*) FROM customers WHERE is_churned=0").fetchone()[0]
            churned  = total - active
            mrr      = conn.execute("SELECT COALESCE(SUM(mrr),0) FROM customers WHERE is_churned=0").fetchone()[0]
            ltv      = conn.execute("SELECT COALESCE(SUM(ltv_estimate),0) FROM ml_predictions").fetchone()[0]
            hi_risk  = conn.execute("SELECT COUNT(*) FROM ml_predictions WHERE churn_score>=0.6").fetchone()[0]
            hot_conv = conn.execute("SELECT COUNT(*) FROM ml_predictions WHERE conversion_score>=0.7").fetchone()[0]
            executed = conn.execute("SELECT COUNT(*) FROM agent_actions WHERE status='executed'").fetchone()[0]
            pending  = conn.execute("SELECT COUNT(*) FROM agent_actions WHERE status='pending'").fetchone()[0]

            seg_rows = conn.execute(
                "SELECT segment, COUNT(*) as n FROM ml_predictions GROUP BY segment"
            ).fetchall()
            segments = {r["segment"]: r["n"] for r in seg_rows}

            return DashboardSummary(
                total_customers=total,
                active_customers=active,
                churned_customers=churned,
                churn_rate=round(churned / total * 100, 2) if total else 0,
                total_mrr=round(mrr, 2),
                total_ltv=round(ltv, 2),
                high_risk_count=hi_risk,
                hot_conversions=hot_conv,
                actions_executed=executed,
                actions_pending=pending,
                segments=segments,
                generated_at=datetime.now().isoformat(),
            )
        finally:
            conn.close()

    # ── Customers ─────────────────────────────────────────────────────────────

    @app.get("/customers", tags=["Customers"])
    def list_customers(
        plan:      Optional[str] = Query(None, description="Filter by plan: Free/Starter/Pro/Enterprise"),
        segment:   Optional[str] = Query(None, description="Filter by segment: Champion/Loyal/At-Risk/Prospect/Lost"),
        min_churn: float         = Query(0.0,  description="Minimum churn score (0–1)"),
        max_churn: float         = Query(1.0,  description="Maximum churn score (0–1)"),
        limit:     int           = Query(50,   le=500),
        offset:    int           = Query(0),
    ):
        """List customers with ML predictions. Supports filtering and pagination."""
        conn = get_db()
        try:
            where = ["c.is_churned = 0", "p.churn_score >= ?", "p.churn_score <= ?"]
            params: list = [min_churn, max_churn]

            if plan:
                where.append("c.plan = ?")
                params.append(plan)
            if segment:
                where.append("p.segment = ?")
                params.append(segment)

            sql = f"""
                SELECT c.customer_id, c.name, c.plan, c.mrr, c.health_score,
                       p.churn_score, p.conversion_score, p.segment, p.ltv_estimate
                FROM customers c
                JOIN ml_predictions p ON p.customer_id = c.customer_id
                WHERE {' AND '.join(where)}
                ORDER BY p.churn_score DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(sql, params + [limit, offset]).fetchall()
            return {"customers": [row_to_dict(r) for r in rows], "count": len(rows), "offset": offset}
        finally:
            conn.close()

    @app.get("/customers/{customer_id}/risk", response_model=CustomerRisk, tags=["Customers"])
    def get_customer_risk(customer_id: str):
        """Full risk profile for a single customer."""
        conn = get_db()
        try:
            row = conn.execute("""
                SELECT c.customer_id, c.name, c.plan, c.mrr, c.health_score,
                       p.churn_score, p.conversion_score, p.segment, p.ltv_estimate, p.predicted_at
                FROM customers c
                JOIN ml_predictions p ON p.customer_id = c.customer_id
                WHERE c.customer_id = ?
            """, (customer_id,)).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

            d = row_to_dict(row)
            d["churn_risk"]      = "High" if d["churn_score"] >= 0.6 else "Medium" if d["churn_score"] >= 0.3 else "Low"
            d["conversion_tier"] = "Hot"  if d["conversion_score"] >= 0.7 else "Warm" if d["conversion_score"] >= 0.4 else "Cold"
            return CustomerRisk(**d)
        finally:
            conn.close()

    @app.get("/customers/{customer_id}/actions", tags=["Customers"])
    def get_customer_actions(customer_id: str):
        """All actions (past + pending) for a specific customer."""
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM agent_actions WHERE customer_id=? ORDER BY created_at DESC",
                (customer_id,)
            ).fetchall()
            return {"customer_id": customer_id, "actions": [row_to_dict(r) for r in rows]}
        finally:
            conn.close()

    # ── Predictions ───────────────────────────────────────────────────────────

    @app.get("/predictions/churn", tags=["Predictions"])
    def get_churn_predictions(
        min_score: float = Query(0.0, description="Minimum churn score"),
        limit:     int   = Query(50,  le=500),
    ):
        """Customers ranked by churn score — highest risk first."""
        conn = get_db()
        try:
            rows = conn.execute("""
                SELECT c.customer_id, c.name, c.plan, c.mrr,
                       p.churn_score, p.segment, p.ltv_estimate
                FROM ml_predictions p
                JOIN customers c ON c.customer_id = p.customer_id
                WHERE p.churn_score >= ? AND c.is_churned = 0
                ORDER BY p.churn_score DESC LIMIT ?
            """, (min_score, limit)).fetchall()
            return {"predictions": [row_to_dict(r) for r in rows], "count": len(rows)}
        finally:
            conn.close()

    @app.get("/predictions/conversion", tags=["Predictions"])
    def get_conversion_predictions(
        min_score: float = Query(0.5, description="Minimum conversion score"),
        limit:     int   = Query(50,  le=500),
    ):
        """Customers ranked by conversion probability — hottest leads first."""
        conn = get_db()
        try:
            rows = conn.execute("""
                SELECT c.customer_id, c.name, c.plan, c.mrr,
                       p.conversion_score, p.segment
                FROM ml_predictions p
                JOIN customers c ON c.customer_id = p.customer_id
                WHERE p.conversion_score >= ? AND c.plan != 'Enterprise'
                ORDER BY p.conversion_score DESC LIMIT ?
            """, (min_score, limit)).fetchall()
            return {"predictions": [row_to_dict(r) for r in rows], "count": len(rows)}
        finally:
            conn.close()

    @app.get("/predictions/segments", tags=["Predictions"])
    def get_segment_distribution():
        """Customer count and average metrics per segment."""
        conn = get_db()
        try:
            rows = conn.execute("""
                SELECT p.segment,
                       COUNT(*) as customer_count,
                       ROUND(AVG(p.churn_score), 3) as avg_churn_score,
                       ROUND(AVG(p.conversion_score), 3) as avg_conversion_score,
                       ROUND(AVG(c.mrr), 2) as avg_mrr,
                       ROUND(SUM(p.ltv_estimate), 2) as total_ltv
                FROM ml_predictions p
                JOIN customers c ON c.customer_id = p.customer_id
                GROUP BY p.segment
                ORDER BY avg_churn_score DESC
            """).fetchall()
            return {"segments": [row_to_dict(r) for r in rows]}
        finally:
            conn.close()

    # ── Actions ───────────────────────────────────────────────────────────────

    @app.get("/actions/queue", tags=["Actions"])
    def get_action_queue(
        status:   str = Query("pending", description="pending | executed | skipped"),
        priority: Optional[int] = Query(None, description="Filter by priority 1–5"),
        limit:    int = Query(50, le=500),
    ):
        """Fetch the action queue filtered by status and priority."""
        conn = get_db()
        try:
            where  = ["status = ?"]
            params: list = [status]
            if priority:
                where.append("priority = ?")
                params.append(priority)
            rows = conn.execute(
                f"SELECT * FROM agent_actions WHERE {' AND '.join(where)} "
                f"ORDER BY urgency_score DESC, priority DESC, created_at ASC LIMIT ?",
                params + [limit]
            ).fetchall()
            return {"actions": [row_to_dict(r) for r in rows], "count": len(rows), "status": status}
        finally:
            conn.close()

    @app.get("/actions/{action_id}", tags=["Actions"])
    def get_action(action_id: str):
        """Get a single action by ID."""
        conn = get_db()
        try:
            row = conn.execute("SELECT * FROM agent_actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Action not found")
            return row_to_dict(row)
        finally:
            conn.close()

    @app.post("/actions/execute", tags=["Actions"])
    def execute_actions(req: ExecuteRequest):
        """
        Execute pending actions from the queue.
        Returns execution results immediately (synchronous).
        """
        from actions.executor import execute_batch, print_execution_report
        results = execute_batch(limit=req.limit, dry_run=req.dry_run, verbose=False)
        summary = {
            "executed":  len(results),
            "success":   sum(1 for r in results if r.status == "success"),
            "failure":   sum(1 for r in results if r.status == "failure"),
            "neutral":   sum(1 for r in results if r.status == "neutral"),
            "dry_run":   req.dry_run,
            "timestamp": datetime.now().isoformat(),
        }
        return summary

    # ── Agent ─────────────────────────────────────────────────────────────────

    @app.post("/agent/run", response_model=AgentCycleResult, tags=["Agent"])
    def run_agent(
        use_llm:    bool = Query(True,  description="Use OpenRouter free models for LLM planning"),
        llm_budget: int  = Query(20,    description="Max LLM calls per cycle"),
    ):
        """
        Trigger a full agent decision cycle:
        Rules → OpenRouter LLM Planner → Scheduler → DB queue.
        Requires OPENROUTER_API_KEY when use_llm=true.
        """
        from agent.runner import run_agent_cycle
        summary = run_agent_cycle(use_llm=use_llm, llm_budget=llm_budget, verbose=False)
        return AgentCycleResult(**summary)

    @app.post("/agent/train", tags=["Agent"])
    def train_models(req: TrainRequest):
        """Retrain ML models and/or re-run inference on all customers."""
        from models.orchestrator import train_all, run_inference_and_store
        result = {}
        if req.retrain_models:
            arts = train_all(verbose=False)
            result["models_trained"] = list(arts.keys())
        if req.run_inference:
            preds = run_inference_and_store(verbose=False)
            result["customers_scored"] = len(preds)
        result["timestamp"] = datetime.now().isoformat()
        return result

    @app.post("/agent/feedback", tags=["Agent"])
    def apply_feedback_endpoint(lookback_hours: int = Query(24)):
        """Process recent outcomes and update health scores (feedback loop)."""
        from actions.feedback import apply_feedback
        summary = apply_feedback(lookback_hours=lookback_hours, verbose=False)
        return summary

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first:  pip install fastapi uvicorn")
    else:
        try:
            import uvicorn
            print("Starting ACIA API on http://localhost:8000")
            print("Interactive docs: http://localhost:8000/docs")
            uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
        except ImportError:
            print("Install uvicorn:  pip install uvicorn")
