"""REST API: contract of every documented endpoint against a live app."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(acia_db):
    from api import create_app
    return TestClient(create_app())


def test_health_and_root(client):
    assert client.get("/").json()["status"] == "ok"
    health = client.get("/health").json()
    assert health["db"] == "connected"
    assert health["customers"] == 500


def test_dashboard_kpis_are_consistent(client):
    d = client.get("/dashboard").json()
    assert d["total_customers"] == d["active_customers"] + d["churned_customers"]
    assert d["total_mrr"] > 0
    assert 0 <= d["churn_rate"] <= 100
    assert isinstance(d["segments"], dict)


def test_customer_risk_profile_and_404(client):
    r = client.get("/customers/C0001/risk")
    assert r.status_code == 200
    body = r.json()
    assert body["churn_risk"] in {"Low", "Medium", "High"}
    assert body["conversion_tier"] in {"Cold", "Warm", "Hot"}
    assert 0 <= body["churn_score"] <= 1

    assert client.get("/customers/NOPE/risk").status_code == 404


def test_customer_list_filters(client):
    free = client.get("/customers", params={"plan": "Free", "limit": 5}).json()
    assert free["count"] <= 5
    assert all(c["plan"] == "Free" for c in free["customers"])

    paged = client.get("/customers", params={"limit": 5, "offset": 5}).json()
    assert paged["offset"] == 5


def test_churn_predictions_ranked_descending(client):
    preds = client.get("/predictions/churn", params={"limit": 10}).json()["predictions"]
    scores = [p["churn_score"] for p in preds]
    assert scores == sorted(scores, reverse=True)


def test_segment_distribution(client):
    segs = client.get("/predictions/segments").json()["segments"]
    assert len(segs) >= 1
    assert all("customer_count" in s for s in segs)


def test_action_queue_and_lookup(client):
    q = client.get("/actions/queue", params={"status": "executed", "limit": 5}).json()
    assert q["count"] <= 5
    assert all(a["status"] == "executed" for a in q["actions"])
    if q["actions"]:
        aid = q["actions"][0]["action_id"]
        assert client.get(f"/actions/{aid}").status_code == 200
    assert client.get("/actions/does-not-exist").status_code == 404


def test_customer_actions_history(client):
    body = client.get("/customers/C0001/actions").json()
    assert body["customer_id"] == "C0001"
    assert isinstance(body["actions"], list)


def test_execute_endpoint_dry_run(client):
    r = client.post("/actions/execute", json={"limit": 5, "dry_run": True})
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is True
    assert body["executed"] >= 0


def test_agent_run_cycle(client):
    r = client.post("/agent/run", params={"use_llm": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["customers"] == 500
    for key in ("candidates", "plans", "queued_actions", "llm_enhanced",
                "critical", "high", "elapsed_sec"):
        assert key in body
    assert body["queued_actions"] >= 0
    assert body["llm_enhanced"] == 0        # LLM disabled for this request


def test_agent_feedback_endpoint(client):
    r = client.post("/agent/feedback", params={"lookback_hours": 48})
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_agent_train_endpoint(client):
    r = client.post("/agent/train", json={"retrain_models": False, "run_inference": True})
    assert r.status_code == 200
    body = r.json()
    assert body["customers_scored"] == 500


def test_ui_dashboard_is_served(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "<title>ACIA — Live Dashboard</title>" in r.text
    assert "/dashboard" in r.text           # page fetches this API's own data
