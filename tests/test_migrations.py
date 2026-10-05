"""Schema migrations: databases created before the metadata/urgency/feedback
columns existed gain them on connect, without losing their rows."""
from __future__ import annotations

import sqlite3

import data.db as dbmod

LEGACY_SCHEMA = """
CREATE TABLE agent_actions (
    action_id       TEXT PRIMARY KEY,
    customer_id     TEXT,
    action_type     TEXT,
    reason          TEXT,
    priority        INTEGER,
    status          TEXT DEFAULT 'pending',
    outcome         TEXT,
    created_at      TEXT,
    executed_at     TEXT
);
INSERT INTO agent_actions (action_id, customer_id, action_type, status)
VALUES ('legacy-1', 'C0001', 'send_retention_email', 'pending');
"""

NEW_COLUMNS = {"metadata", "urgency_score", "feedback_applied_at"}


def test_legacy_db_is_migrated_in_place(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.db"
    seed = sqlite3.connect(legacy)
    seed.executescript(LEGACY_SCHEMA)
    seed.close()

    monkeypatch.setattr(dbmod, "DB_PATH", legacy)

    conn = dbmod.connect()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_actions)")}
    assert NEW_COLUMNS <= cols, "migration did not add new columns"

    row = conn.execute(
        "SELECT metadata, urgency_score FROM agent_actions WHERE action_id='legacy-1'"
    ).fetchone()
    assert row["metadata"] is None          # existing rows survive untouched
    assert row["urgency_score"] is None

    # Running the migration again must be a no-op (no duplicate-column errors)
    dbmod._ensure_schema(conn)
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(agent_actions)")}
    assert cols_after == cols
    conn.close()


def test_connect_returns_rows_addressable_by_name(tmp_path, monkeypatch):
    fresh = tmp_path / "fresh.db"
    monkeypatch.setattr(dbmod, "DB_PATH", fresh)

    conn = dbmod.connect()
    assert conn.execute("SELECT 1 AS one").fetchone()["one"] == 1
    conn.close()


def test_fresh_schema_contains_new_columns():
    from data.generate import SCHEMA
    for col in NEW_COLUMNS:
        assert col in SCHEMA, f"{col} missing from CREATE TABLE agent_actions"
