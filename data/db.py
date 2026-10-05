"""Shared SQLite connection + schema migrations for the whole ACIA project."""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Project root = parent of the data/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "acia.db"

# Columns added after the first release. Applied to databases created before
# these fields existed, so existing installs keep working without a rebuild.
_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "agent_actions": [
        ("metadata",          "TEXT"),            # planner context / LLM notes (JSON)
        ("urgency_score",     "REAL"),            # scheduler composite score, used for queue ordering
        ("feedback_applied_at", "TEXT"),          # marks outcomes already folded into health scores
    ],
}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """ADD COLUMN migrations for databases created before a field existed."""
    for table, columns in _MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table not created yet (fresh DB); schema DDL carries the columns
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def connect() -> sqlite3.Connection:
    """Open the project DB with migrations applied and row access by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)
    return conn
