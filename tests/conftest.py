"""Shared fixtures: the whole suite runs against a throwaway SQLite DB,
never the project's data/acia.db."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def acia_db(tmp_path_factory):
    """Build a fresh synthetic database in a temp dir and point the app at it."""
    import data.db as dbmod
    import data.generate as gen

    db_path = tmp_path_factory.mktemp("acia") / "test.db"
    old_db, old_gen = dbmod.DB_PATH, gen.DB_PATH
    dbmod.DB_PATH = db_path
    gen.DB_PATH = db_path
    try:
        gen.build_database()
        # Mirror `python main.py --setup`: models + predictions, so rule and
        # API tests exercise real scores instead of an empty predictions table.
        from models.orchestrator import train_all, run_inference_and_store
        train_all(verbose=False)
        run_inference_and_store(verbose=False)
        yield db_path
    finally:
        dbmod.DB_PATH = old_db
        gen.DB_PATH = old_gen
