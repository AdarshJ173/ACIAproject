"""Shared SQLite path for the whole ACIA project."""
from pathlib import Path

# Project root = parent of the data/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "acia.db"
