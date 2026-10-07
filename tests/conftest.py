"""Shared test fixtures.

`isolated_db` exists because `app.security` and friends import the module-level
`database` singleton, which is bound to the configured path at import time. Without a
fixture, any test that registers or authenticates writes to the real `data/eidomira.db`
— which is a good reason nobody wrote them.

Rebinding `app.database.database` would not help: every module did
`from app.database import database`, so each holds its own reference to the same object.
Mutating that object's `path` in place is what redirects all of them at once.
"""
from __future__ import annotations

import pytest

from app.database import SCHEMA, database


@pytest.fixture
def isolated_db(tmp_path):
    """Point the database singleton at a throwaway file for the duration of a test."""
    original = database.path
    path = tmp_path / "eidomira-test.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    database.path = path
    with database.connect() as db:
        db.executescript(SCHEMA)
    try:
        yield database
    finally:
        database.path = original
        # release the temp file before pytest cleans the directory up, mostly on Windows
        with database.lock:
            pass
