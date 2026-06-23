"""SQLite database connection and schema bootstrapping.

``Database`` opens a connection, enables foreign keys, and applies the schema
(idempotently — schema.sql uses ``CREATE TABLE IF NOT EXISTS``). ``get_db``
returns a process-wide singleton backed by the app data directory.
"""

from __future__ import annotations

import os
import sqlite3

from ..utils.logging import get_logger
from ..utils.paths import db_path

_log = get_logger(__name__)

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class Database:
    """A thin wrapper around a sqlite3 connection.

    The schema is created on construction if it does not yet exist. Rows are
    returned as ``sqlite3.Row`` so repositories can access columns by name.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or db_path()
        # check_same_thread=False so the Flask dev server (multiple threads)
        # can share the singleton; repositories use short-lived cursors.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._apply_schema()
        self._migrate()

    def _apply_schema(self) -> None:
        try:
            with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
                script = fh.read()
        except OSError:
            _log.exception("Could not read schema.sql at %s", _SCHEMA_PATH)
            raise
        self._conn.executescript(script)
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so new
        columns must be added explicitly. Each step is idempotent (guarded by a
        column-existence check) and safe to run on every startup.
        """

        self._add_column_if_missing(
            "track_features", "harmonic_ratio", "REAL NOT NULL DEFAULT 0.5"
        )

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        cols = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            self._conn.commit()
            _log.info("Migration: added %s.%s", table, column)

    @property
    def conn(self) -> sqlite3.Connection:
        """The underlying sqlite3 connection."""

        return self._conn

    @property
    def path(self) -> str:
        """The filesystem path of this database (or ``:memory:``)."""

        return self._path

    def close(self) -> None:
        """Close the underlying connection."""

        try:
            self._conn.close()
        except Exception:  # pragma: no cover - defensive
            _log.exception("Error closing database connection")


_DB_SINGLETON: Database | None = None


def get_db() -> Database:
    """Return a process-wide :class:`Database` singleton (app data dir)."""

    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        _DB_SINGLETON = Database(db_path())
    return _DB_SINGLETON
