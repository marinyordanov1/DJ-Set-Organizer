"""Filesystem paths for application data.

Resolves a per-OS application data directory and the SQLite database path.
The data directory is created on first access.
"""

from __future__ import annotations

import os
import sys

_APP_DIR_NAME = "DJSetPlanner"


def app_data_dir() -> str:
    """Return (creating if necessary) the per-user app data directory.

    - macOS:   ~/Library/Application Support/DJSetPlanner
    - Windows: %APPDATA%\\DJSetPlanner
    - Linux/other: $XDG_DATA_HOME/DJSetPlanner or ~/.local/share/DJSetPlanner
    """

    home = os.path.expanduser("~")

    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif os.name == "nt":  # Windows
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    else:  # Linux / other POSIX
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")

    path = os.path.join(base, _APP_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def db_path() -> str:
    """Return the absolute path to the SQLite database file."""

    return os.path.join(app_data_dir(), "dj_set_planner.db")
