"""Centralized logging.

Use ``get_logger(__name__)`` everywhere. Logging is configured exactly once,
on the root of the ``dj_set_planner`` logger tree, so handlers are not added
repeatedly. NEVER silently swallow exceptions elsewhere — log them via a
logger obtained here.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def _configure_once() -> None:
    """Attach a single StreamHandler to the package root logger."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("DJ_SET_PLANNER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("dj_set_planner")
    root.setLevel(level)

    # Only add a handler if none exist, so reconfiguration / re-import is safe.
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)

    # Avoid double-emission through the global root logger.
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Names are namespaced under ``dj_set_planner`` so a single handler on the
    package root captures everything.
    """

    _configure_once()
    if not name.startswith("dj_set_planner"):
        name = f"dj_set_planner.{name}"
    return logging.getLogger(name)
