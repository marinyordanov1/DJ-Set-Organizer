"""Application entry point.

Run with ``python -m dj_set_planner.main`` (or the ``dj-set-planner`` console
script). Creates the Flask app, prints the local URL, opens the default browser
to it, and runs the server on ``127.0.0.1:5000``.
"""

from __future__ import annotations

import threading
import webbrowser

from .app import run_server
from .utils.logging import get_logger

_log = get_logger(__name__)

HOST = "127.0.0.1"
PORT = 5000


def main() -> None:
    """Start the local web server and open the browser to it.

    The browser is opened from a short-lived background timer so the page load
    races against (and lands just after) the server coming up — the blocking
    ``run_server`` call below keeps the process alive in the foreground.
    """

    url = f"http://{HOST}:{PORT}/"
    print(f"AI DJ Set Planner is starting at {url}")
    print("Press Ctrl+C to stop the server.")

    # Open the browser slightly after the server has had a moment to bind, so
    # the first request doesn't beat the listener. webbrowser.open is best-effort
    # (headless environments simply have nothing to open).
    def _open_browser() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - environment without a browser
            _log.warning("Could not open a web browser automatically; visit %s", url)

    threading.Timer(1.0, _open_browser).start()

    run_server(host=HOST, port=PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
