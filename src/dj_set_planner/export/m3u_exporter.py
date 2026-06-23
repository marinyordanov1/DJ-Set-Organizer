"""Extended-M3U export.

Writes a Rekordbox-importable ``.m3u`` playlist from a :class:`SetPlan`:

    #EXTM3U
    #EXTINF:<seconds>,<artist> - <title>
    <absolute file path>
    ...

The order of the file follows ``plan.tracks`` (already ordered by position).
Absolute paths and the ``#EXTINF`` duration line are what Rekordbox (and most
players) need to import a playlist correctly.
"""

from __future__ import annotations

import os

from ..domain.models import SetPlan, Track
from ..utils.logging import get_logger

_log = get_logger(__name__)


def _extinf_label(track: Track) -> str:
    """Build the ``<artist> - <title>`` label for an #EXTINF line.

    Falls back to the file stem for the title and to a generic artist when the
    tag is missing, so the line is always well-formed.
    """

    title = (track.title or "").strip()
    if not title:
        # Last-resort title: the file name without extension.
        title = os.path.splitext(os.path.basename(track.file_path))[0]

    artist = (track.artist or "").strip() or "Unknown Artist"
    return f"{artist} - {title}"


def export_m3u(
    plan: SetPlan,
    tracks_by_id: dict[int, Track],
    out_path: str,
) -> str:
    """Write ``plan`` to ``out_path`` as an extended M3U playlist.

    Parameters
    ----------
    plan:
        The set plan whose ordered ``tracks`` drive the playlist order.
    tracks_by_id:
        Lookup from ``track_id`` to the full :class:`Track` (for path / tags).
    out_path:
        Destination ``.m3u`` path. Parent directories are created if missing.

    Returns
    -------
    str
        The absolute path actually written.

    Notes
    -----
    Tracks whose id is not present in ``tracks_by_id`` are skipped with a
    warning rather than aborting the whole export.
    """

    # Ensure the parent directory exists (create it if needed).
    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines: list[str] = ["#EXTM3U"]

    for slot in plan.tracks:
        track = tracks_by_id.get(slot.track_id)
        if track is None:
            _log.warning(
                "export_m3u: no Track for track_id=%s (position %s); skipping",
                slot.track_id,
                slot.position,
            )
            continue

        # #EXTINF wants whole seconds; -1 signals "unknown length" per the
        # extended-M3U convention.
        duration = track.duration_seconds
        seconds = int(duration) if duration is not None else -1

        lines.append(f"#EXTINF:{seconds},{_extinf_label(track)}")
        # Absolute path on its own line — Rekordbox resolves these directly.
        lines.append(os.path.abspath(track.file_path))

    # Trailing newline so the file ends cleanly.
    content = "\n".join(lines) + "\n"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    _log.info("export_m3u: wrote %d tracks to %s", len(plan.tracks), out_path)
    return out_path
