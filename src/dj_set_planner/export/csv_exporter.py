"""CSV export of a generated set plan.

Writes one header row plus one data row per ordered track. The column list is
fixed by CONTRACTS.md:

    position, title, artist, file_path, role, duration_seconds, bpm, key,
    energy_score, transition_score, position_score, explanation

Uses the standard-library :mod:`csv` module so quoting/escaping is handled
correctly for fields that contain commas (e.g. the explanation sentence).
"""

from __future__ import annotations

import csv
import os

from ..domain.models import SetPlan, Track, TrackFeatures
from ..utils.logging import get_logger

_log = get_logger(__name__)

# The exact, ordered column list from CONTRACTS.md. Keep in sync.
CSV_COLUMNS: list[str] = [
    "position",
    "title",
    "artist",
    "file_path",
    "role",
    "duration_seconds",
    "bpm",
    "key",
    "energy_score",
    "transition_score",
    "position_score",
    "explanation",
]


def export_csv(
    plan: SetPlan,
    tracks_by_id: dict[int, Track],
    features_by_id: dict[int, TrackFeatures],
    out_path: str,
) -> str:
    """Write ``plan`` to ``out_path`` as a CSV.

    Parameters
    ----------
    plan:
        The set plan whose ordered ``tracks`` become the data rows.
    tracks_by_id:
        Lookup from ``track_id`` to the full :class:`Track`.
    features_by_id:
        Lookup from ``track_id`` to its :class:`TrackFeatures` (for the
        ``energy_score`` column).
    out_path:
        Destination ``.csv`` path. Parent directories are created if missing.

    Returns
    -------
    str
        The absolute path actually written.

    Notes
    -----
    Slots whose id is missing from ``tracks_by_id`` are skipped with a warning.
    The ``key`` column uses the Camelot key when available, otherwise the raw
    musical key.
    """

    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)

        for slot in plan.tracks:
            track = tracks_by_id.get(slot.track_id)
            if track is None:
                _log.warning(
                    "export_csv: no Track for track_id=%s (position %s); skipping",
                    slot.track_id,
                    slot.position,
                )
                continue

            features = features_by_id.get(slot.track_id)
            # energy_score comes from features; default to neutral 0.5 when the
            # features blob is missing for this track.
            energy_score = (
                float(features.energy_score) if features is not None else 0.5
            )

            # Prefer the Camelot key for the DJ-facing "key" column; fall back
            # to the raw musical key, then to an empty string.
            key = track.camelot_key or track.musical_key or ""

            writer.writerow(
                [
                    slot.position,
                    track.title or "",
                    track.artist or "",
                    track.file_path,
                    slot.role,
                    track.duration_seconds
                    if track.duration_seconds is not None
                    else "",
                    track.bpm if track.bpm is not None else "",
                    key,
                    round(energy_score, 4),
                    round(float(slot.transition_score), 4),
                    round(float(slot.position_score), 4),
                    slot.explanation,
                ]
            )

    _log.info("export_csv: wrote %d tracks to %s", len(plan.tracks), out_path)
    return out_path
