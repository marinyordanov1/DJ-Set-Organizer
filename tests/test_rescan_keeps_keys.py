"""Re-scanning a folder must not wipe BPM/key that came from Rekordbox."""

from __future__ import annotations

from dj_set_planner.db import database


def test_rescan_keeps_imported_bpm_and_key(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "_DB_SINGLETON", database.Database(":memory:"))
    from dj_set_planner.app import PlannerService

    (tmp_path / "a.mp3").write_bytes(b"")  # no tags at all
    service = PlannerService()
    (track,) = service.scan(str(tmp_path))
    track.bpm, track.musical_key, track.camelot_key = 130.0, "1A", "1A"  # as XML import
    track.analyzed_at = "2026-10-12T20:00:00+00:00"
    service.tracks.upsert(track)

    (again,) = service.scan(str(tmp_path))

    assert (again.bpm, again.camelot_key, again.analyzed_at) == (
        130.0, "1A", "2026-10-12T20:00:00+00:00"
    )
