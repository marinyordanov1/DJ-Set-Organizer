"""Clearing the library wipes tracks AND all derived rows (via FK cascade)."""

from __future__ import annotations

from dj_set_planner.db.database import Database
from dj_set_planner.db.repositories import (
    ConstraintRepository,
    FeatureOverrideRepository,
    FeatureRepository,
    TrackRepository,
)
from dj_set_planner.domain.models import Track, TrackFeatures


def test_clear_all_cascades() -> None:
    db = Database(":memory:")
    tracks = TrackRepository(db)
    feats = FeatureRepository(db)
    cons = ConstraintRepository(db)
    overrides = FeatureOverrideRepository(db)

    t = tracks.upsert(Track(id=None, file_path="/x/a.mp3"))
    feats.upsert(TrackFeatures(track_id=t.id, energy_score=0.6))
    cons.set_for_track(t.id, "MUST_PLAY")
    overrides.set_energy(t.id, 0.3)
    assert tracks.get_all() and feats.get_all() and cons.get_all() and overrides.get_all()

    removed = tracks.clear_all()

    assert removed == 1
    assert tracks.get_all() == []
    assert feats.get_all() == {}      # cascaded
    assert cons.get_all() == []       # cascaded
    assert overrides.get_all() == {}  # cascaded


def test_remove_not_in_syncs_to_scanned_folder() -> None:
    db = Database(":memory:")
    tracks = TrackRepository(db)
    feats = FeatureRepository(db)
    a = tracks.upsert(Track(id=None, file_path="/old/a.mp3"))
    tracks.upsert(Track(id=None, file_path="/new/b.mp3"))
    feats.upsert(TrackFeatures(track_id=a.id))

    removed = tracks.remove_not_in({"/new/b.mp3"})

    assert removed == 1
    assert [t.file_path for t in tracks.get_all()] == ["/new/b.mp3"]
    assert a.id not in feats.get_all()  # stale track's features cascaded away
