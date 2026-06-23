"""Tests for the M3U and CSV exporters.

Builds a tiny two-track :class:`SetPlan` (no real audio files needed) and
exports it in both formats into pytest's ``tmp_path``, asserting the structure
and content of each output file.
"""

from __future__ import annotations

import csv

import pytest

from dj_set_planner.domain.models import (
    EventProfile,
    SetPlan,
    SetPlanTrack,
    SetSegment,
    Track,
    TrackFeatures,
)
from dj_set_planner.export.csv_exporter import CSV_COLUMNS, export_csv
from dj_set_planner.export.m3u_exporter import export_m3u


@pytest.fixture
def tiny_plan() -> tuple[
    SetPlan, dict[int, Track], dict[int, TrackFeatures]
]:
    """A minimal 2-track plan plus matching Track/TrackFeatures lookups."""

    profile = EventProfile(
        id=None,
        name="Test Profile",
        venue_type="RESTAURANT",
        time_of_day="DAY",
        crowd_state="EATING+TALKING+PARTIAL_DANCING",
        desired_energy="BALANCED",
        peak_strategy="ONE_MAIN_PEAK",
        target_duration_minutes=120,
        min_energy=0.30,
        max_energy=0.78,
        main_peak_energy=0.75,
    )

    # Two tracks with absolute paths so the m3u is Rekordbox-importable.
    track_a = Track(
        id=1,
        file_path="/library/intro/opening_groove.mp3",
        title="Opening Groove",
        artist="Artist A",
        album="Album A",
        genre="Deep House",
        duration_seconds=185,
        bpm=90.0,
        musical_key="Am",
        camelot_key="8A",
    )
    track_b = Track(
        id=2,
        file_path="/library/peak/the_big_one.mp3",
        # Note the comma in the title — exercises CSV quoting.
        title="The Big One, Pt. 1",
        artist="Artist B",
        album="Album B",
        genre="Melodic House",
        duration_seconds=240,
        bpm=92.0,
        musical_key="C",
        camelot_key="8B",
    )

    tracks_by_id = {1: track_a, 2: track_b}

    features_by_id = {
        1: TrackFeatures(track_id=1, energy_score=0.34),
        2: TrackFeatures(track_id=2, energy_score=0.76),
    }

    slot_a = SetPlanTrack(
        track_id=1,
        position=0,
        role="INTRO",
        transition_score=0.0,
        position_score=0.81,
        explanation="Opens the set as a gentle intro.",
    )
    slot_b = SetPlanTrack(
        track_id=2,
        position=1,
        role="MAIN_PEAK",
        transition_score=0.72,
        position_score=0.88,
        explanation="Lands the main peak, mixing in smoothly.",
    )

    segments = [
        SetSegment("intro", 0.0, 0.5, 0.30, 0.40, "INTRO"),
        SetSegment("main_peak", 0.5, 1.0, 0.72, 0.78, "MAIN_PEAK"),
    ]

    plan = SetPlan(
        event_profile=profile,
        tracks=[slot_a, slot_b],
        total_duration_seconds=425,
        target_duration_seconds=7200,
        total_score=0.845,
        segments=segments,
        energy_points=[0.34, 0.76],
    )

    return plan, tracks_by_id, features_by_id


def test_export_m3u(tiny_plan, tmp_path):
    plan, tracks_by_id, _ = tiny_plan

    out_path = tmp_path / "set.m3u"
    returned = export_m3u(plan, tracks_by_id, str(out_path))

    # Returns the absolute path it wrote.
    assert returned == str(out_path)
    assert out_path.exists()

    text = out_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Extended-M3U header on the first line.
    assert lines[0] == "#EXTM3U"
    assert "#EXTM3U" in text

    # Both absolute file paths present, and in plan order.
    path_a = "/library/intro/opening_groove.mp3"
    path_b = "/library/peak/the_big_one.mp3"
    assert path_a in text
    assert path_b in text
    assert text.index(path_a) < text.index(path_b)

    # Each track has an #EXTINF line with its duration and "artist - title".
    assert "#EXTINF:185,Artist A - Opening Groove" in lines
    assert "#EXTINF:240,Artist B - The Big One, Pt. 1" in lines

    # Structure: header + (EXTINF + path) per track == 1 + 2*2 == 5 lines.
    assert len(lines) == 5


def test_export_csv(tiny_plan, tmp_path):
    plan, tracks_by_id, features_by_id = tiny_plan

    out_path = tmp_path / "set.csv"
    returned = export_csv(
        plan, tracks_by_id, features_by_id, str(out_path)
    )

    assert returned == str(out_path)
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    # Header row matches the exact contract column list.
    assert rows[0] == CSV_COLUMNS
    assert rows[0] == [
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

    # Exactly two data rows.
    data_rows = rows[1:]
    assert len(data_rows) == 2

    # First data row — the intro track.
    row0 = dict(zip(CSV_COLUMNS, data_rows[0]))
    assert row0["position"] == "0"
    assert row0["title"] == "Opening Groove"
    assert row0["artist"] == "Artist A"
    assert row0["file_path"] == "/library/intro/opening_groove.mp3"
    assert row0["role"] == "INTRO"
    assert row0["duration_seconds"] == "185"
    assert float(row0["bpm"]) == 90.0
    # key column prefers the Camelot key.
    assert row0["key"] == "8A"
    assert float(row0["energy_score"]) == 0.34
    assert float(row0["transition_score"]) == 0.0
    assert float(row0["position_score"]) == 0.81
    assert row0["explanation"] == "Opens the set as a gentle intro."

    # Second data row — the main-peak track (title contains a comma).
    row1 = dict(zip(CSV_COLUMNS, data_rows[1]))
    assert row1["position"] == "1"
    assert row1["title"] == "The Big One, Pt. 1"
    assert row1["artist"] == "Artist B"
    assert row1["file_path"] == "/library/peak/the_big_one.mp3"
    assert row1["role"] == "MAIN_PEAK"
    assert row1["key"] == "8B"
    assert float(row1["energy_score"]) == 0.76
    assert float(row1["transition_score"]) == 0.72


def test_m3u_creates_parent_dir(tiny_plan, tmp_path):
    """The exporter creates missing parent directories."""

    plan, tracks_by_id, _ = tiny_plan
    nested = tmp_path / "deep" / "nested" / "dir" / "set.m3u"

    assert not nested.parent.exists()
    export_m3u(plan, tracks_by_id, str(nested))
    assert nested.exists()
    assert nested.read_text(encoding="utf-8").startswith("#EXTM3U")
