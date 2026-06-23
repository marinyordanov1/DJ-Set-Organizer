"""Tests for analysis.rekordbox_import.

Uses a small inline Rekordbox collection XML written to ``tmp_path`` (no real
audio files). Covers: parsing, URL-decoding of ``file://`` Locations with
``%20`` spaces, Tonality -> Camelot mapping, applying data onto Tracks (by
basename and by full path), and graceful handling of malformed/missing XML.
"""

from __future__ import annotations

from dj_set_planner.analysis.rekordbox_import import (
    apply_rekordbox_to_tracks,
    parse_rekordbox_xml,
)
from dj_set_planner.domain.models import Track

# A minimal-but-realistic collection export. Note:
#   * Location is a file:// URL with %20 encoded spaces (and a localhost host).
#   * Track 1 has Tonality "Am" -> Camelot 8A.
#   * Track 2 has Tonality "C"  -> Camelot 8B.
#   * Track 3 has AverageBpm="0.00" (un-analysed) -> bpm should be None.
#   * The PLAYLISTS section contains TRACK *references* (Key only) that must be
#     ignored by the parser.
_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.0.0" Company="AlphaTheta"/>
  <COLLECTION Entries="3">
    <TRACK TrackID="1" Name="Sunset Groove" Artist="DJ One" Album="Beach Vol 1"
           Genre="Organic House" AverageBpm="120.00" Tonality="Am"
           TotalTime="312"
           Location="file://localhost/Users/dj/Music/Sunset%20Groove.mp3"/>
    <TRACK TrackID="2" Name="Daylight" Artist="DJ Two" Album="Day"
           Genre="Deep House" AverageBpm="122.50" Tonality="C"
           TotalTime="305"
           Location="file:///Users/dj/Music/sub%20folder/Daylight.flac"/>
    <TRACK TrackID="3" Name="Unanalysed" Artist="DJ Three"
           AverageBpm="0.00" Tonality=""
           Location="file://localhost/Users/dj/Music/Unanalysed.wav"/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Name="My List" Type="1" KeyType="0" Entries="1">
        <TRACK Key="1"/>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
"""


def _write_xml(tmp_path, text: str) -> str:
    p = tmp_path / "collection.xml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parse_decodes_location_and_maps_camelot(tmp_path):
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    rows = parse_rekordbox_xml(xml_path)

    # Three TRACK rows in COLLECTION (the PLAYLISTS reference is ignored).
    assert len(rows) == 3
    by_title = {r["title"]: r for r in rows}

    sunset = by_title["Sunset Groove"]
    # %20 must be decoded back to a real space and the file:// scheme stripped.
    assert sunset["file_path"] == "/Users/dj/Music/Sunset Groove.mp3"
    assert sunset["artist"] == "DJ One"
    assert sunset["genre"] == "Organic House"
    assert sunset["bpm"] == 120.0
    assert sunset["duration_seconds"] == 312
    assert sunset["musical_key"] == "Am"
    # Tonality "Am" -> Camelot 8A.
    assert sunset["camelot_key"] == "8A"

    daylight = by_title["Daylight"]
    assert daylight["file_path"] == "/Users/dj/Music/sub folder/Daylight.flac"
    assert daylight["camelot_key"] == "8B"  # "C" major -> 8B

    # Un-analysed track: zero BPM and empty Tonality become None.
    unanalysed = by_title["Unanalysed"]
    assert unanalysed["bpm"] is None
    assert unanalysed["musical_key"] is None
    assert unanalysed["camelot_key"] is None


def test_apply_fills_bpm_and_camelot_by_basename(tmp_path):
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)

    # Track lives in a DIFFERENT folder than Rekordbox recorded, so only the
    # basename matches — exercises the basename-fallback path. Missing bpm/key.
    track = Track(id=1, file_path="/some/other/place/Sunset Groove.mp3")
    matched = apply_rekordbox_to_tracks([track], xml_path)

    assert matched == 1
    assert track.bpm == 120.0
    assert track.musical_key == "Am"
    assert track.camelot_key == "8A"
    assert track.duration_seconds == 312


def test_apply_overrides_wrong_bpm_key(tmp_path):
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)

    # Exact full-path match. The Track has a WRONG existing bpm/key (e.g. a
    # half-time tag — 61.25 instead of the real 122.5). Rekordbox is
    # AUTHORITATIVE and must overwrite them; duration is filled (was missing).
    track = Track(
        id=2,
        file_path="/Users/dj/Music/sub folder/Daylight.flac",
        bpm=61.25,
        musical_key="Em",
        camelot_key="9A",
    )
    matched = apply_rekordbox_to_tracks([track], xml_path)

    assert matched == 1
    assert track.bpm == 122.5          # overridden, not the wrong half-time value
    assert track.musical_key == "C"
    assert track.camelot_key == "8B"
    assert track.duration_seconds == 305


def test_apply_returns_zero_for_unmatched(tmp_path):
    xml_path = _write_xml(tmp_path, _SAMPLE_XML)
    track = Track(id=9, file_path="/library/nothing_here.mp3")
    assert apply_rekordbox_to_tracks([track], xml_path) == 0


def test_missing_file_returns_empty(tmp_path):
    missing = str(tmp_path / "does_not_exist.xml")
    assert parse_rekordbox_xml(missing) == []
    assert apply_rekordbox_to_tracks([Track(id=1, file_path="/x.mp3")], missing) == 0


def test_malformed_xml_is_tolerated(tmp_path):
    bad = _write_xml(tmp_path, "<DJ_PLAYLISTS><COLLECTION><TRACK ")  # truncated
    assert parse_rekordbox_xml(bad) == []
    assert apply_rekordbox_to_tracks([Track(id=1, file_path="/x.mp3")], bad) == 0
