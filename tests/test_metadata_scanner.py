"""Tests for analysis.scanner and analysis.metadata_reader.

These run WITHOUT any real music: the scanner is exercised with empty dummy
files of various extensions, and the metadata reader is exercised against a
tiny silent WAV synthesized with the stdlib :mod:`wave` module so mutagen can
read a real (if trivial) duration.
"""

from __future__ import annotations

import os
import wave

import pytest

from dj_set_planner.analysis.scanner import (
    SUPPORTED_EXTENSIONS,
    scan_folder,
)
from dj_set_planner.analysis.metadata_reader import read_metadata
from dj_set_planner.domain.models import Track


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _touch(path: str) -> None:
    """Create an empty file, making parent directories as needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def _write_silent_wav(path: str, *, seconds: float = 1.0, rate: int = 8000) -> None:
    """Write a tiny mono 16-bit silent WAV so mutagen can read its duration."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_frames = int(seconds * rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


# A single valid MPEG-1 Layer III frame: mono, 128 kbps, 44.1 kHz. The 4-byte
# header FF FB 90 64 yields a 417-byte frame; the body is silence. Repeating it
# gives mutagen real MPEG frames to sync to so it recognizes a true MP3.
_MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * (417 - 4)


def _write_silent_mp3(path: str, *, n_frames: int = 20) -> None:
    """Write a tiny valid silent MP3 (no tags) that mutagen recognizes."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(_MP3_FRAME * n_frames)


# --------------------------------------------------------------------------- #
# scanner.scan_folder
# --------------------------------------------------------------------------- #
def test_scan_picks_supported_extensions_and_ignores_others(tmp_path):
    base = str(tmp_path)
    # Supported files (one per supported extension), including upper-case ext.
    supported = [
        os.path.join(base, "a.mp3"),
        os.path.join(base, "b.wav"),
        os.path.join(base, "c.flac"),
        os.path.join(base, "d.aiff"),
        os.path.join(base, "e.aif"),
        os.path.join(base, "f.m4a"),
        os.path.join(base, "g.ogg"),
        os.path.join(base, "UPPER.MP3"),  # extension match is case-insensitive
    ]
    # Unsupported / junk files that must be ignored.
    unsupported = [
        os.path.join(base, "notes.txt"),
        os.path.join(base, "cover.jpg"),
        os.path.join(base, "playlist.m3u"),
        os.path.join(base, "video.mp4"),
        os.path.join(base, "no_extension"),
    ]
    for p in supported + unsupported:
        _touch(p)

    found = scan_folder(base)

    found_set = set(found)
    for p in supported:
        assert os.path.abspath(p) in found_set
    for p in unsupported:
        assert os.path.abspath(p) not in found_set
    assert len(found) == len(supported)


def test_scan_is_recursive_and_sorted(tmp_path):
    base = str(tmp_path)
    _touch(os.path.join(base, "z_top.mp3"))
    _touch(os.path.join(base, "sub", "a_deep.flac"))
    _touch(os.path.join(base, "sub", "nested", "m_mid.m4a"))

    found = scan_folder(base)

    assert len(found) == 3
    # All returned paths are absolute.
    assert all(os.path.isabs(p) for p in found)
    # Recursion reached the nested file.
    assert any(p.endswith(os.path.join("nested", "m_mid.m4a")) for p in found)
    # Results are sorted.
    assert found == sorted(found)


def test_scan_ignores_dotfiles_and_dot_dirs(tmp_path):
    base = str(tmp_path)
    _touch(os.path.join(base, "real.mp3"))
    _touch(os.path.join(base, ".hidden.mp3"))  # dotfile -> ignored
    _touch(os.path.join(base, "._AppleDouble.wav"))  # dotfile -> ignored
    _touch(os.path.join(base, ".cache", "buried.flac"))  # in dot-dir -> ignored

    found = scan_folder(base)

    assert len(found) == 1
    assert found[0].endswith("real.mp3")


def test_scan_missing_folder_returns_empty(tmp_path):
    missing = os.path.join(str(tmp_path), "does_not_exist")
    assert scan_folder(missing) == []
    assert scan_folder("") == []


def test_supported_extensions_match_contract():
    assert SUPPORTED_EXTENSIONS == frozenset(
        {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
    )


# --------------------------------------------------------------------------- #
# metadata_reader.read_metadata
# --------------------------------------------------------------------------- #
def test_read_metadata_no_tags_falls_back_to_filename_stem(tmp_path):
    path = os.path.join(str(tmp_path), "My Untagged Track.wav")
    _write_silent_wav(path)

    track = read_metadata(path)

    assert isinstance(track, Track)
    assert track.id is None
    assert track.file_path == os.path.abspath(path)
    # No tags -> title is the filename stem.
    assert track.title == "My Untagged Track"
    assert track.artist is None
    # mutagen can read the duration of a real (silent) WAV.
    assert track.duration_seconds is not None
    assert track.duration_seconds >= 1


def test_read_metadata_returns_absolute_path(tmp_path):
    path = os.path.join(str(tmp_path), "x.wav")
    _write_silent_wav(path)
    track = read_metadata(path)
    assert os.path.isabs(track.file_path)


def test_read_metadata_never_raises_on_garbage_file(tmp_path):
    # A non-audio file with an audio extension must not raise.
    path = os.path.join(str(tmp_path), "broken.mp3")
    with open(path, "wb") as fh:
        fh.write(b"this is definitely not a valid mp3 stream")

    track = read_metadata(path)

    assert isinstance(track, Track)
    assert track.title == "broken"  # falls back to stem
    assert track.file_path == os.path.abspath(path)


def test_read_metadata_reads_id3_tags_bpm_key_and_camelot(tmp_path):
    """An MP3 with ID3 title/artist/TBPM/TKEY should populate the Track."""

    pytest.importorskip("mutagen.id3")
    from mutagen.id3 import ID3, TBPM, TCON, TIT2, TKEY, TPE1, ID3NoHeaderError

    # Write a real (silent) MP3 so mutagen recognizes the container, then
    # attach ID3 frames. This exercises the production native-tag path.
    path = os.path.join(str(tmp_path), "tagged.mp3")
    _write_silent_mp3(path)

    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.add(TIT2(encoding=3, text="Sunset Groove"))
    tags.add(TPE1(encoding=3, text="DJ Test"))
    tags.add(TCON(encoding=3, text="Deep House"))
    tags.add(TBPM(encoding=3, text="122"))
    tags.add(TKEY(encoding=3, text="Am"))
    tags.save(path)

    track = read_metadata(path)

    assert track.title == "Sunset Groove"
    assert track.artist == "DJ Test"
    assert track.genre == "Deep House"
    assert track.bpm == 122.0
    assert track.musical_key == "Am"
    # to_camelot("Am") -> "8A"
    assert track.camelot_key == "8A"
