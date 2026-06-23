"""Tests for the harmonic_ratio descriptor (melodic <-> percussive).

Covers: value range, the librosa HPSS path (a tone reads more melodic than
noise), the heuristic fallback, that derive_energy_dependent preserves it, and
that it round-trips through the DB (which also exercises the schema/migration).
"""

from __future__ import annotations

import math
import struct
import wave

import pytest

from dj_set_planner.analysis.heuristic_scorer import (
    derive_energy_dependent,
    score_from_metadata,
)
from dj_set_planner.db.database import Database
from dj_set_planner.db.repositories import FeatureRepository, TrackRepository
from dj_set_planner.domain.models import Track, TrackFeatures


def test_heuristic_harmonic_ratio_in_range() -> None:
    for genre in ("Deep House", "Techno", "Ambient", None):
        f = score_from_metadata(
            Track(id=1, file_path=f"/x/{genre}.mp3", genre=genre, bpm=122.0)
        )
        assert 0.0 <= f.harmonic_ratio <= 1.0


def test_derive_energy_dependent_preserves_harmonic_ratio() -> None:
    base = TrackFeatures(track_id=1, harmonic_ratio=0.82)
    out = derive_energy_dependent(base, 0.4)
    # harmonic_ratio is NOT an energy-dependent field — it must survive untouched.
    assert out.harmonic_ratio == 0.82


def test_harmonic_ratio_round_trips_through_db() -> None:
    db = Database(":memory:")
    track = TrackRepository(db).upsert(Track(id=None, file_path="/x/a.mp3"))
    fr = FeatureRepository(db)
    fr.upsert(TrackFeatures(track_id=track.id, harmonic_ratio=0.73))
    got = fr.get(track.id)
    assert got is not None
    assert got.harmonic_ratio == pytest.approx(0.73)


def _write_wav(path: str, samples: list[float], sr: int = 22050) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples
        )
        w.writeframes(frames)


def test_librosa_tone_is_more_melodic_than_noise(tmp_path) -> None:
    pytest.importorskip("librosa")
    import random

    from dj_set_planner.analysis.librosa_extractor import LibrosaFeatureExtractor

    ex = LibrosaFeatureExtractor()
    if not ex.is_available():
        pytest.skip("librosa not importable")

    sr = 22050
    n = sr * 5
    # Pure sustained tone -> harmonic. Deterministic white noise -> percussive.
    tone = [0.6 * math.sin(2 * math.pi * 220 * i / sr) for i in range(n)]
    rng = random.Random(42)
    noise = [rng.uniform(-0.6, 0.6) for _ in range(n)]

    tone_path = str(tmp_path / "tone.wav")
    noise_path = str(tmp_path / "noise.wav")
    _write_wav(tone_path, tone, sr)
    _write_wav(noise_path, noise, sr)

    tone_f = ex.extract(Track(id=1, file_path=tone_path))
    noise_f = ex.extract(Track(id=2, file_path=noise_path))

    assert tone_f.harmonic_ratio > noise_f.harmonic_ratio


def test_librosa_fills_missing_bpm_and_key(tmp_path) -> None:
    pytest.importorskip("librosa")
    from dj_set_planner.analysis.librosa_extractor import LibrosaFeatureExtractor

    ex = LibrosaFeatureExtractor()
    if not ex.is_available():
        pytest.skip("librosa not importable")

    sr = 22050
    n = sr * 5
    # A C-major triad (C-E-G) so key estimation has something to lock onto.
    def _tone(f):
        return [math.sin(2 * math.pi * f * i / sr) for i in range(n)]
    chord = [(a + b + c) / 3 for a, b, c in zip(_tone(261.63), _tone(329.63), _tone(392.0))]
    path = str(tmp_path / "cmaj.wav")
    _write_wav(path, chord, sr)

    t = Track(id=1, file_path=path)  # no bpm, no key
    ex.extract(t)
    assert t.musical_key is not None        # key was estimated
    assert t.camelot_key is not None        # and mapped to Camelot


def test_librosa_does_not_override_existing_bpm_key(tmp_path) -> None:
    pytest.importorskip("librosa")
    from dj_set_planner.analysis.librosa_extractor import LibrosaFeatureExtractor

    ex = LibrosaFeatureExtractor()
    if not ex.is_available():
        pytest.skip("librosa not importable")

    sr = 22050
    chord = [math.sin(2 * math.pi * 261.63 * i / sr) for i in range(sr * 5)]
    path = str(tmp_path / "c.wav")
    _write_wav(path, chord, sr)

    t = Track(id=1, file_path=path, bpm=124.0, musical_key="Am", camelot_key="8A")
    ex.extract(t)
    assert t.bpm == 124.0 and t.camelot_key == "8A"  # real data preserved
