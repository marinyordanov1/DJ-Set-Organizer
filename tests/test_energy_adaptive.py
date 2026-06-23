"""Tests for library-relative energy normalization + manual energy override.

These guard the fix for the real-world failure where a library of uniformly
loud club tracks (energy 0.55..0.91) gave no "light" intro material, so the
planner opened on the wrong track. Adaptive energy rescales the library onto the
profile band so the relatively-lightest track opens; the manual override lets
the DJ correct any single track by hand.
"""

from __future__ import annotations

import pytest

from dj_set_planner.analysis.heuristic_scorer import derive_energy_dependent
from dj_set_planner.domain.models import Track, TrackFeatures
from dj_set_planner.planning.beam_search_planner import plan_set
from dj_set_planner.planning.context_profiles import default_profile
from dj_set_planner.planning.energy_normalize import relativize_features


def _feat(track_id: int, energy: float) -> TrackFeatures:
    return TrackFeatures(
        track_id=track_id,
        energy_score=energy,
        danceability_score=0.65,
        mood_brightness=0.55,
        groove_score=0.6,
        vocal_density=0.5,
        intro_suitability=0.5,
        outro_suitability=0.5,
        peak_potential=0.5,
        restaurant_safety_score=0.5,
        mixability_score=0.6,
    )


def _loud_library(n: int = 14):
    """A library of uniformly energetic tracks (energy 0.55..0.91)."""

    keys = ["8A", "8B", "9A", "9B", "10A", "10B", "11A"]
    tracks, feats = [], {}
    for i in range(n):
        tid = i + 1
        energy = 0.55 + (0.91 - 0.55) * (i / (n - 1))
        tracks.append(
            Track(
                id=tid,
                file_path=f"/lib/loud_{tid:02d}.mp3",
                title=f"Loud {tid}",
                artist=f"Artist {tid}",
                album=None,
                genre="Tech House",
                duration_seconds=330,
                bpm=124.0,
                musical_key=None,
                camelot_key=keys[i % len(keys)],
                analyzed_at="2026-06-21T12:00:00",
            )
        )
        feats[tid] = _feat(tid, energy)
    return tracks, feats


def test_derive_energy_dependent_is_consistent() -> None:
    base = _feat(1, 0.9)
    light = derive_energy_dependent(base, 0.30)
    heavy = derive_energy_dependent(base, 0.85)

    assert light.energy_score == pytest.approx(0.30)
    assert heavy.energy_score == pytest.approx(0.85)
    # Calmer track is a better intro, worse peak, and more restaurant-safe.
    assert light.intro_suitability > heavy.intro_suitability
    assert light.peak_potential < heavy.peak_potential
    assert light.restaurant_safety_score > heavy.restaurant_safety_score
    # Energy-independent fields are preserved untouched.
    assert light.danceability_score == base.danceability_score
    assert light.mood_brightness == base.mood_brightness


def test_relativize_maps_library_extremes_onto_profile_band() -> None:
    tracks, feats = _loud_library()
    profile = default_profile()  # min 0.30, max 0.78

    out = relativize_features(tracks, feats, profile)
    energies = sorted(out[t.id].energy_score for t in tracks)

    # The lightest track lands at (or near) the profile floor, the loudest at
    # the ceiling — the absolute 0.55..0.91 range is stretched onto 0.30..0.78.
    assert energies[0] == pytest.approx(profile.min_energy, abs=0.03)
    assert energies[-1] == pytest.approx(profile.max_energy, abs=0.03)
    # Order is preserved (track 1 was lightest, track n was loudest).
    assert out[1].energy_score < out[len(tracks)].energy_score


def test_too_small_library_is_left_unchanged() -> None:
    tracks, feats = _loud_library(2)
    profile = default_profile()
    out = relativize_features(tracks, feats, profile)
    # Fewer than 3 tracks -> no normalization, original features returned as-is.
    assert out[1].energy_score == feats[1].energy_score


def test_loud_library_opens_light_and_builds() -> None:
    tracks, feats = _loud_library()
    profile = default_profile()

    plan = plan_set(tracks, feats, profile, [])
    assert plan.tracks, "expected a non-empty plan"

    pts = plan.energy_points
    # The opener sits in the lower half of the (normalized) energy band even
    # though every track is absolutely loud — this is the core fix.
    assert pts[0] <= 0.5
    # And the set genuinely builds: its peak is well above the opener.
    assert max(pts) >= pts[0] + 0.2


def test_adaptive_off_keeps_absolute_energies() -> None:
    tracks, feats = _loud_library()
    profile = default_profile()
    plan = plan_set(tracks, feats, profile, [], adaptive_energy=False)
    # Without normalization every emitted point is one of the raw loud energies.
    raw = {round(f.energy_score, 6) for f in feats.values()}
    for e in plan.energy_points:
        assert round(e, 6) in raw
