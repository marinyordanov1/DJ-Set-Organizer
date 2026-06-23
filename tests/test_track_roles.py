"""Tests for planning.track_roles.

These tests construct small, explicit synthetic Track/TrackFeatures objects so
the assertions are self-contained and robust (no reliance on real audio). They
verify the three spec scenarios:

* a low-energy, high-intro-suitability track ranks INTRO as a top role;
* a high-energy, high-peak-potential track *within the energy cap* ranks
  MAIN_PEAK among its top roles;
* a high-outro-suitability, low-energy track ranks OUTRO as a top role.

Plus basic invariants: all eight roles present, every fit in 0..1, and the
MAIN_PEAK hard-cap rule (a too-loud track cannot be the main peak).
"""

from __future__ import annotations

from dj_set_planner.domain.enums import TrackRole
from dj_set_planner.domain.models import Track, TrackFeatures
from dj_set_planner.planning.context_profiles import default_profile
from dj_set_planner.planning.track_roles import best_role, role_fit_scores


def _track(track_id: int) -> Track:
    """Minimal Track (the role rules read features, not tags)."""

    return Track(id=track_id, file_path=f"/library/sample_{track_id}.mp3")


def _features(track_id: int, **overrides: float) -> TrackFeatures:
    """TrackFeatures defaulting to neutral 0.5, with selected overrides."""

    return TrackFeatures(track_id=track_id, **overrides)


def _rank(scores: dict[str, float]) -> list[str]:
    """Roles ordered best-first."""

    return [r for r, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def test_all_roles_present_and_in_range() -> None:
    profile = default_profile()
    f = _features(1)
    scores = role_fit_scores(_track(1), f, profile)

    # Every TrackRole value is scored.
    assert set(scores.keys()) == {r.value for r in TrackRole}
    # Every fit is a valid 0..1 float.
    for value in scores.values():
        assert 0.0 <= value <= 1.0


def test_low_energy_high_intro_suitability_is_intro() -> None:
    """A gentle, restaurant-safe opener should top out at INTRO."""

    profile = default_profile()
    f = _features(
        1,
        energy_score=0.32,
        intro_suitability=0.92,
        outro_suitability=0.40,
        restaurant_safety_score=0.95,
        peak_potential=0.15,
        danceability_score=0.45,
    )
    scores = role_fit_scores(_track(1), f, profile)

    # INTRO is the single best role, and best_role agrees.
    assert _rank(scores)[0] == TrackRole.INTRO.value
    assert best_role(_track(1), f, profile) == TrackRole.INTRO.value


def test_high_energy_high_peak_within_cap_is_main_peak() -> None:
    """A danceable peak track inside the energy window/cap ranks MAIN_PEAK top."""

    profile = default_profile()
    # energy 0.75 sits in [0.70, 0.80] and below the 0.85 hard cap.
    f = _features(
        24,
        energy_score=0.75,
        peak_potential=0.90,
        danceability_score=0.90,
        groove_score=0.70,
        intro_suitability=0.20,
        outro_suitability=0.20,
        restaurant_safety_score=0.55,
    )
    scores = role_fit_scores(_track(24), f, profile)

    # MAIN_PEAK is among the top two roles (a strong small_peak score is OK,
    # but the peak track must register clearly as a main-peak candidate).
    assert TrackRole.MAIN_PEAK.value in _rank(scores)[:2]
    # And it should be a high absolute fit.
    assert scores[TrackRole.MAIN_PEAK.value] >= 0.7


def test_high_outro_suitability_low_energy_is_outro() -> None:
    """A gentle closer should top out at OUTRO."""

    profile = default_profile()
    f = _features(
        30,
        energy_score=0.40,
        outro_suitability=0.93,
        intro_suitability=0.45,
        restaurant_safety_score=0.90,
        peak_potential=0.15,
        danceability_score=0.45,
    )
    scores = role_fit_scores(_track(30), f, profile)

    assert _rank(scores)[0] == TrackRole.OUTRO.value
    assert best_role(_track(30), f, profile) == TrackRole.OUTRO.value


def test_main_peak_disqualified_above_hard_cap() -> None:
    """A track louder than the harshness cap cannot be the main peak."""

    profile = default_profile()  # avoid_energy_above 0.85, max_energy 0.78
    f = _features(
        99,
        energy_score=0.95,  # above both caps
        peak_potential=0.95,
        danceability_score=0.95,
    )
    scores = role_fit_scores(_track(99), f, profile)

    # Hard rule: over the ceiling -> main-peak fit is exactly 0.
    assert scores[TrackRole.MAIN_PEAK.value] == 0.0
    assert best_role(_track(99), f, profile) != TrackRole.MAIN_PEAK.value


def test_best_role_is_deterministic() -> None:
    """Same inputs always yield the same role (stable tie-breaks)."""

    profile = default_profile()
    f = _features(7, energy_score=0.5)  # near-neutral, likely several close fits
    first = best_role(_track(7), f, profile)
    for _ in range(5):
        assert best_role(_track(7), f, profile) == first


def test_fixture_peak_track_ranks_main_peak_high(
    sample_tracks, sample_features, sofia_profile
) -> None:
    """Sanity check against the shared fixtures.

    Fixture track 24 (energy 0.775, high peak/dance, under the 0.85 cap) should
    have a meaningfully high MAIN_PEAK fit and rank it among its top roles.
    """

    track = next(t for t in sample_tracks if t.id == 24)
    f = sample_features[24]
    scores = role_fit_scores(track, f, sofia_profile)

    assert scores[TrackRole.MAIN_PEAK.value] > 0.5
    assert TrackRole.MAIN_PEAK.value in _rank(scores)[:3]
