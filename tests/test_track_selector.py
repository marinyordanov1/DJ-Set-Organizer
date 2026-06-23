"""Tests for planning.track_selector.

Covers the weighted :func:`compute_position_score` (its 0..1 range, sensitivity
to energy-curve fit, neighbour transition handling, and penalties) plus the
candidate-pool / constraint helpers (AVOID dropped; MUST_PLAY / PREFERRED_* /
LOCK_POSITION surfaced). Tests use small explicit objects and the shared
fixtures; no real audio.
"""

from __future__ import annotations

from dj_set_planner.domain.enums import ConstraintType
from dj_set_planner.domain.models import (
    DjConstraint,
    SetSegment,
    Track,
    TrackFeatures,
)
from dj_set_planner.planning.context_profiles import default_profile
from dj_set_planner.planning.energy_curve import generate_energy_curve, segment_at
from dj_set_planner.planning.track_selector import (
    build_candidate_pool,
    compute_position_score,
    index_constraints,
)


def _track(track_id: int, **kw) -> Track:
    return Track(id=track_id, file_path=f"/library/sample_{track_id}.mp3", **kw)


def _features(track_id: int, **overrides: float) -> TrackFeatures:
    return TrackFeatures(track_id=track_id, **overrides)


def _segment(name: str, role: str, lo: float, hi: float) -> SetSegment:
    return SetSegment(
        name=name, start_pct=0.0, end_pct=1.0, min_energy=lo, max_energy=hi, role=role
    )


# --------------------------------------------------------------------------- #
# compute_position_score
# --------------------------------------------------------------------------- #


def test_position_score_in_range_and_has_breakdown() -> None:
    profile = default_profile()
    seg = _segment("intro", "INTRO", 0.30, 0.40)
    f = _features(1, energy_score=0.34, intro_suitability=0.9, restaurant_safety_score=0.9)

    score, breakdown = compute_position_score(_track(1), f, seg, profile, None, [])

    assert 0.0 <= score <= 1.0
    # Breakdown exposes each weighted component plus the transition slot.
    for key in (
        "context_fit",
        "energy_curve_fit",
        "role_fit",
        "mood_fit",
        "mixability_with_neighbors",
        "dj_preference",
    ):
        assert 0.0 <= breakdown[key] <= 1.0
    # No previous track -> no incoming transition.
    assert breakdown["transition"] is None


def test_energy_in_segment_band_scores_higher_than_out_of_band() -> None:
    """A track whose energy sits inside the segment band beats one outside it."""

    profile = default_profile()
    seg = _segment("main_peak", "MAIN_PEAK", 0.72, 0.78)

    # In-band peak track (energy 0.75, strong peak/dance).
    good = _features(
        1, energy_score=0.75, peak_potential=0.9, danceability_score=0.9, groove_score=0.7
    )
    # Out-of-band low-energy track in the same (peak) slot.
    bad = _features(
        2, energy_score=0.30, peak_potential=0.1, danceability_score=0.4, groove_score=0.4
    )

    good_score, _ = compute_position_score(_track(1), good, seg, profile, None, [])
    bad_score, _ = compute_position_score(_track(2), bad, seg, profile, None, [])

    assert good_score > bad_score


def test_incoming_transition_is_scored_with_previous_track() -> None:
    """With a previous track, the breakdown carries an incoming TransitionScore."""

    profile = default_profile()
    seg = _segment("warm_groove", "WARM_GROOVE", 0.40, 0.52)

    prev = _track(1, bpm=120.0, camelot_key="8A")
    prev_f = _features(1, energy_score=0.45)
    nxt = _track(2, bpm=121.0, camelot_key="8A")
    nxt_f = _features(2, energy_score=0.48)

    score, breakdown = compute_position_score(
        nxt, nxt_f, seg, profile, (prev, prev_f), []
    )

    assert breakdown["transition"] is not None
    assert 0.0 <= breakdown["transition"].score <= 1.0
    assert 0.0 <= score <= 1.0


def test_energy_cap_penalty_lowers_score() -> None:
    """A track above the harshness cap is penalised vs an identical in-cap one."""

    profile = default_profile()  # avoid_energy_above 0.85
    seg = _segment("main_peak", "MAIN_PEAK", 0.72, 0.78)

    over = _features(1, energy_score=0.95, peak_potential=0.9, danceability_score=0.9)
    under = _features(2, energy_score=0.75, peak_potential=0.9, danceability_score=0.9)

    over_score, over_bd = compute_position_score(_track(1), over, seg, profile, None, [])
    under_score, _ = compute_position_score(_track(2), under, seg, profile, None, [])

    assert over_bd["penalty"] > 0.0
    assert "over energy cap" in over_bd["notes"]
    assert over_score < under_score


def test_repeat_artist_penalty() -> None:
    """Back-to-back same-artist incurs a penalty."""

    profile = default_profile()
    seg = _segment("warm_groove", "WARM_GROOVE", 0.40, 0.52)

    prev = _track(1, artist="DJ Foo", bpm=120.0, camelot_key="8A")
    prev_f = _features(1, energy_score=0.45)
    same = _track(2, artist="DJ Foo", bpm=120.0, camelot_key="8A")
    same_f = _features(2, energy_score=0.46)

    _score, bd = compute_position_score(same, same_f, seg, profile, (prev, prev_f), [])
    assert "repeats previous artist" in bd["notes"]
    assert bd["penalty"] > 0.0


def test_dj_preference_boost_for_must_play() -> None:
    """A MUST_PLAY constraint maxes out the dj_preference component."""

    profile = default_profile()
    seg = _segment("warm_groove", "WARM_GROOVE", 0.40, 0.52)
    f = _features(1, energy_score=0.45)

    cons = [DjConstraint(id=None, track_id=1, constraint_type=ConstraintType.MUST_PLAY.value)]
    _score, bd = compute_position_score(_track(1), f, seg, profile, None, cons)
    assert bd["dj_preference"] == 1.0


def test_position_score_is_deterministic() -> None:
    profile = default_profile()
    seg = _segment("progressive_build", "PROGRESSIVE_BUILD", 0.52, 0.65)
    f = _features(1, energy_score=0.58)
    first, _ = compute_position_score(_track(1), f, seg, profile, None, [])
    for _ in range(5):
        again, _ = compute_position_score(_track(1), f, seg, profile, None, [])
        assert again == first


def test_position_score_uses_fixture(sample_tracks, sample_features, sofia_profile) -> None:
    """Smoke test against the shared fixtures: every track scores a valid 0..1."""

    curve = generate_energy_curve(sofia_profile)
    seg = segment_at(curve, 0.05)  # intro
    for track in sample_tracks:
        f = sample_features[track.id]
        score, _bd = compute_position_score(track, f, seg, sofia_profile, None, [])
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Candidate-pool / constraint helpers
# --------------------------------------------------------------------------- #


def test_index_constraints_partitions_seeds() -> None:
    cons = [
        DjConstraint(id=None, track_id=1, constraint_type=ConstraintType.AVOID.value),
        DjConstraint(id=None, track_id=2, constraint_type=ConstraintType.MUST_PLAY.value),
        DjConstraint(id=None, track_id=3, constraint_type=ConstraintType.PREFERRED_INTRO.value),
        DjConstraint(id=None, track_id=4, constraint_type=ConstraintType.PREFERRED_OUTRO.value),
        DjConstraint(id=None, track_id=5, constraint_type=ConstraintType.PREFERRED_PEAK.value),
        DjConstraint(id=None, track_id=6, constraint_type=ConstraintType.LOCK_POSITION.value, value="7"),
    ]
    seeds = index_constraints(cons)

    assert seeds.avoid == {1}
    assert seeds.must_play == {2}
    assert seeds.preferred_intro == {3}
    assert seeds.preferred_outro == {4}
    assert seeds.preferred_peak == {5}
    assert seeds.lock_positions == {6: 7}


def test_index_constraints_ignores_unparseable_lock_value() -> None:
    cons = [
        DjConstraint(id=None, track_id=6, constraint_type=ConstraintType.LOCK_POSITION.value, value="not-a-number"),
        DjConstraint(id=None, track_id=7, constraint_type=ConstraintType.LOCK_POSITION.value, value=None),
    ]
    seeds = index_constraints(cons)
    assert seeds.lock_positions == {}


def test_build_candidate_pool_drops_avoid() -> None:
    library = [_track(i) for i in range(1, 6)]
    seeds = index_constraints(
        [DjConstraint(id=None, track_id=3, constraint_type=ConstraintType.AVOID.value)]
    )
    pool = build_candidate_pool(library, seeds)
    ids = [t.id for t in pool]
    assert 3 not in ids
    assert ids == [1, 2, 4, 5]  # AVOID removed, rest in stable id order


def test_build_candidate_pool_is_id_sorted_regardless_of_input_order() -> None:
    library = [_track(4), _track(1), _track(3), _track(2)]
    seeds = index_constraints([])
    pool = build_candidate_pool(library, seeds)
    assert [t.id for t in pool] == [1, 2, 3, 4]


def test_build_candidate_pool_on_real_fixture(sample_tracks) -> None:
    seeds = index_constraints(
        [DjConstraint(id=None, track_id=10, constraint_type=ConstraintType.AVOID.value)]
    )
    pool = build_candidate_pool(sample_tracks, seeds)
    assert all(t.id != 10 for t in pool)
    assert len(pool) == len(sample_tracks) - 1
