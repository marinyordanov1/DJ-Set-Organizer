"""Tests for planning.beam_search_planner.

Driven by the shared fixtures (``sample_tracks`` / ``sample_features`` /
``sofia_profile``). Asserts the spec acceptance criteria:

  (a) the set's total runtime lands within +/-5 minutes of the target,
  (b) a track marked AVOID never appears,
  (c) a MUST_PLAY track always appears,
  (d) a PREFERRED_PEAK track lands inside the main_peak segment,
  (e) the plan is deterministic across two runs.

Plus structural invariants (positions, energy_points/segments populated,
LOCK_POSITION honoured, locked flag set).

Fixture note
------------
The shared fixture has 30 synthetic tracks totalling ~150 minutes of runtime.
After dropping the few tracks above the day-party harshness cap there is still
comfortably more than 120 minutes available, so the flagship 120-minute Sofia
target is used directly (the spec's primary acceptance target). The +/-5 minute
assertion is made around that 120-minute target.
"""

from __future__ import annotations

from dataclasses import replace

from dj_set_planner.domain.enums import ConstraintType
from dj_set_planner.domain.models import DjConstraint
from dj_set_planner.planning.beam_search_planner import plan_set
from dj_set_planner.planning.energy_curve import generate_energy_curve, segment_at

# +/-5 minutes, expressed in seconds, per the spec acceptance criterion.
_FIVE_MINUTES = 5 * 60


def _ids(plan) -> list[int]:
    return [t.track_id for t in plan.tracks]


# --------------------------------------------------------------------------- #
# (a) duration within +/-5 min of the 120-minute target
# --------------------------------------------------------------------------- #


def test_total_duration_within_five_minutes_of_target(
    sample_tracks, sample_features, sofia_profile
) -> None:
    # Use a CLUB venue (harshness ceiling 1.0) so no track is filtered out and
    # the full ~150 min library is available — this isolates the duration logic
    # from venue character filtering (a restaurant set may legitimately have less
    # on-character runtime; that is covered elsewhere).
    profile = replace(sofia_profile, venue_type="CLUB")
    plan = plan_set(sample_tracks, sample_features, profile, [])

    target = profile.target_duration_minutes * 60
    assert plan.target_duration_seconds == target
    assert abs(plan.total_duration_seconds - target) <= _FIVE_MINUTES, (
        f"runtime {plan.total_duration_seconds / 60:.1f} min is more than "
        f"5 min from the {sofia_profile.target_duration_minutes} min target"
    )


# --------------------------------------------------------------------------- #
# (b) AVOID never appears
# --------------------------------------------------------------------------- #


def test_avoid_track_never_selected(
    sample_tracks, sample_features, sofia_profile
) -> None:
    # Track 5 is normally one of the first picks; explicitly AVOID it.
    cons = [
        DjConstraint(id=None, track_id=5, constraint_type=ConstraintType.AVOID.value)
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    assert 5 not in _ids(plan)


def test_multiple_avoids_all_excluded(
    sample_tracks, sample_features, sofia_profile
) -> None:
    avoided = {3, 8, 15, 22}
    cons = [
        DjConstraint(id=None, track_id=tid, constraint_type=ConstraintType.AVOID.value)
        for tid in avoided
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    assert avoided.isdisjoint(_ids(plan))


# --------------------------------------------------------------------------- #
# (c) MUST_PLAY appears
# --------------------------------------------------------------------------- #


def test_must_play_track_included(
    sample_tracks, sample_features, sofia_profile
) -> None:
    # Track 1 is a very-low-energy track that the unconstrained planner might
    # otherwise leave out; force it in.
    cons = [
        DjConstraint(id=None, track_id=1, constraint_type=ConstraintType.MUST_PLAY.value)
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    assert 1 in _ids(plan)


def test_must_play_survives_with_avoid_combo(
    sample_tracks, sample_features, sofia_profile
) -> None:
    cons = [
        DjConstraint(id=None, track_id=2, constraint_type=ConstraintType.MUST_PLAY.value),
        DjConstraint(id=None, track_id=7, constraint_type=ConstraintType.MUST_PLAY.value),
        DjConstraint(id=None, track_id=5, constraint_type=ConstraintType.AVOID.value),
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    ids = _ids(plan)
    assert 2 in ids and 7 in ids
    assert 5 not in ids


# --------------------------------------------------------------------------- #
# (d) PREFERRED_PEAK lands in the main_peak segment
# --------------------------------------------------------------------------- #


def test_preferred_peak_lands_in_main_peak_segment(
    sample_tracks, sample_features, sofia_profile
) -> None:
    # Track 24 is a strong in-cap peak candidate (energy ~0.775).
    cons = [
        DjConstraint(
            id=None, track_id=24, constraint_type=ConstraintType.PREFERRED_PEAK.value
        )
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    ids = _ids(plan)
    assert 24 in ids

    curve = generate_energy_curve(sofia_profile)
    n = len(plan.tracks)
    pos = next(t.position for t in plan.tracks if t.track_id == 24)
    frac = pos / (n - 1) if n > 1 else 0.0
    seg = segment_at(curve, frac)
    assert seg.name == "main_peak", (
        f"PREFERRED_PEAK track placed at position {pos}/{n} -> segment "
        f"{seg.name!r}, expected main_peak"
    )
    # Its narrative role should also read as the main peak.
    assert next(t.role for t in plan.tracks if t.track_id == 24) == "MAIN_PEAK"


# --------------------------------------------------------------------------- #
# (e) deterministic across runs
# --------------------------------------------------------------------------- #


def test_plan_is_deterministic(
    sample_tracks, sample_features, sofia_profile
) -> None:
    plan_a = plan_set(sample_tracks, sample_features, sofia_profile, [])
    plan_b = plan_set(sample_tracks, sample_features, sofia_profile, [])

    assert _ids(plan_a) == _ids(plan_b)
    assert [t.role for t in plan_a.tracks] == [t.role for t in plan_b.tracks]
    assert plan_a.energy_points == plan_b.energy_points
    assert plan_a.total_duration_seconds == plan_b.total_duration_seconds


def test_plan_is_deterministic_with_constraints(
    sample_tracks, sample_features, sofia_profile
) -> None:
    cons = [
        DjConstraint(id=None, track_id=1, constraint_type=ConstraintType.MUST_PLAY.value),
        DjConstraint(id=None, track_id=5, constraint_type=ConstraintType.AVOID.value),
        DjConstraint(id=None, track_id=24, constraint_type=ConstraintType.PREFERRED_PEAK.value),
    ]
    a = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    b = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    assert _ids(a) == _ids(b)


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #


def test_plan_structure_is_well_formed(
    sample_tracks, sample_features, sofia_profile
) -> None:
    plan = plan_set(sample_tracks, sample_features, sofia_profile, [])

    # Positions are 0..n-1 in order.
    positions = [t.position for t in plan.tracks]
    assert positions == list(range(len(plan.tracks)))

    # energy_points parallel the tracks and are valid normalized energies.
    assert len(plan.energy_points) == len(plan.tracks)
    for e in plan.energy_points:
        assert 0.0 <= e <= 1.0

    # With adaptive energy OFF the planner uses the raw feature energies, so the
    # emitted points equal each track's original energy_score exactly.
    raw_plan = plan_set(
        sample_tracks, sample_features, sofia_profile, [], adaptive_energy=False
    )
    for spt, e in zip(raw_plan.tracks, raw_plan.energy_points):
        assert e == sample_features[spt.track_id].energy_score

    # The Sofia curve's eight segments are attached.
    assert len(plan.segments) == 8
    assert plan.segments[0].name == "intro"
    assert plan.segments[-1].name == "outro"

    # Per-track scores are in range and explanations non-empty.
    for t in plan.tracks:
        assert 0.0 <= t.position_score <= 1.0
        assert 0.0 <= t.transition_score <= 1.0
        assert t.explanation.strip()

    # The first track has no incoming transition.
    assert plan.tracks[0].transition_score == 0.0


def test_lock_position_is_honoured(
    sample_tracks, sample_features, sofia_profile
) -> None:
    cons = [
        DjConstraint(
            id=None,
            track_id=12,
            constraint_type=ConstraintType.LOCK_POSITION.value,
            value="5",
        )
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    slot = next(t for t in plan.tracks if t.track_id == 12)
    assert slot.position == 5
    assert slot.is_locked is True


def test_preferred_intro_and_outro_placement(
    sample_tracks, sample_features, sofia_profile
) -> None:
    cons = [
        DjConstraint(id=None, track_id=10, constraint_type=ConstraintType.PREFERRED_INTRO.value),
        DjConstraint(id=None, track_id=3, constraint_type=ConstraintType.PREFERRED_OUTRO.value),
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    intro_pos = next(t.position for t in plan.tracks if t.track_id == 10)
    outro_pos = next(t.position for t in plan.tracks if t.track_id == 3)
    assert intro_pos == 0
    assert outro_pos == len(plan.tracks) - 1


def test_empty_library_yields_empty_plan(sofia_profile) -> None:
    plan = plan_set([], {}, sofia_profile, [])
    assert plan.tracks == []
    assert plan.energy_points == []
    assert plan.total_duration_seconds == 0
    # A valid (non-empty) curve is still attached so the UI can render it.
    assert len(plan.segments) == 8
