"""Tests for planning.energy_curve.

Focus is the flagship Sofia ONE_MAIN_PEAK template, plus the shared invariants
that every template (and the segment lookup helpers) must satisfy.
"""

from __future__ import annotations

import pytest

from dj_set_planner.domain.enums import PeakStrategy, TrackRole
from dj_set_planner.domain.models import EventProfile
from dj_set_planner.planning.context_profiles import builtin_presets
from dj_set_planner.planning.energy_curve import (
    generate_energy_curve,
    segment_at,
    target_energy_at,
)

# The exact, ordered story the Sofia ONE_MAIN_PEAK template must tell.
_EXPECTED_SOFIA_ORDER = [
    "intro",
    "warm_groove",
    "progressive_build",
    "small_peak",
    "breathing_space",
    "main_peak",
    "release",
    "outro",
]


def _make_profile(strategy: str) -> EventProfile:
    """A profile with the Sofia envelope but a chosen peak strategy."""

    return EventProfile(
        id=None,
        name=f"test-{strategy}",
        venue_type="RESTAURANT",
        time_of_day="DAY",
        crowd_state="EATING+TALKING+PARTIAL_DANCING",
        desired_energy="BALANCED",
        peak_strategy=strategy,
        target_duration_minutes=120,
        min_energy=0.30,
        max_energy=0.78,
        main_peak_energy=0.75,
    )


# --------------------------------------------------------------------------- #
# Sofia ONE_MAIN_PEAK — the contract-specified shape.
# --------------------------------------------------------------------------- #


def test_sofia_has_exactly_eight_segments(sofia_profile: EventProfile) -> None:
    assert sofia_profile.target_duration_minutes == 120
    assert sofia_profile.peak_strategy == PeakStrategy.ONE_MAIN_PEAK.value

    curve = generate_energy_curve(sofia_profile)
    assert len(curve) == 8


def test_sofia_segments_in_expected_order(sofia_profile: EventProfile) -> None:
    curve = generate_energy_curve(sofia_profile)
    names = [seg.name for seg in curve]
    assert names == _EXPECTED_SOFIA_ORDER


def test_sofia_segments_contiguous_cover_0_to_1(
    sofia_profile: EventProfile,
) -> None:
    curve = generate_energy_curve(sofia_profile)

    # Starts at 0%, ends at 100%.
    assert curve[0].start_pct == pytest.approx(0.0)
    assert curve[-1].end_pct == pytest.approx(1.0)

    # No gaps / overlaps: each segment starts where the previous ended.
    for prev, nxt in zip(curve, curve[1:]):
        assert nxt.start_pct == pytest.approx(prev.end_pct)
        # Each segment has positive width.
        assert prev.end_pct > prev.start_pct
    assert curve[-1].end_pct > curve[-1].start_pct


def test_sofia_main_peak_starts_at_or_after_70pct(
    sofia_profile: EventProfile,
) -> None:
    curve = generate_energy_curve(sofia_profile)
    main_peak = next(seg for seg in curve if seg.name == "main_peak")
    assert main_peak.start_pct >= 0.70
    assert main_peak.role == TrackRole.MAIN_PEAK.value


def test_sofia_outro_is_last_and_ends_at_100pct(
    sofia_profile: EventProfile,
) -> None:
    curve = generate_energy_curve(sofia_profile)
    last = curve[-1]
    assert last.name == "outro"
    assert last.role == TrackRole.OUTRO.value
    assert last.end_pct == pytest.approx(1.0)


def test_sofia_every_band_min_le_max(sofia_profile: EventProfile) -> None:
    curve = generate_energy_curve(sofia_profile)
    for seg in curve:
        assert seg.min_energy <= seg.max_energy


def test_sofia_exact_band_values(sofia_profile: EventProfile) -> None:
    """The Sofia envelope (0.30..0.78) contains every template band, so the
    spec's exact numbers should pass through unclamped."""

    curve = generate_energy_curve(sofia_profile)
    bands = {seg.name: (seg.min_energy, seg.max_energy) for seg in curve}
    assert bands["intro"] == pytest.approx((0.30, 0.40))
    assert bands["warm_groove"] == pytest.approx((0.40, 0.52))
    assert bands["progressive_build"] == pytest.approx((0.52, 0.65))
    assert bands["small_peak"] == pytest.approx((0.65, 0.72))
    assert bands["breathing_space"] == pytest.approx((0.50, 0.60))
    assert bands["main_peak"] == pytest.approx((0.72, 0.78))
    assert bands["release"] == pytest.approx((0.55, 0.65))
    assert bands["outro"] == pytest.approx((0.35, 0.48))


# --------------------------------------------------------------------------- #
# Shared invariants across ALL strategies.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", [s.value for s in PeakStrategy])
def test_every_template_is_contiguous_and_valid(strategy: str) -> None:
    curve = generate_energy_curve(_make_profile(strategy))

    assert curve, "curve must be non-empty"
    assert curve[0].start_pct == pytest.approx(0.0)
    assert curve[-1].end_pct == pytest.approx(1.0)

    for prev, nxt in zip(curve, curve[1:]):
        assert nxt.start_pct == pytest.approx(prev.end_pct)

    for seg in curve:
        assert seg.end_pct > seg.start_pct
        assert seg.min_energy <= seg.max_energy
        # Bands stay within the profile envelope.
        assert seg.min_energy >= 0.30 - 1e-9
        assert seg.max_energy <= 0.78 + 1e-9


def test_unknown_strategy_falls_back_to_one_main_peak() -> None:
    curve = generate_energy_curve(_make_profile("NOT_A_REAL_STRATEGY"))
    names = [seg.name for seg in curve]
    assert names == _EXPECTED_SOFIA_ORDER


def test_clamping_respects_narrow_envelope() -> None:
    """A profile with a tight envelope clamps bands but keeps min<=max and the
    contiguous 0..1 coverage."""

    profile = _make_profile(PeakStrategy.ONE_MAIN_PEAK.value)
    profile.min_energy = 0.45
    profile.max_energy = 0.60

    curve = generate_energy_curve(profile)
    for seg in curve:
        assert 0.45 - 1e-9 <= seg.min_energy <= seg.max_energy <= 0.60 + 1e-9
    assert curve[0].start_pct == pytest.approx(0.0)
    assert curve[-1].end_pct == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# segment_at / target_energy_at helpers.
# --------------------------------------------------------------------------- #


def test_segment_at_maps_fractions_to_segments(
    sofia_profile: EventProfile,
) -> None:
    curve = generate_energy_curve(sofia_profile)

    assert segment_at(curve, 0.0).name == "intro"
    assert segment_at(curve, 0.05).name == "intro"
    assert segment_at(curve, 0.20).name == "warm_groove"
    assert segment_at(curve, 0.55).name == "small_peak"
    assert segment_at(curve, 0.75).name == "main_peak"
    # End boundary (1.0) maps to the closing outro, not off the end.
    assert segment_at(curve, 1.0).name == "outro"


def test_segment_at_clamps_out_of_range(sofia_profile: EventProfile) -> None:
    curve = generate_energy_curve(sofia_profile)
    assert segment_at(curve, -0.5).name == "intro"
    assert segment_at(curve, 2.0).name == "outro"


def test_segment_at_boundary_belongs_to_following_segment(
    sofia_profile: EventProfile,
) -> None:
    """Spans are half-open [start, end): the exact boundary belongs to the
    next segment."""

    curve = generate_energy_curve(sofia_profile)
    # 0.10 is intro.end == warm_groove.start -> belongs to warm_groove.
    assert segment_at(curve, 0.10).name == "warm_groove"
    # 0.70 is breathing_space.end == main_peak.start -> belongs to main_peak.
    assert segment_at(curve, 0.70).name == "main_peak"


def test_target_energy_at_returns_band(sofia_profile: EventProfile) -> None:
    curve = generate_energy_curve(sofia_profile)
    assert target_energy_at(curve, 0.75) == pytest.approx((0.72, 0.78))
    assert target_energy_at(curve, 0.05) == pytest.approx((0.30, 0.40))


def test_segment_at_empty_curve_raises() -> None:
    with pytest.raises(ValueError):
        segment_at([], 0.5)


def test_builtin_sofia_preset_matches_default(
    sofia_profile: EventProfile,
) -> None:
    """The builtin registry's Sofia preset yields the same 8-segment curve as
    the default profile used by the fixture."""

    sofia = builtin_presets()["Sofia Day Party Restaurant"]
    curve = generate_energy_curve(sofia)
    assert [s.name for s in curve] == _EXPECTED_SOFIA_ORDER
    fixture_curve = generate_energy_curve(sofia_profile)
    assert [s.name for s in fixture_curve] == [s.name for s in curve]
