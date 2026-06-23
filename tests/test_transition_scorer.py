"""Tests for ``planning.transition_scorer``.

These run with no audio files: we hand-build :class:`Track` /
:class:`TrackFeatures` instances for each scenario and assert on the resulting
:class:`TransitionScore`.
"""

from __future__ import annotations

from dj_set_planner.domain.models import Track, TrackFeatures
from dj_set_planner.planning.transition_scorer import (
    TransitionScore,
    score_transition,
)


def _track(
    track_id: int,
    *,
    bpm: float | None,
    camelot: str | None,
) -> Track:
    """Build a minimal Track with just the fields the scorer reads."""

    return Track(
        id=track_id,
        file_path=f"/library/t{track_id}.mp3",
        title=f"T{track_id}",
        bpm=bpm,
        camelot_key=camelot,
    )


def _features(
    track_id: int,
    *,
    energy: float,
    brightness: float = 0.5,
    groove: float = 0.5,
    vocal: float = 0.5,
    intro: float = 0.7,
    outro: float = 0.7,
) -> TrackFeatures:
    """Build TrackFeatures with controllable energy/mood/edges/vocals."""

    return TrackFeatures(
        track_id=track_id,
        energy_score=energy,
        mood_brightness=brightness,
        groove_score=groove,
        vocal_density=vocal,
        intro_suitability=intro,
        outro_suitability=outro,
    )


def test_smooth_transition_scores_high(sofia_profile) -> None:
    """Same BPM + same camelot + tiny energy increase -> score > 0.7."""

    prev = _track(1, bpm=122.0, camelot="8A")
    nxt = _track(2, bpm=122.0, camelot="8A")
    prev_f = _features(1, energy=0.50, brightness=0.5, groove=0.5)
    nxt_f = _features(2, energy=0.55, brightness=0.5, groove=0.5)

    result = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)

    assert isinstance(result, TransitionScore)
    assert result.score > 0.7, f"expected smooth transition high, got {result.score}"
    # The matching key/tempo should be reflected in the sub-scores too.
    assert result.bpm > 0.95
    assert result.key == 1.0


def test_clashing_transition_scores_low(sofia_profile) -> None:
    """120 vs 140 BPM + opposite-energy big jump + clashing key -> < 0.45."""

    prev = _track(1, bpm=120.0, camelot="8A")
    nxt = _track(2, bpm=140.0, camelot="3B")  # 8A vs 3B clashes harmonically
    # Big energy jump from low to high, not entering a peak segment.
    prev_f = _features(1, energy=0.20, brightness=0.2, groove=0.2)
    nxt_f = _features(2, energy=0.85, brightness=0.9, groove=0.9)

    result = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)

    assert result.score < 0.45, f"expected clashing transition low, got {result.score}"


def test_half_time_bpm_is_compatible(sofia_profile) -> None:
    """Half-time (140 vs 70) is treated as a compatible tempo."""

    prev = _track(1, bpm=140.0, camelot="8A")
    nxt = _track(2, bpm=70.0, camelot="8A")  # exactly half-time
    prev_f = _features(1, energy=0.50)
    nxt_f = _features(2, energy=0.52)

    result = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)

    # The folded BPM difference is ~0, so the tempo sub-score should be near 1.
    assert result.bpm > 0.95, f"half-time should be compatible, got bpm={result.bpm}"
    # And the whole transition should read as smooth.
    assert result.score > 0.7


def test_double_time_bpm_is_compatible(sofia_profile) -> None:
    """Double-time (70 vs 140) is also treated as compatible."""

    prev = _track(1, bpm=70.0, camelot="8A")
    nxt = _track(2, bpm=140.0, camelot="8A")
    prev_f = _features(1, energy=0.50)
    nxt_f = _features(2, energy=0.52)

    result = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)

    assert result.bpm > 0.95, f"double-time should be compatible, got bpm={result.bpm}"


def test_big_energy_jump_ok_into_peak(sofia_profile) -> None:
    """A big energy jump is rewarded when entering the main peak segment."""

    prev = _track(1, bpm=122.0, camelot="8A")
    nxt = _track(2, bpm=123.0, camelot="8A")
    prev_f = _features(1, energy=0.55)
    nxt_f = _features(2, energy=0.78)  # +0.23, a big jump

    into_peak = score_transition(
        prev, prev_f, nxt, nxt_f, sofia_profile,
        entering_segment_role="MAIN_PEAK",
    )
    no_context = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)

    # The same jump is penalised in normal flow but welcomed into the peak.
    assert into_peak.energy > no_context.energy
    assert into_peak.energy >= 0.8


def test_unknown_bpm_is_neutral(sofia_profile) -> None:
    """Missing tempo data yields a neutral (not punishing) bpm sub-score."""

    prev = _track(1, bpm=None, camelot="8A")
    nxt = _track(2, bpm=122.0, camelot="8A")
    prev_f = _features(1, energy=0.50)
    nxt_f = _features(2, energy=0.52)

    result = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)
    assert 0.55 <= result.bpm <= 0.65


def test_vocal_clash_lowers_score(sofia_profile) -> None:
    """Two vocal-heavy tracks incur a penalty vs two instrumental ones."""

    prev = _track(1, bpm=122.0, camelot="8A")
    nxt = _track(2, bpm=122.0, camelot="8A")

    vocal_prev = _features(1, energy=0.50, vocal=1.0)
    vocal_nxt = _features(2, energy=0.52, vocal=1.0)
    instr_prev = _features(1, energy=0.50, vocal=0.0)
    instr_nxt = _features(2, energy=0.52, vocal=0.0)

    vocal_result = score_transition(prev, vocal_prev, nxt, vocal_nxt, sofia_profile)
    instr_result = score_transition(prev, instr_prev, nxt, instr_nxt, sofia_profile)

    assert vocal_result.vocal_penalty > 0.5
    assert instr_result.vocal_penalty == 0.0
    assert vocal_result.score < instr_result.score


def test_deterministic(sofia_profile) -> None:
    """Same inputs always produce the same score (no randomness/clock)."""

    prev = _track(1, bpm=122.0, camelot="8A")
    nxt = _track(2, bpm=124.0, camelot="9A")
    prev_f = _features(1, energy=0.50)
    nxt_f = _features(2, energy=0.56)

    a = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)
    b = score_transition(prev, prev_f, nxt, nxt_f, sofia_profile)
    assert a == b
