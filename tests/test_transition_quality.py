"""Track-to-track flow: directional key table, edge/timbre-aware transitions,
and the exact re-ordering step."""

from __future__ import annotations

import itertools
import random

from dj_set_planner.domain.models import Track, TrackFeatures
from dj_set_planner.planning.beam_search_planner import _NEG, _held_karp
from dj_set_planner.planning.transition_scorer import score_transition
from dj_set_planner.utils.camelot import key_compatibility


def test_key_table_is_directional_and_musical():
    assert key_compatibility("8A", "8A") == 1.0
    assert key_compatibility("8A", "9A") == key_compatibility("8A", "7A") == 0.90
    assert key_compatibility("8A", "8B") == key_compatibility("8B", "8A") == 0.90
    # Mixed In Key diagonals: A->B subtracts one, B->A adds one.
    assert key_compatibility("8A", "7B") == key_compatibility("8B", "9A") == 0.75
    assert key_compatibility("8A", "9B") == key_compatibility("8B", "7A") == 0.50
    assert key_compatibility("8A", "11B") == key_compatibility("8B", "5A") == 0.60  # parallel
    assert key_compatibility("8A", "10A") == 0.50  # energy boost
    assert key_compatibility("8A", "2A") == 0.05  # tritone
    assert key_compatibility("12A", "1A") == 0.90  # wraps around the wheel
    assert key_compatibility(None, "8A") == 0.6


def _t(i, bpm, key):
    return Track(id=i, file_path=f"/x/{i}.mp3", bpm=bpm, camelot_key=key)


def test_drum_only_edges_soften_a_key_clash(sofia_profile):
    a, b = _t(1, 124.0, "8A"), _t(2, 124.0, "2A")  # tritone clash
    tonal = TrackFeatures(1, intro_percussive=0.2, outro_percussive=0.2)
    drums = TrackFeatures(2, intro_percussive=0.98, outro_percussive=0.98)
    clash = score_transition(a, tonal, b, TrackFeatures(2, intro_percussive=0.2), sofia_profile)
    soft = score_transition(a, drums, b, drums, sofia_profile)
    assert soft.key > clash.key and soft.score > clash.score


def test_similar_timbre_beats_different_timbre(sofia_profile):
    a, b = _t(1, 124.0, "8A"), _t(2, 124.0, "8A")
    base = TrackFeatures(1, timbre=(1.0, 0.5, -0.2))
    same = score_transition(a, base, b, TrackFeatures(2, timbre=(0.9, 0.6, -0.1)), sofia_profile)
    diff = score_transition(a, base, b, TrackFeatures(2, timbre=(-1.0, -0.5, 0.3)), sofia_profile)
    assert same.mood > 0.9 > 0.2 > diff.mood


def test_energy_jump_uses_audible_raw_energy(sofia_profile):
    """A stretched +0.30 that was really +0.05 is a gentle rise, not a jump."""
    a, b = _t(1, 124.0, "8A"), _t(2, 124.0, "8A")
    pf = TrackFeatures(1, energy_score=0.40, raw_energy=0.70)
    nf = TrackFeatures(2, energy_score=0.70, raw_energy=0.75)
    assert score_transition(a, pf, b, nf, sofia_profile).energy >= 0.95


def test_held_karp_matches_brute_force_with_pins_and_neighbours():
    rnd = random.Random(7)
    n = 6
    table = {(s, i, j): rnd.random() for s in range(n + 2) for i in [None, *range(n + 1)] for j in range(n + 1)}

    def step(slot, i, j):
        if slot == 3 and j != 4 or slot != 3 and j == 4:  # item 4 pinned to slot 3
            return _NEG
        return table[(slot, i, j)]

    def total(order):  # block at slots 1..n, nothing before, fixed item 6 after
        s = step(1, None, order[0]) + step(1 + n, order[-1], 6)
        return s + sum(step(1 + k, order[k - 1], order[k]) for k in range(1, n))

    items = list(range(n))
    best = max(itertools.permutations(items), key=total)
    got = _held_karp(items, 1, None, 6, step)
    assert got[2] == 4 and abs(total(got) - total(best)) < 1e-12


def test_unmeasured_tracks_score_neutral_not_favoured(sofia_profile):
    a, b = _t(1, 124.0, "8A"), _t(2, 124.0, "8A")
    stale = score_transition(a, TrackFeatures(1), b, TrackFeatures(2), sofia_profile)
    assert stale.mood == 0.5


def test_half_time_fold_is_symmetric_without_cliff(sofia_profile):
    f1, f2 = TrackFeatures(1), TrackFeatures(2)
    fwd = score_transition(_t(1, 124.0, "8A"), f1, _t(2, 66.0, "8A"), f2, sofia_profile)
    rev = score_transition(_t(1, 66.0, "8A"), f1, _t(2, 124.0, "8A"), f2, sofia_profile)
    assert fwd.bpm == rev.bpm > 0.3  # 132 vs 124 = 6.5%, not the 0.05 floor


def test_lock_is_never_evicted_by_a_preference():
    from dj_set_planner.domain.models import SetSegment
    from dj_set_planner.planning.beam_search_planner import _seed_positions
    from dj_set_planner.planning.track_selector import index_constraints
    from dj_set_planner.domain.models import DjConstraint

    seeds = index_constraints([
        DjConstraint(None, 27, "LOCK_POSITION", "5"),
        DjConstraint(None, 8, "PREFERRED_PEAK"),  # lower id, same slot
    ])
    curve = [SetSegment("main", 0.0, 1.0, 0.6, 0.8, "MAIN_PEAK")]
    pins = _seed_positions(seeds, {8: None, 27: None}, curve, 11)  # peak centre = slot 5
    assert pins[5] == 27


def test_reorder_only_improves_with_conflicting_seeds(
    monkeypatch, sample_tracks, sample_features, sofia_profile
):
    """Two closers, or a lock at 0 plus a preferred intro: _finalize must not
    undo the optimizer, so the plan never scores below the beam's own order."""
    from dataclasses import replace

    import dj_set_planner.planning.beam_search_planner as bsp
    from dj_set_planner.domain.models import DjConstraint

    profile = replace(sofia_profile, target_duration_minutes=45)
    cases = [
        [DjConstraint(None, 1, "PREFERRED_OUTRO"), DjConstraint(None, 16, "PREFERRED_OUTRO")],
        [DjConstraint(None, 2, "LOCK_POSITION", "0"), DjConstraint(None, 9, "PREFERRED_INTRO")],
    ]
    for cons in cases:
        with_opt = bsp.plan_set(sample_tracks, sample_features, profile, cons)
        with monkeypatch.context() as m:
            m.setattr(bsp, "_optimize_order", lambda *a, **k: None)
            without = bsp.plan_set(sample_tracks, sample_features, profile, cons)
        assert with_opt.total_score >= without.total_score - 1e-9


def test_must_play_survives_forced_positional_seeds(
    sample_tracks, sample_features, sofia_profile
):
    """Two preferred intros (only one can open): forcing seeds in must never
    evict a MUST_PLAY the beam placed itself."""
    from dj_set_planner.domain.models import DjConstraint
    from dj_set_planner.planning.beam_search_planner import plan_set

    cons = [
        DjConstraint(None, 27, "MUST_PLAY"),
        DjConstraint(None, 17, "PREFERRED_INTRO"),
        DjConstraint(None, 30, "PREFERRED_INTRO"),
    ]
    plan = plan_set(sample_tracks, sample_features, sofia_profile, cons)
    assert 27 in {t.track_id for t in plan.tracks}


def test_one_unmeasured_edge_is_neutral_not_penalised(sofia_profile):
    a, b = _t(1, 124.0, "8A"), _t(2, 124.0, "11A")  # +3, a weak key move
    typical = TrackFeatures(1, outro_percussive=0.9)
    measured = score_transition(a, typical, b, TrackFeatures(2, intro_percussive=0.9), sofia_profile)
    missing = score_transition(a, typical, b, TrackFeatures(2), sofia_profile)
    assert abs(measured.score - missing.score) < 1e-9
