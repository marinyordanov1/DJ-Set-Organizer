"""Beam-search set planner — the orchestrator that turns a library into a story.

:func:`plan_set` is the top-level entry point. Given a music library, its
analysed features, an :class:`EventProfile`, and a list of :class:`DjConstraint`,
it produces an ordered :class:`SetPlan` that tells the spec's narrative arc
(intro -> warm groove -> build -> small peak -> breathing space -> main peak ->
release -> outro) while respecting the DJ's constraints.

Algorithm (per CONTRACTS.md)
----------------------------
1. **Drop AVOID** tracks — they are never selectable.
2. **Generate the energy curve** for the profile (the target story shape).
3. **Decide the slot count** so the set's total runtime lands near the target
   duration (number of tracks ~= target_seconds / average_track_seconds).
4. **Seed** the locked / preferred tracks:
     * ``LOCK_POSITION`` pins a track to an exact slot,
     * ``PREFERRED_INTRO`` near the start, ``PREFERRED_OUTRO`` at the end,
     * ``PREFERRED_PEAK`` inside the ``main_peak`` segment,
     * ``MUST_PLAY`` is forced in (assigned to the slot where it fits best and
       is still free).
5. **Beam search** the remaining slots left-to-right: for each beam (a partial
   ordering) and each free candidate, score the candidate for the next slot as
   ``position_score + incoming transition_score``, keep the top
   ``max_candidates_per_step`` expansions per beam, then prune all expansions to
   the global top ``beam_width`` by cumulative score.
6. **Stop** once a beam's cumulative duration is within
   ``duration_tolerance_seconds`` of the target (or it runs out of slots /
   candidates), then pick the highest-scoring complete beam.
7. Build the :class:`SetPlan` with per-track roles, transition/position scores,
   explanations, the actual ``energy_points`` curve, and the segments.

Determinism
-----------
Everything is deterministic: candidate iteration is id-ordered, and every score
comparison breaks ties by track id so the same inputs always yield the same set
(a hard CONTRACTS.md requirement — no ``random``, no wall-clock).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..domain.models import (
    DjConstraint,
    EventProfile,
    SetPlan,
    SetPlanTrack,
    SetSegment,
    Track,
    TrackFeatures,
)
from ..utils.logging import get_logger
from . import context_profiles
from .energy_curve import generate_energy_curve, segment_at
from .energy_normalize import relativize_features, relativize_timbre
from .explanations import explain_track
from .track_roles import best_role
from .venue_profiles import apply_time_modifier, character_fit, character_profile
from .track_selector import (
    ConstraintSeeds,
    build_candidate_pool,
    compute_position_score,
    index_constraints,
)
from .transition_scorer import TransitionScore

_log = get_logger(__name__)


# Fallback per-track duration (seconds) when a track has no duration tag, so the
# slot-count maths still works. 5 minutes is a sensible day-party average.
_DEFAULT_TRACK_SECONDS = 300


# --------------------------------------------------------------------------- #
# Internal search state.
# --------------------------------------------------------------------------- #


@dataclass
class _Placement:
    """One placed slot inside a partial/complete beam."""

    track: Track
    features: TrackFeatures
    position: int
    role: str
    segment: SetSegment
    position_score: float
    transition: TransitionScore | None
    breakdown: dict
    is_locked: bool


@dataclass
class _Beam:
    """A partial ordering under construction.

    ``cumulative_score`` is the running sum of (position_score + transition
    score) over placed slots; ``duration`` is the running runtime in seconds;
    ``used_ids`` is the set of already-placed track ids for O(1) "already used"
    checks. ``placements`` is ordered by slot position.
    """

    placements: list[_Placement] = field(default_factory=list)
    used_ids: set[int] = field(default_factory=set)
    cumulative_score: float = 0.0
    duration: int = 0

    def last_pair(self) -> tuple[Track, TrackFeatures] | None:
        if not self.placements:
            return None
        last = self.placements[-1]
        return last.track, last.features

    def clone(self) -> "_Beam":
        """Shallow-copy the beam so an expansion doesn't mutate its parent."""

        return _Beam(
            placements=list(self.placements),
            used_ids=set(self.used_ids),
            cumulative_score=self.cumulative_score,
            duration=self.duration,
        )


def _track_seconds(track: Track) -> int:
    """Track runtime in seconds, falling back to a sensible default."""

    if track.duration_seconds and track.duration_seconds > 0:
        return int(track.duration_seconds)
    return _DEFAULT_TRACK_SECONDS


def _drop_over_cap_tracks(
    pool: list[Track],
    features: dict[int, TrackFeatures],
    profile: EventProfile,
    seeds: ConstraintSeeds,
) -> list[Track]:
    """Remove tracks whose energy exceeds the context's HARSHNESS ceiling.

    The ceiling is ``avoid_energy_above`` (0.85 for the day-party context) — the
    point past which a track is considered harsh and unsuitable for a restaurant
    crowd. This is intentionally the *harshness* cap, NOT the curve's
    ``max_energy`` envelope (0.78): tracks between 0.78 and 0.85 are still usable
    (they just can't be the main peak), so we keep them. A MUST_PLAY track is
    never dropped (explicit DJ override). If filtering would leave the pool empty
    we keep the original pool (better an imperfect set than none).
    """

    # The venue's harshness ceiling on ABSOLUTE energy: restaurant rejects
    # genuinely loud tracks, club allows everything. Applied before adaptive
    # normalization so it means "this track is objectively too hot for here".
    hard_cap = character_profile(profile.venue_type).harshness_ceiling
    exempt = seeds.dj_chosen()
    kept: list[Track] = []
    dropped = 0
    for t in pool:
        f = features.get(t.id) if t.id is not None else None
        energy = f.energy_score if f is not None else 0.5
        if energy > hard_cap and (t.id not in exempt):
            dropped += 1
            continue
        kept.append(t)

    if not kept:
        # Don't strand the planner with nothing to place.
        return pool
    if dropped:
        _log.info(
            "Dropped %d track(s) above the energy cap %.2f for this context.",
            dropped,
            hard_cap,
        )
    return kept


def _apply_strict_filter(
    pool: list[Track],
    features: dict[int, TrackFeatures],
    profile: EventProfile,
    seeds: ConstraintSeeds,
    threshold: float = 0.4,
) -> list[Track]:
    """Keep only tracks that fit the venue's character (character_fit >= threshold).

    MUST_PLAY tracks are exempt. A neutral venue (no character prefs) keeps
    everything. If the filter would empty the pool, fall back to the full pool
    and log it — the same "better an imperfect set than none" safety net as the
    energy-cap filter.
    """

    venue = character_profile(profile.venue_type)
    if not venue.prefs:
        return pool  # neutral venue: nothing to be strict about
    exempt = seeds.dj_chosen()
    kept: list[Track] = []
    dropped = 0
    for t in pool:
        f = features.get(t.id) if t.id is not None else None
        fit = character_fit(f, venue) if f is not None else 0.5
        if fit >= threshold or (t.id in exempt):
            kept.append(t)
        else:
            dropped += 1
    if not kept:
        _log.info("Strict character filter emptied the pool; falling back to soft.")
        return pool
    if dropped:
        _log.info(
            "Strict character: kept %d/%d on-venue track(s).", len(kept), len(pool)
        )
    return kept


def _estimate_slot_count(
    pool: list[Track], target_seconds: int
) -> int:
    """Estimate how many slots best approximate ``target_seconds`` of runtime.

    Uses the *average* track duration of the pool so the estimate is robust to a
    library of mixed-length tracks. Clamped to ``[1, len(pool)]`` so we never ask
    for more tracks than exist or fewer than one.
    """

    if not pool:
        return 0
    avg = sum(_track_seconds(t) for t in pool) / len(pool)
    if avg <= 0:
        avg = _DEFAULT_TRACK_SECONDS
    # Round to nearest whole number of average-length tracks.
    n = round(target_seconds / avg)
    return max(1, min(int(n), len(pool)))


def _max_slots_for_duration(
    pool: list[Track], target_seconds: int, tolerance_seconds: int
) -> int:
    """Upper bound on slots so runtime can reach (target + tolerance).

    The search is duration-driven and stops once it enters the target window,
    but it needs enough head-room to *get* there even if the chosen tracks are
    shorter than the pool average. We size the bound using the pool's SHORTEST
    tracks (the worst case for reaching a duration) so the search can always fill
    up to the upper tolerance edge, capped at the pool size.
    """

    if not pool:
        return 0
    shortest = min(_track_seconds(t) for t in pool)
    if shortest <= 0:
        shortest = _DEFAULT_TRACK_SECONDS
    upper = target_seconds + tolerance_seconds
    # +1 so we can cross into the window rather than stop just short of it.
    n = int(upper // shortest) + 1
    return max(1, min(n, len(pool)))


def _position_fraction(slot: int, slot_count: int) -> float:
    """Map a 0-based slot index to its progression fraction in 0..1.

    The first slot sits at 0.0 and the last at 1.0 so the curve is sampled across
    its full span (with a single slot mapping to 0.0 -> the intro).
    """

    if slot_count <= 1:
        return 0.0
    return slot / (slot_count - 1)


def _seed_positions(
    seeds: ConstraintSeeds,
    pool_by_id: dict[int, Track],
    curve: list[SetSegment],
    slot_count: int,
) -> dict[int, int]:
    """Decide a fixed slot for each locked/preferred *positional* seed.

    Returns a mapping ``{slot_index: track_id}`` for seeds that pin a track to a
    specific slot:
      * LOCK_POSITION -> its exact (clamped) slot,
      * PREFERRED_INTRO -> slot 0 (the very start),
      * PREFERRED_OUTRO -> the last slot,
      * PREFERRED_PEAK -> a slot inside the ``main_peak`` segment.

    MUST_PLAY tracks are NOT pinned here — they are forced in later at their
    best free slot (a fixed position would over-constrain them). On a collision
    (two seeds want the same slot) the lower track id wins deterministically and
    the loser is left for the normal search / must-play pass.
    """

    pinned: dict[int, int] = {}

    def _try_pin(slot: int, track_id: int) -> None:
        slot = max(0, min(slot, slot_count - 1))
        existing = pinned.get(slot)
        if existing is None:
            pinned[slot] = track_id
        elif track_id < existing and (
            existing not in seeds.lock_positions or track_id in seeds.lock_positions
        ):
            # Deterministic tie-break: lower id keeps the slot — but a
            # preference never evicts a LOCK_POSITION.
            pinned[slot] = track_id

    # LOCK_POSITION is the strongest: honour the DJ's exact slot first.
    for track_id, slot in sorted(seeds.lock_positions.items()):
        if track_id in pool_by_id:
            _try_pin(slot, track_id)

    # PREFERRED_INTRO -> the opening slot.
    for track_id in sorted(seeds.preferred_intro):
        if track_id in pool_by_id and track_id not in pinned.values():
            _try_pin(0, track_id)

    # PREFERRED_OUTRO -> the closing slot.
    for track_id in sorted(seeds.preferred_outro):
        if track_id in pool_by_id and track_id not in pinned.values():
            _try_pin(slot_count - 1, track_id)

    # PREFERRED_PEAK -> the centre of the main_peak segment.
    peak_slot = _main_peak_slot(curve, slot_count)
    if peak_slot is not None:
        for track_id in sorted(seeds.preferred_peak):
            if track_id in pool_by_id and track_id not in pinned.values():
                _try_pin(peak_slot, track_id)

    return pinned


def _main_peak_slot(curve: list[SetSegment], slot_count: int) -> int | None:
    """Return a representative slot index inside the ``main_peak`` segment.

    Picks the slot whose progression fraction is closest to the centre of the
    main_peak segment, so a PREFERRED_PEAK track lands squarely inside it. None
    if the curve has no main_peak segment (e.g. a flat-lounge template).
    """

    main = next((s for s in curve if s.role == "MAIN_PEAK"), None)
    if main is None:
        return None
    centre = (main.start_pct + main.end_pct) / 2.0
    # Find the slot whose fraction is nearest the segment centre.
    best_slot = 0
    best_dist = 2.0
    for slot in range(slot_count):
        frac = _position_fraction(slot, slot_count)
        dist = abs(frac - centre)
        if dist < best_dist:
            best_dist = dist
            best_slot = slot
    return best_slot


def _make_placement(
    track: Track,
    f: TrackFeatures,
    slot: int,
    slot_count: int,
    curve: list[SetSegment],
    profile: EventProfile,
    prev_pair: tuple[Track, TrackFeatures] | None,
    constraints_for_track: list[DjConstraint],
    is_locked: bool,
) -> _Placement:
    """Score ``track`` at ``slot`` and package it into a :class:`_Placement`."""

    frac = _position_fraction(slot, slot_count)
    segment = segment_at(curve, frac)
    score, breakdown = compute_position_score(
        track, f, segment, profile, prev_pair, constraints_for_track
    )
    transition = breakdown.get("transition")
    # Narrative role for this slot: prefer the segment's role when the track is a
    # genuine fit for it, else fall back to the track's intrinsic best role. This
    # keeps the displayed role honest (a low-energy track forced into a peak slot
    # won't be mislabelled MAIN_PEAK).
    role = _slot_role(track, f, segment, profile)
    return _Placement(
        track=track,
        features=f,
        position=slot,
        role=role,
        segment=segment,
        position_score=score,
        transition=transition,
        breakdown=breakdown,
        is_locked=is_locked,
    )


def _slot_role(
    track: Track, f: TrackFeatures, segment: SetSegment, profile: EventProfile
) -> str:
    """Pick the narrative role label for a placed track.

    Default to the segment's own role (so the set reads as the intended story),
    but if the track is a poor fit for that role we defer to its intrinsic
    best role so explanations stay truthful.
    """

    from .track_roles import role_fit_scores  # local import: avoid cycle at top

    seg_role = segment.role
    fits = role_fit_scores(track, f, profile)
    if fits.get(seg_role, 0.0) >= 0.45:
        return seg_role
    return best_role(track, f, profile)


def _expand_beam(
    beam: _Beam,
    slot: int,
    slot_count: int,
    pinned: dict[int, int],
    pool: list[Track],
    features: dict[int, TrackFeatures],
    curve: list[SetSegment],
    profile: EventProfile,
    seeds: ConstraintSeeds,
    max_candidates_per_step: int,
) -> list[_Beam]:
    """Return up to ``max_candidates_per_step`` child beams for the next slot.

    When the slot is pinned (a positional seed lives here) the only expansion is
    that exact track. Otherwise we score every unused candidate for this slot,
    keep the best ``max_candidates_per_step`` (tie-broken by track id), and
    return one child beam per kept candidate.
    """

    prev_pair = beam.last_pair()
    pinned_for_slot = pinned.get(slot)

    # --- pinned slot: forced single expansion ---------------------------- #
    if pinned_for_slot is not None:
        if pinned_for_slot in beam.used_ids:
            # Already placed earlier (shouldn't happen, but stay safe): skip the
            # pin and let this slot be filled by the normal search below.
            pass
        else:
            f = features.get(
                pinned_for_slot, TrackFeatures(track_id=pinned_for_slot)
            )
            track = next((t for t in pool if t.id == pinned_for_slot), None)
            if track is not None:
                placement = _make_placement(
                    track,
                    f,
                    slot,
                    slot_count,
                    curve,
                    profile,
                    prev_pair,
                    seeds.by_track.get(pinned_for_slot, []),
                    is_locked=True,
                )
                child = beam.clone()
                _apply_placement(child, placement)
                return [child]

    # --- free slot: score all unused candidates -------------------------- #
    # Track ids reserved for a pin in a strictly later slot must not be consumed
    # now, so the planner can still honour that pin when it reaches that slot.
    reserved = {
        track_id for s, track_id in pinned.items() if s > slot
    }
    scored: list[tuple[float, int, _Placement]] = []
    for track in pool:
        tid = track.id
        if tid is None or tid in beam.used_ids:
            continue
        if tid in reserved:
            continue
        f = features.get(tid, TrackFeatures(track_id=tid))
        placement = _make_placement(
            track,
            f,
            slot,
            slot_count,
            curve,
            profile,
            prev_pair,
            seeds.by_track.get(tid, []),
            is_locked=False,
        )
        combined = _placement_step_score(placement)
        # Sort key: higher combined score first, then lower id (deterministic).
        scored.append((combined, tid, placement))

    if not scored:
        return []

    # Best-first by score, deterministic tie-break by ascending track id.
    scored.sort(key=lambda s: (-s[0], s[1]))
    children: list[_Beam] = []
    for _combined, _tid, placement in scored[:max_candidates_per_step]:
        child = beam.clone()
        _apply_placement(child, placement)
        children.append(child)
    return children


def _placement_step_score(placement: _Placement) -> float:
    """Combined per-step score = position_score + incoming transition_score.

    The transition score is 0 for the first slot (no previous track to blend
    from), matching the CONTRACTS description "position_score + incoming
    transition_score".
    """

    trans = placement.transition.score if placement.transition is not None else 0.0
    return placement.position_score + trans


def _apply_placement(beam: _Beam, placement: _Placement) -> None:
    """Mutate ``beam`` in place to append ``placement``."""

    beam.placements.append(placement)
    if placement.track.id is not None:
        beam.used_ids.add(placement.track.id)
    beam.cumulative_score += _placement_step_score(placement)
    beam.duration += _track_seconds(placement.track)


def _force_must_play(
    beam: _Beam,
    seeds: ConstraintSeeds,
    pool: list[Track],
    features: dict[int, TrackFeatures],
    curve: list[SetSegment],
    profile: EventProfile,
    slot_count: int,
) -> None:
    """Ensure every MUST_PLAY / positional-seed track is in the completed beam.

    Called once on the chosen beam after the search. Any MUST_PLAY track not yet
    present is swapped in for the *weakest non-locked* placed slot whose segment
    it fits best, so the DJ's mandatory tracks are always included while
    disturbing the story as little as possible.
    """

    pool_by_id = {t.id: t for t in pool if t.id is not None}
    placed_ids = {p.track.id for p in beam.placements}

    # MUST_PLAY first, then positional seeds that hold a pin at the final length
    # (the beam can stop before reaching a pinned slot; _optimize_order then
    # moves each to its slot). Seeds that lost their pin — a 2nd preferred
    # intro, a peak with no MAIN_PEAK — stay optional, as before.
    pinned = set(_seed_positions(seeds, pool_by_id, curve, slot_count).values())
    protected = seeds.dj_chosen()  # never evict a track the DJ asked for
    for track_id in sorted(seeds.must_play) + sorted(pinned - seeds.must_play):
        if track_id in placed_ids or track_id not in pool_by_id:
            continue
        track = pool_by_id[track_id]
        f = features.get(track_id, TrackFeatures(track_id=track_id))

        # Find the best replaceable (non-locked) slot for this track: the slot
        # where its position score is highest, breaking ties toward replacing the
        # weakest currently-placed track.
        best_idx = None
        best_gain = None
        for idx, placement in enumerate(beam.placements):
            if placement.is_locked or placement.track.id in protected:
                continue
            frac = _position_fraction(placement.position, slot_count)
            segment = segment_at(curve, frac)
            score, _bd = compute_position_score(
                track, f, segment, profile, None, seeds.by_track.get(track_id, [])
            )
            # Gain = how much better the must-play scores here than the incumbent.
            gain = score - placement.position_score
            if best_gain is None or gain > best_gain or (
                gain == best_gain and best_idx is not None
                and placement.position_score < beam.placements[best_idx].position_score
            ):
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            _log.warning(
                "Could not place MUST_PLAY track_id=%s: no free slot.", track_id
            )
            continue

        # Swap it in.
        old = beam.placements[best_idx]
        if old.track.id is not None:
            beam.used_ids.discard(old.track.id)
        frac = _position_fraction(old.position, slot_count)
        new_placement = _make_placement(
            track,
            f,
            old.position,
            slot_count,
            curve,
            profile,
            None,  # neighbour rescoring happens in the final rescore pass
            seeds.by_track.get(track_id, []),
            is_locked=True,  # lock so a later must-play won't evict it
        )
        beam.placements[best_idx] = new_placement
        beam.used_ids.add(track_id)
        placed_ids.discard(old.track.id)
        placed_ids.add(track_id)


def _move_match(
    placements: list[_Placement], ids: set[int], *, to_end: bool
) -> None:
    """Move the first placement whose track id is in ``ids`` to an end.

    ``to_end=True`` moves it to the back of the list (the closer), ``False`` to
    the front (the opener). No-op if no placement matches. Mutates in place.
    """

    idx = next(
        (i for i, p in enumerate(placements) if p.track.id in ids), None
    )
    if idx is None:
        return
    p = placements.pop(idx)
    if to_end:
        placements.append(p)
    else:
        placements.insert(0, p)


def _finalize(
    beam: _Beam,
    slot_count: int,
    curve: list[SetSegment],
    profile: EventProfile,
    seeds: ConstraintSeeds,
    features: dict[int, TrackFeatures],
) -> SetPlan:
    """Rescore the chosen ordering end-to-end and build the :class:`SetPlan`.

    After must-play swaps the neighbour relationships may have changed, so we
    recompute each slot's transition/position score against its *actual* final
    neighbour, regenerate explanations, and assemble the plan with the real
    ``energy_points`` curve and the segments.
    """

    placements = sorted(beam.placements, key=lambda p: p.position)

    # Guarantee the DJ's preferred opener/closer actually bookend the set. The
    # outro is seeded at the *estimated* last slot, but the realized length can
    # differ (the snapshot selector may pick a longer/shorter set), so a pinned
    # outro can end up one or two slots from the end. Re-anchor them here, after
    # the final length is known, so PREFERRED_INTRO always opens and
    # PREFERRED_OUTRO always closes.
    # Skip when that end already holds a preferred track (or a LOCK) — the
    # optimizer anchored it there, and moving another one would undo its work.
    def _anchored(p: _Placement, ids: set[int]) -> bool:
        return p.track.id in ids or p.track.id in seeds.lock_positions

    if seeds.preferred_intro and placements and not _anchored(placements[0], seeds.preferred_intro):
        _move_match(placements, seeds.preferred_intro, to_end=False)
    if seeds.preferred_outro and placements and not _anchored(placements[-1], seeds.preferred_outro):
        _move_match(placements, seeds.preferred_outro, to_end=True)

    set_tracks: list[SetPlanTrack] = []
    energy_points: list[float] = []
    total_score = 0.0
    total_duration = 0

    prev_pair: tuple[Track, TrackFeatures] | None = None
    for new_pos, placement in enumerate(placements):
        track = placement.track
        f = placement.features
        frac = _position_fraction(new_pos, slot_count)
        segment = segment_at(curve, frac)

        # Rescore against the real final neighbour so the displayed numbers and
        # explanation match the order the DJ will actually see.
        score, breakdown = compute_position_score(
            track, f, segment, profile, prev_pair,
            seeds.by_track.get(track.id, []) if track.id is not None else [],
        )
        transition: TransitionScore | None = breakdown.get("transition")
        role = _slot_role(track, f, segment, profile)

        explanation = explain_track(
            track, f, role, segment, transition, score, profile
        )

        set_tracks.append(
            SetPlanTrack(
                track_id=track.id if track.id is not None else -1,
                position=new_pos,
                role=role,
                transition_score=(transition.score if transition is not None else 0.0),
                position_score=score,
                explanation=explanation,
                is_locked=placement.is_locked,
            )
        )
        energy_points.append(float(f.energy_score))
        total_score += score + (transition.score if transition is not None else 0.0)
        total_duration += _track_seconds(track)

        prev_pair = (track, f)

    target_seconds = profile.target_duration_minutes * 60

    return SetPlan(
        event_profile=profile,
        tracks=set_tracks,
        total_duration_seconds=total_duration,
        target_duration_seconds=target_seconds,
        total_score=total_score,
        segments=curve,
        energy_points=energy_points,
    )


def plan_set(
    library: list[Track],
    features: dict[int, TrackFeatures],
    profile: EventProfile,
    constraints: list[DjConstraint],
    *,
    beam_width: int = 20,
    max_candidates_per_step: int = 10,
    duration_tolerance_seconds: int = 300,
    adaptive_energy: bool = True,
    strict: bool = False,
) -> SetPlan:
    """Plan a complete set for ``profile`` from ``library``.

    See the module docstring for the full algorithm. Returns a :class:`SetPlan`
    whose total duration lands within ``duration_tolerance_seconds`` of the
    profile's target where the library allows, never includes an ``AVOID`` track,
    includes every feasible ``MUST_PLAY`` track, and places a ``PREFERRED_PEAK``
    track inside the ``main_peak`` segment.

    Deterministic: identical inputs always yield an identical plan.
    """

    # 0) Time-of-day modifier: shift the energy band lighter (DAY) -> stronger
    #    (NIGHT) BEFORE the curve, normalization, and scoring run. Work on a COPY
    #    so the caller's profile is never mutated.
    eff_lo, eff_hi = apply_time_modifier(
        profile.min_energy, profile.max_energy, profile.time_of_day
    )
    profile = replace(profile, min_energy=eff_lo, max_energy=eff_hi)

    target_seconds = profile.target_duration_minutes * 60

    # 1) Parse constraints and drop AVOID tracks.
    seeds = index_constraints(constraints)
    pool = build_candidate_pool(library, seeds)

    # 1a/1b) Venue filtering is STRICT-ONLY. Soft mode (the default) keeps every
    #     track and only RANKS by character via the position score — so a
    #     restaurant set still uses your whole library, just melodic-first. Strict
    #     additionally drops tracks above the venue's harshness ceiling AND tracks
    #     that don't fit the venue character. DJ-chosen tracks are exempt; an
    #     emptied pool falls back to the full pool.
    if strict:
        pool = _drop_over_cap_tracks(pool, features, profile, seeds)
        pool = _apply_strict_filter(pool, features, profile, seeds)

    # Empty library / fully avoided pool -> an empty but valid plan.
    if not pool:
        _log.warning("plan_set: candidate pool is empty; returning empty plan.")
        curve = generate_energy_curve(profile)
        return SetPlan(
            event_profile=profile,
            tracks=[],
            total_duration_seconds=0,
            target_duration_seconds=target_seconds,
            total_score=0.0,
            segments=curve,
            energy_points=[],
        )

    # 1c) Library-relative energy normalization over the KEPT pool: rescale each
    #     track's energy to where it sits WITHIN this library, onto the profile's
    #     (time-adjusted) band, so the arc works for any library's loudness.
    #     Done on a copy; the DB's absolute energies are untouched.
    #     The target is the CURVE's own span (its lowest intro band .. highest
    #     peak band), not the raw profile envelope: otherwise the loudest tracks
    #     land above every band, fit nowhere, and drift into the intro.
    curve = generate_energy_curve(profile)  # 2) Target story shape.
    if adaptive_energy:
        span = replace(
            profile,
            min_energy=min(s.min_energy for s in curve),
            max_energy=max(s.max_energy for s in curve),
        )
        features = relativize_features(pool, features, span)
    features = relativize_timbre(pool, features)

    pool_by_id = {t.id: t for t in pool if t.id is not None}

    # 3) The story is mapped across ``slot_estimate`` slots — the energy-curve
    #    fraction at each slot uses this, so positional seeds (preferred outro at
    #    the last slot, preferred peak inside the main_peak segment) line up with
    #    the arc. ``max_slots`` is a hard upper bound (using the SHORTEST tracks)
    #    so a duration-driven search always has the head-room to reach the target
    #    even when it picks shorter-than-average tracks.
    slot_estimate = _estimate_slot_count(pool, target_seconds)
    max_slots = _max_slots_for_duration(
        pool, target_seconds, duration_tolerance_seconds
    )
    plan_slots = max(slot_estimate, max_slots)

    # 4) Positional seeds (locked / preferred intro/outro/peak) laid out against
    #    the EXPECTED realized length so they fall inside the actual set, not a
    #    padded upper bound.
    pinned = _seed_positions(seeds, pool_by_id, curve, slot_estimate)
    # Slot beyond which no positional pin remains to be honoured (we must not
    # finalize at a length that drops a pinned slot).
    last_pinned_slot = max(pinned.keys()) if pinned else -1

    # 5/6) Beam search, slot by slot. We expand up to ``plan_slots`` (the
    #      duration head-room bound), snapshotting the best beam at EACH length so
    #      we can later pick the length whose runtime lands closest to the target.
    #      Candidate scoring uses the intended length ``slot_estimate`` for the
    #      curve fraction, so the narrative arc stays stable regardless of how
    #      many extra tail slots the duration target demands.
    beams: list[_Beam] = [_Beam()]
    # snapshots: list of (length, best_beam_at_that_length)
    snapshots: list[tuple[int, _Beam]] = []
    for slot in range(plan_slots):
        next_beams: list[_Beam] = []
        for beam in beams:
            next_beams.extend(
                _expand_beam(
                    beam,
                    slot,
                    slot_estimate,
                    pinned,
                    pool,
                    features,
                    curve,
                    profile,
                    seeds,
                    max_candidates_per_step,
                )
            )

        if not next_beams:
            # Ran out of candidates (e.g. tiny library) — keep what we have.
            break

        # Prune to the global top ``beam_width`` by cumulative score, with a
        # deterministic tie-break on the ordered tuple of placed track ids.
        next_beams.sort(
            key=lambda b: (
                -b.cumulative_score,
                tuple(p.track.id if p.track.id is not None else -1 for p in b.placements),
            )
        )
        beams = next_beams[:beam_width]
        snapshots.append((slot + 1, beams[0]))

        # Stop once the best beam's runtime has reached the upper tolerance edge:
        # going further can only overshoot. We always have enough snapshots by
        # now to pick the length closest to target.
        if beams[0].duration >= target_seconds + duration_tolerance_seconds:
            break

    # Choose the snapshot whose duration is closest to the target (deterministic
    # tie-break preferring the shorter set, then higher score). The chosen length
    # must include every positional pin.
    best_beam = _select_best_snapshot(
        snapshots, target_seconds, last_pinned_slot, duration_tolerance_seconds
    )

    # The story is mapped across however many slots we actually filled, so the
    # arc (intro..outro) spans the real set rather than a padded estimate.
    final_slot_count = max(1, len(best_beam.placements))

    # 7) Force in any missing MUST_PLAY tracks, re-order the chosen tracks for
    #    the best flow (exact), then rescore + assemble.
    _force_must_play(
        best_beam, seeds, pool, features, curve, profile, final_slot_count
    )
    _optimize_order(best_beam, curve, profile, seeds)
    return _finalize(best_beam, final_slot_count, curve, profile, seeds, features)


# Exact re-ordering is Held-Karp, O(2^n * n^2): 16 tracks ~0.3 s in pure Python.
# Bigger sets are improved with exact sliding windows of this size instead.
_EXACT_MAX_TRACKS = 16
_WINDOW = 12
_NEG = float("-inf")


def _held_karp(items, start, prev, nxt, step):
    """Best order of ``items`` for slots ``start..`` between fixed neighbours.

    ``step(slot, i, j)`` scores placing item ``j`` at ``slot`` after ``i``
    (``i`` None = nothing before). ``nxt`` (or None) is the fixed item right
    after the block, so its incoming step is part of the objective. Returns the
    best ordering, or None when the pins make every ordering infeasible.
    Deterministic: strict ``>`` keeps the first-found optimum.
    """

    n = len(items)
    full = (1 << n) - 1
    # tbl[k][a][b]: item b at slot start+k right after item a (plain lists so
    # the hot loop below is pure indexing).
    tbl = [None] + [
        [[step(start + k, items[a], items[b]) for b in range(n)] for a in range(n)]
        for k in range(1, n)
    ]
    # dp[mask][last] = (score, prev_last) over orderings of ``mask`` ending in last
    dp: list[dict[int, tuple[float, int]]] = [{} for _ in range(full + 1)]
    for a in range(n):
        v = step(start, prev, items[a])
        if v > _NEG:
            dp[1 << a][a] = (v, -1)
    for mask in range(1, full + 1):
        if not mask & (mask - 1):
            continue  # single-item orderings were seeded above
        t = tbl[mask.bit_count() - 1]
        row = dp[mask]
        for b in range(n):
            if not mask >> b & 1:
                continue
            best, arg = _NEG, -1
            for a, (v, _) in dp[mask ^ (1 << b)].items():
                s = v + t[a][b]
                if s > best:
                    best, arg = s, a
            if arg >= 0:
                row[b] = (best, arg)
    best, last = _NEG, -1
    for a, (v, _) in dp[full].items():
        if nxt is not None:
            v += step(start + n, items[a], nxt)
        if v > best:
            best, last = v, a
    if last < 0:
        return None
    order, mask = [], full
    while last >= 0:
        order.append(items[last])
        mask, last = mask ^ (1 << last), dp[mask][last][1]
    return order[::-1]


def _optimize_order(
    beam: _Beam,
    curve: list[SetSegment],
    profile: EventProfile,
    seeds: ConstraintSeeds,
) -> None:
    """Re-order the chosen tracks to maximise the set's total score.

    The beam decides WHICH tracks play; its left-to-right greed rarely finds the
    best ORDER (on real 15-track sets it never did). The objective is exactly
    what ``_finalize`` reports — sum of position + incoming transition scores
    for the final length — so this can only improve the plan. Positional pins
    (lock / preferred intro / outro / peak) are hard constraints. Mutates
    ``beam.placements`` in place.
    """

    pl = sorted(beam.placements, key=lambda p: p.position)
    n = len(pl)
    if n < 3:
        return

    segs = [segment_at(curve, _position_fraction(k, n)) for k in range(n)]
    seg_idx = [next(i for i, s in enumerate(curve) if s is seg) for seg in segs]
    pins = _seed_positions(seeds, {p.track.id: p.track for p in pl}, curve, n)
    pinned_ids = set(pins.values())
    cons = [seeds.by_track.get(p.track.id, []) for p in pl]
    memo: dict[tuple, float] = {}

    def step(slot: int, i: int | None, j: int) -> float:
        tid = pl[j].track.id
        if slot in pins and pins[slot] != tid or slot not in pins and tid in pinned_ids:
            return _NEG
        key = (seg_idx[slot], i, j)
        v = memo.get(key)
        if v is None:
            prev = None if i is None else (pl[i].track, pl[i].features)
            score, bd = compute_position_score(
                pl[j].track, pl[j].features, segs[slot], profile, prev, cons[j]
            )
            t = bd.get("transition")
            v = memo[key] = score + (t.score if t is not None else 0.0)
        return v

    def total(order: list[int]) -> float:
        return sum(step(k, order[k - 1] if k else None, j) for k, j in enumerate(order))

    order = list(range(n))
    if n <= _EXACT_MAX_TRACKS:
        order = _held_karp(order, 0, None, None, step) or order
    else:
        # Start pin-feasible (pinned tracks at their final-length slots, the
        # rest in beam order) so every window can only improve it.
        at = {k: j for k in pins for j in range(n) if pl[j].track.id == pins[k]}
        rest = iter(j for j in range(n) if j not in at.values())
        order = [at[k] if k in at else next(rest) for k in range(n)]
        # ponytail: windowed exact passes, not a global optimum for long sets
        # (~2% short on 37 tracks); add or-opt moves / restarts if that matters.
        for _ in range(4):
            before = total(order)
            for a in range(0, n - 1, _WINDOW // 2):
                b = min(n, a + _WINDOW)
                best = _held_karp(
                    order[a:b], a, order[a - 1] if a else None,
                    order[b] if b < n else None, step,
                )
                if best:
                    order[a:b] = best
            if total(order) <= before + 1e-9:
                break

    if total(order) > total(list(range(n))) + 1e-9:
        beam.placements = [replace(pl[j], position=k) for k, j in enumerate(order)]


def _select_best_snapshot(
    snapshots: list[tuple[int, _Beam]],
    target_seconds: int,
    last_pinned_slot: int,
    tolerance_seconds: int,
) -> _Beam:
    """Pick the best set length given the duration target.

    ``snapshots`` is ``[(length, best_beam_at_that_length), ...]`` in increasing
    length order. We only consider snapshots long enough to include every
    positional pin (``length > last_pinned_slot``).

    Selection is two-tier so we honour the duration target *and* tell a good
    story tail:

      * Among snapshots whose runtime is WITHIN tolerance of the target, prefer
        the one that ENDS BEST — i.e. whose last track has the lowest energy (an
        outro should wind down, never end on a peak) — then highest cumulative
        score, then closest to target. This avoids padding the set's end with a
        leftover high-energy track just to hit the runtime exactly.
      * If NO snapshot is within tolerance, fall back to pure duration proximity
        (closest runtime to target) so we still get as near the target as the
        library allows.

    Deterministic throughout (final tie-break on ordered track ids).
    """

    if not snapshots:
        return _Beam()

    # Only lengths that have placed every pinned slot are valid choices.
    valid = [
        (length, beam)
        for length, beam in snapshots
        if length > last_pinned_slot
    ]
    if not valid:
        valid = snapshots  # degrade: better a slightly-short set than none

    def _last_energy(b: _Beam) -> float:
        # Lower is better for an outro; an empty beam can't end well -> 1.0.
        if not b.placements:
            return 1.0
        return float(b.placements[-1].features.energy_score)

    def _ordered_ids(b: _Beam) -> tuple:
        return tuple(p.track.id if p.track.id is not None else -1 for p in b.placements)

    in_window = [
        (length, b)
        for length, b in valid
        if abs(b.duration - target_seconds) <= tolerance_seconds
    ]

    if in_window:
        # Prefer the lowest-energy ending, then quality, then closeness to target.
        return min(
            in_window,
            key=lambda item: (
                round(_last_energy(item[1]), 3),
                -item[1].cumulative_score,
                abs(item[1].duration - target_seconds),
                _ordered_ids(item[1]),
            ),
        )[1]

    # Nothing in tolerance: get as close to the target runtime as possible.
    return min(
        valid,
        key=lambda item: (
            abs(item[1].duration - target_seconds),
            -item[1].cumulative_score,
            _ordered_ids(item[1]),
        ),
    )[1]
