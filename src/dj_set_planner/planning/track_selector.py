"""Per-position track scoring and candidate-pool construction.

This module answers two questions the beam-search planner repeatedly asks:

1. *How well does this track fit THIS slot?* — :func:`compute_position_score`
   blends six context-aware sub-scores (context fit, energy-curve fit, role fit,
   mood fit, neighbour mixability, DJ preference) and subtracts penalties for
   things we want to discourage (energy-cap breaches, repeated artists, hard
   harmonic clashes). The result is a 0..1 placement score plus a breakdown
   dict for explanations/debugging.

2. *Which tracks are even allowed, and which are pinned?* — the candidate-pool
   helpers read the DJ's :class:`DjConstraint` list and:
     * DROP every ``AVOID`` track (never selectable),
     * surface the ``MUST_PLAY`` / ``PREFERRED_INTRO`` / ``PREFERRED_OUTRO`` /
       ``PREFERRED_PEAK`` / ``LOCK_POSITION`` seeds the planner must honour.

Design notes
------------
* Everything here is **pure / deterministic**: no wall-clock, no randomness.
  Given the same inputs the scores and pools are identical, so the planner
  stays reproducible (a hard requirement from CONTRACTS.md).
* Every scoring formula is commented inline with its weight, as required by the
  project style.
* We never silently swallow exceptions — :func:`compute_position_score` logs and
  degrades to a neutral score on unexpected input rather than crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.enums import ConstraintType
from ..domain.models import (
    DjConstraint,
    EventProfile,
    SetSegment,
    Track,
    TrackFeatures,
)
from ..utils.camelot import key_compatibility
from ..utils.logging import get_logger
from . import context_profiles
from .track_roles import role_fit_scores
from .transition_scorer import TransitionScore, score_transition
from .venue_profiles import character_fit, character_profile

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Position-score component weights (the positive parts sum to 1.0). These come
# straight from CONTRACTS.md:
#
#   context_fit               * 0.30
# + energy_curve_fit          * 0.25
# + role_fit                  * 0.20
# + mood_fit                  * 0.10
# + mixability_with_neighbors * 0.10
# + dj_preference             * 0.05
#
# ... then the penalties below are SUBTRACTED on top.
# --------------------------------------------------------------------------- #
_W_CONTEXT = 0.30
_W_ENERGY_CURVE = 0.25
_W_ROLE = 0.20
_W_MOOD = 0.10
_W_MIXABILITY = 0.10
_W_DJ_PREF = 0.05

# Penalty magnitudes (subtracted from the weighted blend, then clamped to 0..1).
#   * Energy cap breach: the track is louder than the day-party harshness
#     ceiling — strongly discouraged everywhere (it should usually have been
#     filtered out, but a penalty keeps it last-resort only).
#   * Repeated artist: the immediately-previous track shares this artist —
#     mild penalty to avoid back-to-back same-artist plays.
#   * Hard key clash: the incoming key clashes harmonically with the previous
#     track (key_compatibility very low) — mild penalty (transition scoring also
#     handles this, but at the position level it discourages painting the beam
#     into a harmonic corner).
_PEN_ENERGY_CAP = 0.40
_PEN_REPEAT_ARTIST = 0.10
_PEN_KEY_CLASH = 0.10

# A key_compatibility at/under this is treated as a "hard clash" for the penalty.
_KEY_CLASH_THRESHOLD = 0.30


def _clamp01(x: float) -> float:
    """Clamp ``x`` into the inclusive 0.0..1.0 range."""

    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _band_fit(value: float, low: float, high: float, tol: float = 0.12) -> float:
    """Triangular membership of ``value`` in the band ``[low, high]``.

    1.0 anywhere inside the band; decays linearly to 0.0 once ``value`` is more
    than ``tol`` past the nearest edge. Mirrors the helper in ``track_roles`` so
    energy-curve fit and role fit speak the same language.
    """

    if low <= value <= high:
        return 1.0
    dist = (low - value) if value < low else (value - high)
    if tol <= 0.0:
        return 0.0
    return _clamp01(1.0 - dist / tol)


def _context_fit(f: TrackFeatures, profile: EventProfile, segment: SetSegment) -> float:
    """How well the track suits the **venue's character** (0..1).

    This is the venue-aware character fit (see ``planning.venue_profiles``):
    melodic/gentle for a restaurant, percussive for a bar, strong/danceable for
    a club, hypnotic/low-vocal for an afterparty, etc. Energy itself is handled
    separately by ``_energy_curve_fit`` + adaptive normalization, so this term
    focuses purely on character.

    ``position_fraction`` is taken from the segment's midpoint so the only
    position-varying venue (OUTDOOR_DAY_PARTY: melodic early → percussive later)
    is scored at roughly where the track sits in the set.
    """

    venue = character_profile(profile.venue_type)
    position_fraction = (segment.start_pct + segment.end_pct) / 2.0
    return character_fit(f, venue, position_fraction)


def _energy_curve_fit(f: TrackFeatures, segment: SetSegment) -> float:
    """How well the track's energy lands in this segment's target band (0..1).

    This is the heart of telling the story: a track placed in the ``main_peak``
    segment should have ``energy_score`` inside ``[0.72, 0.78]``; a track in the
    intro should sit low. Triangular membership gives a smooth preference rather
    than a brittle in/out test.
    """

    return _band_fit(
        _clamp01(f.energy_score), segment.min_energy, segment.max_energy, tol=0.12
    )


def _mood_fit(f: TrackFeatures, profile: EventProfile) -> float:
    """Mood suitability for the context (0..1).

    A balanced day party wants a *warm, bright-but-not-blinding* mood. We score
    brightness with a soft preference for the upper-middle of the range and add
    a little groove so a flat, grooveless track doesn't read as "good mood".

      0.70 * brightness-near-0.60   (warm, inviting, not ice-cold or harsh)
    + 0.30 * groove
    """

    bright = _clamp01(f.mood_brightness)
    # Peak preference around 0.60 brightness; linearly falls off either side.
    bright_pref = _clamp01(1.0 - abs(bright - 0.60) / 0.40)
    groove = _clamp01(f.groove_score)
    return _clamp01(0.70 * bright_pref + 0.30 * groove)


def _mixability_with_neighbors(
    track: Track,
    f: TrackFeatures,
    prev_pair: tuple[Track, TrackFeatures] | None,
    profile: EventProfile,
    segment: SetSegment,
) -> tuple[float, TransitionScore | None]:
    """Neighbour mixability (0..1) and the incoming :class:`TransitionScore`.

    With a previous track we delegate to the transition scorer (the single
    source of truth for "how smooth is this blend") and also fold in the track's
    own intrinsic ``mixability_score`` so an easy-to-mix track is rewarded even
    when the specific transition is only average:

      0.70 * incoming_transition_score
    + 0.30 * track.mixability_score

    With NO previous track (the first slot) there's nothing to blend against, so
    we fall back to the track's intrinsic mixability alone and return ``None`` for
    the transition.
    """

    intrinsic = _clamp01(f.mixability_score)
    if prev_pair is None:
        return intrinsic, None

    prev, prev_f = prev_pair
    trans = score_transition(
        prev, prev_f, track, f, profile, entering_segment_role=segment.role
    )
    mix = 0.70 * _clamp01(trans.score) + 0.30 * intrinsic
    return _clamp01(mix), trans


def _dj_preference(constraints_for_track: list[DjConstraint], segment: SetSegment) -> float:
    """DJ-preference bonus (0..1) from this track's own constraints.

    A track the DJ explicitly wants (MUST_PLAY) or that they earmarked for the
    role of the current segment (PREFERRED_INTRO in an intro segment, etc.)
    scores high here, nudging the planner to honour the DJ's taste. A track with
    no relevant preference is neutral (0.5).
    """

    if not constraints_for_track:
        return 0.5

    types = {c.constraint_type for c in constraints_for_track}
    seg_role = (segment.role or "").upper()

    # An explicit must-play is the strongest signal of DJ preference.
    if ConstraintType.MUST_PLAY.value in types:
        return 1.0

    # A "preferred for this kind of slot" earmark that matches the current
    # segment role is a strong, but slightly weaker, signal.
    if ConstraintType.PREFERRED_INTRO.value in types and seg_role == "INTRO":
        return 0.95
    if ConstraintType.PREFERRED_OUTRO.value in types and seg_role == "OUTRO":
        return 0.95
    if ConstraintType.PREFERRED_PEAK.value in types and seg_role == "MAIN_PEAK":
        return 0.95

    # The track is earmarked for *some* preferred role, just not this slot's —
    # mildly above neutral (the DJ clearly likes it).
    preferred = {
        ConstraintType.PREFERRED_INTRO.value,
        ConstraintType.PREFERRED_OUTRO.value,
        ConstraintType.PREFERRED_PEAK.value,
        ConstraintType.LOCK_POSITION.value,
    }
    if types & preferred:
        return 0.6

    return 0.5


def compute_position_score(
    track: Track,
    f: TrackFeatures,
    segment: SetSegment,
    profile: EventProfile,
    prev_pair: tuple[Track, TrackFeatures] | None,
    constraints_for_track: list[DjConstraint] | None,
) -> tuple[float, dict]:
    """Score how well ``track`` fits the slot defined by ``segment``.

    Returns ``(score, breakdown)`` where ``score`` is the final 0..1 placement
    score and ``breakdown`` is a dict of the individual components (and the
    incoming :class:`TransitionScore` under key ``"transition"``, or ``None``
    for the first slot) so the planner can build explanations and the UI can
    show *why*.

    Weighted blend (positive weights sum to 1.0), then penalties subtracted::

        score = context_fit               * 0.30
              + energy_curve_fit          * 0.25
              + role_fit                  * 0.20
              + mood_fit                  * 0.10
              + mixability_with_neighbors * 0.10
              + dj_preference             * 0.05
              - energy_cap_penalty
              - repeat_artist_penalty
              - key_clash_penalty

    ``prev_pair`` is the previously placed ``(Track, TrackFeatures)`` or ``None``
    for the first slot. ``constraints_for_track`` is just the constraints
    targeting *this* track (may be empty/None).
    """

    constraints_for_track = constraints_for_track or []

    try:
        # --- positive components ----------------------------------------- #
        context = _context_fit(f, profile, segment)
        energy_curve = _energy_curve_fit(f, segment)

        # Role fit: how well this track suits the narrative role of the segment
        # it's being placed in (e.g. its MAIN_PEAK fit when placed in the
        # main_peak segment). role_fit_scores is keyed by TrackRole value.
        role_scores = role_fit_scores(track, f, profile)
        role_fit = _clamp01(role_scores.get(segment.role, 0.5))

        mood = _mood_fit(f, profile)
        mixability, transition = _mixability_with_neighbors(
            track, f, prev_pair, profile, segment
        )
        dj_pref = _dj_preference(constraints_for_track, segment)

        positive = (
            context * _W_CONTEXT
            + energy_curve * _W_ENERGY_CURVE
            + role_fit * _W_ROLE
            + mood * _W_MOOD
            + mixability * _W_MIXABILITY
            + dj_pref * _W_DJ_PREF
        )

        # --- penalties ---------------------------------------------------- #
        penalty = 0.0
        notes: list[str] = []

        # 1) Energy cap breach: louder than the day-party harshness ceiling
        #    (min of the named constant and the profile's own max_energy).
        hard_cap = min(context_profiles.avoid_energy_above, profile.max_energy)
        if f.energy_score > hard_cap:
            penalty += _PEN_ENERGY_CAP
            notes.append("over energy cap")

        # 2) Repeated artist back-to-back: discourage same artist twice in a row.
        if prev_pair is not None:
            prev, _prev_f = prev_pair
            if (
                track.artist
                and prev.artist
                and track.artist.strip().lower() == prev.artist.strip().lower()
            ):
                penalty += _PEN_REPEAT_ARTIST
                notes.append("repeats previous artist")

        # 3) Hard harmonic clash with the previous track.
        if prev_pair is not None:
            prev, _prev_f = prev_pair
            compat = key_compatibility(
                prev.camelot_key or prev.musical_key,
                track.camelot_key or track.musical_key,
            )
            if compat <= _KEY_CLASH_THRESHOLD:
                penalty += _PEN_KEY_CLASH
                notes.append("harmonic clash")

        final = _clamp01(positive - penalty)

        breakdown: dict = {
            "context_fit": context,
            "energy_curve_fit": energy_curve,
            "role_fit": role_fit,
            "mood_fit": mood,
            "mixability_with_neighbors": mixability,
            "dj_preference": dj_pref,
            "penalty": penalty,
            "transition": transition,
            "notes": notes,
        }
        return final, breakdown

    except Exception:  # never let one bad track crash the whole plan
        _log.exception(
            "Failed to compute position score for track_id=%s; neutral fallback",
            getattr(track, "id", None),
        )
        return 0.5, {
            "context_fit": 0.5,
            "energy_curve_fit": 0.5,
            "role_fit": 0.5,
            "mood_fit": 0.5,
            "mixability_with_neighbors": 0.5,
            "dj_preference": 0.5,
            "penalty": 0.0,
            "transition": None,
            "notes": ["error scoring position (neutral fallback)"],
        }


# --------------------------------------------------------------------------- #
# Candidate-pool / constraint helpers.
# --------------------------------------------------------------------------- #


@dataclass
class ConstraintSeeds:
    """Parsed constraint seeds the planner needs to honour.

    All track-id collections refer to ``Track.id``. ``lock_positions`` maps a
    track id to the 0-based position the DJ pinned it to (parsed from the
    ``LOCK_POSITION`` constraint's ``value``); unparseable values are ignored.
    ``by_track`` is the raw per-track constraint index so the position scorer can
    apply DJ-preference bonuses.
    """

    avoid: set[int]
    must_play: set[int]
    preferred_intro: set[int]
    preferred_outro: set[int]
    preferred_peak: set[int]
    lock_positions: dict[int, int]
    by_track: dict[int, list[DjConstraint]]

    def dj_chosen(self) -> set[int]:
        """Track ids the DJ explicitly wants (must-play / preferred / locked).

        These are exempt from venue-driven filtering (harshness ceiling, strict
        character) — an explicit choice always wins over an automatic rule.
        """

        return (
            self.must_play
            | self.preferred_intro
            | self.preferred_outro
            | self.preferred_peak
            | set(self.lock_positions)
        )


def index_constraints(constraints: list[DjConstraint] | None) -> ConstraintSeeds:
    """Parse a flat constraint list into the :class:`ConstraintSeeds` index.

    Pure and deterministic. Unknown constraint types (e.g. DO_NOT_PLAY_BEFORE /
    DO_NOT_PLAY_AFTER, which the MVP planner doesn't act on) are still indexed
    under ``by_track`` but otherwise ignored here.
    """

    seeds = ConstraintSeeds(
        avoid=set(),
        must_play=set(),
        preferred_intro=set(),
        preferred_outro=set(),
        preferred_peak=set(),
        lock_positions={},
        by_track={},
    )

    for c in constraints or []:
        seeds.by_track.setdefault(c.track_id, []).append(c)
        ctype = c.constraint_type
        if ctype == ConstraintType.AVOID.value:
            seeds.avoid.add(c.track_id)
        elif ctype == ConstraintType.MUST_PLAY.value:
            seeds.must_play.add(c.track_id)
        elif ctype == ConstraintType.PREFERRED_INTRO.value:
            seeds.preferred_intro.add(c.track_id)
        elif ctype == ConstraintType.PREFERRED_OUTRO.value:
            seeds.preferred_outro.add(c.track_id)
        elif ctype == ConstraintType.PREFERRED_PEAK.value:
            seeds.preferred_peak.add(c.track_id)
        elif ctype == ConstraintType.LOCK_POSITION.value:
            # value carries the pinned 0-based position as a string.
            pos = _parse_int(c.value)
            if pos is not None and pos >= 0:
                seeds.lock_positions[c.track_id] = pos
            else:
                _log.warning(
                    "Ignoring LOCK_POSITION for track_id=%s: bad value %r",
                    c.track_id,
                    c.value,
                )

    return seeds


def _parse_int(value: str | None) -> int | None:
    """Best-effort parse of an int from a constraint ``value`` (else None)."""

    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_candidate_pool(
    library: list[Track],
    seeds: ConstraintSeeds,
) -> list[Track]:
    """Return the selectable tracks: ``library`` minus every ``AVOID`` track.

    The result is sorted by ``Track.id`` (None ids sorted last, by file_path) so
    the planner iterates candidates in a stable, deterministic order regardless
    of the library's incoming order.
    """

    pool = [t for t in library if t.id not in seeds.avoid]
    # Deterministic order: by id, with a stable secondary key on file_path so
    # ties (e.g. unsaved tracks with id None) never depend on input order.
    pool.sort(key=lambda t: (t.id is None, t.id if t.id is not None else 0, t.file_path))
    return pool
