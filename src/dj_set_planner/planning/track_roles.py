"""Track-role fitting.

Given a :class:`Track`, its :class:`TrackFeatures`, and the target
:class:`EventProfile`, decide how well the track suits each of the eight
narrative :class:`TrackRole` values (INTRO, WARM_GROOVE, PROGRESSIVE_BUILD,
SMALL_PEAK, BREATHING_SPACE, MAIN_PEAK, RELEASE, OUTRO).

The planner uses these fits as one ingredient of the position score: a track
that is a natural "intro" should score highly for the INTRO slot and poorly for
the MAIN_PEAK slot, and so on.

Design notes
------------
* Every role returns a fit in the inclusive range 0.0..1.0.
* Each rule is a transparent, *deterministic* combination of the normalized
  features (no randomness, no wall-clock). Same inputs -> same outputs.
* Energy "windows" are scored with a smooth triangular membership: a track
  whose energy sits at the centre of a role's preferred band scores 1.0, and
  the score falls off linearly to 0.0 at the band edges (plus a configurable
  tolerance). This keeps the ranking continuous rather than a hard yes/no.
* MAIN_PEAK additionally respects the profile's energy ceiling: a track louder
  than the hard cap (``avoid_energy_above``) is disqualified from the main
  peak, because in a day-party restaurant context we never want a harsh track
  as the centrepiece.

Every scoring formula below is commented inline.
"""

from __future__ import annotations

from ..domain.enums import TrackRole
from ..domain.models import EventProfile, Track, TrackFeatures
from . import context_profiles
from ..utils.logging import get_logger

_log = get_logger(__name__)


def _clamp01(x: float) -> float:
    """Clamp ``x`` into the inclusive 0.0..1.0 range."""

    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _band_membership(value: float, low: float, high: float, tol: float = 0.12) -> float:
    """Triangular membership of ``value`` in the preferred band ``[low, high]``.

    Returns 1.0 when ``value`` is anywhere inside ``[low, high]``; outside the
    band the score decays linearly, reaching 0.0 once ``value`` is more than
    ``tol`` beyond the nearest edge. This gives a soft, continuous preference
    for "energy in this window" without a brittle hard cutoff.

    Example: band [0.40, 0.52], tol 0.12 -> value 0.46 => 1.0,
    value 0.58 => 0.5, value 0.64 => 0.0.
    """

    if low <= value <= high:
        return 1.0
    # Distance to the nearest edge of the band.
    dist = (low - value) if value < low else (value - high)
    if tol <= 0.0:
        return 0.0
    return _clamp01(1.0 - dist / tol)


def role_fit_scores(
    track: Track, f: TrackFeatures, profile: EventProfile
) -> dict[str, float]:
    """Return a 0..1 fit for each of the eight :class:`TrackRole` values.

    Keys are the ``TrackRole`` *string values* (e.g. ``"INTRO"``). Every role
    is present in the returned dict.

    The rules below read the normalized features and combine them with weights
    chosen so that the dominant signal for each role (e.g. ``intro_suitability``
    for INTRO, ``peak_potential`` for MAIN_PEAK) carries the most weight, while
    secondary signals (energy window, restaurant safety, groove) refine the
    ranking. The energy windows mirror the energy-curve segmentation so a
    track's role fit and its position on the curve agree.
    """

    # --- pull the features once, all already normalized to 0..1 ---------- #
    energy = _clamp01(f.energy_score)
    dance = _clamp01(f.danceability_score)
    groove = _clamp01(f.groove_score)
    bright = _clamp01(f.mood_brightness)
    intro_suit = _clamp01(f.intro_suitability)
    outro_suit = _clamp01(f.outro_suitability)
    peak_pot = _clamp01(f.peak_potential)
    rest_safe = _clamp01(f.restaurant_safety_score)

    # Profile-derived energy ceiling for the main peak. The named tuning
    # constant (0.85) is the harshness cap for the flagship day-party context;
    # we also never let a peak exceed the profile's own ``max_energy``.
    hard_cap = min(context_profiles.avoid_energy_above, profile.max_energy)

    scores: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # INTRO — open the set: low/medium energy, very restaurant-safe, and a
    # track that was *built* to be an intro (high intro_suitability).
    #   0.50 * intro_suitability   (the dominant cue)
    # + 0.25 * energy-in-[0.30,0.42] window (low/med energy)
    # + 0.25 * restaurant_safety   (must be gentle in a restaurant)
    # ------------------------------------------------------------------ #
    intro_energy = _band_membership(energy, 0.30, 0.42, tol=0.15)
    scores[TrackRole.INTRO.value] = _clamp01(
        0.50 * intro_suit + 0.25 * intro_energy + 0.25 * rest_safe
    )

    # ------------------------------------------------------------------ #
    # WARM_GROOVE — settle the room into a groove: groovy + danceable, still
    # gentle (low/med energy), restaurant-safe. This is the long cruising
    # section right after the intro.
    #   0.35 * groove
    # + 0.25 * danceability
    # + 0.25 * energy-in-[0.40,0.52] window
    # + 0.15 * restaurant_safety
    # ------------------------------------------------------------------ #
    warm_energy = _band_membership(energy, 0.40, 0.52, tol=0.14)
    scores[TrackRole.WARM_GROOVE.value] = _clamp01(
        0.35 * groove + 0.25 * dance + 0.25 * warm_energy + 0.15 * rest_safe
    )

    # ------------------------------------------------------------------ #
    # PROGRESSIVE_BUILD — start lifting the room: mid-rising energy, solid
    # danceability and groove, and a hint of peak potential (it's heading
    # somewhere). Brightness helps the sense of forward motion.
    #   0.30 * energy-in-[0.52,0.65] window  (the build band)
    # + 0.25 * danceability
    # + 0.20 * groove
    # + 0.15 * peak_potential               (momentum toward a peak)
    # + 0.10 * mood_brightness              (lifting, opening up)
    # ------------------------------------------------------------------ #
    build_energy = _band_membership(energy, 0.52, 0.65, tol=0.13)
    scores[TrackRole.PROGRESSIVE_BUILD.value] = _clamp01(
        0.30 * build_energy
        + 0.25 * dance
        + 0.20 * groove
        + 0.15 * peak_pot
        + 0.10 * bright
    )

    # ------------------------------------------------------------------ #
    # SMALL_PEAK — a smaller interest-building peak before the main one:
    # elevated (but not maximal) energy, high danceability, some peak
    # potential. We deliberately weight danceability over raw peak potential
    # so small peaks stay fun rather than overwhelming.
    #   0.30 * energy-in-[0.65,0.72] window
    # + 0.30 * danceability
    # + 0.25 * peak_potential
    # + 0.15 * groove
    # ------------------------------------------------------------------ #
    small_peak_energy = _band_membership(energy, 0.65, 0.72, tol=0.12)
    scores[TrackRole.SMALL_PEAK.value] = _clamp01(
        0.30 * small_peak_energy + 0.30 * dance + 0.25 * peak_pot + 0.15 * groove
    )

    # ------------------------------------------------------------------ #
    # BREATHING_SPACE — a deliberate dip after a peak: pull energy back down,
    # keep it groovy and pleasant (mood + groove), stay restaurant-safe. This
    # is a mini-release in the middle of the set so the crowd can recover.
    #   0.35 * energy-in-[0.50,0.60] window  (the dip band)
    # + 0.25 * groove
    # + 0.20 * mood_brightness
    # + 0.20 * restaurant_safety
    # ------------------------------------------------------------------ #
    breathe_energy = _band_membership(energy, 0.50, 0.60, tol=0.13)
    scores[TrackRole.BREATHING_SPACE.value] = _clamp01(
        0.35 * breathe_energy + 0.25 * groove + 0.20 * bright + 0.20 * rest_safe
    )

    # ------------------------------------------------------------------ #
    # MAIN_PEAK — the single centrepiece: high peak_potential + high
    # danceability, energy parked in the peak window [0.70, 0.80]. Crucially
    # the track must NOT exceed the profile's hard energy cap — a harsher
    # track is disqualified from being the main peak (score 0).
    #   0.40 * peak_potential
    # + 0.30 * danceability
    # + 0.30 * energy-in-[0.70,0.80] window
    #   ... then zeroed if energy > hard_cap.
    # ------------------------------------------------------------------ #
    peak_energy = _band_membership(energy, 0.70, 0.80, tol=0.10)
    main_peak_raw = 0.40 * peak_pot + 0.30 * dance + 0.30 * peak_energy
    if energy > hard_cap:
        # Over the harshness ceiling -> never the main peak in this context.
        main_peak_fit = 0.0
    else:
        main_peak_fit = main_peak_raw
    scores[TrackRole.MAIN_PEAK.value] = _clamp01(main_peak_fit)

    # ------------------------------------------------------------------ #
    # RELEASE — come down from the main peak: descending mid energy, still
    # danceable and groovy so the floor doesn't empty, with some brightness
    # for a warm, satisfied feel. Lower peak_potential is fine here.
    #   0.35 * energy-in-[0.55,0.65] window
    # + 0.25 * groove
    # + 0.20 * danceability
    # + 0.20 * mood_brightness
    # ------------------------------------------------------------------ #
    release_energy = _band_membership(energy, 0.55, 0.65, tol=0.13)
    scores[TrackRole.RELEASE.value] = _clamp01(
        0.35 * release_energy + 0.25 * groove + 0.20 * dance + 0.20 * bright
    )

    # ------------------------------------------------------------------ #
    # OUTRO — close the set: a track built to be an outro (high
    # outro_suitability) with low energy and a gentle, restaurant-safe feel.
    #   0.50 * outro_suitability  (the dominant cue)
    # + 0.30 * energy-in-[0.35,0.48] window (low energy wind-down)
    # + 0.20 * restaurant_safety
    # ------------------------------------------------------------------ #
    outro_energy = _band_membership(energy, 0.35, 0.48, tol=0.18)
    scores[TrackRole.OUTRO.value] = _clamp01(
        0.50 * outro_suit + 0.30 * outro_energy + 0.20 * rest_safe
    )

    return scores


def best_role(track: Track, f: TrackFeatures, profile: EventProfile) -> str:
    """Return the :class:`TrackRole` value with the highest fit (argmax).

    Ties are broken deterministically by the canonical declaration order of
    :class:`TrackRole` (so the same inputs always yield the same role).
    """

    scores = role_fit_scores(track, f, profile)
    # Iterate in TrackRole declaration order so that on an exact tie the
    # earlier-declared role wins -> stable, deterministic argmax.
    best_value = TrackRole.INTRO.value
    best_fit = -1.0
    for role in TrackRole:
        fit = scores[role.value]
        if fit > best_fit:
            best_fit = fit
            best_value = role.value
    return best_value
