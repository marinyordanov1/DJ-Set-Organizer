"""Transition scoring between two adjacent tracks.

Given an outgoing track (``prev``) and an incoming track (``nxt``) plus the
event context, produce a :class:`TransitionScore` describing how musically
smooth the move from one to the next is for THIS context (a daytime restaurant
party, by default).

The overall score is a weighted blend of five "good" sub-scores minus one
penalty (see :func:`score_transition` for the exact weights). Every component
is normalized to 0.0..1.0 so the weights are directly interpretable.

Design notes
------------
* Tuned for **smooth, energy-preserving** day-party transitions: small energy
  increases are rewarded, big jumps/drops are penalised unless the set is
  *deliberately* entering a peak (big jump OK) or a breathing/outro section
  (big drop OK).
* BPM compatibility understands **half-/double-time** mixing: 70 BPM into 140
  BPM is treated as a perfect tempo match because they share a pulse.
* Harmonic compatibility is delegated to
  :func:`utils.camelot.key_compatibility` (the single source of truth for key
  scoring).
* The function is **pure / deterministic**: no wall-clock, no randomness — the
  same inputs always yield the same score.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import EventProfile, Track, TrackFeatures
from ..utils.camelot import key_compatibility
from ..utils.logging import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Component weights (must sum the "positive" parts to 1.0; the vocal penalty is
# subtracted on top). These come straight from CONTRACTS.md.
# --------------------------------------------------------------------------- #
_W_BPM = 0.25          # tempo continuity is the backbone of a smooth blend
_W_KEY = 0.25          # harmonic continuity is equally important
_W_ENERGY = 0.20       # energy-delta shaping (context-aware)
_W_MOOD = 0.15         # mood/groove continuity (brightness + groove)
_W_INTRO_OUTRO = 0.10  # does prev end well and nxt begin well?
_W_VOCAL = 0.05        # penalty: two vocal-heavy tracks clashing

# --------------------------------------------------------------------------- #
# BPM tuning.
# --------------------------------------------------------------------------- #
# Half-/double-time detection tolerance: if nxt's bpm is within this fraction of
# 2x or 0.5x of prev's bpm, we fold it back to the same pulse before scoring
# (e.g. 70 vs 140 -> treated as 140 vs 140). 6% covers normal tempo spread.
_HALF_DOUBLE_TOLERANCE = 0.06

# BPM-difference bands (absolute difference after half/double folding), mapped
# to a 0..1 sub-score. Bands per CONTRACTS:
#   0-3 BPM   excellent  -> ~1.0
#   3-6 BPM   good       -> ~0.85..0.7
#   6-10 BPM  acceptable -> ~0.7..0.4
#   10+ BPM   penalty    -> decays toward 0
_BPM_EXCELLENT = 3.0
_BPM_GOOD = 6.0
_BPM_ACCEPTABLE = 10.0


@dataclass
class TransitionScore:
    """The result of scoring a single track-to-track transition.

    ``score`` is the final 0..1 blend; the remaining fields expose the
    individual (already 0..1) sub-scores so the UI can explain *why* a
    transition is good or bad. ``vocal_penalty`` is the raw 0..1 penalty
    magnitude (it is subtracted, weighted, from the blend).
    """

    score: float
    bpm: float
    key: float
    energy: float
    mood: float
    intro_outro: float
    vocal_penalty: float
    notes: str


def _bpm_compatibility(prev_bpm: float | None, nxt_bpm: float | None) -> tuple[float, str]:
    """Score tempo continuity in 0..1, honouring half-/double-time.

    Returns ``(score, note)``. When either BPM is unknown we return a neutral
    0.6 (missing tempo data should not dominate the blend, mirroring the key
    util's "unknown" handling).
    """

    # Missing data -> neutral, not punishing.
    if not prev_bpm or not nxt_bpm or prev_bpm <= 0 or nxt_bpm <= 0:
        return 0.6, "tempo unknown"

    # Fold half-/double-time onto the same pulse so a 70->140 move reads as a
    # perfect tempo match. We pick whichever of {nxt, nxt*2, nxt/2} lands
    # closest to prev, but only accept the doubled/halved variant if it is
    # within tolerance of an exact 2x relationship (so we don't accidentally
    # "fix" an unrelated tempo).
    candidates: list[tuple[float, str]] = [(nxt_bpm, "")]

    doubled = nxt_bpm * 2.0
    if abs(doubled - prev_bpm) <= prev_bpm * _HALF_DOUBLE_TOLERANCE * 2:
        candidates.append((doubled, "double-time"))
    halved = nxt_bpm / 2.0
    if abs(halved - prev_bpm) <= prev_bpm * _HALF_DOUBLE_TOLERANCE:
        candidates.append((halved, "half-time"))

    # Choose the interpretation with the smallest absolute BPM gap to prev.
    folded_bpm, fold_note = min(candidates, key=lambda c: abs(c[0] - prev_bpm))
    diff = abs(folded_bpm - prev_bpm)

    # Piecewise-linear mapping of the (folded) BPM difference to 0..1.
    if diff <= _BPM_EXCELLENT:
        # 0 BPM -> 1.0, 3 BPM -> 0.9 (still "excellent").
        score = 1.0 - (diff / _BPM_EXCELLENT) * 0.10
        band = "excellent"
    elif diff <= _BPM_GOOD:
        # 3 BPM -> 0.9, 6 BPM -> 0.7.
        score = 0.9 - ((diff - _BPM_EXCELLENT) / (_BPM_GOOD - _BPM_EXCELLENT)) * 0.20
        band = "good"
    elif diff <= _BPM_ACCEPTABLE:
        # 6 BPM -> 0.7, 10 BPM -> 0.40.
        score = 0.7 - ((diff - _BPM_GOOD) / (_BPM_ACCEPTABLE - _BPM_GOOD)) * 0.30
        band = "acceptable"
    else:
        # 10 BPM -> 0.40, decaying ~0.02 per extra BPM, floored at 0.05.
        score = max(0.05, 0.40 - (diff - _BPM_ACCEPTABLE) * 0.02)
        band = "wide tempo gap"

    note = f"{band} ({diff:.1f} BPM"
    note += f", {fold_note})" if fold_note else ")"
    return max(0.0, min(1.0, score)), note


# Energy-delta thresholds (difference in 0..1 energy_score between nxt and
# prev). "Small" rises are ideal for a day party; anything beyond these is a
# "big" jump/drop that needs contextual justification.
_SMALL_RISE_IDEAL = 0.10   # +0 to +0.10 is the sweet spot
_BIG_DELTA = 0.20          # magnitude beyond which a move is "big"


def _energy_delta_score(
    prev_f: TrackFeatures,
    nxt_f: TrackFeatures,
    entering_segment_role: str | None,
) -> tuple[float, str]:
    """Score the energy change in 0..1, tuned for a day party.

    Philosophy (day-party):
      * a gentle RISE (0..+0.10) is best -> ~1.0
      * staying flat is fine             -> ~0.9
      * a BIG JUMP up is penalised UNLESS we are deliberately entering a peak
      * a BIG DROP is penalised UNLESS we are entering breathing space / outro
      * mild drops/rises in between scale smoothly
    """

    delta = nxt_f.energy_score - prev_f.energy_score
    role = (entering_segment_role or "").upper()

    is_peak_entry = role in ("MAIN_PEAK", "SMALL_PEAK")
    is_cooldown_entry = role in ("BREATHING_SPACE", "RELEASE", "OUTRO")

    # --- gentle rise: the ideal day-party move ---------------------------- #
    if 0.0 <= delta <= _SMALL_RISE_IDEAL:
        # +0.00 -> 0.95, +0.10 -> 1.00 (a slight lift is *better* than flat).
        return 0.95 + (delta / _SMALL_RISE_IDEAL) * 0.05, "gentle energy rise"

    # --- mild fall: acceptable, gently penalised -------------------------- #
    if -_SMALL_RISE_IDEAL <= delta < 0.0:
        # -0.10 -> 0.80, -0.00 -> 0.95.
        return 0.95 + (delta / _SMALL_RISE_IDEAL) * 0.15, "slight energy dip"

    # --- moderate rise (0.10..0.20): still good but not ideal ------------- #
    if _SMALL_RISE_IDEAL < delta < _BIG_DELTA:
        # +0.10 -> 0.85, +0.20 -> 0.55.
        frac = (delta - _SMALL_RISE_IDEAL) / (_BIG_DELTA - _SMALL_RISE_IDEAL)
        return 0.85 - frac * 0.30, "noticeable energy rise"

    # --- moderate fall (-0.20..-0.10) ------------------------------------- #
    if -_BIG_DELTA < delta <= -_SMALL_RISE_IDEAL:
        if is_cooldown_entry:
            # Wanted here (we're cooling the room) -> reward.
            return 0.90, "intentional cool-down"
        frac = (-delta - _SMALL_RISE_IDEAL) / (_BIG_DELTA - _SMALL_RISE_IDEAL)
        return 0.80 - frac * 0.35, "energy dip"

    # --- BIG jump up (>= +0.20) ------------------------------------------- #
    if delta >= _BIG_DELTA:
        if is_peak_entry:
            # Big lift INTO a peak is exactly what we want -> strong score,
            # but cap below a perfect blend since it's still abrupt.
            return 0.85, "big lift into peak"
        # Otherwise jarring for a day party; decay with how big the jump is.
        # +0.20 -> 0.40, +0.50 -> ~0.10.
        score = max(0.10, 0.40 - (delta - _BIG_DELTA) * 1.0)
        return score, "abrupt energy jump"

    # --- BIG drop (<= -0.20) ---------------------------------------------- #
    # (delta <= -0.20)
    if is_cooldown_entry:
        return 0.85, "deliberate energy release"
    # Otherwise it kills momentum unexpectedly.
    score = max(0.10, 0.40 - (-delta - _BIG_DELTA) * 1.0)
    return score, "energy crash"


def _mood_continuity(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> tuple[float, str]:
    """Score mood/groove continuity in 0..1.

    Smooth transitions keep a similar *brightness* and *groove* feel. We
    measure the average absolute difference of ``mood_brightness`` and
    ``groove_score`` and map "more similar -> higher score". A full 1.0
    difference would be a total mood whiplash.
    """

    bright_diff = abs(nxt_f.mood_brightness - prev_f.mood_brightness)
    groove_diff = abs(nxt_f.groove_score - prev_f.groove_score)
    # Average the two feels; 0 diff -> identical mood, 1 diff -> opposite.
    avg_diff = (bright_diff + groove_diff) / 2.0
    # Linear: identical mood -> 1.0, fully opposite -> 0.0.
    score = 1.0 - avg_diff
    note = "consistent mood" if avg_diff <= 0.15 else "mood shift"
    return max(0.0, min(1.0, score)), note


def _intro_outro_fit(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> tuple[float, str]:
    """Score how well prev *ends* and nxt *begins* in 0..1.

    A clean blend wants the outgoing track to have a usable outro
    (``outro_suitability`` high) AND the incoming track to have a usable intro
    (``intro_suitability`` high). We average the two suitabilities.
    """

    fit = (prev_f.outro_suitability + nxt_f.intro_suitability) / 2.0
    note = "clean outro->intro" if fit >= 0.6 else "weak edges"
    return max(0.0, min(1.0, fit)), note


def _vocal_clash_penalty(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> tuple[float, str]:
    """Compute a 0..1 vocal-clash penalty from both tracks' vocal density.

    Two heavily-vocal tracks back-to-back risk lyrical/melodic clash during a
    blend. The penalty grows only when BOTH tracks are vocal-heavy: we take the
    product of the two densities (so a vocal track followed by an instrumental
    one carries little penalty) and scale it.
    """

    # product is high only when prev AND nxt are both vocal-rich.
    clash = prev_f.vocal_density * nxt_f.vocal_density
    # Only the part of the product above 0.25 (i.e. both densities ~>0.5)
    # actually counts as a clash; rescale that tail to 0..1.
    penalty = max(0.0, (clash - 0.25) / 0.75)
    note = "vocal clash risk" if penalty > 0.3 else ""
    return min(1.0, penalty), note


def score_transition(
    prev: Track,
    prev_f: TrackFeatures,
    nxt: Track,
    nxt_f: TrackFeatures,
    profile: EventProfile,
    *,
    entering_segment_role: str | None = None,
) -> TransitionScore:
    """Score the transition from ``prev`` into ``nxt`` for this ``profile``.

    Final score (all components in 0..1):

        score = bpm        * 0.25
              + key        * 0.25
              + energy     * 0.20
              + mood       * 0.15
              + intro_outro* 0.10
              - vocal_pen  * 0.05

    ``entering_segment_role`` is the TrackRole/segment role the incoming track
    is entering (e.g. ``"MAIN_PEAK"``). It only affects the *energy* component:
    a big energy jump is welcome into a peak, and a big drop is welcome into a
    breathing/outro section.
    """

    try:
        bpm_score, bpm_note = _bpm_compatibility(prev.bpm, nxt.bpm)

        # Harmonic compatibility — single source of truth in utils.camelot.
        # Prefer the precomputed camelot_key, fall back to raw musical_key.
        key_score = key_compatibility(
            prev.camelot_key or prev.musical_key,
            nxt.camelot_key or nxt.musical_key,
        )

        energy_score, energy_note = _energy_delta_score(
            prev_f, nxt_f, entering_segment_role
        )
        mood_score, mood_note = _mood_continuity(prev_f, nxt_f)
        intro_outro_score, io_note = _intro_outro_fit(prev_f, nxt_f)
        vocal_penalty, vocal_note = _vocal_clash_penalty(prev_f, nxt_f)

        # Weighted blend (positive weights sum to 1.0; vocal penalty subtracts).
        raw = (
            bpm_score * _W_BPM
            + key_score * _W_KEY
            + energy_score * _W_ENERGY
            + mood_score * _W_MOOD
            + intro_outro_score * _W_INTRO_OUTRO
            - vocal_penalty * _W_VOCAL
        )
        final = max(0.0, min(1.0, raw))

        # Human-readable note summarising the dominant factors.
        parts = [f"BPM {bpm_note}", f"key {key_score:.2f}", energy_note]
        if vocal_note:
            parts.append(vocal_note)
        notes = "; ".join(parts)

        return TransitionScore(
            score=final,
            bpm=bpm_score,
            key=key_score,
            energy=energy_score,
            mood=mood_score,
            intro_outro=intro_outro_score,
            vocal_penalty=vocal_penalty,
            notes=notes,
        )
    except Exception:  # never silently swallow — log and degrade gracefully
        _log.exception(
            "Failed to score transition %s -> %s; returning neutral score",
            getattr(prev, "file_path", "?"),
            getattr(nxt, "file_path", "?"),
        )
        return TransitionScore(
            score=0.5,
            bpm=0.5,
            key=0.5,
            energy=0.5,
            mood=0.5,
            intro_outro=0.5,
            vocal_penalty=0.0,
            notes="error scoring transition (neutral fallback)",
        )
