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
* BPM compatibility is scored in **percent** stretch and understands
  **half-/double-time**: 70 BPM into 140 BPM shares a pulse.
* Harmonic compatibility is delegated to
  :func:`utils.camelot.key_compatibility` (the single source of truth for key
  scoring).
* The function is **pure / deterministic**: no wall-clock, no randomness — the
  same inputs always yield the same score.
"""

from __future__ import annotations

import math
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
_W_MOOD = 0.20         # sound-palette continuity (timbre; DJs order by it most)
_W_INTRO_OUTRO = 0.10  # are the overlapping edges easy to blend (drum-led)?

# Neutral values for UNMEASURED tracks, set near the median of measured pairs so
# a track that was never (re-)analyzed is neither favoured nor punished.
_MOOD_NEUTRAL = 0.5     # median z-scored timbre similarity ~0.48
_EDGES_NEUTRAL = 0.90   # median edge percussive share on real house/techno
_W_VOCAL = 0.05        # penalty: two vocal-heavy tracks clashing

# One bad factor ruins a blend no matter how good the rest is: the weaker of
# tempo/key below this floor costs extra (a soft minimum).
_CLASH_FLOOR = 0.45
_CLASH_WEIGHT = 0.30

# --------------------------------------------------------------------------- #
# BPM tuning.
# --------------------------------------------------------------------------- #
# Tempo stretch is scored in PERCENT (what the pitch fader / key lock feels):
# 1 / (1 + (pct/6)^4) -> 2% 0.99, 3% 0.94, 4% 0.83, 5% 0.68, 6% 0.50, 8% 0.24.
_BPM_HALF_PCT = 6.0


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
    # perfect tempo match (usually it's a half-time BPM read). No cut-off: an
    # unrelated tempo lands on the curve's floor anyway. Ratio-based, so the
    # score is the same in both directions.
    def gap(a: float, b: float) -> float:
        return max(a, b) / min(a, b) - 1.0

    rel, fold_note = gap(prev_bpm, nxt_bpm), ""
    for factor, label in ((2.0, "double-time"), (0.5, "half-time")):
        r = gap(prev_bpm, nxt_bpm * factor)
        if r < rel:
            rel, fold_note = r, label

    pct = rel * 100.0
    score = max(0.05, 1.0 / (1.0 + (pct / _BPM_HALF_PCT) ** 4))
    band = (
        "excellent" if pct <= 2 else "good" if pct <= 4
        else "stretch" if pct <= 6 else "wide tempo gap"
    )
    note = f"{band} ({pct:.1f}%"
    note += f", {fold_note})" if fold_note else ")"
    return score, note


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

    # Judge the AUDIBLE jump: the planner stretches energies across the profile
    # band for the arc, which would make a homogeneous download look jumpy.
    def _e(f: TrackFeatures) -> float:
        return f.raw_energy if f.raw_energy is not None else f.energy_score

    actual = _e(nxt_f) - _e(prev_f)
    word = "rise" if actual >= 0 else "fall"
    role = (entering_segment_role or "").upper()

    is_peak_entry = role in ("MAIN_PEAK", "SMALL_PEAK")
    # Building out of the intro is as intended as dropping into the outro —
    # otherwise the quiet tracks all get spent on the ending.
    is_build_entry = role in ("WARM_GROOVE", "PROGRESSIVE_BUILD")
    is_cooldown_entry = role in ("BREATHING_SPACE", "RELEASE", "OUTRO")

    # ``delta`` is measured WITH the intended flow: up into a build/peak, down
    # into a cool-down (so a gentle fall is the ideal move while winding down).
    delta = -actual if is_cooldown_entry else actual
    # A BIG move is only intended into a peak / cool-down; a build welcomes a
    # moderate lift but a +0.25 jump out of the intro is still abrupt.
    intended_big = is_peak_entry or is_cooldown_entry
    # Score at exactly +0.20 with the flow; the bands meet here continuously.
    edge = 0.85 if intended_big else 0.70 if is_build_entry else 0.55

    # --- gentle step with the flow: the ideal move (+0.00 0.95 -> +0.10 1.00)
    if 0.0 <= delta <= _SMALL_RISE_IDEAL:
        return 0.95 + (delta / _SMALL_RISE_IDEAL) * 0.05, f"gentle energy {word}"

    # --- slight step against the flow (-0.10 0.80 -> 0.00 0.95)
    if -_SMALL_RISE_IDEAL <= delta < 0.0:
        return 0.95 + (delta / _SMALL_RISE_IDEAL) * 0.15, f"slight energy {word}"

    # --- moderate step with the flow (+0.10 1.00 -> +0.20 ``edge``)
    if _SMALL_RISE_IDEAL < delta < _BIG_DELTA:
        frac = (delta - _SMALL_RISE_IDEAL) / (_BIG_DELTA - _SMALL_RISE_IDEAL)
        return 1.0 - frac * (1.0 - edge), f"noticeable energy {word}"

    # --- moderate step against the flow (-0.10 0.80 -> -0.20 0.45)
    if -_BIG_DELTA < delta <= -_SMALL_RISE_IDEAL:
        frac = (-delta - _SMALL_RISE_IDEAL) / (_BIG_DELTA - _SMALL_RISE_IDEAL)
        return 0.80 - frac * 0.35, f"energy {word} against the flow"

    # --- BIG step with the flow (>= 0.20)
    if delta >= _BIG_DELTA:
        if intended_big:
            # Welcome up to ~0.30; beyond that even a planned lift/drop is abrupt.
            score = max(0.30, 0.85 - max(0.0, delta - 0.30) * 2.0)
            return score, ("big lift into peak" if is_peak_entry else "deliberate energy release")
        return max(0.10, edge - (delta - _BIG_DELTA) * 1.5), "abrupt energy jump"

    # --- BIG step against the flow (<= -0.20): kills momentum
    return max(0.10, 0.45 - (-delta - _BIG_DELTA) * 1.5), (
        "energy crash" if actual < 0 else "abrupt energy jump"
    )


def _mood_continuity(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> tuple[float, str]:
    """Score sound-palette continuity in 0..1.

    The cosine similarity of the two timbre vectors (the planner z-scores them
    across the pool) mapped to 0..1 — "do these two sound like they belong
    together". Unmeasured on either side -> neutral, on the same scale, so a
    stale/failed track is never favoured over a measured one.
    """

    a, b = prev_f.timbre, nxt_f.timbre
    if a and b and len(a) == len(b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na > 0 and nb > 0:
            cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
            score = (cos + 1.0) / 2.0
            return score, "similar sound" if score >= 0.6 else "different sound"

    return _MOOD_NEUTRAL, ""


def _intro_outro_fit(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> tuple[float, str]:
    """Score how blendable the overlapping edges are in 0..1.

    The DJ overlaps prev's last ~30 s with nxt's first ~30 s; drum-led edges
    blend cleanly. Uses the measured percussive share of those edges. Not
    measured -> neutral (``intro/outro_suitability`` mean "good set
    opener/closer", i.e. low energy, so they must not be used here).
    """

    po, ni = prev_f.outro_percussive, nxt_f.intro_percussive
    if po is None or ni is None:
        return _EDGES_NEUTRAL, ""
    fit = (po + ni) / 2.0
    return max(0.0, min(1.0, fit)), "drum-led edges" if fit >= 0.95 else ""


def _tonal_overlap_gate(prev_f: TrackFeatures, nxt_f: TrackFeatures) -> float:
    """How much the key matters for this blend (0.75..1.0).

    Key clashes are only audible when tonal material overlaps. Drum-only edges
    (percussive share ~1) soften a clash a little; real harmonic content on
    both edges keeps the full key weight. Kept mild: the
    edge measure is energy-weighted, so a quiet pad under loud hats still reads
    "drums", and the key also shapes the set's harmonic flow beyond the overlap.
    """

    # An unmeasured side counts as a typical edge, so it isn't penalised.
    po = prev_f.outro_percussive if prev_f.outro_percussive is not None else _EDGES_NEUTRAL
    ni = nxt_f.intro_percussive if nxt_f.intro_percussive is not None else _EDGES_NEUTRAL
    tonal = math.sqrt(max(0.0, 1.0 - po) * max(0.0, 1.0 - ni))
    return 0.75 + 0.25 * min(1.0, tonal / _TONAL_FULL)


# Tonal overlap (geometric mean of 1 - percussive share) at/above which the key
# counts fully. Real house/techno edges (>=150 Hz) read median ~0.9 percussive,
# i.e. tonal ~0.1; clearly melodic edges read tonal 0.5+.
_TONAL_FULL = 0.25


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
              + key        * 0.25   (gated by the edges' tonal overlap)
              + energy     * 0.20
              + mood       * 0.20   (timbre similarity when measured)
              + intro_outro* 0.10   (drum-led overlapping edges)
              - vocal_pen  * 0.05
              - 0.30 * max(0, 0.45 - min(bpm, key))   (one clash dominates)

    ``entering_segment_role`` is the TrackRole/segment role the incoming track
    is entering (e.g. ``"MAIN_PEAK"``). It only affects the *energy* component:
    a big energy jump is welcome into a peak, and a big drop is welcome into a
    breathing/outro section.
    """

    try:
        bpm_score, bpm_note = _bpm_compatibility(prev.bpm, nxt.bpm)

        # Harmonic compatibility — single source of truth in utils.camelot.
        # Prefer the precomputed camelot_key, fall back to raw musical_key.
        key_raw = key_compatibility(
            prev.camelot_key or prev.musical_key,
            nxt.camelot_key or nxt.musical_key,
        )
        # A clash only hurts as much as the edges' tonal content lets it.
        key_score = 1.0 - _tonal_overlap_gate(prev_f, nxt_f) * (1.0 - key_raw)

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
            # Soft minimum: a tempo or key clash dominates the whole blend.
            - _CLASH_WEIGHT * max(0.0, _CLASH_FLOOR - min(bpm_score, key_score))
        )
        final = max(0.0, min(1.0, raw))

        # Human-readable note summarising the dominant factors.
        parts = [f"BPM {bpm_note}", f"key {key_raw:.2f}", energy_note]
        parts += [n for n in (mood_note, io_note, vocal_note) if n]
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
