"""Deterministic heuristic feature scorer.

When real audio analysis (librosa) is unavailable, this module derives a full
set of :class:`TrackFeatures` from *tags only* — BPM, genre, key, duration —
plus a small, **stable** per-track jitter computed from a hash of the file
path. The result is fully deterministic: the same :class:`Track` in always
yields the same features (no ``random``, no wall-clock), which keeps the
planner and its tests reproducible.

Design goals
============
* Every field is clamped to the inclusive range ``0.0..1.0``.
* Genre and BPM are the dominant signals for energy; key/duration refine it.
* The jitter is tiny (``+/- ~0.04``) so it only breaks ties / spreads out
  same-genre tracks; it never overrides the genre/BPM signal.
* Every formula is commented with its rationale and weights.

The scoring is intentionally simple and explainable rather than "accurate":
its job is to make the app fully usable with no heavy dependencies installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from ..domain.models import Track, TrackFeatures
from ..utils.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Genre energy/brightness priors
# ---------------------------------------------------------------------------
# Each genre maps to (energy, brightness) priors in 0..1. These are the
# *baseline* feel of the genre before BPM/key/duration adjustments. Keys are
# lowercase; lookup is substring-based so "Deep House (Extended Mix)" still
# matches "deep house". Order matters for substring matching: more specific
# multi-word genres come first so e.g. "tech house" wins over "house".
_GENRE_PRIORS: list[tuple[str, tuple[float, float]]] = [
    # genre substring        (energy, brightness)
    ("melodic techno", (0.72, 0.55)),
    ("melodic house", (0.55, 0.62)),
    ("organic house", (0.45, 0.55)),
    ("soulful house", (0.55, 0.65)),
    ("jazzy house", (0.50, 0.66)),
    ("funky house", (0.62, 0.70)),
    ("deep house", (0.48, 0.45)),
    ("tech house", (0.70, 0.50)),
    ("afro house", (0.66, 0.58)),
    ("nu disco", (0.62, 0.72)),
    ("progressive house", (0.66, 0.58)),
    ("downtempo", (0.25, 0.45)),
    ("ambient", (0.15, 0.50)),
    ("lounge", (0.28, 0.55)),
    ("chillout", (0.25, 0.50)),
    ("disco", (0.65, 0.74)),
    ("funk", (0.62, 0.72)),
    ("soul", (0.50, 0.66)),
    ("jazz", (0.40, 0.62)),
    ("techno", (0.80, 0.48)),
    ("trance", (0.78, 0.62)),
    ("drum and bass", (0.88, 0.60)),
    ("dnb", (0.88, 0.60)),
    ("house", (0.58, 0.55)),  # generic catch-all, kept last among "*house"
    ("pop", (0.60, 0.70)),
    ("rock", (0.65, 0.60)),
    ("hip hop", (0.55, 0.50)),
    ("hip-hop", (0.55, 0.50)),
    ("rnb", (0.45, 0.58)),
    ("r&b", (0.45, 0.58)),
]

# Neutral fallback when a genre is missing or unrecognized.
_DEFAULT_GENRE_PRIOR: tuple[float, float] = (0.50, 0.55)

# BPM normalization window. Day-party / open-format material mostly lives in
# 90..128 BPM; we normalize within a slightly wider window so out-of-range
# values still map sensibly into 0..1.
_BPM_MIN = 80.0
_BPM_MAX = 135.0


def _clamp01(x: float) -> float:
    """Clamp ``x`` to the inclusive range 0.0..1.0."""

    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _stable_jitter(file_path: str, salt: str) -> float:
    """Deterministic, path-derived jitter in the range ``[-0.04, +0.04]``.

    We hash ``salt + file_path`` with blake2b and read the first two bytes as
    an integer in 0..65535, then map that to ``[-0.04, 0.04]``. Using a salt
    per feature means different features get *different* (but still stable)
    jitter, so two same-genre/same-BPM tracks separate slightly across fields
    rather than moving in lockstep. This is purely for tie-breaking spread; it
    is small enough never to flip a genre/BPM decision.
    """

    digest = hashlib.blake2b(
        (salt + "|" + file_path).encode("utf-8"), digest_size=2
    ).digest()
    raw = int.from_bytes(digest, "big")  # 0..65535
    unit = raw / 65535.0  # 0..1
    # Map 0..1 -> -0.04..+0.04 (width 0.08 centred on 0).
    return (unit - 0.5) * 0.08


def _genre_prior(genre: str | None) -> tuple[float, float]:
    """Return ``(energy, brightness)`` priors for ``genre`` via substring match."""

    if not genre:
        return _DEFAULT_GENRE_PRIOR
    g = genre.strip().lower()
    if not g:
        return _DEFAULT_GENRE_PRIOR
    for needle, prior in _GENRE_PRIORS:
        if needle in g:
            return prior
    return _DEFAULT_GENRE_PRIOR


def _bpm_factor(bpm: float | None) -> float:
    """Normalize BPM to 0..1 within ``[_BPM_MIN, _BPM_MAX]``.

    Returns the neutral 0.5 when BPM is missing. Faster tempo -> higher value.
    """

    if bpm is None or bpm <= 0:
        return 0.5  # unknown tempo -> neutral
    # Some tags store double/half time; fold extremes back into a sane band so
    # a mis-tagged 180 still reads as energetic rather than saturating.
    b = float(bpm)
    if b >= 2 * _BPM_MIN:
        # Looks like double-time; halving usually restores the musical tempo.
        b = b / 2.0
    return _clamp01((b - _BPM_MIN) / (_BPM_MAX - _BPM_MIN))


def score_from_metadata(track: Track) -> TrackFeatures:
    """Derive deterministic :class:`TrackFeatures` from a track's tags.

    All fields are clamped to 0..1. The function is pure and deterministic:
    same ``track`` in -> identical features out.
    """

    fp = track.file_path or ""
    genre_energy, genre_brightness = _genre_prior(track.genre)
    bpm_f = _bpm_factor(track.bpm)

    # -- energy_score -------------------------------------------------------
    # Energy is mostly genre feel blended with tempo. We weight genre 0.55 and
    # BPM 0.45 because genre captures production density/aggression that BPM
    # alone misses (a 122 BPM deep-house track is calmer than 122 BPM techno).
    energy = 0.55 * genre_energy + 0.45 * bpm_f
    energy = _clamp01(energy + _stable_jitter(fp, "energy"))

    # -- mood_brightness ----------------------------------------------------
    # Brightness comes from genre timbre, nudged up for major keys and down for
    # minor keys (minor reads "darker"/moodier). Camelot "B" ring == major.
    brightness = genre_brightness
    cam = track.camelot_key
    if cam:
        ring = cam.strip().upper()[-1:] if cam.strip() else ""
        if ring == "B":  # major key -> a touch brighter
            brightness += 0.05
        elif ring == "A":  # minor key -> a touch darker
            brightness -= 0.05
    brightness = _clamp01(brightness + _stable_jitter(fp, "brightness"))

    # -- danceability_score -------------------------------------------------
    # Danceability peaks for "four-on-the-floor" club tempos (~118..126 BPM)
    # and tapers for very slow or very fast tracks. We model it as a triangular
    # response centred at 122 BPM with a +/-30 BPM half-width, blended with a
    # genre groove floor so danceable genres stay danceable even off-centre.
    if track.bpm and track.bpm > 0:
        b = float(track.bpm)
        if b >= 2 * _BPM_MIN:
            b = b / 2.0
        # Triangular: 1.0 at 122 BPM, linearly down to 0 at +/-30 BPM away.
        tempo_dance = _clamp01(1.0 - abs(b - 122.0) / 30.0)
    else:
        tempo_dance = 0.5  # unknown tempo -> neutral
    # Blend: 0.6 tempo response + 0.4 genre-energy floor (energetic genres tend
    # to be more danceable). Comment: this keeps a slow soulful-house groove
    # respectable while still rewarding peak-time tempos.
    danceability = 0.6 * tempo_dance + 0.4 * genre_energy
    danceability = _clamp01(danceability + _stable_jitter(fp, "dance"))

    # -- groove_score -------------------------------------------------------
    # Groove ~= danceability with a brightness penalty (very bright/harsh
    # tracks groove a little less in a warm day-party context) and a small
    # boost for mid energy (groove lives in the pocket, not the peak). We take
    # danceability, subtract 0.1 * how-far-brightness-exceeds-0.7, and add a
    # mild bonus for energy near the 0.5..0.65 sweet spot.
    bright_penalty = 0.1 * max(0.0, brightness - 0.7)
    mid_energy_bonus = 0.08 * (1.0 - abs(energy - 0.58) / 0.58)
    groove = danceability - bright_penalty + _clamp01(mid_energy_bonus)
    groove = _clamp01(groove + _stable_jitter(fp, "groove"))

    # -- vocal_density ------------------------------------------------------
    # We cannot detect vocals from tags, so the spec fixes the default at 0.5.
    # We apply only a tiny stable jitter so same-value ties still break, but
    # keep it centred on 0.5 (the documented neutral default).
    vocal_density = _clamp01(0.5 + 0.5 * _stable_jitter(fp, "vocal"))

    # -- intro_suitability --------------------------------------------------
    # Good intros are low-energy, on the longer side, and not too bright. We
    # invert energy (calmer = better intro) and add a small bonus for longer
    # tracks (more runway to ease in). Weight: 0.8 calmness + 0.2 length.
    length_bonus = _length_factor(track.duration_seconds)
    intro = 0.8 * (1.0 - energy) + 0.2 * length_bonus
    intro = _clamp01(intro + _stable_jitter(fp, "intro"))

    # -- outro_suitability --------------------------------------------------
    # Good outros are also lower energy and ideally a bit brighter/uplifting to
    # send people off well. 0.7 calmness + 0.3 brightness, lightly jittered.
    outro = 0.7 * (1.0 - energy) + 0.3 * brightness
    outro = _clamp01(outro + _stable_jitter(fp, "outro"))

    # -- peak_potential -----------------------------------------------------
    # Peak material is high energy AND danceable. We multiply-ish blend the two
    # (0.6 energy + 0.4 danceability) so a track needs both to score high; a
    # fast-but-undanceable or danceable-but-low-energy track won't top out.
    peak = 0.6 * energy + 0.4 * danceability
    peak = _clamp01(peak + _stable_jitter(fp, "peak"))

    # -- restaurant_safety_score -------------------------------------------
    # "Restaurant safe" = NOT too energetic and NOT too harsh/bright, so people
    # can eat and talk. It must DECREASE as energy rises. We penalize energy
    # heavily and brightness mildly: safety = 1 - (0.75*energy + 0.25*excess
    # brightness above 0.6). The brightness term only bites for genuinely harsh
    # tracks (brightness > 0.6), scaled into 0..1 over the 0.6..1.0 band.
    harsh = max(0.0, brightness - 0.6) / 0.4  # 0 at 0.6, 1 at 1.0
    safety = 1.0 - (0.75 * energy + 0.25 * harsh)
    # No jitter here: we want a clean monotonic relationship with energy so the
    # "safety decreases as energy rises" invariant holds exactly per track.
    restaurant_safety = _clamp01(safety)

    # -- mixability_score ---------------------------------------------------
    # Mixability rewards a steady, mid-tempo, four-on-the-floor feel that's easy
    # to beatmatch and blend: high danceability, mid energy (extreme energy is
    # harder to blend smoothly), and a known BPM. 0.6 danceability + 0.25
    # mid-energy-closeness + 0.15 has-bpm bonus.
    mid_energy_close = 1.0 - abs(energy - 0.55) / 0.55  # 1 at 0.55, ->0 at extremes
    has_bpm = 1.0 if (track.bpm and track.bpm > 0) else 0.4
    mixability = (
        0.6 * danceability
        + 0.25 * _clamp01(mid_energy_close)
        + 0.15 * has_bpm
    )
    mixability = _clamp01(mixability + _stable_jitter(fp, "mix"))

    # -- harmonic_ratio -----------------------------------------------------
    # Melodic <-> percussive. We can't run HPSS from tags, so approximate: a
    # brighter, less groove-driven track reads more melodic; a heavy-groove
    # track reads more percussive. Centred on 0.5 and clamped to ~0.2..0.8.
    harmonic_ratio = _clamp01(0.5 + 0.6 * (brightness - groove))

    return TrackFeatures(
        track_id=track.id if track.id is not None else 0,
        energy_score=energy,
        danceability_score=danceability,
        mood_brightness=brightness,
        groove_score=groove,
        vocal_density=vocal_density,
        intro_suitability=intro,
        outro_suitability=outro,
        peak_potential=peak,
        restaurant_safety_score=restaurant_safety,
        mixability_score=mixability,
        harmonic_ratio=harmonic_ratio,
    )


def _length_factor(duration_seconds: int | None) -> float:
    """Map track length to a 0..1 "runway" factor (longer -> higher).

    Saturates at 8 minutes (480s). Unknown length -> neutral 0.5. Used to
    favour longer tracks for intro/outro roles.
    """

    if duration_seconds is None or duration_seconds <= 0:
        return 0.5
    return _clamp01(float(duration_seconds) / 480.0)


def derive_energy_dependent(f: TrackFeatures, energy: float) -> TrackFeatures:
    """Return a copy of ``f`` with ``energy`` set and every energy-DEPENDENT
    field recomputed consistently from it.

    Used in two places that change a track's energy after analysis:
      * a manual DJ energy override (the DJ knows a track is lighter than the
        analyzer thinks), and
      * library-relative energy normalization (see ``planning.energy_normalize``).

    Only the fields that are *functions of energy* are recomputed
    (``intro_suitability``, ``outro_suitability``, ``peak_potential``,
    ``restaurant_safety_score``, ``mixability_score``); the genuinely-measured,
    energy-independent fields (``danceability``, ``mood_brightness``,
    ``groove``, ``vocal_density``) are preserved. The formulas mirror the
    extractors so a recomputed track behaves exactly like one analyzed at that
    energy to begin with.
    """

    e = _clamp01(energy)
    bright = _clamp01(f.mood_brightness)
    dance = _clamp01(f.danceability_score)

    # Calmer + less bright -> better intro (mirrors the librosa intro formula).
    intro = _clamp01(0.75 * (1.0 - e) + 0.25 * (1.0 - bright))
    # Calmer + a touch brighter -> better outro (send people off warmly).
    outro = _clamp01(0.70 * (1.0 - e) + 0.30 * bright)
    # Peak material is high energy AND danceable.
    peak = _clamp01(0.60 * e + 0.40 * dance)
    # Restaurant safety decreases with energy (and harsh brightness).
    harsh = max(0.0, bright - 0.6) / 0.4
    safety = _clamp01(1.0 - (0.75 * e + 0.25 * harsh))
    # Mixability rewards danceable, mid-energy tracks (extremes blend harder).
    mid_close = _clamp01(1.0 - abs(e - 0.55) / 0.55)
    mixability = _clamp01(0.6 * dance + 0.4 * mid_close)

    return replace(
        f,
        energy_score=e,
        intro_suitability=intro,
        outro_suitability=outro,
        peak_potential=peak,
        restaurant_safety_score=safety,
        mixability_score=mixability,
    )
