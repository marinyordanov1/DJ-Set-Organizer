"""Librosa-backed feature extractor (optional dependency).

``librosa``/``numpy`` are an OPTIONAL extra. This module is written so that:

* Importing it NEVER imports librosa/numpy at module top level — both are
  imported lazily *inside* methods. So ``import
  dj_set_planner.analysis.librosa_extractor`` works even with librosa absent.
* :meth:`LibrosaFeatureExtractor.is_available` returns ``False`` (not raises)
  when librosa cannot be imported.
* :meth:`LibrosaFeatureExtractor.extract` computes real audio features
  (tempo, RMS energy, spectral centroid, onset strength, zero-crossing rate)
  and maps them to :class:`TrackFeatures` in 0..1. On ANY failure (missing
  dep, unreadable/missing audio file, decode error, etc.) it logs and falls
  back to :func:`heuristic_scorer.score_from_metadata` — it never raises out
  of ``extract``.

Each mapping formula is commented with what raw feature it uses and why.
"""

from __future__ import annotations

import os
from dataclasses import replace

from ..domain.models import Track, TrackFeatures
from ..utils.camelot import to_camelot
from ..utils.logging import get_logger
from .feature_extractor import FeatureExtractor
from .heuristic_scorer import _clamp01, score_from_metadata

_log = get_logger(__name__)

# How many seconds of audio to load for analysis. Loading the whole file is
# wasteful for long tracks; a representative window from the body of the track
# is enough for aggregate features and keeps analysis fast. We skip the first
# 30s (often a sparse intro) and analyze up to 90s.
_ANALYSIS_OFFSET_SECONDS = 30.0
_ANALYSIS_DURATION_SECONDS = 90.0

# Spectral-centroid normalization band (Hz). Centroid ~ perceived brightness.
# ~1.5 kHz reads warm/dark; ~6 kHz reads bright/harsh.
_CENTROID_MIN_HZ = 1500.0
_CENTROID_MAX_HZ = 6000.0

# BPM normalization band (mirrors the heuristic scorer for consistency).
_BPM_MIN = 80.0
_BPM_MAX = 135.0

# Krumhansl-Schmuckler key profiles (major / minor) for key estimation from
# chroma. Pitch classes are C..B; the profile is rolled to each tonic.
_KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _estimate_key(y, sr) -> str | None:
    """Estimate the musical key as ``"C"`` / ``"Am"`` from a signal (or None).

    Correlates the mean chroma vector against the 24 rotated Krumhansl-Schmuckler
    profiles and returns the best match. Approximate (~70-80% on clean material).
    """

    try:
        import librosa
        import numpy as np

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        if not np.any(chroma):
            return None
        best_corr = None
        best = None
        for i in range(12):
            for profile, is_minor in ((_KS_MAJOR, False), (_KS_MINOR, True)):
                rolled = np.roll(np.array(profile), i)
                corr = float(np.corrcoef(chroma, rolled)[0, 1])
                if best_corr is None or corr > best_corr:
                    best_corr = corr
                    best = (i, is_minor)
        if best is None:
            return None
        idx, minor = best
        return _PITCHES[idx] + ("m" if minor else "")
    except Exception:  # never let key estimation break analysis
        return None


class LibrosaFeatureExtractor(FeatureExtractor):
    """Concrete :class:`FeatureExtractor` backed by librosa, with safe fallback."""

    def is_available(self) -> bool:
        """Return ``True`` iff librosa (and numpy) import successfully."""

        try:
            import librosa  # noqa: F401  (lazy availability probe)
            import numpy  # noqa: F401
        except Exception:  # ImportError or any transitive import failure.
            _log.debug("librosa/numpy not importable", exc_info=True)
            return False
        return True

    def extract(self, track: Track) -> TrackFeatures:
        """Extract real audio features, falling back to heuristics on any error.

        The whole body is guarded: anything that goes wrong (no librosa, no
        file, decode failure, empty signal) is logged and we return the
        deterministic heuristic features instead, so callers always get a valid
        :class:`TrackFeatures`.
        """

        try:
            return self._extract_with_librosa(track)
        except Exception:
            # NEVER swallow silently: log with traceback, then degrade.
            _log.warning(
                "librosa extraction failed for %s; using heuristic fallback",
                track.file_path,
                exc_info=True,
            )
            # timbre=() marks "analysed by this version, nothing measurable", so
            # the app doesn't keep asking to re-analyse an undecodable file. A
            # MISSING file (unmounted drive) stays None: re-analyse it later.
            fallback = score_from_metadata(track)
            return replace(fallback, timbre=()) if os.path.isfile(track.file_path) else fallback

    # ------------------------------------------------------------------
    # Internal: the actual librosa work. Raises on any problem; the public
    # ``extract`` converts that into a heuristic fallback.
    # ------------------------------------------------------------------
    def _extract_with_librosa(self, track: Track) -> TrackFeatures:
        import librosa
        import numpy as np

        # Decode ONCE and slice: the body window (skip a sparse first 30 s, use
        # 90 s) plus the first/last ~30 s the DJ actually overlaps. Very long
        # files (full mixes) only get the body window — no whole-hour decode.
        # librosa raises if the file is missing/undecodable -> fallback.
        total = track.duration_seconds or librosa.get_duration(path=track.file_path)
        intro_perc = outro_perc = None
        if total <= _FULL_LOAD_MAX_SECONDS:
            y_full, sr = librosa.load(track.file_path, sr=None, mono=True)
            o = int(_ANALYSIS_OFFSET_SECONDS * sr)
            d = int(_ANALYSIS_DURATION_SECONDS * sr)
            y = y_full[o:o + d]
            if len(y) == 0:  # short track: analyse from the top
                y = y_full[:d]
            intro_perc, outro_perc = _edge_percussive(y_full, sr)
        else:
            y, sr = librosa.load(
                track.file_path,
                sr=None,
                mono=True,
                offset=_ANALYSIS_OFFSET_SECONDS,
                duration=_ANALYSIS_DURATION_SECONDS,
            )
        if y is None or len(y) == 0:
            raise ValueError("empty audio signal")

        timbre = _timbre(y, sr)  # sound palette (() on failure, never fatal)

        # -- Raw features ------------------------------------------------
        # Tempo (BPM): prefer the tag BPM if present (more reliable than
        # estimation for clean libraries); else estimate from onset envelope.
        if track.bpm and track.bpm > 0:
            bpm = float(track.bpm)
        else:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0

        # Persist estimated BPM/key back onto the track when tags + Rekordbox
        # didn't supply them, so the UI and transition scoring have values to
        # work with. Only fills MISSING fields — never overrides real data.
        # (librosa estimates are approximate.)
        if (not track.bpm or track.bpm <= 0) and bpm > 0:
            # Fold half/double-time into a typical DJ range. ponytail: assumes
            # 90-175 BPM (house/techno); a genuine 80 BPM track would fold to 160.
            est = bpm
            while est < 90.0:
                est *= 2.0
            while est >= 175.0:
                est /= 2.0
            track.bpm = round(est, 1)
        if not track.camelot_key:
            key = _estimate_key(y, sr)
            if key:
                track.musical_key = track.musical_key or key
                track.camelot_key = to_camelot(key)

        # RMS energy: loudness/density proxy. Mean RMS over frames.
        rms = float(np.mean(librosa.feature.rms(y=y)))

        # Spectral centroid: brightness proxy (Hz).
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # Onset strength: rhythmic activity / drive proxy.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset = float(np.mean(onset_env)) if onset_env.size else 0.0

        # Zero-crossing rate: noisiness/percussiveness proxy.
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        # Harmonic vs percussive balance (melodic <-> drum-driven). HPSS splits
        # the ALREADY-loaded signal (no second file load); ratio =
        # harmonic_rms / (harmonic_rms + percussive_rms), so 1.0 = fully melodic.
        try:
            h_e, p_e = _hpss_energies(y)
            h_rms, p_rms = float(np.sqrt(h_e)), float(np.sqrt(p_e))
            harmonic_ratio = _clamp01(h_rms / (h_rms + p_rms)) if (h_rms + p_rms) > 0 else 0.5
        except Exception:
            # Don't lose the other features if HPSS chokes on a clip.
            harmonic_ratio = 0.5

        # -- Normalize raw features into 0..1 ---------------------------
        # RMS is unbounded-ish; map through a fixed reference. Typical mastered
        # dance RMS sits ~0.05..0.30; we scale so 0.30 -> ~1.0.
        rms_n = _clamp01(rms / 0.30)

        # BPM normalized within the day-party band (fold double-time down).
        b = bpm
        if b >= 2 * _BPM_MIN:
            b = b / 2.0
        bpm_n = _clamp01((b - _BPM_MIN) / (_BPM_MAX - _BPM_MIN)) if b > 0 else 0.5

        # Centroid -> brightness within the warm..harsh band.
        brightness = _clamp01(
            (centroid - _CENTROID_MIN_HZ) / (_CENTROID_MAX_HZ - _CENTROID_MIN_HZ)
        )

        # Onset strength: reference-scaled. Onset envelopes commonly average a
        # few units for busy material; 3.0 is a reasonable "very driving" ref.
        onset_n = _clamp01(onset / 3.0)

        # ZCR -> "harshness" contribution; ZCR ~0.15 is already quite noisy.
        zcr_n = _clamp01(zcr / 0.15)

        # -- Map normalized features -> TrackFeatures -------------------
        # energy: loudness (RMS) + tempo + rhythmic drive (onset). RMS is the
        # strongest cue, so 0.45 RMS + 0.30 BPM + 0.25 onset.
        energy = _clamp01(0.45 * rms_n + 0.30 * bpm_n + 0.25 * onset_n)

        # danceability: steady strong onsets at a danceable tempo. 0.5 onset +
        # 0.5 tempo-closeness to ~122 BPM (triangular, +/-30 BPM half-width).
        tempo_dance = _clamp01(1.0 - abs(b - 122.0) / 30.0) if b > 0 else 0.5
        danceability = _clamp01(0.5 * onset_n + 0.5 * tempo_dance)

        # groove: rhythmic drive (onset) tempered by not-too-harsh timbre.
        # 0.7 onset + 0.3 (1 - zcr harshness).
        groove = _clamp01(0.7 * onset_n + 0.3 * (1.0 - zcr_n))

        # vocal_density: librosa can't cheaply separate vocals here; keep the
        # documented neutral 0.5 default (heuristic also defaults to 0.5).
        vocal_density = 0.5

        # intro_suitability: calmer (low energy) + lower brightness make better
        # intros. 0.7 (1-energy) + 0.3 (1-brightness).
        intro = _clamp01(0.7 * (1.0 - energy) + 0.3 * (1.0 - brightness))

        # outro_suitability: calmer + a little brighter to send people off.
        outro = _clamp01(0.7 * (1.0 - energy) + 0.3 * brightness)

        # peak_potential: high energy AND danceable. 0.6 energy + 0.4 dance.
        peak = _clamp01(0.6 * energy + 0.4 * danceability)

        # restaurant_safety_score: decreases with energy and harshness.
        # 1 - (0.75 energy + 0.25 zcr-harshness). Monotonic in energy.
        restaurant_safety = _clamp01(1.0 - (0.75 * energy + 0.25 * zcr_n))

        # mixability: danceable + mid-energy (extreme energy blends harder).
        mid_energy_close = _clamp01(1.0 - abs(energy - 0.55) / 0.55)
        mixability = _clamp01(0.6 * danceability + 0.4 * mid_energy_close)

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
            intro_percussive=intro_perc,
            outro_percussive=outro_perc,
            timbre=timbre,
        )


_EDGE_SECONDS = 30.0
# Above this a file is a full mix, not a track: body window only.
_FULL_LOAD_MAX_SECONDS = 20 * 60


def _edge_percussive(y_full, sr) -> tuple[float | None, float | None]:
    """Percussive share of the first and last ~30 s (the parts a DJ overlaps).

    The outro window stops 2 s before the end to skip the fade/silence tail.
    """

    try:
        n = int(_EDGE_SECONDS * sr)
        tail = int(2 * sr)
        if len(y_full) < 2 * n + tail:
            return None, None
        return (
            _percussive_share(y_full[:n], sr),
            _percussive_share(y_full[len(y_full) - n - tail:len(y_full) - tail], sr),
        )
    except Exception:
        _log.debug("edge analysis failed", exc_info=True)
        return None, None


def _hpss_energies(y, sr=None, fmin: float = 0.0) -> tuple[float, float]:
    """(harmonic, percussive) energy of a window via HPSS on the magnitude
    spectrogram, over bins >= ``fmin`` Hz. hop 1024 + kernel 17 is ~5x faster
    than ``effects.hpss`` (no inverse STFT) and plenty to tell drums from pads."""

    import librosa
    import numpy as np

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    H, P = librosa.decompose.hpss(S, kernel_size=17)
    if fmin > 0:
        keep = librosa.fft_frequencies(sr=sr, n_fft=2048) >= fmin
        H, P = H[keep], P[keep]
    return float(np.sum(H ** 2)), float(np.sum(P ** 2))


# A sustained kick fundamental (~50-100 Hz) reads as "harmonic" to HPSS, and the
# bass is EQ-swapped in a blend anyway: judge edges on what clashes audibly.
_EDGE_FMIN_HZ = 150.0


def _percussive_share(y, sr) -> float | None:
    """Percussive / (harmonic + percussive) energy of an edge window, or None."""

    if y is None or len(y) == 0:
        return None
    h, p = _hpss_energies(y, sr, _EDGE_FMIN_HZ)
    return round(p / (h + p), 4) if (h + p) > 0 else None


def _timbre(y, sr) -> tuple[float, ...]:
    """Sound-palette vector: mean MFCC 1-12 (c0 = loudness dropped) + spectral
    contrast, computed at a fixed 22.05 kHz so the file's sample rate / format
    doesn't masquerade as a different sound. () when it can't be measured."""

    try:
        import librosa
        import numpy as np

        if sr != _TIMBRE_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=_TIMBRE_SR)
        mfcc = librosa.feature.mfcc(y=y, sr=_TIMBRE_SR, n_mfcc=13)[1:].mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y=y, sr=_TIMBRE_SR).mean(axis=1)
        return tuple(round(float(v), 4) for v in np.concatenate([mfcc, contrast]))
    except Exception:
        _log.debug("timbre extraction failed", exc_info=True)
        return ()


_TIMBRE_SR = 22050
