"""Library-relative energy normalization.

The energy curve (``energy_curve.py``) describes the set's *target* energy in
ABSOLUTE 0..1 values tuned for a day-party (intro ~0.30, main peak ~0.75). But a
track's measured ``energy_score`` lives on whatever absolute scale the analyzer
and the music happen to produce: a library of loud club/tech-house tracks can
sit entirely at 0.55..0.91, so NOTHING lands in the 0.30 intro band and the
planner can't tell the DJ's lightest opener from their biggest peak.

``relativize_features`` rescales each track's energy from where it sits within
the library's own range onto the profile's target envelope. The library's
lowest-energy tracks map to ``profile.min_energy`` (intro material), the highest
to ``profile.max_energy`` (peaks), and the story's shape is preserved. It anchors
on the 5th/95th percentiles so one outlier doesn't squash everyone else.
``harmonic_ratio`` is spread the same way so the venue character bands can tell a
homogeneous library's tracks apart.

NOTE: only ANALYZED tracks are rescaled. Tracks without features keep the neutral
0.5 default — so an unanalyzed library produces a flat curve. Run "Analyze" first.
"""

from __future__ import annotations

from dataclasses import replace

from ..analysis.heuristic_scorer import _clamp01, derive_energy_dependent
from ..domain.models import EventProfile, Track, TrackFeatures
from ..utils.logging import get_logger

_log = get_logger(__name__)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` (0..1) of a pre-sorted list."""

    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[-1]


def relativize_features(
    pool: list[Track],
    features: dict[int, TrackFeatures],
    profile: EventProfile,
    *,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> dict[int, TrackFeatures]:
    """Return a features dict whose energies are rescaled to the profile band.

    Only pooled tracks WITH features are rescaled; others pass through unchanged.
    If the library has too few tracks or no spread, the original is returned.
    """

    energies = sorted(
        f.energy_score
        for t in pool
        if t.id is not None and (f := features.get(t.id)) is not None
    )
    if len(energies) < 3:
        return features

    lo = _percentile(energies, low_q)
    hi = _percentile(energies, high_q)
    if hi - lo < 1e-6:
        return features

    span = hi - lo
    tgt_lo, tgt_hi = profile.min_energy, profile.max_energy
    if tgt_hi < tgt_lo:  # defensive: never invert the target band
        tgt_lo, tgt_hi = tgt_hi, tgt_lo

    harmonics = sorted(
        f.harmonic_ratio
        for t in pool
        if t.id is not None and (f := features.get(t.id)) is not None
    )
    h_lo = _percentile(harmonics, low_q)
    h_hi = _percentile(harmonics, high_q)
    h_span = h_hi - h_lo

    out: dict[int, TrackFeatures] = dict(features)
    for t in pool:
        if t.id is None:
            continue
        f = features.get(t.id)
        if f is None:
            continue
        rel = _clamp01((f.energy_score - lo) / span)
        new_energy = tgt_lo + rel * (tgt_hi - tgt_lo)
        new_f = derive_energy_dependent(f, new_energy)
        if h_span > 1e-6:
            rel_h = _clamp01((f.harmonic_ratio - h_lo) / h_span)
            new_f = replace(new_f, harmonic_ratio=0.2 + rel_h * 0.6)
        out[t.id] = new_f

    _log.info(
        "Adaptive energy: library p5..p95 [%.2f..%.2f] -> profile band "
        "[%.2f..%.2f] across %d candidate track(s).",
        lo, hi, tgt_lo, tgt_hi, len(pool),
    )
    return out
