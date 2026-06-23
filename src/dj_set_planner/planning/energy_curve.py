"""Energy-curve segmentation for a set.

Turns an :class:`EventProfile` (specifically its ``peak_strategy``) into an
ordered list of :class:`SetSegment` describing how the set's *target* energy
should evolve from start (0%) to end (100%). The planner then places tracks so
that each track's measured ``energy_score`` lands inside the band of the
segment its position falls into.

A "curve" here is a piecewise band: each segment owns a contiguous
``[start_pct, end_pct)`` slice of the set's progression (by track position, not
wall-clock) and a ``[min_energy, max_energy]`` band the placed track should sit
in. Segments are contiguous and tile ``0.0..1.0`` with no gaps or overlaps; the
first segment starts at 0.0 and the last ends at 1.0.

The flagship template is the Sofia ``ONE_MAIN_PEAK`` story:
intro -> warm groove -> progressive build -> small peak -> breathing space ->
one main peak -> release -> outro. Alternative templates are provided for the
other :class:`PeakStrategy` values.

All four templates share the same invariants so downstream code (and the
tests) can rely on them regardless of strategy:
  * non-empty, contiguous coverage of 0.0..1.0 (first start == 0.0,
    last end == 1.0, each segment's start == previous segment's end);
  * every band has ``min_energy <= max_energy``;
  * every band respects the profile's overall ``[min_energy, max_energy]``
    envelope (templates are clamped into it, see ``_clamp_band``).
"""

from __future__ import annotations

from ..domain.enums import PeakStrategy, TrackRole
from ..domain.models import EventProfile, SetSegment
from ..utils.logging import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Template definitions.
#
# Each template is a list of raw tuples:
#   (name, start_pct, end_pct, min_energy, max_energy, role)
# expressed as ABSOLUTE energy values in 0..1 (NOT yet clamped to the profile
# envelope). ``generate_energy_curve`` clamps each band into the profile's
# [min_energy, max_energy] window before returning concrete SetSegments, so a
# template can be authored against canonical day-party numbers and still be
# reused (in shape) by profiles with a different overall energy envelope.
#
# A raw template row is a 6-tuple of (str, float, float, float, float, str).
# --------------------------------------------------------------------------- #

_RawSegment = tuple[str, float, float, float, float, str]


# Sofia ONE_MAIN_PEAK — the EXACT 8-segment story from the spec.
#
#   intro            0-10%    0.30-0.40   ease people in, low and open
#   warm_groove      10-30%   0.40-0.52   establish a comfortable groove
#   progressive_build 30-50%  0.52-0.65   slowly lift the room
#   small_peak       50-60%   0.65-0.72   a first taste of energy
#   breathing_space  60-70%   0.50-0.60   pull back so the main peak lands
#   main_peak        70-82%   0.72-0.78   the single climax of the set
#   release          82-92%   0.55-0.65   come down gracefully
#   outro            92-100%  0.35-0.48   wind down to a soft close
_ONE_MAIN_PEAK: list[_RawSegment] = [
    ("intro", 0.00, 0.10, 0.30, 0.40, TrackRole.INTRO.value),
    ("warm_groove", 0.10, 0.30, 0.40, 0.52, TrackRole.WARM_GROOVE.value),
    ("progressive_build", 0.30, 0.50, 0.52, 0.65, TrackRole.PROGRESSIVE_BUILD.value),
    ("small_peak", 0.50, 0.60, 0.65, 0.72, TrackRole.SMALL_PEAK.value),
    ("breathing_space", 0.60, 0.70, 0.50, 0.60, TrackRole.BREATHING_SPACE.value),
    ("main_peak", 0.70, 0.82, 0.72, 0.78, TrackRole.MAIN_PEAK.value),
    ("release", 0.82, 0.92, 0.55, 0.65, TrackRole.RELEASE.value),
    ("outro", 0.92, 1.00, 0.35, 0.48, TrackRole.OUTRO.value),
]


# MULTIPLE_SMALL_PEAKS — keep the room engaged with several rolling peaks and
# valleys rather than one climax. Energy oscillates around the comfortable
# cruising band, trending gently upward, with a final wind-down.
_MULTIPLE_SMALL_PEAKS: list[_RawSegment] = [
    ("intro", 0.00, 0.10, 0.30, 0.40, TrackRole.INTRO.value),
    ("warm_groove", 0.10, 0.25, 0.40, 0.52, TrackRole.WARM_GROOVE.value),
    ("small_peak", 0.25, 0.35, 0.60, 0.70, TrackRole.SMALL_PEAK.value),
    ("breathing_space", 0.35, 0.45, 0.48, 0.58, TrackRole.BREATHING_SPACE.value),
    ("small_peak", 0.45, 0.55, 0.62, 0.72, TrackRole.SMALL_PEAK.value),
    ("breathing_space", 0.55, 0.65, 0.50, 0.60, TrackRole.BREATHING_SPACE.value),
    ("small_peak", 0.65, 0.78, 0.64, 0.74, TrackRole.SMALL_PEAK.value),
    ("release", 0.78, 0.90, 0.52, 0.62, TrackRole.RELEASE.value),
    ("outro", 0.90, 1.00, 0.35, 0.48, TrackRole.OUTRO.value),
]


# PROGRESSIVE_BUILD — a steady, monotonic climb from a low open to a high close
# with no real release; the "peak" is essentially the end of the set.
_PROGRESSIVE_BUILD: list[_RawSegment] = [
    ("intro", 0.00, 0.12, 0.30, 0.40, TrackRole.INTRO.value),
    ("warm_groove", 0.12, 0.30, 0.40, 0.52, TrackRole.WARM_GROOVE.value),
    ("progressive_build", 0.30, 0.55, 0.52, 0.64, TrackRole.PROGRESSIVE_BUILD.value),
    ("progressive_build", 0.55, 0.75, 0.62, 0.72, TrackRole.PROGRESSIVE_BUILD.value),
    ("small_peak", 0.75, 0.88, 0.70, 0.78, TrackRole.SMALL_PEAK.value),
    ("main_peak", 0.88, 1.00, 0.74, 0.80, TrackRole.MAIN_PEAK.value),
]


# FLAT_LOUNGE — background dinner music: stay low and even the whole way, with
# only the gentlest lift in the middle. Never demands the dancefloor.
_FLAT_LOUNGE: list[_RawSegment] = [
    ("intro", 0.00, 0.15, 0.28, 0.38, TrackRole.INTRO.value),
    ("warm_groove", 0.15, 0.45, 0.36, 0.48, TrackRole.WARM_GROOVE.value),
    ("breathing_space", 0.45, 0.70, 0.40, 0.52, TrackRole.BREATHING_SPACE.value),
    ("warm_groove", 0.70, 0.88, 0.36, 0.48, TrackRole.WARM_GROOVE.value),
    ("outro", 0.88, 1.00, 0.28, 0.40, TrackRole.OUTRO.value),
]


# Registry: PeakStrategy value -> raw template.
_TEMPLATES: dict[str, list[_RawSegment]] = {
    PeakStrategy.ONE_MAIN_PEAK.value: _ONE_MAIN_PEAK,
    PeakStrategy.MULTIPLE_SMALL_PEAKS.value: _MULTIPLE_SMALL_PEAKS,
    PeakStrategy.PROGRESSIVE_BUILD.value: _PROGRESSIVE_BUILD,
    PeakStrategy.FLAT_LOUNGE.value: _FLAT_LOUNGE,
}


def _clamp_band(
    lo: float, hi: float, env_lo: float, env_hi: float
) -> tuple[float, float]:
    """Clamp a template band ``[lo, hi]`` into the profile envelope.

    The template authors absolute energy numbers (tuned for the Sofia
    day-party). A profile may, however, declare a narrower or shifted overall
    ``[min_energy, max_energy]`` envelope. We clamp each band into that
    envelope so no segment ever asks for energy the profile forbids, while
    preserving the *shape* of the curve as much as the envelope allows.

    Robustness: if the envelope itself is inverted (env_lo > env_hi) we swap it
    so the result is always a valid band with ``min <= max``.
    """

    if env_lo > env_hi:
        env_lo, env_hi = env_hi, env_lo

    clamped_lo = min(max(lo, env_lo), env_hi)
    clamped_hi = min(max(hi, env_lo), env_hi)

    # Guarantee min <= max even if the original template band was authored
    # inverted (it never is, but stay defensive).
    if clamped_lo > clamped_hi:
        clamped_lo, clamped_hi = clamped_hi, clamped_lo

    return clamped_lo, clamped_hi


def generate_energy_curve(profile: EventProfile) -> list[SetSegment]:
    """Build the target energy curve for ``profile``.

    Selects a template by ``profile.peak_strategy`` (falling back to the Sofia
    ``ONE_MAIN_PEAK`` template for any unrecognized strategy), then clamps every
    band into the profile's overall ``[min_energy, max_energy]`` envelope and
    returns concrete :class:`SetSegment` objects.

    The returned list is non-empty, ordered, contiguous over 0.0..1.0, and every
    segment has ``min_energy <= max_energy``.
    """

    strategy = profile.peak_strategy
    template = _TEMPLATES.get(strategy)
    if template is None:
        # Unknown / unset strategy: degrade gracefully to the flagship story
        # rather than raising, so the app stays usable. Log it so it's visible.
        _log.warning(
            "Unknown peak_strategy %r; falling back to ONE_MAIN_PEAK template.",
            strategy,
        )
        template = _ONE_MAIN_PEAK

    env_lo = profile.min_energy
    env_hi = profile.max_energy

    segments: list[SetSegment] = []
    for name, start_pct, end_pct, min_e, max_e, role in template:
        clamped_lo, clamped_hi = _clamp_band(min_e, max_e, env_lo, env_hi)
        segments.append(
            SetSegment(
                name=name,
                start_pct=start_pct,
                end_pct=end_pct,
                min_energy=clamped_lo,
                max_energy=clamped_hi,
                role=role,
            )
        )

    return segments


def segment_at(curve: list[SetSegment], position_fraction: float) -> SetSegment:
    """Return the segment whose span contains ``position_fraction`` (0..1).

    A position belongs to a segment when ``start_pct <= f < end_pct``. The
    final segment is treated as *inclusive* of its ``end_pct`` so that
    ``f == 1.0`` (the very last track) maps to the outro rather than falling off
    the end. Out-of-range fractions are clamped to ``[0, 1]``.

    Raises :class:`ValueError` only if ``curve`` is empty (a programming error —
    every profile yields a non-empty curve).
    """

    if not curve:
        raise ValueError("segment_at called with an empty curve")

    # Clamp into range so callers can pass slightly-off values safely.
    f = min(max(position_fraction, 0.0), 1.0)

    for seg in curve:
        if seg.start_pct <= f < seg.end_pct:
            return seg

    # f did not land strictly inside any half-open span. The only way this
    # happens for an in-range f over a contiguous 0..1 curve is f == 1.0 (or
    # floating-point landing exactly on the last boundary): map to the last
    # segment, which owns the closing edge.
    return curve[-1]


def target_energy_at(
    curve: list[SetSegment], fraction: float
) -> tuple[float, float]:
    """Return the ``(min_energy, max_energy)`` target band at ``fraction``.

    Thin convenience wrapper over :func:`segment_at` returning just the energy
    band of the containing segment.
    """

    seg = segment_at(curve, fraction)
    return seg.min_energy, seg.max_energy
