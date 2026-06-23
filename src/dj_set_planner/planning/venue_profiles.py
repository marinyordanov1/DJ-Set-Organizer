"""Venue character profiles + time-of-day modifiers (plain data).

A **venue** defines the *character* of the music — melodic/lounge for a
restaurant, percussive for a bar, strong/danceable for a club, hypnotic for an
afterparty. Each venue is a :class:`CharacterProfile`: a small list of
:class:`FeaturePref` (a preferred range for a track feature, with a weight) plus
a ``harshness_ceiling`` (the energy past which a track is "too hot" for this
venue, replacing the old global constant).

``character_fit`` scores how well a track matches the venue (0..1). Energy itself
is intentionally NOT a character preference here — it's already shaped by the
energy curve, adaptive normalization, and the **time of day** modifier, which
shifts the whole energy band lighter (day) → stronger (night).

Kept as data so the DJ's tuning lives in one readable place, separate from the
scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.enums import TimeOfDay, VenueType


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _band_fit(value: float, low: float, high: float, tol: float = 0.15) -> float:
    """Triangular membership: 1.0 inside ``[low, high]``, decaying to 0 over ``tol``."""

    if low <= value <= high:
        return 1.0
    dist = (low - value) if value < low else (value - high)
    if tol <= 0.0:
        return 0.0
    return _clamp01(1.0 - dist / tol)


@dataclass(frozen=True)
class FeaturePref:
    """A preferred range for one :class:`TrackFeatures` attribute.

    If ``lo_end``/``hi_end`` are set, the band interpolates from
    ``[lo, hi]`` at the start of the set to ``[lo_end, hi_end]`` at the end —
    used only by OUTDOOR_DAY_PARTY (melodic early → percussive later).
    """

    feature: str
    lo: float
    hi: float
    weight: float
    lo_end: float | None = None
    hi_end: float | None = None

    def band_at(self, position_fraction: float) -> tuple[float, float]:
        lo = self.lo if self.lo_end is None else self.lo + (self.lo_end - self.lo) * position_fraction
        hi = self.hi if self.hi_end is None else self.hi + (self.hi_end - self.hi) * position_fraction
        return lo, hi


@dataclass(frozen=True)
class CharacterProfile:
    """The character target for a venue."""

    venue: str
    prefs: tuple[FeaturePref, ...]
    harshness_ceiling: float
    label: str  # one-line hint for the UI, e.g. "melodic · lounge · gentle"


# --------------------------------------------------------------------------- #
# Venue registry. Energy is shaped elsewhere; these target the *character*
# features: harmonic_ratio (melodic↔percussive), vocals, danceability, groove,
# brightness, peak_potential.
# --------------------------------------------------------------------------- #
_PROFILES: dict[str, CharacterProfile] = {
    VenueType.RESTAURANT.value: CharacterProfile(
        venue=VenueType.RESTAURANT.value,
        label="melodic · lounge · gentle",
        harshness_ceiling=0.75,
        prefs=(
            FeaturePref("harmonic_ratio", 0.55, 1.0, 0.40),
            FeaturePref("danceability_score", 0.0, 0.55, 0.20),
            FeaturePref("mood_brightness", 0.45, 0.78, 0.15),
            FeaturePref("groove_score", 0.0, 0.55, 0.15),
            FeaturePref("peak_potential", 0.0, 0.50, 0.10),
        ),
    ),
    VenueType.BAR.value: CharacterProfile(
        venue=VenueType.BAR.value,
        label="percussion house · groovy · relaxed",
        harshness_ceiling=0.80,
        prefs=(
            FeaturePref("harmonic_ratio", 0.30, 0.55, 0.35),
            FeaturePref("groove_score", 0.55, 1.0, 0.30),
            FeaturePref("danceability_score", 0.40, 0.75, 0.20),
            FeaturePref("vocal_density", 0.0, 0.55, 0.15),
        ),
    ),
    VenueType.BEACH.value: CharacterProfile(
        venue=VenueType.BEACH.value,
        label="sunny · organic · percussive",
        harshness_ceiling=0.80,
        prefs=(
            FeaturePref("harmonic_ratio", 0.35, 0.60, 0.30),
            FeaturePref("mood_brightness", 0.55, 1.0, 0.25),
            FeaturePref("groove_score", 0.55, 1.0, 0.25),
            FeaturePref("danceability_score", 0.40, 0.75, 0.10),
            FeaturePref("vocal_density", 0.0, 0.60, 0.10),
        ),
    ),
    VenueType.CLUB.value: CharacterProfile(
        venue=VenueType.CLUB.value,
        label="strong · danceable · peak-driven",
        harshness_ceiling=1.0,  # anything goes
        prefs=(
            FeaturePref("danceability_score", 0.60, 1.0, 0.35),
            FeaturePref("peak_potential", 0.60, 1.0, 0.35),
            FeaturePref("groove_score", 0.55, 1.0, 0.30),
        ),
    ),
    VenueType.AFTERPARTY.value: CharacterProfile(
        venue=VenueType.AFTERPARTY.value,
        label="hypnotic · low-vocal · keep-moving",
        harshness_ceiling=0.85,
        prefs=(
            FeaturePref("vocal_density", 0.0, 0.40, 0.30),
            FeaturePref("harmonic_ratio", 0.50, 0.85, 0.25),
            FeaturePref("groove_score", 0.55, 1.0, 0.25),
            FeaturePref("peak_potential", 0.0, 0.60, 0.20),
        ),
    ),
    # The only position-varying venue: melodic early (restaurant-like) ->
    # percussive later (bar-like). harmonic_ratio band slides high → low.
    VenueType.OUTDOOR_DAY_PARTY.value: CharacterProfile(
        venue=VenueType.OUTDOOR_DAY_PARTY.value,
        label="sunny blend · melodic→percussive",
        harshness_ceiling=0.82,
        prefs=(
            FeaturePref("harmonic_ratio", 0.55, 0.80, 0.35, lo_end=0.30, hi_end=0.55),
            FeaturePref("groove_score", 0.50, 1.0, 0.25),
            FeaturePref("mood_brightness", 0.50, 1.0, 0.20),
            FeaturePref("danceability_score", 0.40, 0.80, 0.20),
        ),
    ),
}

# A neutral profile for any unknown / unset venue: accepts everything equally so
# the planner degrades to pure energy/arc behaviour rather than breaking.
_NEUTRAL = CharacterProfile(
    venue="NEUTRAL",
    label="no venue character",
    harshness_ceiling=0.85,
    prefs=(),
)


def character_profile(venue: str | None) -> CharacterProfile:
    """Return the :class:`CharacterProfile` for ``venue`` (neutral if unknown)."""

    if not venue:
        return _NEUTRAL
    return _PROFILES.get(venue.strip().upper(), _NEUTRAL)


def character_fit(features, profile: CharacterProfile, position_fraction: float = 0.5) -> float:
    """How well ``features`` match the venue's character (0..1).

    Weighted mean of each preference's band membership. A profile with no
    preferences (neutral venue) returns 0.5 so it neither helps nor hurts.
    """

    if not profile.prefs:
        return 0.5
    total_w = 0.0
    acc = 0.0
    for p in profile.prefs:
        value = _clamp01(float(getattr(features, p.feature, 0.5)))
        lo, hi = p.band_at(position_fraction)
        acc += p.weight * _band_fit(value, lo, hi)
        total_w += p.weight
    return _clamp01(acc / total_w) if total_w > 0 else 0.5


# --------------------------------------------------------------------------- #
# Time-of-day modifier: shifts the whole energy band lighter (day) → stronger
# (night). Bounded so it never inverts the band.
# --------------------------------------------------------------------------- #
_TIME_ENERGY_DELTA: dict[str, float] = {
    TimeOfDay.DAY.value: -0.08,        # daytime: keep it light
    TimeOfDay.SUNSET.value: -0.02,     # warming up toward the evening
    TimeOfDay.NIGHT.value: 0.06,       # club intensity
    TimeOfDay.LATE_NIGHT.value: 0.10,  # strongest / toward afterparty
}


def time_energy_delta(time_of_day: str | None) -> float:
    """Energy-band shift for a time of day (0.0 if unknown/unset)."""

    if not time_of_day:
        return 0.0
    return _TIME_ENERGY_DELTA.get(time_of_day.strip().upper(), 0.0)


def apply_time_modifier(min_energy: float, max_energy: float, time_of_day: str | None) -> tuple[float, float]:
    """Shift ``[min_energy, max_energy]`` by the time-of-day delta, clamped.

    Always returns a valid band (``0 <= min <= max <= 1``).
    """

    d = time_energy_delta(time_of_day)
    lo = _clamp01(min_energy + d)
    hi = _clamp01(max_energy + d)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi
