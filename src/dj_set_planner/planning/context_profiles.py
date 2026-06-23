"""Built-in event-context presets.

Provides a registry of ready-made :class:`EventProfile` presets plus the named
tuning constants the planner uses for the flagship "Sofia day-party restaurant"
context. This module is FULLY implemented in the Foundation phase so other
phases can rely on stable defaults.
"""

from __future__ import annotations

from ..domain.enums import (
    DesiredEnergy,
    PeakStrategy,
    TimeOfDay,
    VenueType,
)
from ..domain.models import EventProfile

# --------------------------------------------------------------------------- #
# Named tuning constants (Sofia day-party restaurant).
#
# These encode the "vibe rules" for a daytime restaurant party in Sofia: keep
# things groovy and danceable but never harsh; one clear main peak; smooth,
# harmonic transitions; small peaks allowed to keep interest.
# --------------------------------------------------------------------------- #

# Never let a track's energy exceed this in this context (avoid harshness).
avoid_energy_above: float = 0.85

# The comfortable cruising energy band for most of the set.
average_energy_range: tuple[float, float] = (0.52, 0.62)

# Prefer smooth, harmonically-compatible transitions over bold jumps.
prefer_smooth_transitions: bool = True

# Small interest-building peaks are welcome (not just the single main peak).
allow_small_peaks: bool = True


# Canonical name of the flagship preset.
SOFIA_PRESET_NAME: str = "Sofia Day Party Restaurant"


def _sofia_profile() -> EventProfile:
    """Construct the Sofia day-party restaurant profile.

    venue RESTAURANT, time DAY, crowd "EATING+TALKING+PARTIAL_DANCING",
    desired BALANCED, peak ONE_MAIN_PEAK, 120 min,
    min_energy 0.30, max_energy 0.78, main_peak_energy 0.75.
    """

    return EventProfile(
        id=None,
        name=SOFIA_PRESET_NAME,
        venue_type=VenueType.RESTAURANT.value,
        time_of_day=TimeOfDay.DAY.value,
        # Compound crowd state — restaurant patrons eating/talking with some
        # already starting to dance.
        crowd_state="EATING+TALKING+PARTIAL_DANCING",
        desired_energy=DesiredEnergy.BALANCED.value,
        peak_strategy=PeakStrategy.ONE_MAIN_PEAK.value,
        target_duration_minutes=120,
        min_energy=0.30,
        max_energy=0.78,
        main_peak_energy=0.75,
    )


def builtin_presets() -> dict[str, EventProfile]:
    """Return the registry of built-in presets, keyed by name.

    A fresh dict (with fresh :class:`EventProfile` instances) is returned on
    every call so callers can mutate freely without corrupting the registry.
    """

    presets: dict[str, EventProfile] = {}

    # 1) Flagship Sofia day-party restaurant.
    sofia = _sofia_profile()
    presets[sofia.name] = sofia

    # 2) Sunset beach — progressive, slightly higher ceiling, melodic.
    presets["Sunset Beach"] = EventProfile(
        id=None,
        name="Sunset Beach",
        venue_type=VenueType.BEACH.value,
        time_of_day=TimeOfDay.SUNSET.value,
        crowd_state="PARTIAL_DANCING+DANCING",
        desired_energy=DesiredEnergy.PROGRESSIVE.value,
        peak_strategy=PeakStrategy.PROGRESSIVE_BUILD.value,
        target_duration_minutes=120,
        min_energy=0.35,
        max_energy=0.82,
        main_peak_energy=0.80,
    )

    # 3) Club night — high energy, one big main peak.
    presets["Club Night Peak"] = EventProfile(
        id=None,
        name="Club Night Peak",
        venue_type=VenueType.CLUB.value,
        time_of_day=TimeOfDay.NIGHT.value,
        crowd_state="DANCING+PEAK_DANCING",
        desired_energy=DesiredEnergy.ENERGETIC.value,
        peak_strategy=PeakStrategy.ONE_MAIN_PEAK.value,
        target_duration_minutes=90,
        min_energy=0.45,
        max_energy=0.95,
        main_peak_energy=0.92,
    )

    # 4) Restaurant lounge — flat, low, dinner background.
    presets["Restaurant Lounge"] = EventProfile(
        id=None,
        name="Restaurant Lounge",
        venue_type=VenueType.RESTAURANT.value,
        time_of_day=TimeOfDay.NIGHT.value,
        crowd_state="EATING+TALKING",
        desired_energy=DesiredEnergy.RELAXED.value,
        peak_strategy=PeakStrategy.FLAT_LOUNGE.value,
        target_duration_minutes=180,
        min_energy=0.20,
        max_energy=0.55,
        main_peak_energy=0.50,
    )

    # 5) Bar warm-up — multiple small peaks, building the room.
    presets["Bar Warm-Up"] = EventProfile(
        id=None,
        name="Bar Warm-Up",
        venue_type=VenueType.BAR.value,
        time_of_day=TimeOfDay.NIGHT.value,
        crowd_state="TALKING+WARMING_UP",
        desired_energy=DesiredEnergy.BALANCED.value,
        peak_strategy=PeakStrategy.MULTIPLE_SMALL_PEAKS.value,
        target_duration_minutes=120,
        min_energy=0.30,
        max_energy=0.75,
        main_peak_energy=0.72,
    )

    return presets


def default_profile() -> EventProfile:
    """Return the default :class:`EventProfile` (the Sofia preset)."""

    return _sofia_profile()
