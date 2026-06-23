"""Domain enumerations.

All enums are ``(str, Enum)`` so their values serialize directly to strings
(useful for JSON APIs and SQLite storage). Compare/serialize via ``.value``.
"""

from __future__ import annotations

from enum import Enum


class VenueType(str, Enum):
    """Where the set is being played."""

    RESTAURANT = "RESTAURANT"
    BAR = "BAR"
    BEACH = "BEACH"
    CLUB = "CLUB"
    AFTERPARTY = "AFTERPARTY"
    PRIVATE_PARTY = "PRIVATE_PARTY"
    OUTDOOR_DAY_PARTY = "OUTDOOR_DAY_PARTY"


class TimeOfDay(str, Enum):
    """Rough time-of-day slot for the set."""

    DAY = "DAY"
    SUNSET = "SUNSET"
    NIGHT = "NIGHT"
    LATE_NIGHT = "LATE_NIGHT"


class CrowdState(str, Enum):
    """What the crowd is currently doing / expected to do."""

    EATING = "EATING"
    TALKING = "TALKING"
    WARMING_UP = "WARMING_UP"
    PARTIAL_DANCING = "PARTIAL_DANCING"
    DANCING = "DANCING"
    PEAK_DANCING = "PEAK_DANCING"


class DesiredEnergy(str, Enum):
    """Overall desired energy trajectory of the set."""

    RELAXED = "RELAXED"
    BALANCED = "BALANCED"
    PROGRESSIVE = "PROGRESSIVE"
    ENERGETIC = "ENERGETIC"


class PeakStrategy(str, Enum):
    """How peaks are shaped across the set."""

    ONE_MAIN_PEAK = "ONE_MAIN_PEAK"
    MULTIPLE_SMALL_PEAKS = "MULTIPLE_SMALL_PEAKS"
    PROGRESSIVE_BUILD = "PROGRESSIVE_BUILD"
    FLAT_LOUNGE = "FLAT_LOUNGE"


class TrackRole(str, Enum):
    """Narrative role a track plays inside the set."""

    INTRO = "INTRO"
    WARM_GROOVE = "WARM_GROOVE"
    PROGRESSIVE_BUILD = "PROGRESSIVE_BUILD"
    SMALL_PEAK = "SMALL_PEAK"
    BREATHING_SPACE = "BREATHING_SPACE"
    MAIN_PEAK = "MAIN_PEAK"
    RELEASE = "RELEASE"
    OUTRO = "OUTRO"


class ConstraintType(str, Enum):
    """DJ-imposed constraints on track selection / ordering."""

    MUST_PLAY = "MUST_PLAY"
    AVOID = "AVOID"
    PREFERRED_INTRO = "PREFERRED_INTRO"
    PREFERRED_OUTRO = "PREFERRED_OUTRO"
    PREFERRED_PEAK = "PREFERRED_PEAK"
    LOCK_POSITION = "LOCK_POSITION"
    DO_NOT_PLAY_BEFORE = "DO_NOT_PLAY_BEFORE"
    DO_NOT_PLAY_AFTER = "DO_NOT_PLAY_AFTER"
