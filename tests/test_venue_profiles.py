"""Tests for venue character profiles + time-of-day modifiers."""

from __future__ import annotations

from dj_set_planner.domain.enums import TimeOfDay, VenueType
from dj_set_planner.domain.models import TrackFeatures
from dj_set_planner.planning.venue_profiles import (
    apply_time_modifier,
    character_fit,
    character_profile,
)

_ACTIVE_VENUES = [
    VenueType.RESTAURANT,
    VenueType.BAR,
    VenueType.BEACH,
    VenueType.CLUB,
    VenueType.AFTERPARTY,
    VenueType.OUTDOOR_DAY_PARTY,
]


def test_every_active_venue_has_a_profile() -> None:
    for v in _ACTIVE_VENUES:
        p = character_profile(v.value)
        assert p.venue == v.value
        assert p.prefs, f"{v.value} should have character preferences"
        assert p.label


def test_unknown_venue_is_neutral() -> None:
    p = character_profile("PRIVATE_PARTY")  # removed venue -> neutral
    assert not p.prefs
    # A neutral profile neither helps nor hurts.
    assert character_fit(TrackFeatures(track_id=1), p) == 0.5


def test_harshness_ceilings_ordered() -> None:
    assert (
        character_profile(VenueType.RESTAURANT.value).harshness_ceiling
        < character_profile(VenueType.CLUB.value).harshness_ceiling
    )


def test_time_modifier_direction_and_bounds() -> None:
    lo0, hi0 = 0.30, 0.78
    d_lo, d_hi = apply_time_modifier(lo0, hi0, TimeOfDay.DAY.value)
    n_lo, n_hi = apply_time_modifier(lo0, hi0, TimeOfDay.NIGHT.value)
    assert d_hi < hi0   # day lowers the ceiling
    assert n_hi > hi0   # night raises it
    # Always a valid band, even at the extremes.
    for t in TimeOfDay:
        lo, hi = apply_time_modifier(0.0, 1.0, t.value)
        assert 0.0 <= lo <= hi <= 1.0


def test_character_fit_discriminates_restaurant_vs_club() -> None:
    melodic = TrackFeatures(
        track_id=1, harmonic_ratio=0.85, energy_score=0.40,
        danceability_score=0.40, groove_score=0.40, peak_potential=0.30,
        mood_brightness=0.60, vocal_density=0.50,
    )
    banger = TrackFeatures(
        track_id=2, harmonic_ratio=0.40, energy_score=0.85,
        danceability_score=0.90, groove_score=0.80, peak_potential=0.85,
        mood_brightness=0.50, vocal_density=0.40,
    )
    rest = character_profile(VenueType.RESTAURANT.value)
    club = character_profile(VenueType.CLUB.value)

    assert character_fit(melodic, rest) > character_fit(banger, rest)
    assert character_fit(banger, club) > character_fit(melodic, club)


def test_outdoor_character_slides_melodic_to_percussive() -> None:
    melodic = TrackFeatures(
        track_id=1, harmonic_ratio=0.78, groove_score=0.70,
        mood_brightness=0.70, danceability_score=0.60,
    )
    outdoor = character_profile(VenueType.OUTDOOR_DAY_PARTY.value)
    early = character_fit(melodic, outdoor, position_fraction=0.0)
    late = character_fit(melodic, outdoor, position_fraction=1.0)
    assert early > late  # a melodic track suits the early (melodic) part better
