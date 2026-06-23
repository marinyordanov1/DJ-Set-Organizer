"""Context-aware selection: venue character, strict, time-of-day (PRD success metrics).

Uses a synthetic library with four clear character clusters so the venue effect
is measurable: melodic-lounge, club-banger, hypnotic-afterparty, vocal-house.
Means are taken over the SELECTED tracks' ABSOLUTE features (not the normalized
energy_points), so they measure *selection*, not the energy rescale.
"""

from __future__ import annotations

from dataclasses import replace

from dj_set_planner.domain.enums import PeakStrategy, TimeOfDay, VenueType
from dj_set_planner.domain.models import EventProfile, Track, TrackFeatures
from dj_set_planner.planning.beam_search_planner import plan_set
from dj_set_planner.planning.venue_profiles import character_fit, character_profile

# (energy, danceability, brightness, groove, vocal, peak, harmonic) per cluster.
_CLUSTERS = {
    "melodic_lounge": (0.42, 0.38, 0.62, 0.40, 0.45, 0.30, 0.82),
    "club_banger":    (0.86, 0.92, 0.50, 0.82, 0.40, 0.86, 0.40),
    "hypnotic_after": (0.62, 0.62, 0.55, 0.72, 0.18, 0.45, 0.66),
    "vocal_house":    (0.66, 0.78, 0.62, 0.66, 0.85, 0.62, 0.58),
}


def _library():
    keys = ["8A", "9A", "10A", "11A"]
    tracks, feats = [], {}
    tid = 0
    for _name, (e, d, b, g, v, p, h) in _CLUSTERS.items():
        for k in range(4):  # 4 tracks per cluster -> 16 total
            tid += 1
            tracks.append(Track(
                id=tid, file_path=f"/lib/{tid:02d}.mp3", title=f"T{tid}",
                artist=f"A{tid}", genre="House", duration_seconds=300,
                bpm=123.0, camelot_key=keys[k], analyzed_at="2026-06-21T12:00:00",
            ))
            feats[tid] = TrackFeatures(
                track_id=tid, energy_score=e, danceability_score=d,
                mood_brightness=b, groove_score=g, vocal_density=v,
                peak_potential=p, harmonic_ratio=h,
                intro_suitability=1.0 - e, outro_suitability=1.0 - e,
                restaurant_safety_score=1.0 - e, mixability_score=0.6,
            )
    return tracks, feats


def _profile(venue: str, time: str = TimeOfDay.SUNSET.value, minutes: int = 30) -> EventProfile:
    return EventProfile(
        id=None, name=f"{venue} set", venue_type=venue, time_of_day=time,
        crowd_state="", desired_energy="", peak_strategy=PeakStrategy.ONE_MAIN_PEAK.value,
        target_duration_minutes=minutes, min_energy=0.30, max_energy=0.85, main_peak_energy=0.80,
    )


def _mean(plan, feats, attr: str) -> float:
    vals = [getattr(feats[t.track_id], attr) for t in plan.tracks]
    return sum(vals) / len(vals)


def test_restaurant_is_more_melodic_and_calmer_than_club() -> None:
    tracks, feats = _library()
    rest = plan_set(tracks, feats, _profile(VenueType.RESTAURANT.value), [])
    club = plan_set(tracks, feats, _profile(VenueType.CLUB.value), [])

    assert _mean(rest, feats, "harmonic_ratio") > _mean(club, feats, "harmonic_ratio")
    assert _mean(rest, feats, "energy_score") < _mean(club, feats, "energy_score")


def test_afterparty_is_lower_vocal_than_club() -> None:
    tracks, feats = _library()
    after = plan_set(tracks, feats, _profile(VenueType.AFTERPARTY.value), [])
    club = plan_set(tracks, feats, _profile(VenueType.CLUB.value), [])

    assert _mean(after, feats, "vocal_density") < _mean(club, feats, "vocal_density")


def test_night_raises_energy_vs_day_same_venue() -> None:
    tracks, feats = _library()
    day = plan_set(tracks, feats, _profile(VenueType.CLUB.value, TimeOfDay.DAY.value), [])
    night = plan_set(tracks, feats, _profile(VenueType.CLUB.value, TimeOfDay.NIGHT.value), [])
    # Time-of-day shifts the (normalized) energy band, so compare energy_points.
    mean_day = sum(day.energy_points) / len(day.energy_points)
    mean_night = sum(night.energy_points) / len(night.energy_points)
    assert mean_night > mean_day


def test_strict_keeps_only_on_character_tracks() -> None:
    tracks, feats = _library()
    venue = character_profile(VenueType.RESTAURANT.value)
    plan = plan_set(tracks, feats, _profile(VenueType.RESTAURANT.value), [], strict=True)
    # Every selected track is on-character (>= 0.4 fit), since strict excludes
    # the rest (and this library has enough restaurant-fit tracks, so no fallback).
    for t in plan.tracks:
        assert character_fit(feats[t.track_id], venue) >= 0.4


def test_selection_is_deterministic() -> None:
    tracks, feats = _library()
    p = _profile(VenueType.BAR.value)
    a = [t.track_id for t in plan_set(tracks, feats, p, []).tracks]
    b = [t.track_id for t in plan_set(tracks, feats, p, []).tracks]
    assert a == b
