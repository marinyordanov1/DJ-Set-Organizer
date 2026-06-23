"""Domain models (dataclasses).

These dataclasses are the single source of truth for the shapes that flow
between analysis, planning, persistence, and the web layer. Field names and
types here MUST match CONTRACTS.md exactly.

All ``*_score`` / feature fields are floats in the inclusive range 0.0..1.0
with a sensible default of 0.5 ("unknown / neutral").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Track:
    """A single audio track and its tag-derived metadata.

    ``id`` is ``None`` until the track is persisted. ``camelot_key`` is the
    Camelot-wheel notation derived from ``musical_key`` (see utils.camelot).
    """

    id: int | None
    file_path: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    duration_seconds: int | None = None
    bpm: float | None = None
    musical_key: str | None = None
    camelot_key: str | None = None
    analyzed_at: str | None = None


@dataclass
class TrackFeatures:
    """Per-track musical features, all normalized to 0.0..1.0.

    Produced either by real audio analysis (librosa) or by the deterministic
    heuristic scorer. ``track_id`` links back to the owning Track.
    """

    track_id: int
    energy_score: float = 0.5
    danceability_score: float = 0.5
    mood_brightness: float = 0.5
    groove_score: float = 0.5
    vocal_density: float = 0.5
    intro_suitability: float = 0.5
    outro_suitability: float = 0.5
    peak_potential: float = 0.5
    restaurant_safety_score: float = 0.5
    mixability_score: float = 0.5
    # 1.0 = fully melodic/harmonic, 0.0 = fully percussive/drum-driven.
    harmonic_ratio: float = 0.5


@dataclass
class EventProfile:
    """The context of the event the set is being built for.

    The string fields hold the ``.value`` of the corresponding enum
    (VenueType, TimeOfDay, ... ). ``crowd_state`` may be a compound string
    such as ``"EATING+TALKING+PARTIAL_DANCING"``.
    """

    id: int | None
    name: str
    venue_type: str
    time_of_day: str
    crowd_state: str
    desired_energy: str
    peak_strategy: str
    target_duration_minutes: int
    min_energy: float
    max_energy: float
    main_peak_energy: float


@dataclass
class DjConstraint:
    """A DJ-imposed constraint targeting a specific track.

    ``constraint_type`` holds a ConstraintType value. ``value`` is an optional
    payload (e.g. a position index for LOCK_POSITION, or a track id for
    DO_NOT_PLAY_BEFORE/AFTER).
    """

    id: int | None
    track_id: int
    constraint_type: str
    value: str | None = None


@dataclass
class SetSegment:
    """A contiguous span of the set with a target energy band and role.

    ``start_pct`` / ``end_pct`` are fractions in 0.0..1.0 of the set's
    progression (by track position).
    """

    name: str
    start_pct: float
    end_pct: float
    min_energy: float
    max_energy: float
    role: str


@dataclass
class SetPlanTrack:
    """One ordered slot in a generated set plan."""

    track_id: int
    position: int
    role: str
    transition_score: float
    position_score: float
    explanation: str
    is_locked: bool = False


@dataclass
class SetPlan:
    """A complete generated set plan.

    ``energy_points`` is the actual ``energy_score`` of each ordered track
    (parallel to ``tracks``), used to render the energy curve.
    """

    event_profile: EventProfile
    tracks: list[SetPlanTrack]
    total_duration_seconds: int
    target_duration_seconds: int
    total_score: float
    segments: list[SetSegment] = field(default_factory=list)
    energy_points: list[float] = field(default_factory=list)
