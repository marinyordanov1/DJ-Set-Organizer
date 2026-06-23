"""Repositories: CRUD over the domain dataclasses.

Each repository wraps a :class:`Database` and translates between SQLite rows
and the domain dataclasses defined in ``domain.models``. All SQL is real and
parameterized.
"""

from __future__ import annotations

import sqlite3

from ..domain.models import (
    DjConstraint,
    EventProfile,
    SetPlan,
    SetPlanTrack,
    Track,
    TrackFeatures,
)
from ..utils.logging import get_logger
from .database import Database

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Row -> dataclass mappers
# --------------------------------------------------------------------------- #
def _row_to_track(row: sqlite3.Row) -> Track:
    return Track(
        id=row["id"],
        file_path=row["file_path"],
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        genre=row["genre"],
        duration_seconds=row["duration_seconds"],
        bpm=row["bpm"],
        musical_key=row["musical_key"],
        camelot_key=row["camelot_key"],
        analyzed_at=row["analyzed_at"],
    )


def _row_to_features(row: sqlite3.Row) -> TrackFeatures:
    return TrackFeatures(
        track_id=row["track_id"],
        energy_score=row["energy_score"],
        danceability_score=row["danceability_score"],
        mood_brightness=row["mood_brightness"],
        groove_score=row["groove_score"],
        vocal_density=row["vocal_density"],
        intro_suitability=row["intro_suitability"],
        outro_suitability=row["outro_suitability"],
        peak_potential=row["peak_potential"],
        restaurant_safety_score=row["restaurant_safety_score"],
        mixability_score=row["mixability_score"],
        harmonic_ratio=row["harmonic_ratio"],
    )


def _row_to_profile(row: sqlite3.Row) -> EventProfile:
    return EventProfile(
        id=row["id"],
        name=row["name"],
        venue_type=row["venue_type"],
        time_of_day=row["time_of_day"],
        crowd_state=row["crowd_state"],
        desired_energy=row["desired_energy"],
        peak_strategy=row["peak_strategy"],
        target_duration_minutes=row["target_duration_minutes"],
        min_energy=row["min_energy"],
        max_energy=row["max_energy"],
        main_peak_energy=row["main_peak_energy"],
    )


def _row_to_constraint(row: sqlite3.Row) -> DjConstraint:
    return DjConstraint(
        id=row["id"],
        track_id=row["track_id"],
        constraint_type=row["constraint_type"],
        value=row["value"],
    )


# --------------------------------------------------------------------------- #
# TrackRepository
# --------------------------------------------------------------------------- #
class TrackRepository:
    """CRUD for :class:`Track`. Identity is the unique ``file_path``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, track: Track) -> Track:
        """Insert or update a track by ``file_path``; return it with its id."""

        conn = self._db.conn
        cur = conn.execute(
            """
            INSERT INTO tracks
                (file_path, title, artist, album, genre, duration_seconds,
                 bpm, musical_key, camelot_key, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                title            = excluded.title,
                artist           = excluded.artist,
                album            = excluded.album,
                genre            = excluded.genre,
                duration_seconds = excluded.duration_seconds,
                bpm              = excluded.bpm,
                musical_key      = excluded.musical_key,
                camelot_key      = excluded.camelot_key,
                analyzed_at      = excluded.analyzed_at
            """,
            (
                track.file_path,
                track.title,
                track.artist,
                track.album,
                track.genre,
                track.duration_seconds,
                track.bpm,
                track.musical_key,
                track.camelot_key,
                track.analyzed_at,
            ),
        )
        conn.commit()
        # Fetch the canonical row (id is authoritative after upsert).
        row = conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (track.file_path,)
        ).fetchone()
        result = _row_to_track(row)
        track.id = result.id  # keep caller's object in sync
        _ = cur  # cursor not needed beyond the write
        return result

    def get_all(self) -> list[Track]:
        rows = self._db.conn.execute(
            "SELECT * FROM tracks ORDER BY id"
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def get(self, track_id: int) -> Track | None:
        row = self._db.conn.execute(
            "SELECT * FROM tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return _row_to_track(row) if row else None

    def get_by_path(self, path: str) -> Track | None:
        row = self._db.conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (path,)
        ).fetchone()
        return _row_to_track(row) if row else None

    def delete(self, track_id: int) -> None:
        conn = self._db.conn
        conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
        conn.commit()


# --------------------------------------------------------------------------- #
# FeatureRepository
# --------------------------------------------------------------------------- #
class FeatureRepository:
    """CRUD for :class:`TrackFeatures` (keyed by ``track_id``)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, features: TrackFeatures) -> TrackFeatures:
        conn = self._db.conn
        conn.execute(
            """
            INSERT INTO track_features
                (track_id, energy_score, danceability_score, mood_brightness,
                 groove_score, vocal_density, intro_suitability,
                 outro_suitability, peak_potential, restaurant_safety_score,
                 mixability_score, harmonic_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                energy_score            = excluded.energy_score,
                danceability_score      = excluded.danceability_score,
                mood_brightness         = excluded.mood_brightness,
                groove_score            = excluded.groove_score,
                vocal_density           = excluded.vocal_density,
                intro_suitability       = excluded.intro_suitability,
                outro_suitability       = excluded.outro_suitability,
                peak_potential          = excluded.peak_potential,
                restaurant_safety_score = excluded.restaurant_safety_score,
                mixability_score        = excluded.mixability_score,
                harmonic_ratio          = excluded.harmonic_ratio
            """,
            (
                features.track_id,
                features.energy_score,
                features.danceability_score,
                features.mood_brightness,
                features.groove_score,
                features.vocal_density,
                features.intro_suitability,
                features.outro_suitability,
                features.peak_potential,
                features.restaurant_safety_score,
                features.mixability_score,
                features.harmonic_ratio,
            ),
        )
        conn.commit()
        return features

    def get(self, track_id: int) -> TrackFeatures | None:
        row = self._db.conn.execute(
            "SELECT * FROM track_features WHERE track_id = ?", (track_id,)
        ).fetchone()
        return _row_to_features(row) if row else None

    def get_all(self) -> dict[int, TrackFeatures]:
        rows = self._db.conn.execute("SELECT * FROM track_features").fetchall()
        return {r["track_id"]: _row_to_features(r) for r in rows}


# --------------------------------------------------------------------------- #
# FeatureOverrideRepository
# --------------------------------------------------------------------------- #
class FeatureOverrideRepository:
    """Manual per-track energy overrides (keyed by ``track_id``).

    Stored apart from ``track_features`` so re-analysis never wipes a value the
    DJ set by hand. The service overlays these on top of the analyzed features.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def set_energy(self, track_id: int, energy_score: float) -> None:
        """Insert or replace the energy override for a track."""

        conn = self._db.conn
        conn.execute(
            """
            INSERT INTO feature_overrides (track_id, energy_score)
            VALUES (?, ?)
            ON CONFLICT(track_id) DO UPDATE SET energy_score = excluded.energy_score
            """,
            (track_id, float(energy_score)),
        )
        conn.commit()

    def clear(self, track_id: int) -> None:
        """Remove any override for a track (no-op if none exists)."""

        conn = self._db.conn
        conn.execute("DELETE FROM feature_overrides WHERE track_id = ?", (track_id,))
        conn.commit()

    def get_all(self) -> dict[int, float]:
        """Return ``{track_id: energy_score}`` for every override set."""

        rows = self._db.conn.execute(
            "SELECT track_id, energy_score FROM feature_overrides "
            "WHERE energy_score IS NOT NULL"
        ).fetchall()
        return {r["track_id"]: r["energy_score"] for r in rows}


# --------------------------------------------------------------------------- #
# EventProfileRepository
# --------------------------------------------------------------------------- #
class EventProfileRepository:
    """Persistence for :class:`EventProfile`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, profile: EventProfile) -> EventProfile:
        """Insert (id is None) or update (id set) and return with id."""

        conn = self._db.conn
        if profile.id is None:
            cur = conn.execute(
                """
                INSERT INTO event_profiles
                    (name, venue_type, time_of_day, crowd_state,
                     desired_energy, peak_strategy, target_duration_minutes,
                     min_energy, max_energy, main_peak_energy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.name,
                    profile.venue_type,
                    profile.time_of_day,
                    profile.crowd_state,
                    profile.desired_energy,
                    profile.peak_strategy,
                    profile.target_duration_minutes,
                    profile.min_energy,
                    profile.max_energy,
                    profile.main_peak_energy,
                ),
            )
            conn.commit()
            profile.id = int(cur.lastrowid)
        else:
            conn.execute(
                """
                UPDATE event_profiles SET
                    name = ?, venue_type = ?, time_of_day = ?, crowd_state = ?,
                    desired_energy = ?, peak_strategy = ?,
                    target_duration_minutes = ?, min_energy = ?, max_energy = ?,
                    main_peak_energy = ?
                WHERE id = ?
                """,
                (
                    profile.name,
                    profile.venue_type,
                    profile.time_of_day,
                    profile.crowd_state,
                    profile.desired_energy,
                    profile.peak_strategy,
                    profile.target_duration_minutes,
                    profile.min_energy,
                    profile.max_energy,
                    profile.main_peak_energy,
                    profile.id,
                ),
            )
            conn.commit()
        return profile

    def get(self, profile_id: int) -> EventProfile | None:
        row = self._db.conn.execute(
            "SELECT * FROM event_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return _row_to_profile(row) if row else None

    def get_all(self) -> list[EventProfile]:
        rows = self._db.conn.execute(
            "SELECT * FROM event_profiles ORDER BY id"
        ).fetchall()
        return [_row_to_profile(r) for r in rows]


# --------------------------------------------------------------------------- #
# ConstraintRepository
# --------------------------------------------------------------------------- #
class ConstraintRepository:
    """Persistence for :class:`DjConstraint`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def set_for_track(
        self, track_id: int, constraint_type: str, value: str | None = None
    ) -> DjConstraint:
        """Add (or replace value of) a constraint of a given type on a track.

        Re-setting the same (track_id, constraint_type) replaces the existing
        value rather than creating duplicates.
        """

        conn = self._db.conn
        existing = conn.execute(
            "SELECT id FROM dj_constraints WHERE track_id = ? AND constraint_type = ?",
            (track_id, constraint_type),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE dj_constraints SET value = ? WHERE id = ?",
                (value, existing["id"]),
            )
            conn.commit()
            cid = existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO dj_constraints (track_id, constraint_type, value) "
                "VALUES (?, ?, ?)",
                (track_id, constraint_type, value),
            )
            conn.commit()
            cid = int(cur.lastrowid)
        return DjConstraint(
            id=cid, track_id=track_id, constraint_type=constraint_type, value=value
        )

    def clear(self, track_id: int, constraint_type: str | None = None) -> None:
        """Remove all constraints for a track, or only one type if given."""

        conn = self._db.conn
        if constraint_type is None:
            conn.execute("DELETE FROM dj_constraints WHERE track_id = ?", (track_id,))
        else:
            conn.execute(
                "DELETE FROM dj_constraints WHERE track_id = ? AND constraint_type = ?",
                (track_id, constraint_type),
            )
        conn.commit()

    def get_all(self) -> list[DjConstraint]:
        rows = self._db.conn.execute(
            "SELECT * FROM dj_constraints ORDER BY id"
        ).fetchall()
        return [_row_to_constraint(r) for r in rows]

    def get_for_track(self, track_id: int) -> list[DjConstraint]:
        rows = self._db.conn.execute(
            "SELECT * FROM dj_constraints WHERE track_id = ? ORDER BY id",
            (track_id,),
        ).fetchall()
        return [_row_to_constraint(r) for r in rows]


# --------------------------------------------------------------------------- #
# SetPlanRepository
# --------------------------------------------------------------------------- #
class SetPlanRepository:
    """Persistence for :class:`SetPlan` (header + ordered tracks)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, plan: SetPlan) -> int:
        """Persist a plan (and its ordered tracks); return the new plan id.

        The plan's :class:`EventProfile` is saved first if it has no id.
        """

        conn = self._db.conn

        profile_id = plan.event_profile.id
        if profile_id is None:
            profile_id = EventProfileRepository(self._db).save(plan.event_profile).id

        cur = conn.execute(
            """
            INSERT INTO set_plans
                (event_profile_id, created_at, total_duration_seconds,
                 target_duration_seconds, total_score)
            VALUES (?, datetime('now'), ?, ?, ?)
            """,
            (
                profile_id,
                plan.total_duration_seconds,
                plan.target_duration_seconds,
                plan.total_score,
            ),
        )
        plan_id = int(cur.lastrowid)

        for spt in plan.tracks:
            conn.execute(
                """
                INSERT INTO set_plan_tracks
                    (set_plan_id, track_id, position, role, transition_score,
                     position_score, explanation, is_locked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    spt.track_id,
                    spt.position,
                    spt.role,
                    spt.transition_score,
                    spt.position_score,
                    spt.explanation,
                    1 if spt.is_locked else 0,
                ),
            )
        conn.commit()
        return plan_id

    def get(self, plan_id: int) -> SetPlan | None:
        """Reconstruct a :class:`SetPlan` from storage.

        Note: ``segments`` and ``energy_points`` are not persisted as their own
        rows; they are returned empty and recomputed by the planning layer when
        needed. The ordered tracks, scores, and profile are fully restored.
        """

        conn = self._db.conn
        header = conn.execute(
            "SELECT * FROM set_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if not header:
            return None

        profile = None
        if header["event_profile_id"] is not None:
            profile = EventProfileRepository(self._db).get(header["event_profile_id"])

        rows = conn.execute(
            "SELECT * FROM set_plan_tracks WHERE set_plan_id = ? ORDER BY position",
            (plan_id,),
        ).fetchall()
        tracks = [
            SetPlanTrack(
                track_id=r["track_id"],
                position=r["position"],
                role=r["role"],
                transition_score=r["transition_score"],
                position_score=r["position_score"],
                explanation=r["explanation"],
                is_locked=bool(r["is_locked"]),
            )
            for r in rows
        ]

        return SetPlan(
            event_profile=profile,
            tracks=tracks,
            total_duration_seconds=header["total_duration_seconds"] or 0,
            target_duration_seconds=header["target_duration_seconds"] or 0,
            total_score=header["total_score"] or 0.0,
            segments=[],
            energy_points=[],
        )
