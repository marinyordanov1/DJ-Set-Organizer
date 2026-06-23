"""Flask application + service layer for the AI DJ Set Planner.

This is the WEB UI phase. It wires the JSON API the frontend expects (see
CONTRACTS.md) to the real business-logic modules below it:

  * :mod:`analysis.scanner` / :mod:`analysis.metadata_reader` — discover and read
    tracks,
  * :func:`analysis.get_extractor` — analyse features (librosa if available, else
    deterministic heuristics),
  * :mod:`analysis.rekordbox_import` — pull BPM/key from a Rekordbox XML,
  * :mod:`db.repositories` — persist everything in SQLite,
  * :func:`planning.beam_search_planner.plan_set` — build the ordered set,
  * :mod:`export.m3u_exporter` / :mod:`export.csv_exporter` — write playlists.

Design rules (from the project spec):
  * UI is kept strictly separate from business logic — this module only
    translates HTTP <-> domain calls; it contains no scoring/planning maths.
  * Every endpoint returns JSON. Errors are returned as ``{"error": ...}`` with
    a proper HTTP status — a stack trace is NEVER leaked as HTML.
  * Exceptions are logged via ``utils.logging`` (never silently swallowed).
  * Exports are written into ``utils.paths.app_data_dir()``.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .analysis import get_extractor
from .analysis.heuristic_scorer import derive_energy_dependent
from .analysis.metadata_reader import read_metadata
from .analysis.rekordbox_import import apply_rekordbox_to_tracks
from .analysis.scanner import scan_folder
from .db.database import get_db
from .db.repositories import (
    ConstraintRepository,
    FeatureOverrideRepository,
    FeatureRepository,
    SetPlanRepository,
    TrackRepository,
)
from .domain.enums import ConstraintType
from .domain.models import DjConstraint, EventProfile, SetPlan, Track, TrackFeatures
from .export.csv_exporter import export_csv
from .export.m3u_exporter import export_m3u
from .planning.beam_search_planner import plan_set
from .planning.context_profiles import builtin_presets, default_profile
from .utils.logging import get_logger
from .utils.paths import app_data_dir

_log = get_logger(__name__)

# Frontend assets live next to this module under ui/static.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui", "static")

# The valid constraint-type strings (so the API rejects garbage early).
_VALID_CONSTRAINTS = {c.value for c in ConstraintType}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """A client/server error to be rendered as JSON ``{"error": ...}``.

    Carries an HTTP ``status`` so handlers can raise a single exception type and
    the error handler turns it into the right response. We NEVER let a raw
    exception escape as an HTML stack trace.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------- #
# Service layer — the only place that talks to repositories / planner.
#
# Kept deliberately thin: each method maps one API action to the underlying
# domain calls and returns plain dataclasses / dicts. This keeps HTTP concerns
# (in the route functions) separate from business logic.
# --------------------------------------------------------------------------- #
class PlannerService:
    """Wires the JSON API to scanning, analysis, planning, persistence, export."""

    def __init__(self) -> None:
        db = get_db()
        self.tracks = TrackRepository(db)
        self.features = FeatureRepository(db)
        self.overrides = FeatureOverrideRepository(db)
        self.constraints = ConstraintRepository(db)
        self.plans = SetPlanRepository(db)
        # Remember the most recently generated plan so the export endpoints can
        # write "the current set" without the client re-sending it. Persisted
        # plans are also retrievable by id.
        self._last_plan: SetPlan | None = None
        self._last_plan_id: int | None = None

    # ----- presets -------------------------------------------------------- #
    def list_presets(self) -> list[dict[str, Any]]:
        """Return all built-in presets as plain dicts (EventProfile fields)."""

        presets = builtin_presets()
        # Sofia first (it is the default), then the rest in registry order.
        return [asdict(profile) for profile in presets.values()]

    # ----- scan ----------------------------------------------------------- #
    def scan(self, folder: str) -> list[Track]:
        """Scan ``folder``, read tags, upsert each Track; return them.

        No heavy audio analysis here — just discovery + tag metadata, per the
        ``/api/scan`` contract.
        """

        if not folder or not os.path.isdir(folder):
            raise ApiError(f"Folder not found: {folder!r}", status=400)

        paths = scan_folder(folder)
        saved: list[Track] = []
        for path in paths:
            track = read_metadata(path)  # never raises; best-effort fields
            saved.append(self.tracks.upsert(track))
        _log.info("scan: discovered %d track(s) under %s", len(saved), folder)
        return saved

    # ----- analyze -------------------------------------------------------- #
    def analyze(self, track_ids: list[int] | None) -> tuple[dict[int, TrackFeatures], bool]:
        """Analyse tracks and cache features in the DB.

        Runs :func:`get_extractor` over the requested tracks (or all tracks when
        ``track_ids`` is falsy), upserts the resulting features, and stamps
        ``analyzed_at``. Returns ``(features_by_id, used_librosa)`` so the client
        can report which engine ran.
        """

        extractor = get_extractor()
        # Report which engine actually ran (librosa if importable, else
        # heuristic). is_available() is True for the heuristic too, so we detect
        # librosa by class name to keep the report honest.
        used_librosa = type(extractor).__name__ == "LibrosaFeatureExtractor"

        all_tracks = {t.id: t for t in self.tracks.get_all() if t.id is not None}
        if track_ids:
            wanted = [all_tracks[i] for i in track_ids if i in all_tracks]
        else:
            wanted = list(all_tracks.values())

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result: dict[int, TrackFeatures] = {}
        for track in wanted:
            features = extractor.extract(track)  # extractor falls back internally
            # Ensure the features point at the persisted track id.
            features.track_id = track.id  # type: ignore[assignment]
            self.features.upsert(features)
            # Mark the track analysed (timestamp only — keep tags intact).
            track.analyzed_at = now
            self.tracks.upsert(track)
            result[track.id] = features  # type: ignore[index]

        _log.info(
            "analyze: scored %d track(s) using %s",
            len(result),
            "librosa" if used_librosa else "heuristic",
        )
        return result, used_librosa

    # ----- rekordbox ------------------------------------------------------ #
    def rekordbox_import(self, xml_path: str) -> int:
        """Apply a Rekordbox XML to the stored library; persist any fills.

        Returns the number of tracks matched. Tolerant of a bad path/XML (the
        underlying function logs and returns 0).
        """

        if not xml_path or not os.path.isfile(xml_path):
            raise ApiError(f"Rekordbox XML not found: {xml_path!r}", status=400)

        library = self.tracks.get_all()
        matched = apply_rekordbox_to_tracks(library, xml_path)
        # Persist whatever was filled in (bpm/key/camelot/duration).
        for track in library:
            self.tracks.upsert(track)
        _log.info("rekordbox_import: matched %d track(s) from %s", matched, xml_path)
        return matched

    # ----- features (analyzed + manual overrides) ------------------------ #
    def _overlaid_features(self) -> dict[int, TrackFeatures]:
        """Analyzed features with any manual DJ energy override applied.

        For each override, the track's energy is replaced and every
        energy-dependent field is recomputed consistently (via
        :func:`derive_energy_dependent`), so a hand-set "this is a light track"
        flows through the whole planner. Overrides are stored separately, so
        re-analysis never wipes them.
        """

        features = self.features.get_all()
        for track_id, energy in self.overrides.get_all().items():
            base = features.get(track_id) or TrackFeatures(track_id=track_id)
            features[track_id] = derive_energy_dependent(base, energy)
        return features

    # ----- library snapshot ---------------------------------------------- #
    def library(self) -> dict[str, Any]:
        """Return the full library: tracks + features + constraints, as dicts."""

        tracks = self.tracks.get_all()
        features = self._overlaid_features()
        overrides = self.overrides.get_all()
        constraints = self.constraints.get_all()
        return {
            "tracks": [asdict(t) for t in tracks],
            "features": {tid: asdict(f) for tid, f in features.items()},
            "overrides": {tid: round(float(e), 4) for tid, e in overrides.items()},
            "constraints": [asdict(c) for c in constraints],
        }

    # ----- manual energy override ---------------------------------------- #
    def set_energy_override(self, track_id: int, energy_score: float) -> TrackFeatures:
        """Set a manual energy for a track; return the recomputed features."""

        if self.tracks.get(track_id) is None:
            raise ApiError(f"No track with id {track_id}", status=404)
        if energy_score < 0.0 or energy_score > 1.0:
            raise ApiError("energy_score must be between 0.0 and 1.0", status=400)
        self.overrides.set_energy(track_id, energy_score)
        base = self.features.get(track_id) or TrackFeatures(track_id=track_id)
        return derive_energy_dependent(base, energy_score)

    def clear_energy_override(self, track_id: int) -> TrackFeatures | None:
        """Remove a manual energy override; return the analyzed features (if any)."""

        self.overrides.clear(track_id)
        return self.features.get(track_id)

    # ----- constraints ---------------------------------------------------- #
    def set_constraint(
        self, track_id: int, constraint_type: str, value: str | None
    ) -> DjConstraint:
        """Set (or replace) a constraint of a type on a track."""

        if constraint_type not in _VALID_CONSTRAINTS:
            raise ApiError(f"Unknown constraint_type: {constraint_type!r}", status=400)
        if self.tracks.get(track_id) is None:
            raise ApiError(f"No track with id {track_id}", status=404)
        return self.constraints.set_for_track(track_id, constraint_type, value)

    def clear_constraint(self, track_id: int, constraint_type: str | None) -> None:
        """Remove all constraints on a track, or only one type if given."""

        if constraint_type is not None and constraint_type not in _VALID_CONSTRAINTS:
            raise ApiError(f"Unknown constraint_type: {constraint_type!r}", status=400)
        self.constraints.clear(track_id, constraint_type)

    # ----- generate ------------------------------------------------------- #
    def generate(
        self, profile_payload: dict[str, Any], locks: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Build a set plan and return it as JSON with per-row track details.

        ``profile_payload`` may be either ``{"preset": "<name>"}``, a full set of
        EventProfile fields, or a mix (custom fields override the named preset).
        ``locks`` is an optional list of ``{track_id, position}`` that become
        LOCK_POSITION constraints for this run (layered on top of the stored
        per-track constraints, never persisted).
        """

        profile = self._resolve_profile(profile_payload)

        library = self.tracks.get_all()
        if not library:
            raise ApiError(
                "Library is empty — scan a folder before generating a set.",
                status=400,
            )
        features = self._overlaid_features()

        # Stored constraints + the ad-hoc locks for this run.
        constraints = list(self.constraints.get_all())
        constraints.extend(self._locks_to_constraints(locks))

        # "Strict character" is a generate-time flag (not persisted on the
        # profile) — it rides in the profile payload from the form.
        strict = bool(profile_payload.get("strict"))
        plan = plan_set(library, features, profile, constraints, strict=strict)

        # Persist the plan and remember it as "current" for export endpoints.
        plan_id = self.plans.save(plan)
        self._last_plan = plan
        self._last_plan_id = plan_id

        tracks_by_id = {t.id: t for t in library if t.id is not None}
        return _plan_to_json(plan, plan_id, tracks_by_id, features)

    # ----- export --------------------------------------------------------- #
    def export(self, fmt: str, plan_id: int | None) -> str:
        """Export the current (or a stored) plan to ``fmt`` ('m3u'|'csv').

        Writes into the app data directory and returns the absolute file path.
        """

        plan, resolved_id = self._resolve_plan_for_export(plan_id)
        if plan is None or not plan.tracks:
            raise ApiError("No set plan available to export — generate one first.", 400)

        # Rebuild the track/feature lookups for the plan's tracks.
        library = {t.id: t for t in self.tracks.get_all() if t.id is not None}
        features = self.features.get_all()

        out_dir = app_data_dir()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = f"dj_set_{resolved_id or 'current'}_{stamp}"

        if fmt == "m3u":
            out_path = os.path.join(out_dir, f"{base}.m3u")
            return export_m3u(plan, library, out_path)
        if fmt == "csv":
            out_path = os.path.join(out_dir, f"{base}.csv")
            return export_csv(plan, library, features, out_path)
        raise ApiError(f"Unknown export format: {fmt!r}", status=400)

    # ----- internals ------------------------------------------------------ #
    def _resolve_profile(self, payload: dict[str, Any]) -> EventProfile:
        """Turn an API profile payload into an :class:`EventProfile`.

        Resolution: start from the named preset (or the Sofia default), then
        overlay any explicit EventProfile fields the client supplied. Unknown
        keys are ignored. Numeric fields are coerced defensively.
        """

        payload = payload or {}
        preset_name = payload.get("preset") or payload.get("name")
        presets = builtin_presets()
        if preset_name and preset_name in presets:
            base = presets[preset_name]
        else:
            base = default_profile()

        # Overlay explicit fields. id stays None so a fresh profile row is saved.
        def _str(key: str, fallback: str) -> str:
            val = payload.get(key)
            return str(val) if val not in (None, "") else fallback

        def _num(key: str, fallback: float) -> float:
            val = payload.get(key)
            if val in (None, ""):
                return fallback
            try:
                return float(val)
            except (TypeError, ValueError):
                _log.warning("generate: bad numeric %r=%r; using %s", key, val, fallback)
                return fallback

        return EventProfile(
            id=None,
            name=_str("name", base.name) if not preset_name else base.name,
            venue_type=_str("venue_type", base.venue_type),
            time_of_day=_str("time_of_day", base.time_of_day),
            crowd_state=_str("crowd_state", base.crowd_state),
            desired_energy=_str("desired_energy", base.desired_energy),
            peak_strategy=_str("peak_strategy", base.peak_strategy),
            target_duration_minutes=int(
                _num("target_duration_minutes", base.target_duration_minutes)
            ),
            min_energy=_num("min_energy", base.min_energy),
            max_energy=_num("max_energy", base.max_energy),
            main_peak_energy=_num("main_peak_energy", base.main_peak_energy),
        )

    def _locks_to_constraints(
        self, locks: list[dict[str, Any]] | None
    ) -> list[DjConstraint]:
        """Convert ad-hoc ``{track_id, position}`` locks into LOCK_POSITION
        constraints (not persisted — only used for this generation run)."""

        result: list[DjConstraint] = []
        for lock in locks or []:
            try:
                tid = int(lock["track_id"])
                pos = int(lock["position"])
            except (KeyError, TypeError, ValueError):
                _log.warning("generate: ignoring malformed lock %r", lock)
                continue
            result.append(
                DjConstraint(
                    id=None,
                    track_id=tid,
                    constraint_type=ConstraintType.LOCK_POSITION.value,
                    value=str(pos),
                )
            )
        return result

    def _resolve_plan_for_export(
        self, plan_id: int | None
    ) -> tuple[SetPlan | None, int | None]:
        """Pick the plan to export: an explicit id, else the in-memory current."""

        if plan_id is not None:
            stored = self.plans.get(plan_id)
            if stored is None:
                raise ApiError(f"No set plan with id {plan_id}", status=404)
            return stored, plan_id
        return self._last_plan, self._last_plan_id


# --------------------------------------------------------------------------- #
# JSON shaping helpers
# --------------------------------------------------------------------------- #
def _track_display_key(track: Track) -> str | None:
    """The DJ-facing key for display: Camelot if known, else the raw key."""

    return track.camelot_key or track.musical_key


def _plan_to_json(
    plan: SetPlan,
    plan_id: int | None,
    tracks_by_id: dict[int, Track],
    features_by_id: dict[int, TrackFeatures],
) -> dict[str, Any]:
    """Serialize a :class:`SetPlan` for the frontend.

    Each ordered row is joined to its track's title/artist/bpm/key/duration (and
    energy) so the UI can render the set table without a second lookup, exactly
    as the ``/api/generate`` contract requires.
    """

    rows: list[dict[str, Any]] = []
    for spt in plan.tracks:
        track = tracks_by_id.get(spt.track_id)
        feat = features_by_id.get(spt.track_id)
        rows.append(
            {
                "track_id": spt.track_id,
                "position": spt.position,
                "role": spt.role,
                "transition_score": round(float(spt.transition_score), 4),
                "position_score": round(float(spt.position_score), 4),
                "explanation": spt.explanation,
                "is_locked": spt.is_locked,
                # Joined track display fields:
                "title": track.title if track else None,
                "artist": track.artist if track else None,
                "bpm": track.bpm if track else None,
                "key": _track_display_key(track) if track else None,
                "duration_seconds": track.duration_seconds if track else None,
                "energy_score": round(float(feat.energy_score), 4) if feat else None,
            }
        )

    return {
        "plan": {
            "id": plan_id,
            "event_profile": asdict(plan.event_profile),
            "tracks": rows,
            "total_duration_seconds": plan.total_duration_seconds,
            "target_duration_seconds": plan.target_duration_seconds,
            "total_score": round(float(plan.total_score), 4),
            "segments": [asdict(s) for s in plan.segments],
            "energy_points": [round(float(e), 4) for e in plan.energy_points],
        }
    }


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app() -> Flask:
    """Create and configure the Flask application (with CORS enabled)."""

    app = Flask(__name__, static_folder=None)
    CORS(app)  # local tool: allow the frontend to call the API freely.

    service = PlannerService()

    # ----- error handling: never leak an HTML stack trace ----------------- #
    @app.errorhandler(ApiError)
    def _handle_api_error(err: ApiError):  # type: ignore[unused-ignore]
        # Expected, client-facing errors carry their own status.
        return jsonify({"error": err.message}), err.status

    @app.errorhandler(404)
    def _handle_404(err):  # type: ignore[unused-ignore]
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(Exception)
    def _handle_unexpected(err: Exception):  # type: ignore[unused-ignore]
        # Anything unexpected is logged in full and returned as a clean JSON 500
        # — the client never sees a stack trace.
        _log.exception("Unhandled error serving %s", request.path)
        return jsonify({"error": "Internal server error"}), 500

    # ----- static frontend ------------------------------------------------ #
    @app.get("/")
    def index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(_STATIC_DIR, filename)

    # ----- JSON API ------------------------------------------------------- #
    @app.get("/api/presets")
    def api_presets():
        return jsonify({"presets": service.list_presets()})

    @app.post("/api/scan")
    def api_scan():
        body = request.get_json(silent=True) or {}
        folder = (body.get("folder") or "").strip()
        tracks = service.scan(folder)
        return jsonify({"tracks": [asdict(t) for t in tracks]})

    @app.post("/api/analyze")
    def api_analyze():
        body = request.get_json(silent=True) or {}
        track_ids = body.get("track_ids") or None
        features, used_librosa = service.analyze(track_ids)
        return jsonify(
            {
                "features": {tid: asdict(f) for tid, f in features.items()},
                "used_librosa": used_librosa,
                "engine": "librosa" if used_librosa else "heuristic",
            }
        )

    @app.post("/api/rekordbox-import")
    def api_rekordbox_import():
        body = request.get_json(silent=True) or {}
        xml_path = (body.get("xml_path") or "").strip()
        matched = service.rekordbox_import(xml_path)
        return jsonify({"matched": matched})

    @app.get("/api/tracks")
    def api_tracks():
        return jsonify(service.library())

    @app.post("/api/constraints")
    def api_set_constraint():
        body = request.get_json(silent=True) or {}
        track_id = body.get("track_id")
        constraint_type = body.get("constraint_type")
        value = body.get("value")
        if track_id is None or constraint_type is None:
            raise ApiError("track_id and constraint_type are required", status=400)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            raise ApiError(f"Invalid track_id: {track_id!r}", status=400)
        constraint = service.set_constraint(track_id, str(constraint_type), value)
        return jsonify({"ok": True, "constraint": asdict(constraint)})

    @app.delete("/api/constraints")
    def api_clear_constraint():
        # Accept params from JSON body or query string for flexibility.
        body = request.get_json(silent=True) or {}
        track_id = body.get("track_id", request.args.get("track_id"))
        constraint_type = body.get("constraint_type", request.args.get("constraint_type"))
        if track_id is None:
            raise ApiError("track_id is required", status=400)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            raise ApiError(f"Invalid track_id: {track_id!r}", status=400)
        service.clear_constraint(
            track_id, str(constraint_type) if constraint_type else None
        )
        return jsonify({"ok": True})

    @app.post("/api/features/override")
    def api_set_override():
        body = request.get_json(silent=True) or {}
        track_id = body.get("track_id")
        energy = body.get("energy_score")
        if track_id is None or energy is None:
            raise ApiError("track_id and energy_score are required", status=400)
        try:
            track_id = int(track_id)
            energy = float(energy)
        except (TypeError, ValueError):
            raise ApiError("track_id must be int and energy_score a number", status=400)
        feat = service.set_energy_override(track_id, energy)
        return jsonify({"ok": True, "features": asdict(feat)})

    @app.delete("/api/features/override")
    def api_clear_override():
        body = request.get_json(silent=True) or {}
        track_id = body.get("track_id", request.args.get("track_id"))
        if track_id is None:
            raise ApiError("track_id is required", status=400)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            raise ApiError(f"Invalid track_id: {track_id!r}", status=400)
        feat = service.clear_energy_override(track_id)
        return jsonify({"ok": True, "features": asdict(feat) if feat else None})

    @app.post("/api/generate")
    def api_generate():
        body = request.get_json(silent=True) or {}
        profile_payload = body.get("profile") or {}
        locks = body.get("locks") or None
        return jsonify(service.generate(profile_payload, locks))

    @app.post("/api/export/m3u")
    def api_export_m3u():
        body = request.get_json(silent=True) or {}
        plan_id = body.get("plan_id")
        path = service.export("m3u", _maybe_int(plan_id))
        return jsonify({"path": path})

    @app.post("/api/export/csv")
    def api_export_csv():
        body = request.get_json(silent=True) or {}
        plan_id = body.get("plan_id")
        path = service.export("csv", _maybe_int(plan_id))
        return jsonify({"path": path})

    return app


def _maybe_int(value: Any) -> int | None:
    """Coerce an optional plan-id payload to int, or None."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Server runner (used by main.py)
# --------------------------------------------------------------------------- #
def run_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Create the app and run the development server (no auto-reload).

    ``main.py`` opens the browser before calling this; we keep the reloader off
    so the singleton DB / in-memory "current plan" survive and the process stays
    single-instance.
    """

    app = create_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)
