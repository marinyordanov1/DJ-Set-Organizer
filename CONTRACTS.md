# CONTRACTS — AI DJ Set Planner

**This file is the single source of truth for shared types and function
signatures.** Every phase MUST conform to it EXACTLY: same module paths, class
names, field names, and function signatures. Read this first, then read the
actual foundation files under `src/dj_set_planner/domain/` and
`src/dj_set_planner/db/` before coding.

General rules:
- Python 3.11+ type hints everywhere; `dataclasses` for domain models.
- Keep UI separate from business logic, and analysis separate from planning.
- Comment every scoring formula.
- Never silently swallow exceptions — log them via `utils.logging.get_logger`.
- The app must be fully usable with fallback/mock (heuristic) analysis.
- `librosa`/`numpy`/`scipy` are OPTIONAL and must be lazily imported so the app
  imports and runs without them.
- The planner must be **deterministic** given the same inputs (stable
  tie-breaks by track id; no `random`, no wall-clock in scoring).

---

## `src/dj_set_planner/domain/enums.py`

All are `(str, Enum)` so values serialize to strings.

- `VenueType`: RESTAURANT, BAR, BEACH, CLUB, AFTERPARTY, PRIVATE_PARTY, OUTDOOR_DAY_PARTY
- `TimeOfDay`: DAY, SUNSET, NIGHT, LATE_NIGHT
- `CrowdState`: EATING, TALKING, WARMING_UP, PARTIAL_DANCING, DANCING, PEAK_DANCING
- `DesiredEnergy`: RELAXED, BALANCED, PROGRESSIVE, ENERGETIC
- `PeakStrategy`: ONE_MAIN_PEAK, MULTIPLE_SMALL_PEAKS, PROGRESSIVE_BUILD, FLAT_LOUNGE
- `TrackRole`: INTRO, WARM_GROOVE, PROGRESSIVE_BUILD, SMALL_PEAK, BREATHING_SPACE, MAIN_PEAK, RELEASE, OUTRO
- `ConstraintType`: MUST_PLAY, AVOID, PREFERRED_INTRO, PREFERRED_OUTRO, PREFERRED_PEAK, LOCK_POSITION, DO_NOT_PLAY_BEFORE, DO_NOT_PLAY_AFTER

---

## `src/dj_set_planner/domain/models.py` (dataclasses)

- **Track**: `id: int|None`, `file_path: str`, `title: str|None`,
  `artist: str|None`, `album: str|None`, `genre: str|None`,
  `duration_seconds: int|None`, `bpm: float|None`, `musical_key: str|None`,
  `camelot_key: str|None`, `analyzed_at: str|None`
- **TrackFeatures**: `track_id: int`, `energy_score: float`,
  `danceability_score: float`, `mood_brightness: float`, `groove_score: float`,
  `vocal_density: float`, `intro_suitability: float`, `outro_suitability: float`,
  `peak_potential: float`, `restaurant_safety_score: float`,
  `mixability_score: float`
  (all floats in 0.0..1.0; sensible default 0.5)
- **EventProfile**: `id: int|None`, `name: str`, `venue_type: str`,
  `time_of_day: str`, `crowd_state: str`, `desired_energy: str`,
  `peak_strategy: str`, `target_duration_minutes: int`, `min_energy: float`,
  `max_energy: float`, `main_peak_energy: float`
- **DjConstraint**: `id: int|None`, `track_id: int`, `constraint_type: str`,
  `value: str|None`
- **SetSegment**: `name: str`, `start_pct: float`, `end_pct: float`,
  `min_energy: float`, `max_energy: float`, `role: str`
- **SetPlanTrack**: `track_id: int`, `position: int`, `role: str`,
  `transition_score: float`, `position_score: float`, `explanation: str`,
  `is_locked: bool = False`
- **SetPlan**: `event_profile: EventProfile`, `tracks: list[SetPlanTrack]`,
  `total_duration_seconds: int`, `target_duration_seconds: int`,
  `total_score: float`, `segments: list[SetSegment]`,
  `energy_points: list[float]`
  (`energy_points` = actual `energy_score` of each ordered track, for the curve)

---

## `src/dj_set_planner/analysis/feature_extractor.py`

- `class FeatureExtractor(ABC)`:
  - `def extract(self, track: Track) -> TrackFeatures` — takes a Track so it can
    read tag-derived bpm/key/genre; returns features in 0..1.
  - `def is_available(self) -> bool`

---

## analysis module contracts

- `analysis/scanner.py`: `def scan_folder(folder: str) -> list[str]` — return
  absolute paths of supported audio files (`.mp3 .wav .flac .aiff .aif .m4a
  .ogg`); recursive; sorted; ignore dotfiles.
- `analysis/metadata_reader.py`: `def read_metadata(file_path: str) -> Track` —
  use `mutagen`; fill title/artist/album/genre/duration_seconds/bpm/musical_key
  when present in tags; `id=None`; also set `camelot_key` via
  `utils.camelot.to_camelot` when a key tag exists. Must never raise on
  unreadable files — log and return a Track with best-effort fields (title
  falls back to filename stem).
- `analysis/heuristic_scorer.py`: `def score_from_metadata(track: Track) ->
  TrackFeatures` — deterministic heuristic features derived ONLY from tags + a
  stable hash of `file_path` for spread; used when librosa unavailable.
  Implements the heuristic formulas from the spec (energy from bpm/genre cues,
  danceability, brightness, groove, vocal_density default 0.5, intro/outro
  suitability, peak_potential, `restaurant_safety_score` = inverse of excessive
  energy/harshness, mixability). All 0..1.
- `analysis/librosa_extractor.py`: `class
  LibrosaFeatureExtractor(FeatureExtractor)` — lazy-imports librosa/numpy inside
  methods; `is_available()` returns False if import fails; `extract()` computes
  tempo/RMS energy/spectral centroid/onset strength/zcr and maps to the
  TrackFeatures fields normalized 0..1; on ANY failure falls back to
  `heuristic_scorer.score_from_metadata(track)`.
- `analysis/essentia_extractor.py`: `class
  EssentiaFeatureExtractor(FeatureExtractor)` — `is_available()` False (stub);
  `extract()` delegates to LibrosaFeatureExtractor then heuristic.
- `analysis/__init__` helper: `def get_extractor(prefer: str|None=None) ->
  FeatureExtractor` — returns best available: essentia if available else librosa
  if available else a HeuristicExtractor wrapper around `score_from_metadata`.
- `analysis/rekordbox_import.py`:
  - `def parse_rekordbox_xml(xml_path: str) -> list[dict]` — parse a Rekordbox
    collection XML; each dict has keys: `file_path, title, artist, album, genre,
    bpm, musical_key, camelot_key, duration_seconds` — any may be None.
  - `def apply_rekordbox_to_tracks(tracks: list[Track], xml_path: str) -> int` —
    match by file basename or path, fill in bpm/musical_key/camelot_key/duration
    when present; return count matched.
  - Must tolerate missing/garbage XML by logging and returning empty/0.

---

## utils

- `utils/paths.py`: `def app_data_dir() -> str` (e.g. `~/Library/Application
  Support/DJSetPlanner` on macOS, created if missing); `def db_path() -> str`.
- `utils/logging.py`: `def get_logger(name: str) -> logging.Logger` (configured
  once).
- `utils/camelot.py`:
  - `def to_camelot(musical_key: str|None) -> str|None` — parse `"Am"`, `"A
    min"`, `"Amin"`, `"8A"`, `"A#"`, `"Bbm"`, `"F#min"`, `"Open key"` etc ->
    Camelot like `"8A"`/`"5B"`; return None if unknown.
  - `def camelot_distance(a: str|None, b: str|None) -> int|None` — steps on the
    wheel; harmonic neighbours = 0/1; relative major-minor = 0; None if either
    unknown.
  - `def key_compatibility(a: str|None, b: str|None) -> float` — 0..1: same=1.0,
    harmonic neighbour or relative=~0.85, two steps=~0.5, unknown=0.6 neutral,
    clashing=~0.2.

---

## db

- `db/schema.sql`: EXACTLY the schema from the spec (tracks, track_features,
  event_profiles, set_plans, set_plan_tracks, dj_constraints) with the given
  columns.
- `db/database.py`: `class Database`: `__init__(self, path: str|None=None)` opens
  sqlite (`PRAGMA foreign_keys=ON`), creates schema from schema.sql if missing;
  `.conn` property; `.close()`. Module-level `def get_db() -> Database` singleton
  using `utils.paths.db_path()`.
- `db/repositories.py`: `TrackRepository`, `FeatureRepository`,
  `EventProfileRepository`, `ConstraintRepository`, `SetPlanRepository` — CRUD
  methods returning the domain dataclasses. At minimum:
  - `TrackRepository.upsert(track)->Track`(with id), `.get_all()->list[Track]`,
    `.get_by_path(path)`.
  - `FeatureRepository.upsert(features)`, `.get(track_id)`,
    `.get_all()->dict[int,TrackFeatures]`.
  - `ConstraintRepository.set_for_track(track_id, constraint_type, value=None)`,
    `.clear(track_id, constraint_type=None)`, `.get_all()->list[DjConstraint]`.
  - `EventProfileRepository.save/get`.
  - `SetPlanRepository.save(plan)->id`, `.get(id)`.

---

## planning

- `planning/context_profiles.py`: a builtin preset registry. `def
  builtin_presets() -> dict[str, EventProfile]`. MUST include **"Sofia Day Party
  Restaurant"**: venue RESTAURANT, time DAY, crowd
  `"EATING+TALKING+PARTIAL_DANCING"`, desired BALANCED, peak ONE_MAIN_PEAK,
  `target_duration_minutes 120`, `min_energy 0.30`, `max_energy 0.78`,
  `main_peak_energy 0.75`. Also expose constants `avoid_energy_above=0.85`,
  `average_energy_range=(0.52,0.62)`, `prefer_smooth_transitions`,
  `allow_small_peaks`. `def default_profile() -> EventProfile` (the Sofia
  preset).
- `planning/energy_curve.py`: `def generate_energy_curve(profile: EventProfile)
  -> list[SetSegment]`. For ONE_MAIN_PEAK use EXACTLY the spec segmentation:
  - intro 0-10% e0.30-0.40
  - warm_groove 10-30% 0.40-0.52
  - progressive_build 30-50% 0.52-0.65
  - small_peak 50-60% 0.65-0.72
  - breathing_space 60-70% 0.50-0.60
  - main_peak 70-82% 0.72-0.78
  - release 82-92% 0.55-0.65
  - outro 92-100% 0.35-0.48

  Provide sensible alternative templates for MULTIPLE_SMALL_PEAKS,
  PROGRESSIVE_BUILD, FLAT_LOUNGE. `def segment_at(curve, position_fraction:
  float) -> SetSegment`. `def target_energy_at(curve, fraction) ->
  tuple[float,float]`.
- `planning/transition_scorer.py`: `def score_transition(prev: Track, prev_f:
  TrackFeatures, nxt: Track, nxt_f: TrackFeatures, profile: EventProfile, *,
  entering_segment_role: str|None=None) -> TransitionScore`. `TransitionScore`
  is a dataclass: `score: float` (0..1), `bpm: float`, `key: float`, `energy:
  float`, `mood: float`, `intro_outro: float`, `vocal_penalty: float`, `notes:
  str`. Weights per spec: `bpm*0.25 + key*0.25 + energy_delta*0.20 +
  mood_continuity*0.15 + intro_outro_fit*0.10 - vocal_clash_penalty*0.05`. BPM
  compatibility bands (0-3 excellent, 3-6 good, 6-10 acceptable, 10+ penalty)
  and treat half/double-time (2x within tolerance) as compatible. Key via
  `utils.camelot.key_compatibility`. Energy delta tuned for day-party (small
  increases best; big jumps penalised unless entering peak; big drops penalised
  unless entering breathing_space/outro).
- `planning/track_roles.py`: `def role_fit_scores(track: Track, f:
  TrackFeatures, profile: EventProfile) -> dict[str, float]` (0..1 fit per
  TrackRole, using the spec rules). `def best_role(track, f, profile) -> str`.
- `planning/explanations.py`: `def explain_track(track: Track, f: TrackFeatures,
  role: str, segment: SetSegment, transition_in: TransitionScore|None,
  position_score: float, profile: EventProfile) -> str` (human-readable, in the
  style of the spec examples, English).
- `planning/track_selector.py`: `def compute_position_score(track, f, segment,
  profile, prev_pair, constraints_for_track) -> tuple[float, dict]` using
  weights: `context_fit*0.30 + energy_curve_fit*0.25 + role_fit*0.20 +
  mood_fit*0.10 + mixability_with_neighbors*0.10 + dj_preference*0.05`, minus the
  listed penalties. Also helpers to build the candidate pool (drop AVOID, honour
  MUST_PLAY/PREFERRED_* and LOCK_POSITION).
- `planning/beam_search_planner.py`: `def plan_set(library: list[Track],
  features: dict[int, TrackFeatures], profile: EventProfile, constraints:
  list[DjConstraint], *, beam_width: int=20, max_candidates_per_step: int=10,
  duration_tolerance_seconds: int=300) -> SetPlan`. Algorithm per spec: drop
  AVOID; seed locked/preferred (intro/outro/peak/must_play/lock_position);
  generate energy curve; expand partial playlists choosing top candidates per
  next slot by combined `position_score + incoming transition_score`; keep top
  `beam_width`; stop within duration tolerance of target; rank; return best with
  per-track explanations, energy_points and segments. MUST be deterministic
  (stable tie-breaks by track id). MUST respect AVOID (never selected) and
  include MUST_PLAY when feasible, and place PREFERRED_PEAK in the main_peak
  segment when possible.

---

## web layer (Foundation only writes CONTRACTS; WebUI phase implements)

- `app.py`: builds the Flask app and a service layer.
- `main.py`: `def main():` start server on `127.0.0.1:5000` and open the
  browser; runnable via `python -m dj_set_planner.main`.

JSON API the frontend expects:

```
GET  /api/presets -> {presets:[{name, ...EventProfile fields}]}
POST /api/scan {folder} -> {tracks:[Track...]}            # scan + read metadata + upsert, no heavy analysis yet
POST /api/analyze {track_ids?:[]} -> {features:{track_id:TrackFeatures}}   # run get_extractor over tracks, cache in db; report whether librosa was used
POST /api/rekordbox-import {xml_path} -> {matched:int}
GET  /api/tracks -> {tracks:[...], features:{...}, constraints:[...]}
POST /api/constraints {track_id, constraint_type, value?}  # and DELETE semantics -> ok
POST /api/generate {profile:{...}, locks?:[{track_id,position}]} -> {plan: SetPlan-as-json with joined track title/artist/bpm/key/duration per row}
POST /api/export/m3u {plan_id? or current} -> writes file, returns {path}
POST /api/export/csv -> {path}
```

---

## export

- `export/m3u_exporter.py`: `def export_m3u(plan: SetPlan, tracks_by_id:
  dict[int,Track], out_path: str) -> str` (`#EXTM3U` + `#EXTINF` lines +
  absolute file paths, ordered; Rekordbox-importable).
- `export/csv_exporter.py`: `def export_csv(plan: SetPlan, tracks_by_id,
  features_by_id, out_path: str) -> str` (columns:
  `position,title,artist,file_path,role,duration_seconds,bpm,key,energy_score,transition_score,position_score,explanation`).

---

## tests (pytest)

Tests must run WITHOUT any real audio files, using
`tests/fixtures/sample_tracks.json` (Foundation writes it: ~30 synthetic tracks
with bpm/key/genre/duration and a parallel features blob). `tests/conftest.py`
exposes pytest fixtures: `sample_tracks` (`list[Track]`), `sample_features`
(`dict[int,TrackFeatures]`), `sofia_profile` (`EventProfile`).

Required test modules:
- `test_energy_curve.py` (Sofia 120min -> 8 ordered segments, main_peak after
  70%, outro last)
- `test_transition_scorer.py` (close bpm + compatible key -> high; far bpm + big
  energy jump -> low)
- `test_track_roles.py`
- `test_track_selector.py`
- `test_beam_search_planner.py` (respects target duration +/-5min; AVOID never
  selected; MUST_PLAY included; PREFERRED_PEAK lands in main_peak segment)
- `test_metadata_scanner.py`
- `test_heuristic_scorer.py`
- `test_export.py`
