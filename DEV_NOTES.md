# DEV NOTES

Quick orientation for the build phases. The authoritative type/signature
contracts live in `CONTRACTS.md` — read that first.

## Virtualenv

A virtualenv already exists at `./.venv` (Python 3.14). **Always activate it**
before running python/pip:

```bash
. .venv/bin/activate
pip install -e .          # editable install (core deps only)
python -m dj_set_planner.main
pytest                    # run the test suite
```

Core deps that MUST install cleanly on Python 3.14: `mutagen`, `flask`,
`flask-cors` (confirmed working). `pytest` for tests.

## librosa is OPTIONAL

`librosa` / `numpy` / `scipy` / `scikit-learn` / `soundfile` are an **optional
extra** (`pip install -e ".[analysis]"`). The whole app must run with full
functionality **without** them, falling back to tag data + deterministic
heuristics.

Rules:
- NEVER import librosa/numpy/scipy at module top-level in a way that breaks the
  import when they're absent. Import them **lazily inside methods**.
- `analysis.get_extractor()` already degrades: essentia (stub, unavailable) →
  librosa (if importable) → `HeuristicExtractor` (always works).
- `essentia` is DROPPED for the MVP: keep only a thin `essentia_extractor.py`
  stub that reports unavailable and defers to librosa/heuristic.

## Determinism

The planner must be deterministic given the same inputs. No `random`, no
`Date.now`/wall-clock in scoring. Tie-break by track id. The heuristic scorer
derives spread from a **stable hash of `file_path`**, not randomness.

## Style

- Python 3.11+ type hints everywhere; dataclasses for domain models.
- UI separate from business logic; analysis separate from planning.
- Comment every scoring formula.
- Never silently swallow exceptions — log via `utils.logging.get_logger`.

---

## Module ownership map (phases)

Foundation has created the skeleton + shared contracts + the small deterministic
pieces. Later phases own the marked modules. **Conform to CONTRACTS.md exactly.**

### Foundation (DONE — do not redo)
- `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`
- `CONTRACTS.md`, `DEV_NOTES.md`
- All package `__init__.py`
- `domain/enums.py`, `domain/models.py`
- `db/schema.sql`, `db/database.py`, `db/repositories.py` (fully implemented)
- `utils/paths.py`, `utils/logging.py`, `utils/camelot.py` (fully implemented)
- `analysis/feature_extractor.py` (ABC + `HeuristicExtractor`), `analysis/__init__.get_extractor`
- `planning/context_profiles.py` (fully implemented, incl. Sofia preset)
- `main.py` (entry stub; WebUI replaces internals)
- `tests/fixtures/sample_tracks.json`, `tests/conftest.py`

### Analysis phase
- `analysis/scanner.py` — `scan_folder`
- `analysis/metadata_reader.py` — `read_metadata` (mutagen; never raises)
- `analysis/heuristic_scorer.py` — `score_from_metadata` (deterministic)
- `analysis/librosa_extractor.py` — `LibrosaFeatureExtractor` (lazy imports)
- `analysis/essentia_extractor.py` — `EssentiaFeatureExtractor` (stub)
- `analysis/rekordbox_import.py` — `parse_rekordbox_xml`, `apply_rekordbox_to_tracks`
- Tests: `test_metadata_scanner.py`, `test_heuristic_scorer.py`

### Planning phase
- `planning/energy_curve.py` — `generate_energy_curve`, `segment_at`, `target_energy_at`
- `planning/transition_scorer.py` — `TransitionScore`, `score_transition`
- `planning/track_roles.py` — `role_fit_scores`, `best_role`
- `planning/explanations.py` — `explain_track`
- `planning/track_selector.py` — `compute_position_score` + candidate-pool helpers
- `planning/beam_search_planner.py` — `plan_set` (deterministic)
- Tests: `test_energy_curve.py`, `test_transition_scorer.py`,
  `test_track_roles.py`, `test_track_selector.py`, `test_beam_search_planner.py`

### Export phase
- `export/m3u_exporter.py` — `export_m3u`
- `export/csv_exporter.py` — `export_csv`
- Tests: `test_export.py`

### WebUI phase
- `app.py` — Flask app + service layer + JSON API (see CONTRACTS.md endpoints)
- `main.py` — replace the Foundation stub's internals to actually start the
  server (`run_server`) and open the browser
- `ui/` — frontend assets (HTML/CSS/JS)

---

## Cross-phase reminders

- `Track.camelot_key` is derived from `musical_key` via
  `utils.camelot.to_camelot`. Set it in `metadata_reader` and `rekordbox_import`.
- All `TrackFeatures` fields are 0..1, default 0.5.
- `EventProfile.crowd_state` may be a compound string like
  `"EATING+TALKING+PARTIAL_DANCING"` — parse defensively if you split it.
- `SetPlan.energy_points` = ordered list of each track's `energy_score` (for the
  UI curve). `SetPlan.segments` = the generated energy curve.
- The DB `set_plans` table does not persist segments/energy_points; the planning
  layer recomputes them. `SetPlanRepository.get` returns them empty.
