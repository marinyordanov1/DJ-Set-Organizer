# Tasks: Context-Aware Track Selection

Based on `prd-context-aware-selection.md`. Working directly on the main tree (no
feature branch, per the DJ's request).

## Relevant Files

- `src/dj_set_planner/domain/models.py` - `TrackFeatures` dataclass gains the new `harmonic_ratio` field.
- `src/dj_set_planner/db/schema.sql` - `track_features` gains a `harmonic_ratio` column.
- `src/dj_set_planner/db/database.py` - Lightweight migration to add `harmonic_ratio` to existing DBs (CREATE TABLE IF NOT EXISTS won't alter an existing table).
- `src/dj_set_planner/db/repositories.py` - `FeatureRepository` mapper + upsert SQL include `harmonic_ratio`.
- `src/dj_set_planner/analysis/librosa_extractor.py` - Compute `harmonic_ratio` via HPSS on the already-loaded signal.
- `src/dj_set_planner/analysis/heuristic_scorer.py` - Heuristic fallback for `harmonic_ratio`; confirm `derive_energy_dependent` preserves it.
- `src/dj_set_planner/planning/venue_profiles.py` - **NEW.** Venue character profiles (per-feature targets + harshness ceiling) and time-of-day modifiers, as plain data.
- `src/dj_set_planner/planning/track_selector.py` - `character_fit()` + venue-aware `context_fit` in `compute_position_score`.
- `src/dj_set_planner/planning/beam_search_planner.py` - Per-venue harshness ceiling, Strict filter, time-of-day band modifier, outdoor position-varying character; thread venue/time/strict through `plan_set`.
- `src/dj_set_planner/planning/context_profiles.py` - Presets stop relying on removed fields; expose strict default.
- `src/dj_set_planner/app.py` - `generate()` accepts/forwards `strict`; `_resolve_profile` tolerates removed fields.
- `src/dj_set_planner/ui/static/index.html` - Remove Crowd state + Desired energy fields and the PRIVATE_PARTY option; add Strict checkbox.
- `src/dj_set_planner/ui/static/app.js` - Drop removed fields; read the Strict checkbox; optional venue hint.
- `tests/test_harmonic_ratio.py` - **NEW.** Descriptor range, melodic>percussive, persistence round-trip.
- `tests/test_venue_profiles.py` - **NEW.** Profiles load; time modifiers bounded; character_fit behaves.
- `tests/test_context_selection.py` - **NEW.** Restaurant vs Club character differs; Strict excludes; time-of-day shifts energy; removed fields don't break load/generate.
- `tests/test_beam_search_planner.py` / `tests/test_track_selector.py` - Update for the new scoring inputs (keep existing assertions green).

### Notes

- This is a Python project — tests live in `tests/` (not alongside source). Run with `python -m pytest -q` inside the venv (`source .venv/bin/activate`). There is no jest.
- librosa is an optional extra; everything must still import and run without it (heuristic fallback). Re-run "Analyze" to populate `harmonic_ratio` for the existing library.
- `strict_character` is passed as a generate-time flag (like `locks`), NOT persisted on `EventProfile`, to avoid a schema migration for it.
- Removed context fields (`crowd_state`, `desired_energy`, `PRIVATE_PARTY`) leave their DB columns/enum value dormant with safe defaults — no destructive migration (PRD §4.18).
- Keep the planner deterministic (no randomness) so tests stay reproducible.

## Tasks

- [x] 1.0 Add the `harmonic_ratio` audio descriptor (data layer)
  - [x] 1.1 Add `harmonic_ratio: float = 0.5` to the `TrackFeatures` dataclass in `domain/models.py`.
  - [x] 1.2 Add `harmonic_ratio REAL NOT NULL DEFAULT 0.5` to `track_features` in `schema.sql`.
  - [x] 1.3 In `db/database.py`, add a tiny idempotent migration: check `PRAGMA table_info(track_features)` and `ALTER TABLE ... ADD COLUMN harmonic_ratio` if it's missing (so existing DBs upgrade).
  - [x] 1.4 Update `FeatureRepository` in `repositories.py`: `_row_to_features`, the upsert column list/VALUES, and the `ON CONFLICT` set, to include `harmonic_ratio`.
  - [x] 1.5 In `librosa_extractor.py`, compute `harmonic_ratio` from HPSS on the already-loaded `y` (e.g. `librosa.effects.hpss`), as `rms(harmonic) / (rms(harmonic) + rms(percussive))`; reuse the signal, don't re-load the file.
  - [x] 1.6 In `heuristic_scorer.py`, add a deterministic `harmonic_ratio` estimate (from `mood_brightness` + inverse `groove_score`) in `score_from_metadata`; confirm `derive_energy_dependent` preserves `harmonic_ratio` (it should, via `replace`).
  - [x] 1.7 Verify the planner treats a missing/neutral `harmonic_ratio` (0.5) safely; re-run Analyze to populate the existing library.
  - [x] 1.8 Write `tests/test_harmonic_ratio.py`: value in [0,1]; a synthetic harmonic signal scores higher than a percussive one; repo upsert→get round-trips the field.

- [x] 2.0 Define venue character profiles + time-of-day modifiers; remove dead context fields
  - [x] 2.1 Create `planning/venue_profiles.py` with a `CharacterProfile` dataclass (per-feature target/range + weight, plus `harshness_ceiling`) and a registry for all venues (RESTAURANT, BAR, BEACH, CLUB, AFTERPARTY, OUTDOOR_DAY_PARTY) per PRD §4.6.
  - [x] 2.2 Add a time-of-day modifier table (DAY/SUNSET/NIGHT/LATE_NIGHT → energy-band delta + optional character nudge for LATE_NIGHT toward afterparty), bounded so `min<=max`.
  - [x] 2.3 Remove the `PRIVATE_PARTY` `<option>` from the Venue select in `index.html` (keep the enum value dormant in `enums.py`).
  - [x] 2.4 Remove `crowd_state` and `desired_energy` from the context form (`index.html`) and from `app.js` (`readProfileFromForm`, `applyPresetToForm`, the slider/listener wiring); keep the model/schema fields dormant with defaults.
  - [x] 2.5 Decide strict as a generate-time flag: add a `strict` parameter to the generate path (API → service → planner), default `False`; do not persist on `EventProfile`.
  - [x] 2.6 Write `tests/test_venue_profiles.py`: every venue has a profile; harshness ceilings ordered (restaurant < club); time modifiers never invert the band.

- [x] 3.0 Implement character-fit scoring, per-venue harshness ceiling, Strict filter, and time-of-day modifier in the planner
  - [x] 3.1 Add `character_fit(features, venue_profile, position_fraction=None) -> float` (0–1) in `venue_profiles.py` or `track_selector.py`; support the outdoor position-varying target.
  - [x] 3.2 In `track_selector.compute_position_score`, replace the generic `_context_fit` with venue-aware character-fit (keep the positive weights summed to 1.0); thread the venue profile in.
  - [x] 3.3 In `beam_search_planner._drop_over_cap_tracks`, use the venue's `harshness_ceiling` instead of the global `context_profiles.avoid_energy_above`.
  - [x] 3.4 In `plan_set`, when `strict` is on, drop pool tracks with `character_fit < 0.4`; if that empties the pool, fall back to soft and `log` it (mirror the energy-cap safety net). DJ-chosen tracks (must-play/preferred/locked) are exempt.
  - [x] 3.5 Apply the time-of-day energy-band modifier to the profile's `min_energy`/`max_energy` BEFORE the curve + adaptive normalization run; clamp to keep `min<=max`.
  - [x] 3.6 Thread `venue`/`time_of_day`/`strict` into `plan_set` (venue+time come from the `EventProfile`; `strict` is the new kwarg); pass `position_fraction` to character-fit for outdoor.
  - [x] 3.7 Update `tests/test_track_selector.py` and `tests/test_beam_search_planner.py` for the new scoring inputs; keep existing acceptance assertions green.
  - [x] 3.8 Write `tests/test_context_selection.py`: Restaurant set has higher mean `harmonic_ratio` + lower mean energy than Club on the same fixture; Afterparty lower mean `vocal_density` than Club; Strict ON yields only on-character tracks (or logged fallback); Day→Night raises mean energy; plans deterministic.

- [x] 4.0 Wire the context form + API/service
  - [x] 4.1 `index.html`: ensure the form keeps Preset, Target duration, Venue, Time of day, Peak strategy, Energy min/max; add a `Strict character` checkbox; (optional) a one-line venue-character hint under the Venue select.
  - [x] 4.2 `app.js`: read the Strict checkbox into the generate payload; (optional) update the venue hint on change; remove all references to the deleted fields.
  - [x] 4.3 `app.py`: `api_generate`/`service.generate` accept `strict` and forward it to `plan_set`; `_resolve_profile` tolerates missing `crowd_state`/`desired_energy` from the payload.
  - [x] 4.4 Browser smoke (Playwright on a temp port): switch Venue restaurant↔club and regenerate — confirm the selection character visibly changes; toggle Strict and confirm filtering.

- [x] 5.0 Tests & verification
  - [x] 5.1 `python -m pytest -q` fully green (old + new tests).
  - [x] 5.2 All-modules import smoke WITH and WITHOUT librosa (no top-level hard imports).
  - [x] 5.3 Re-analyze the real library; spot-check that `harmonic_ratio` has a sensible spread (melodic tracks high, percussive low).
  - [x] 5.4 End-to-end on the real library: generate Restaurant vs Club and assert the PRD success metrics (mean `harmonic_ratio`/energy differ in the expected direction).
  - [x] 5.5 Confirm the existing DB upgrades cleanly (migration adds the column) and generate still works after the field removals.
