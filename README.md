# AI DJ Set Planner

A **local** desktop tool (served as a local web app) that helps a DJ pick and
order ~25-30 tracks from a folder of ~100, telling a musical story:

> intro → build → small peaks → one main peak → release → outro

This is **not** a live mixing app — there are no decks and no realtime
beatmatching. It analyzes a folder of audio, scores each track, and plans an
ordered set for a given event context.

The UI is a small **Flask** web app served locally; you drive it from your
browser. (There is no PySide6 / native desktop window.)

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Optional real audio analysis (heavy, not required — the app works on tag data
+ deterministic heuristics without it):

```bash
pip install -e ".[analysis]"   # pulls in librosa / numpy / scipy
```

To run the tests you also want the dev extra:

```bash
pip install -e ".[dev]"        # pytest
```

## Run

```bash
python -m dj_set_planner.main
```

This starts the local server on **http://127.0.0.1:5000/** and opens your
default browser to it. Press `Ctrl+C` to stop. All data (the SQLite library and
exported playlists) lives under the per-user app-data directory
(`~/Library/Application Support/DJSetPlanner` on macOS).

## Workflow

The browser UI walks you through the same JSON API the app exposes:

1. **Scan** a folder of audio — discovers supported files
   (`.mp3 .wav .flac .aiff .aif .m4a .ogg`, recursive) and reads their tags
   (title/artist/genre/bpm/key/duration) into the library. No heavy analysis
   yet. (`POST /api/scan {folder}`)
2. **Analyze** the library — scores each track's energy, danceability, mood,
   groove, intro/outro suitability, peak potential, restaurant-safety, and
   mixability. Uses **librosa** if installed, otherwise deterministic
   tag-based heuristics; the response reports which engine ran.
   (`POST /api/analyze`)
3. **Import Rekordbox XML** (optional) — fills in BPM/key/duration from a
   Rekordbox collection export, matching by file path/basename.
   (`POST /api/rekordbox-import {xml_path}`)
4. **Generate** a set — pick a preset (e.g. *Sofia Day Party Restaurant*),
   optionally lock or constrain tracks (MUST_PLAY / AVOID / PREFERRED_PEAK /
   LOCK_POSITION …), and the beam-search planner orders ~25-30 tracks along the
   target energy curve, with a per-track explanation and an energy curve.
   (`POST /api/generate {profile, locks?}`)
5. **Export** the generated set as an `.m3u` (Rekordbox-importable) or `.csv`.
   Files are written into the app-data directory.
   (`POST /api/export/m3u`, `POST /api/export/csv`)

The planner is **deterministic**: the same library + features + profile +
constraints always produce the same set (stable tie-breaks by track id).

## Testing

```bash
. .venv/bin/activate
python -m pytest -q
```

The suite runs with **no real audio files** — it uses synthetic fixtures in
`tests/fixtures/sample_tracks.json` — and passes whether or not librosa is
installed.

## MVP limitations

- **Not a live mixing tool**: no decks, no realtime beatmatching, no waveform
  scrubbing. It plans an ordered set; you play it in your own software.
- **librosa is optional.** Without it, analysis falls back to tag data +
  deterministic heuristics. Real audio features (RMS energy, spectral centroid,
  onset strength, etc.) require `pip install -e ".[analysis]"`.
- **Essentia is dropped for the MVP.** `essentia_extractor.py` is a thin stub
  that always reports unavailable and defers to librosa/heuristics.
- Single local user, single SQLite library; no auth, no multi-user, no cloud.

## Layout

```
src/dj_set_planner/
  domain/     shared dataclasses + enums (the contracts)
  db/         sqlite schema, connection, repositories
  analysis/   scanning, metadata, feature extraction (librosa optional)
  planning/   energy curve, scoring, beam-search planner
  export/     m3u / csv exporters
  ui/         web frontend assets
  utils/      paths, logging, camelot-wheel helpers
```

See `CONTRACTS.md` for the authoritative shared types and signatures, and
`DEV_NOTES.md` for the per-phase module-ownership map.
