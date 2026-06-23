# PRD: Context-Aware Track Selection

## 1. Introduction / Overview

Today the set planner only really uses four inputs from the event context:
`peak_strategy`, `min_energy`, `max_energy`, and `target_duration_minutes`.
The other context controls (Venue, Time of day, Desired energy, Crowd state)
are stored but **ignored** by the algorithm — selecting "Restaurant" produces
the same kind of tracks as "Club".

This feature makes **Venue** and **Time of day** genuinely shape the set, so the
software arranges music with the *character* the DJ wants — melodic and lounge
for a restaurant, strong and danceable for a club, etc. The DJ picks the moment;
the planner picks tracks that fit that moment and orders them into a story.

The core new idea: every track gets a **melodic ↔ percussive** descriptor
(missing today), and every Venue defines a **character profile** (preferred
ranges for energy, melodic-ness, danceability, groove, vocals, and harshness).
Selection scores each track against the venue's character; Time of day nudges
the energy band on top.

## 2. Goals

1. Selecting a different **Venue** measurably changes *which* tracks are chosen
   and their character (not just the energy band).
2. **Restaurant** sets skew melodic/lounge; **Club** sets skew strong/danceable;
   **Afterparty** sets skew hypnotic/low-vocal — verifiably, on the DJ's own library.
3. **Time of day** shifts the overall feel lighter (day) → stronger (night).
4. Keep it controllable: a **Strict** toggle turns the character preference from a
   soft bias into a hard filter.
5. Remove dead/confusing controls (Crowd state, Desired energy, Private party).
6. No regression to the existing arc (peak strategy), adaptive energy, manual
   energy override, or duration targeting.

## 3. User Stories

- As a DJ playing a **restaurant day party**, I select Venue=Restaurant,
  Time=Day, and the planner gives me melodic, lounge-leaning tracks that build
  gently — without me hand-marking each one.
- As a DJ playing a **club night**, I select Venue=Club, Time=Night, and the
  planner picks my strongest, most danceable tracks and orders them to peak.
- As a DJ playing an **afterparty**, I get hypnotic, groovy, low-vocal tracks
  that keep people moving and talking, that "don't make them want to leave".
- As a DJ with a **mixed open-format library**, I flip Strict on when I want the
  planner to *only* use on-character tracks, and off when I want it to fill the
  full set even if some tracks are off-character.
- As a DJ, when I change Venue from Restaurant to Club and re-generate, I can see
  the selection change character, so the context controls feel real.

## 4. Functional Requirements

### A. New audio descriptor: melodic vs percussive

1. The system MUST add a new per-track feature `harmonic_ratio` (0.0–1.0):
   `1.0` = fully melodic/harmonic, `0.0` = fully percussive/drum-driven.
2. When librosa is available, `harmonic_ratio` MUST be computed via
   harmonic–percussive source separation (e.g. `librosa.effects.hpss`),
   as `harmonic_energy / (harmonic_energy + percussive_energy)` over the
   analysis window.
3. When librosa is NOT available, `harmonic_ratio` MUST fall back to a
   deterministic heuristic estimate from existing tags/features (e.g. from
   `mood_brightness` and `groove_score`) so the app still runs.
4. `harmonic_ratio` MUST be persisted in `track_features` and survive restarts,
   exactly like the other features.
5. Because this is a new descriptor, the existing analyzed library MUST be
   re-analyzable to populate it (re-run "Analyze"); tracks without it MUST be
   treated as neutral (`0.5`) rather than breaking the planner.

### B. Venue character profiles

6. Each Venue MUST map to a **character profile** describing preferred feature
   ranges/targets and a harshness ceiling. The confident definitions are:
   - **RESTAURANT** — melodic high (`harmonic_ratio` high), energy low–mid,
     danceability low–mid, vocals OK, harshness ceiling LOW (no banging tracks).
   - **CLUB** — energy high, danceability high, `peak_potential` high,
     `harmonic_ratio` neutral (doesn't matter), harshness ceiling HIGH.
   - **AFTERPARTY** — `harmonic_ratio` mid–high (hypnotic/melodic), energy mid,
     groove high, **vocals LOW** (so people talk over it), not too peaky.
   - **BAR** — "percussion house": percussive (`harmonic_ratio` 0.30–0.50),
     energy 0.45–0.65, groove high, vocals low–mid, relaxed but rhythmic,
     harshness ceiling mid-low.
   - **BEACH** — like BAR but brighter/sunnier: `harmonic_ratio` 0.35–0.55,
     energy 0.45–0.65, **higher `mood_brightness`**, organic feel, groove high,
     harshness ceiling mid-low.
   - **OUTDOOR_DAY_PARTY** — a blend whose character shifts across the set:
     more melodic early (restaurant-like), more percussive later (bar-like).
     Overall `harmonic_ratio` 0.45–0.65, energy 0.40–0.70, sunny brightness.
     This is the only venue whose character target varies with set position.
7. The planner MUST compute a **character-fit score** (0–1) for each track
   against the active venue's profile, and fold it into the existing
   position/selection score so on-character tracks rank higher.
8. The venue's **harshness ceiling** MUST replace the current hardcoded global
   `avoid_energy_above = 0.85`. Restaurant uses a low ceiling; Club a high one.
   This fixes today's bug where strong club tracks are dropped regardless of venue.
9. Character-fit MUST operate on the same library-relative (adaptive) energy the
   planner already uses, so it works for any library's absolute loudness.

### C. Strict toggle (soft by default)

10. The context form MUST include a **"Strict character"** checkbox, default OFF.
11. **OFF (soft):** off-character tracks are down-ranked but still selectable to
    fill the set's duration.
12. **ON (strict):** tracks whose character-fit is below the threshold
    (default **character-fit < 0.4**) are EXCLUDED from selection — UNLESS
    excluding them would make the set impossible to fill, in which case the
    planner MUST fall back to soft and log that it did (same safety pattern as
    the existing energy-cap filter).

### D. Time of day modifier

13. **Time of day** MUST act as a modifier layered on top of the venue (not a
    standalone character). It MUST adjust the energy band / overall intensity:
    - **DAY** → lighter (lower the effective energy band).
    - **SUNSET** → mid / warming up (slight increase vs day).
    - **NIGHT** → stronger (raise the band; aligns with club intensity).
    - **LATE_NIGHT** → strongest + nudge character toward Afterparty
      (more hypnotic, lower vocals).
14. The modifier MUST be bounded so it never inverts the band
    (`min_energy <= max_energy` always holds).

### E. Removals & UI

15. **Crowd state** MUST be removed from the UI and from the profile inputs.
16. **Desired energy** MUST be removed from the UI and from the profile inputs.
17. **PRIVATE_PARTY** MUST be removed from the Venue options.
18. Removed fields MUST NOT break the database or existing saved data (leave
    dormant columns with safe defaults rather than forcing a destructive migration).
19. The context form MUST keep: Preset, Target duration, **Venue**,
    **Time of day**, **Peak strategy**, Energy min/max sliders, and the new
    **Strict character** checkbox.
20. Re-generating after changing Venue or Time of day MUST visibly change the set.

## 5. Non-Goals (Out of Scope)

- No genre/mood AI model or external tagging service — descriptors stay local
  and heuristic/DSP-based.
- No new audio descriptors beyond `harmonic_ratio` in this iteration (vocal
  detection stays a placeholder).
- No change to the M3U/CSV export formats, Rekordbox import, or the beam-search
  ordering algorithm itself (only the *scoring inputs* change).
- Crowd state and Desired energy are not being re-designed — they are removed.
- Final BAR / BEACH / OUTDOOR character values are not finalized here (5A).

## 6. Design Considerations

- The Venue character profiles SHOULD live as plain data (a table/dict in
  `planning/context_profiles.py` or a new `planning/venue_profiles.py`), so they
  are easy to read and tweak without touching the scoring code.
- The "Strict character" checkbox sits with the other context controls in the
  left form; keep the form's current 2-column layout.
- Consider showing the active venue's character as a one-line hint under the
  Venue dropdown (e.g. "melodic · lounge · gentle") so the DJ understands what
  it will do. (Nice-to-have, not required.)
- The library table could optionally show a small "melodic/percussive" indicator
  per track once `harmonic_ratio` exists. (Nice-to-have.)

## 7. Technical Considerations

- `TrackFeatures` (dataclass), `track_features` schema, the librosa extractor,
  the heuristic scorer, and `derive_energy_dependent` all need the new
  `harmonic_ratio` field threaded through.
- HPSS on a 90s window is more expensive than current features; keep the
  existing analysis window and reuse the already-loaded signal (don't load the
  file twice).
- Character-fit should slot into `track_selector.compute_position_score` as the
  `context_fit` term (currently a generic restaurant-safety blend) — replace it
  with venue-aware character-fit so weights stay summed to 1.0.
- The harshness ceiling change touches `beam_search_planner._drop_over_cap_tracks`
  (use the venue ceiling instead of the global constant).
- Time-of-day modifier should adjust the profile's energy band *before* the
  energy curve + adaptive normalization run, so everything downstream stays
  consistent.
- Keep everything deterministic (no randomness) — tests rely on it.
- Add tests: venue changes selection character; strict excludes off-character;
  time-of-day shifts energy; removed fields don't break load/generate.

## 8. Success Metrics

1. On the DJ's own library, a **Restaurant** set has a measurably higher mean
   `harmonic_ratio` and lower mean energy than a **Club** set generated from the
   same library.
2. An **Afterparty** set has measurably lower mean `vocal_density` than a Club set.
3. Switching **Time of day** from Day to Night raises the set's mean energy.
4. With **Strict ON**, every selected track is above the character-fit threshold
   (or the planner logged a soft fallback).
5. Subjective: the DJ accepts the opener and overall character without manual
   energy overrides more often than before.

## 9. Open Questions

1. Are **Venue** and **Time of day** both worth keeping long-term, or should they
   eventually merge into combined "moment" presets? (Kept separate for now per 3A.)
2. Should the harshness ceiling per venue be a single number, or derived from the
   venue's energy profile? (Implementation detail — decide at build time.)
3. Should `harmonic_ratio` also influence **transitions** (e.g. avoid jumping
   melodic → banging percussive), or only selection in this iteration?
   (Selection-only for this iteration unless decided otherwise.)

*Resolved during clarification: BAR/BEACH/OUTDOOR character (§4.6), Strict
threshold = 0.4 (§4.12).*
