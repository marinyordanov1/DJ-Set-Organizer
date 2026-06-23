-- DJ Set Planner SQLite schema.
-- Loaded by db.database.Database on first connection.
-- Foreign keys are enforced (PRAGMA foreign_keys=ON set at connect time).

CREATE TABLE IF NOT EXISTS tracks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT    NOT NULL UNIQUE,
    title            TEXT,
    artist           TEXT,
    album            TEXT,
    genre            TEXT,
    duration_seconds INTEGER,
    bpm              REAL,
    musical_key      TEXT,
    camelot_key      TEXT,
    analyzed_at      TEXT
);

CREATE TABLE IF NOT EXISTS track_features (
    track_id                INTEGER PRIMARY KEY,
    energy_score            REAL NOT NULL DEFAULT 0.5,
    danceability_score      REAL NOT NULL DEFAULT 0.5,
    mood_brightness         REAL NOT NULL DEFAULT 0.5,
    groove_score            REAL NOT NULL DEFAULT 0.5,
    vocal_density           REAL NOT NULL DEFAULT 0.5,
    intro_suitability       REAL NOT NULL DEFAULT 0.5,
    outro_suitability       REAL NOT NULL DEFAULT 0.5,
    peak_potential          REAL NOT NULL DEFAULT 0.5,
    restaurant_safety_score REAL NOT NULL DEFAULT 0.5,
    mixability_score        REAL NOT NULL DEFAULT 0.5,
    harmonic_ratio          REAL NOT NULL DEFAULT 0.5,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

-- Manual per-track energy overrides set by the DJ in the UI. Kept SEPARATE
-- from track_features so re-analysis never clobbers a manual value; the service
-- overlays these on top of the analyzed features at read/plan time.
CREATE TABLE IF NOT EXISTS feature_overrides (
    track_id     INTEGER PRIMARY KEY,
    energy_score REAL,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS event_profiles (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    name                    TEXT NOT NULL,
    venue_type              TEXT NOT NULL,
    time_of_day             TEXT NOT NULL,
    crowd_state             TEXT NOT NULL,
    desired_energy          TEXT NOT NULL,
    peak_strategy           TEXT NOT NULL,
    target_duration_minutes INTEGER NOT NULL,
    min_energy              REAL NOT NULL,
    max_energy              REAL NOT NULL,
    main_peak_energy        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS set_plans (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_profile_id       INTEGER,
    created_at             TEXT,
    total_duration_seconds INTEGER,
    target_duration_seconds INTEGER,
    total_score            REAL,
    FOREIGN KEY (event_profile_id) REFERENCES event_profiles (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS set_plan_tracks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    set_plan_id      INTEGER NOT NULL,
    track_id         INTEGER NOT NULL,
    position         INTEGER NOT NULL,
    role             TEXT,
    transition_score REAL,
    position_score   REAL,
    explanation      TEXT,
    is_locked        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (set_plan_id) REFERENCES set_plans (id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dj_constraints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id        INTEGER NOT NULL,
    constraint_type TEXT NOT NULL,
    value           TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_set_plan_tracks_plan ON set_plan_tracks (set_plan_id);
CREATE INDEX IF NOT EXISTS idx_dj_constraints_track ON dj_constraints (track_id);
