"""Tests for the deterministic heuristic feature scorer.

These tests run with NO real audio files and NO librosa installed. They pin the
contract that:

* ``score_from_metadata`` is deterministic (same Track -> identical features),
* every feature field is within the inclusive range [0, 1],
* an energetic, high-BPM track scores higher ``energy_score`` than a slow
  ambient/lounge track,
* ``restaurant_safety_score`` decreases as a track's energy rises,
* and the whole ``analysis`` package imports cleanly without librosa.
"""

from __future__ import annotations

import dataclasses

from dj_set_planner.analysis.heuristic_scorer import score_from_metadata
from dj_set_planner.domain.models import Track, TrackFeatures

# All numeric fields of TrackFeatures except the integer track_id.
_FEATURE_FIELDS = [
    f.name
    for f in dataclasses.fields(TrackFeatures)
    if f.name != "track_id"
]


def _track(
    track_id: int,
    file_path: str,
    *,
    genre: str | None = None,
    bpm: float | None = None,
    musical_key: str | None = None,
    camelot_key: str | None = None,
    duration_seconds: int | None = None,
) -> Track:
    """Build a minimal Track for scoring."""

    return Track(
        id=track_id,
        file_path=file_path,
        title=f"Track {track_id}",
        genre=genre,
        bpm=bpm,
        musical_key=musical_key,
        camelot_key=camelot_key,
        duration_seconds=duration_seconds,
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_scoring_is_deterministic() -> None:
    """Same input Track yields identical features across repeated calls."""

    track = _track(
        1,
        "/library/tech_house/track_a.mp3",
        genre="Tech House",
        bpm=124.0,
        musical_key="Am",
        camelot_key="8A",
        duration_seconds=360,
    )

    first = score_from_metadata(track)
    for _ in range(5):
        again = score_from_metadata(track)
        assert again == first  # dataclass equality over all fields


def test_two_distinct_paths_get_distinct_spread() -> None:
    """Identical tags but different file paths differ via the stable jitter.

    This guards the "stable hash of file_path for spread" contract: the jitter
    must actually depend on the path (so same-genre tracks separate), while
    still being deterministic per path.
    """

    a = _track(1, "/library/house/a.mp3", genre="House", bpm=120.0)
    b = _track(2, "/library/house/b.mp3", genre="House", bpm=120.0)

    fa = score_from_metadata(a)
    fb = score_from_metadata(b)
    # At least one jittered field should differ between the two paths.
    assert (fa.energy_score, fa.groove_score, fa.mixability_score) != (
        fb.energy_score,
        fb.groove_score,
        fb.mixability_score,
    )


# ---------------------------------------------------------------------------
# Range invariants
# ---------------------------------------------------------------------------
def test_all_fields_within_unit_interval() -> None:
    """Every feature field stays within [0, 1] across varied inputs."""

    tracks = [
        _track(1, "/l/ambient/a.flac", genre="Ambient", bpm=70.0, duration_seconds=600),
        _track(2, "/l/dnb/b.wav", genre="Drum and Bass", bpm=174.0, duration_seconds=300),
        _track(3, "/l/lounge/c.mp3", genre="Lounge", bpm=None, duration_seconds=None),
        _track(4, "/l/unknown/d.aiff", genre=None, bpm=0.0),
        _track(5, "/l/techno/e.mp3", genre="Techno", bpm=132.0, camelot_key="9B"),
        _track(6, "/l/deep/f.m4a", genre="Deep House", bpm=118.0, camelot_key="4A"),
        _track(7, "/l/weird/g.ogg", genre="Polka", bpm=-5.0),
    ]
    for t in tracks:
        feats = score_from_metadata(t)
        for field_name in _FEATURE_FIELDS:
            value = getattr(feats, field_name)
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0, f"{field_name}={value} out of range for {t.file_path}"


def test_track_id_propagates_and_handles_none() -> None:
    """track_id is carried through; a None id maps to 0 (default)."""

    assert score_from_metadata(_track(42, "/l/x.mp3", genre="House")).track_id == 42
    no_id = Track(id=None, file_path="/l/y.mp3", genre="House")
    assert score_from_metadata(no_id).track_id == 0


# ---------------------------------------------------------------------------
# Energy ordering: energetic/high-BPM > slow/ambient
# ---------------------------------------------------------------------------
def test_energetic_genre_scores_higher_energy_than_ambient() -> None:
    """A fast, energetic-genre track out-scores a slow ambient track on energy."""

    energetic = _track(
        1, "/library/techno/peak.mp3", genre="Techno", bpm=130.0
    )
    ambient = _track(
        2, "/library/ambient/calm.mp3", genre="Ambient", bpm=72.0
    )

    e_feats = score_from_metadata(energetic)
    a_feats = score_from_metadata(ambient)

    assert e_feats.energy_score > a_feats.energy_score
    # The gap should be clearly meaningful, not a jitter coin-flip.
    assert e_feats.energy_score - a_feats.energy_score > 0.2


def test_lounge_track_is_more_restaurant_safe_than_techno() -> None:
    """Low-energy lounge is safer for a restaurant than high-energy techno."""

    lounge = _track(1, "/library/lounge/soft.mp3", genre="Lounge", bpm=95.0)
    techno = _track(2, "/library/techno/hard.mp3", genre="Techno", bpm=130.0)

    assert (
        score_from_metadata(lounge).restaurant_safety_score
        > score_from_metadata(techno).restaurant_safety_score
    )


# ---------------------------------------------------------------------------
# restaurant_safety_score decreases as energy rises
# ---------------------------------------------------------------------------
def test_restaurant_safety_decreases_as_energy_rises() -> None:
    """Across a BPM sweep (same genre/path family), higher energy => lower safety.

    We hold genre and brightness influences roughly constant by reusing one
    genre and one file path, varying only BPM. Energy must rise monotonically
    with BPM and safety must fall as energy rises.
    """

    # Same path so the deterministic jitter is identical across the sweep,
    # isolating the BPM -> energy -> safety relationship.
    path = "/library/house/sweep.mp3"
    bpms = [95.0, 105.0, 115.0, 125.0, 132.0]

    pairs = []
    for bpm in bpms:
        feats = score_from_metadata(
            Track(id=1, file_path=path, genre="House", bpm=bpm)
        )
        pairs.append((feats.energy_score, feats.restaurant_safety_score))

    # Energy strictly increases with BPM in this band.
    energies = [e for e, _ in pairs]
    assert energies == sorted(energies)
    assert energies[0] < energies[-1]

    # Safety strictly decreases as energy increases.
    safeties = [s for _, s in pairs]
    assert safeties == sorted(safeties, reverse=True)
    assert safeties[0] > safeties[-1]


def test_safety_is_monotonic_in_energy_general() -> None:
    """Directly: a higher-energy track is never *safer* than a lower-energy one.

    Build two tracks on the same path differing only in BPM so the brightness/
    jitter terms match; the higher-BPM (higher-energy) one must be less safe.
    """

    path = "/library/x/same.mp3"
    low = score_from_metadata(Track(id=1, file_path=path, genre="House", bpm=98.0))
    high = score_from_metadata(Track(id=1, file_path=path, genre="House", bpm=128.0))

    assert high.energy_score > low.energy_score
    assert high.restaurant_safety_score < low.restaurant_safety_score


# ---------------------------------------------------------------------------
# Package import safety (no librosa required)
# ---------------------------------------------------------------------------
def test_analysis_package_imports_without_librosa() -> None:
    """The analysis package and its extractors import without librosa present.

    ``get_extractor()`` must degrade to the always-available HeuristicExtractor
    when librosa is absent, and that extractor must produce valid features.
    """

    import dj_set_planner.analysis as analysis
    from dj_set_planner.analysis.feature_extractor import HeuristicExtractor

    # Importing the extractor modules must not require librosa/numpy.
    import dj_set_planner.analysis.librosa_extractor as lib_mod
    import dj_set_planner.analysis.essentia_extractor as ess_mod

    # Essentia is always unavailable in the MVP.
    assert ess_mod.EssentiaFeatureExtractor().is_available() is False

    extractor = analysis.get_extractor()
    # With librosa absent in CI, we expect the heuristic fallback.
    if not lib_mod.LibrosaFeatureExtractor().is_available():
        assert isinstance(extractor, HeuristicExtractor)

    feats = extractor.extract(
        Track(id=7, file_path="/l/house/z.mp3", genre="House", bpm=122.0)
    )
    assert isinstance(feats, TrackFeatures)
    for field_name in _FEATURE_FIELDS:
        assert 0.0 <= getattr(feats, field_name) <= 1.0
