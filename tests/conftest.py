"""Shared pytest fixtures.

Loads ``tests/fixtures/sample_tracks.json`` (30 synthetic tracks + a parallel
features blob) and exposes them as domain dataclasses. Tests run WITHOUT any
real audio files.
"""

from __future__ import annotations

import json
import os

import pytest

from dj_set_planner.domain.models import EventProfile, Track, TrackFeatures
from dj_set_planner.planning.context_profiles import default_profile

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_tracks.json"
)


def _load_fixture() -> dict:
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def fixture_data() -> dict:
    """Raw parsed fixture JSON (``{"tracks": [...], "features": [...]}``)."""

    return _load_fixture()


@pytest.fixture
def sample_tracks(fixture_data: dict) -> list[Track]:
    """The 30 synthetic tracks as :class:`Track` instances."""

    return [Track(**row) for row in fixture_data["tracks"]]


@pytest.fixture
def sample_features(fixture_data: dict) -> dict[int, TrackFeatures]:
    """Features keyed by ``track_id`` as :class:`TrackFeatures` instances."""

    feats = [TrackFeatures(**row) for row in fixture_data["features"]]
    return {f.track_id: f for f in feats}


@pytest.fixture
def sofia_profile() -> EventProfile:
    """The default Sofia day-party restaurant :class:`EventProfile`."""

    return default_profile()
