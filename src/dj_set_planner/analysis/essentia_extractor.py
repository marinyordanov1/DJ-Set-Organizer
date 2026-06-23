"""Essentia feature extractor — thin stub (DROPPED for the MVP).

Essentia is not part of the MVP. This module exists only to satisfy the
extractor chain in :func:`analysis.get_extractor`. It:

* always reports ``is_available() == False`` (so the factory skips it), and
* if ever called directly, delegates to :class:`LibrosaFeatureExtractor`
  (which itself falls back to the deterministic heuristic on any failure),
  so it never raises and always returns valid :class:`TrackFeatures`.

Importing this module pulls in NO heavy dependencies.
"""

from __future__ import annotations

from ..domain.models import Track, TrackFeatures
from ..utils.logging import get_logger
from .feature_extractor import FeatureExtractor

_log = get_logger(__name__)


class EssentiaFeatureExtractor(FeatureExtractor):
    """Stub extractor: unavailable in the MVP; delegates if invoked."""

    def is_available(self) -> bool:
        """Always ``False`` — Essentia is intentionally dropped for the MVP."""

        return False

    def extract(self, track: Track) -> TrackFeatures:
        """Delegate to librosa (which itself falls back to the heuristic).

        This path is not normally reached because the factory skips
        unavailable extractors, but we keep it correct and non-raising for
        defensive callers.
        """

        # Lazy import keeps essentia_extractor import-light and avoids any
        # circulars at package import time.
        from .librosa_extractor import LibrosaFeatureExtractor

        librosa_extractor = LibrosaFeatureExtractor()
        if librosa_extractor.is_available():
            # extract() already falls back to the heuristic on any error.
            return librosa_extractor.extract(track)

        # Librosa unavailable too: go straight to the deterministic heuristic.
        from .heuristic_scorer import score_from_metadata

        return score_from_metadata(track)
