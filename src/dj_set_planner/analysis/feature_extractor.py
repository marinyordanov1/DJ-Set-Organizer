"""Feature-extractor interface and the heuristic wrapper.

``FeatureExtractor`` is the ABC every concrete extractor implements. The
``HeuristicExtractor`` always works (no heavy deps) by delegating to
``heuristic_scorer.score_from_metadata``. It imports that module LAZILY inside
``extract`` because the scorer is implemented in a later phase — the contract
(this interface) must import cleanly today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.models import Track, TrackFeatures


class FeatureExtractor(ABC):
    """Abstract base for anything that turns a :class:`Track` into features.

    ``extract`` takes a whole Track (not just a path) so the extractor can use
    tag-derived bpm/key/genre as priors. It must return features normalized to
    0..1 and must never raise for an analyzable-but-degenerate input — falling
    back to heuristics instead.
    """

    @abstractmethod
    def extract(self, track: Track) -> TrackFeatures:
        """Return :class:`TrackFeatures` (all fields in 0..1) for ``track``."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this extractor's backing engine is usable in this env."""
        raise NotImplementedError


class HeuristicExtractor(FeatureExtractor):
    """Always-available extractor backed by the deterministic heuristic scorer.

    It is the universal fallback: no librosa/numpy required.
    """

    def is_available(self) -> bool:
        return True

    def extract(self, track: Track) -> TrackFeatures:
        # Lazy import: heuristic_scorer is delivered by the Analysis phase.
        from .heuristic_scorer import score_from_metadata

        return score_from_metadata(track)
