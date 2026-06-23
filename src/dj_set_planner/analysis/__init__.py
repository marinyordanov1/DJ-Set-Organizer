"""Analysis package.

Exposes ``get_extractor`` which picks the best AVAILABLE feature extractor:
Essentia (stub, never available in MVP) -> Librosa (optional) -> Heuristic
(always available). All concrete extractors are imported lazily so importing
this package never pulls in librosa/numpy.
"""

from __future__ import annotations

from .feature_extractor import FeatureExtractor, HeuristicExtractor
from ..utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["FeatureExtractor", "HeuristicExtractor", "get_extractor"]


def get_extractor(prefer: str | None = None) -> FeatureExtractor:
    """Return the best available :class:`FeatureExtractor`.

    Resolution order (subject to ``prefer``):
      1. Essentia (stub) — only if it reports available (never, in the MVP).
      2. Librosa — only if its optional deps import successfully.
      3. Heuristic — always available, deterministic fallback.

    ``prefer`` may be one of ``"essentia"``, ``"librosa"``, ``"heuristic"`` to
    bias selection; if the preferred engine is unavailable we degrade
    gracefully down the chain. Concrete extractors are imported lazily so a
    missing optional dependency never breaks this call.
    """

    prefer = (prefer or "").strip().lower() or None

    def _try_essentia() -> FeatureExtractor | None:
        try:
            from .essentia_extractor import EssentiaFeatureExtractor

            ex = EssentiaFeatureExtractor()
            return ex if ex.is_available() else None
        except Exception:  # pragma: no cover - defensive, deps absent
            _log.debug("Essentia extractor unavailable", exc_info=True)
            return None

    def _try_librosa() -> FeatureExtractor | None:
        try:
            from .librosa_extractor import LibrosaFeatureExtractor

            ex = LibrosaFeatureExtractor()
            return ex if ex.is_available() else None
        except Exception:
            _log.debug("Librosa extractor unavailable", exc_info=True)
            return None

    # Honour an explicit preference first (still degrading if unavailable).
    if prefer == "heuristic":
        return HeuristicExtractor()
    if prefer == "librosa":
        return _try_librosa() or HeuristicExtractor()
    if prefer == "essentia":
        return _try_essentia() or _try_librosa() or HeuristicExtractor()

    # Default best-available chain.
    return _try_essentia() or _try_librosa() or HeuristicExtractor()
