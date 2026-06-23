"""Recursive audio-file scanner.

Walks a folder tree and returns the absolute paths of every supported audio
file, sorted, ignoring dotfiles (and dot-directories). This is the cheap
"discovery" step — no tags are read and no audio is decoded here; that is the
job of :mod:`metadata_reader` and the feature extractors.
"""

from __future__ import annotations

import os

from ..utils.logging import get_logger

_log = get_logger(__name__)

# Supported audio extensions (lower-cased, including the leading dot). These
# are exactly the formats mutagen can read tags/duration for in this MVP.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
)


def _is_supported(name: str) -> bool:
    """True if ``name`` has a supported audio extension (case-insensitive)."""

    _, ext = os.path.splitext(name)
    return ext.lower() in SUPPORTED_EXTENSIONS


def scan_folder(folder: str) -> list[str]:
    """Return sorted absolute paths of supported audio files under ``folder``.

    The walk is recursive. Dotfiles and dot-directories (names starting with
    ``"."``, e.g. ``.DS_Store`` or ``.Trash``) are ignored so we never pick up
    OS/junk files. Unsupported extensions are skipped. A missing or
    unreadable folder yields an empty list (logged, never raised) so callers
    stay robust.
    """

    if not folder or not os.path.isdir(folder):
        _log.warning("scan_folder: %r is not a directory; returning []", folder)
        return []

    root_abs = os.path.abspath(folder)
    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Prune dot-directories in place so os.walk does not descend into them.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            if filename.startswith("."):
                continue  # ignore dotfiles (e.g. ._resource forks, .DS_Store)
            if not _is_supported(filename):
                continue
            results.append(os.path.abspath(os.path.join(dirpath, filename)))

    results.sort()
    return results
