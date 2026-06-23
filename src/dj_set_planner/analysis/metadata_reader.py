"""Tag-based metadata reader (mutagen).

Reads title/artist/album/genre/duration plus BPM and musical key from an audio
file's tags across the formats mutagen supports (MP3/ID3, FLAC, MP4/m4a, OGG
Vorbis, WAV/AIFF best-effort). It NEVER raises on a bad/unreadable file: it
logs the problem and returns a :class:`Track` populated with whatever could be
recovered, falling back to the filename stem for the title.

Why two passes?
---------------
``mutagen.File(path)`` returns the format-native object, exposing
format-specific frames (ID3 ``TBPM``/``TKEY``, MP4 ``tmpo`` and freeform
atoms, Vorbis comments). ``mutagen.File(path, easy=True)`` gives a normalized
EasyTag mapping (``title``/``artist``/...) but hides BPM/key. We read the easy
mapping for the common text tags and dig into the native tags for BPM/key.
"""

from __future__ import annotations

import os

from ..domain.models import Track
from ..utils.camelot import to_camelot
from ..utils.logging import get_logger

_log = get_logger(__name__)


# Custom tag names commonly used for BPM across taggers / formats. Compared
# case-insensitively. Used for FLAC/OGG (Vorbis) and MP4 freeform atoms.
_BPM_KEYS = ("bpm", "tempo", "tbpm")

# Custom tag names commonly used for the musical key.
_KEY_KEYS = ("key", "initialkey", "initial_key", "tkey")


def _first(value) -> str | None:
    """Return the first non-empty scalar from a mutagen tag value.

    mutagen tag values are usually lists (``["Title"]``); ID3 frames are
    objects whose ``str()`` is the text; MP4 freeform atoms are ``bytes``.
    """

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            got = _first(item)
            if got:
                return got
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - decode is very defensive
            return None
        return text or None
    text = str(value).strip()
    return text or None


def _to_float(text: str | None) -> float | None:
    """Parse a BPM string to float; tolerate ``"128"``, ``"128.0"``, junk."""

    if not text:
        return None
    cleaned = text.strip().replace(",", ".")
    try:
        val = float(cleaned)
    except (TypeError, ValueError):
        return None
    # Guard against absurd values that are clearly not a real BPM.
    if val <= 0 or val > 400:
        return None
    return val


def _easy_get(easy_tags, name: str) -> str | None:
    """Read a normalized EasyTag field (e.g. ``"title"``) safely."""

    if easy_tags is None:
        return None
    try:
        return _first(easy_tags.get(name))
    except Exception:  # pragma: no cover - defensive against odd tag objects
        return None


def _lookup_native(tags, candidate_keys: tuple[str, ...]) -> str | None:
    """Find the first matching tag value across ``candidate_keys``.

    Matches keys case-insensitively and also tolerates the MP4 freeform
    prefix (``"----:com.apple.iTunes:BPM"``) by comparing on the trailing
    component of the atom name.
    """

    if not tags:
        return None

    wanted = {k.lower() for k in candidate_keys}

    try:
        items = list(tags.items())
    except Exception:  # pragma: no cover - some tag containers are odd
        return None

    for raw_key, raw_val in items:
        key_str = str(raw_key).lower()
        # MP4 freeform atoms look like "----:com.apple.iTunes:BPM" — compare
        # on the final ":"-delimited segment too.
        tail = key_str.rsplit(":", 1)[-1]
        if key_str in wanted or tail in wanted:
            got = _first(raw_val)
            if got:
                return got
    return None


def _read_bpm(native, easy) -> float | None:
    """Extract BPM from native (ID3 TBPM / MP4 tmpo / Vorbis) then easy tags."""

    tags = getattr(native, "tags", None)

    # ID3: the TBPM frame. ``tags.get("TBPM")`` returns the frame whose str()
    # is the numeric text.
    raw = _lookup_native(tags, _BPM_KEYS)
    bpm = _to_float(raw)
    if bpm is not None:
        return bpm

    # MP4: the integer ``tmpo`` atom (a list of ints).
    if tags is not None:
        try:
            if "tmpo" in tags:
                return _to_float(_first(tags["tmpo"]))
        except Exception:  # pragma: no cover - defensive
            pass

    # EasyTag exposes "bpm" for some formats.
    return _to_float(_easy_get(easy, "bpm"))


def _read_key(native, easy) -> str | None:
    """Extract the musical key from native (ID3 TKEY / Vorbis / MP4) tags."""

    tags = getattr(native, "tags", None)
    raw = _lookup_native(tags, _KEY_KEYS)
    if raw:
        return raw
    # Some EasyMP4/EasyID3 setups register a "key"/"initialkey" alias.
    for name in ("key", "initialkey"):
        got = _easy_get(easy, name)
        if got:
            return got
    return None


def _read_duration(native) -> int | None:
    """Return integer duration in seconds from the audio stream info."""

    info = getattr(native, "info", None)
    length = getattr(info, "length", None)
    if length is None:
        return None
    try:
        seconds = int(round(float(length)))
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def read_metadata(file_path: str) -> Track:
    """Read tag metadata for ``file_path`` and return a :class:`Track`.

    Best-effort and exception-safe: any failure to open or parse the file is
    logged and yields a Track with ``id=None`` and the filename stem as the
    title. Tag fields that are present are filled in; ``camelot_key`` is set
    via :func:`utils.camelot.to_camelot` whenever a key tag was found.
    """

    abs_path = os.path.abspath(file_path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]

    # Import mutagen lazily and defensively so a packaging issue degrades to a
    # filename-only Track rather than crashing the scan.
    try:
        import mutagen
    except Exception:  # pragma: no cover - mutagen is a core dep
        _log.exception("mutagen import failed; returning filename-only Track")
        return Track(id=None, file_path=abs_path, title=stem)

    native = None
    easy = None
    try:
        native = mutagen.File(abs_path)
    except Exception:
        # Corrupt/unsupported container: keep going with a filename-only Track.
        _log.exception("Failed to read tags from %s", abs_path)
        return Track(id=None, file_path=abs_path, title=stem)

    if native is None:
        # mutagen returned None: not a recognized audio file.
        _log.warning("Unrecognized audio file (mutagen returned None): %s", abs_path)
        return Track(id=None, file_path=abs_path, title=stem)

    # EasyTag normalized view for the common text fields. Failure here is
    # non-fatal — we still have the native object for duration/bpm/key.
    try:
        easy = mutagen.File(abs_path, easy=True)
    except Exception:
        _log.debug("Easy-tag read failed for %s; using native tags only", abs_path)
        easy = None

    title = _easy_get(easy, "title")
    artist = _easy_get(easy, "artist")
    album = _easy_get(easy, "album")
    genre = _easy_get(easy, "genre")

    # Native-tag fallback for files where the easy view is empty (e.g. AIFF/WAV
    # or formats without an EasyTag mapping).
    tags = getattr(native, "tags", None)
    if title is None:
        title = _lookup_native(tags, ("title", "tit2"))
    if artist is None:
        artist = _lookup_native(tags, ("artist", "tpe1"))
    if album is None:
        album = _lookup_native(tags, ("album", "talb"))
    if genre is None:
        genre = _lookup_native(tags, ("genre", "tcon"))

    duration_seconds = _read_duration(native)
    bpm = _read_bpm(native, easy)
    musical_key = _read_key(native, easy)
    camelot_key = to_camelot(musical_key) if musical_key else None

    return Track(
        id=None,
        file_path=abs_path,
        title=title or stem,  # title falls back to the filename stem
        artist=artist,
        album=album,
        genre=genre,
        duration_seconds=duration_seconds,
        bpm=bpm,
        musical_key=musical_key,
        camelot_key=camelot_key,
        analyzed_at=None,
    )
