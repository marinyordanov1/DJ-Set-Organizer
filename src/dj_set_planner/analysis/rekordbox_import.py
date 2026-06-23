"""Rekordbox collection-XML import.

Rekordbox can export the whole collection (and playlists) to an XML file with
the structure::

    <DJ_PLAYLISTS Version="1.0.0">
      <PRODUCT .../>
      <COLLECTION Entries="N">
        <TRACK TrackID="1" Name="..." Artist="..." Album="..." Genre="..."
               AverageBpm="124.00" Tonality="Am" TotalTime="312"
               Location="file://localhost/Users/dj/Music/track%20one.mp3"/>
        ...
      </COLLECTION>
      <PLAYLISTS> ... </PLAYLISTS>
    </DJ_PLAYLISTS>

This module reads only the ``COLLECTION/TRACK`` elements. Rekordbox already
analyses BPM and key, so importing it lets us reuse that data instead of
running librosa — and it works for tracks we cannot decode locally.

Two public functions:

``parse_rekordbox_xml(xml_path)``
    Parse the XML into a list of plain dicts (one per ``TRACK``).

``apply_rekordbox_to_tracks(tracks, xml_path)``
    Match parsed rows against an in-memory list of :class:`Track` objects and
    fill in ``bpm`` / ``musical_key`` / ``camelot_key`` / ``duration_seconds``
    where the Track is currently missing them.

Both tolerate a missing or malformed file: they log the problem and return an
empty list / a count of ``0`` rather than raising.
"""

from __future__ import annotations

import os
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from ..domain.models import Track
from ..utils.camelot import to_camelot
from ..utils.logging import get_logger

_log = get_logger(__name__)


def _location_to_path(location: str | None) -> str | None:
    """Convert a Rekordbox ``Location`` value into a real filesystem path.

    Rekordbox stores locations as ``file://`` URLs, URL-encoded (spaces become
    ``%20`` etc.) and usually with a ``localhost`` (or empty) host, e.g.::

        file://localhost/Users/dj/Music/track%20one.mp3
        file:///Users/dj/Music/track%20one.mp3

    Returns an absolute path, or ``None`` if ``location`` is empty.
    """

    if not location:
        return None

    raw = location.strip()
    if not raw:
        return None

    if raw.startswith("file:"):
        parsed = urlparse(raw)
        # ``parsed.path`` is the URL-encoded path component; ``netloc`` is the
        # host ("localhost" or ""). We ignore the host entirely. unquote turns
        # "%20" back into a space, "%C3%A9" back into "é", etc.
        path = unquote(parsed.path)
    else:
        # Not a URL — assume it is already a plain path (be lenient).
        path = unquote(raw)

    if not path:
        return None

    # On Windows, Rekordbox produces paths like "/C:/Music/..."; strip the
    # leading slash so it becomes a valid drive path. Harmless on POSIX.
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]

    return path


def _to_float(value: str | None) -> float | None:
    """Parse a numeric attribute to ``float``; return ``None`` on failure/zero.

    Rekordbox writes ``AverageBpm="0.00"`` for un-analysed tracks; we treat a
    zero/blank BPM as "unknown" so we never overwrite real data with 0.
    """

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


def _to_int_seconds(value: str | None) -> int | None:
    """Parse ``TotalTime`` (whole seconds) to ``int``; ``None`` on failure/zero."""

    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # Some exporters write a float; round to nearest whole second.
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


def _clean_str(value: str | None) -> str | None:
    """Return a trimmed string, or ``None`` when empty."""

    if value is None:
        return None
    value = value.strip()
    return value or None


def _iter_track_elements(root: ET.Element):
    """Yield every ``TRACK`` element under the ``COLLECTION`` node.

    We look specifically inside ``COLLECTION`` so we don't accidentally pick up
    the ``TRACK`` reference stubs that live under ``PLAYLISTS`` (those only
    carry a ``Key`` attribute pointing back at a TrackID).
    """

    # The root is usually <DJ_PLAYLISTS>; COLLECTION is a direct child. Fall
    # back to a recursive search to tolerate minor structural variations.
    collection = root.find("COLLECTION")
    if collection is None:
        collection = root.find(".//COLLECTION")
    if collection is None:
        # No COLLECTION node — nothing we recognise to import.
        return
    yield from collection.findall("TRACK")


def parse_rekordbox_xml(xml_path: str) -> list[dict]:
    """Parse a Rekordbox collection XML into a list of dicts.

    Each dict has the keys ``file_path, title, artist, album, genre, bpm,
    musical_key, camelot_key, duration_seconds`` — any value may be ``None``.

    Tolerates a missing path or malformed XML: logs the issue and returns an
    empty list. Rows with no resolvable ``Location`` are skipped (a track we
    cannot match to a file is useless to us).
    """

    if not xml_path or not os.path.isfile(xml_path):
        _log.warning("Rekordbox XML not found: %r", xml_path)
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        # Malformed XML — log and bail (never raise to the caller).
        _log.warning("Failed to parse Rekordbox XML %r: %s", xml_path, exc)
        return []
    except OSError as exc:
        _log.warning("Could not read Rekordbox XML %r: %s", xml_path, exc)
        return []

    rows: list[dict] = []
    for el in _iter_track_elements(root):
        location = el.get("Location")
        file_path = _location_to_path(location)
        if file_path is None:
            # Skip entries we cannot tie to an actual file.
            continue

        # Rekordbox uses "Tonality" for the musical key; some exports use "Key".
        musical_key = _clean_str(el.get("Tonality")) or _clean_str(el.get("Key"))
        camelot_key = to_camelot(musical_key) if musical_key else None

        rows.append(
            {
                "file_path": file_path,
                "title": _clean_str(el.get("Name")),
                "artist": _clean_str(el.get("Artist")),
                "album": _clean_str(el.get("Album")),
                "genre": _clean_str(el.get("Genre")),
                "bpm": _to_float(el.get("AverageBpm")),
                "musical_key": musical_key,
                "camelot_key": camelot_key,
                "duration_seconds": _to_int_seconds(el.get("TotalTime")),
            }
        )

    _log.info("Parsed %d track(s) from Rekordbox XML %r", len(rows), xml_path)
    return rows


def apply_rekordbox_to_tracks(tracks: list[Track], xml_path: str) -> int:
    """Fill missing analysis fields on ``tracks`` from a Rekordbox XML.

    Matching strategy (case-insensitive):
      1. Exact full-path match (normalised case) — strongest signal.
      2. Otherwise, match by file *basename* (filename only) — handles the
         common case where the library was moved to a different folder.

    For each matched Track we fill ``bpm``, ``musical_key``, ``camelot_key``
    and ``duration_seconds`` **only when the Track is currently missing them**
    (we never clobber data the Track already has). ``camelot_key`` is also
    derived from the Track's own key if Rekordbox lacked one.

    Returns the number of Tracks that matched a Rekordbox row. Tolerates a
    missing/garbage XML by returning ``0``.
    """

    rows = parse_rekordbox_xml(xml_path)
    if not rows:
        return 0

    # Build lookup tables from the parsed rows. Full-path keys take precedence
    # over basename keys, so we keep them separate.
    by_path: dict[str, dict] = {}
    by_basename: dict[str, dict] = {}
    for row in rows:
        rp = row["file_path"]
        if not rp:
            continue
        by_path[rp.lower()] = row
        base = os.path.basename(rp).lower()
        if base:
            # First write wins for a basename collision; this is deterministic
            # given the input ordering and avoids surprising overwrites.
            by_basename.setdefault(base, row)

    matched = 0
    for track in tracks:
        if not track.file_path:
            continue

        row = by_path.get(track.file_path.lower())
        if row is None:
            base = os.path.basename(track.file_path).lower()
            row = by_basename.get(base)
        if row is None:
            continue

        matched += 1

        # Fill only when the Track is missing the field. ``bpm``/``duration``
        # treat a falsy/zero value as missing too.
        if not track.bpm and row["bpm"] is not None:
            track.bpm = row["bpm"]
        if not track.duration_seconds and row["duration_seconds"] is not None:
            track.duration_seconds = row["duration_seconds"]
        if not track.musical_key and row["musical_key"] is not None:
            track.musical_key = row["musical_key"]

        # Camelot: prefer Rekordbox's derived value, else derive from whatever
        # musical_key the Track now has.
        if not track.camelot_key:
            if row["camelot_key"] is not None:
                track.camelot_key = row["camelot_key"]
            elif track.musical_key:
                track.camelot_key = to_camelot(track.musical_key)

    _log.info(
        "Applied Rekordbox data to %d/%d track(s) from %r",
        matched,
        len(tracks),
        xml_path,
    )
    return matched
