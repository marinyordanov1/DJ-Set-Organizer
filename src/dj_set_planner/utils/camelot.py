"""Camelot-wheel key utilities.

Parses a wide range of musical-key notations into Camelot notation and
provides harmonic-distance / compatibility scoring used by the transition
scorer.

THE CAMELOT WHEEL
=================
The wheel has 12 numbered positions (1..12). Each position has an inner "A"
ring (minor keys) and an outer "B" ring (major keys). Adjacent numbers are a
perfect fifth apart; the same number on A and B are relative minor/major.

    Camelot   Major (B)    Minor (A)
    -------   ---------    ---------
    1B  B major          1A  G#/Ab minor
    2B  F#/Gb major      2A  D#/Eb minor
    3B  Db/C# major      3A  A#/Bb minor
    4B  Ab/G# major      4A  F minor
    5B  Eb/D# major      5A  C minor
    6B  Bb/A# major      6A  G minor
    7B  F major          7A  D minor
    8B  C major          8A  A minor
    9B  G major          9A  E minor
    10B D major          10A B minor
    11B A major          11A F#/Gb minor
    12B E major          12A C#/Db minor

Harmonic "compatible" moves (energy-preserving):
  * same key              (e.g. 8A -> 8A)
  * +/- 1 on the number   (e.g. 8A -> 7A or 9A)  [perfect fifth neighbours]
  * switch ring same num  (e.g. 8A -> 8B)        [relative major/minor]
"""

from __future__ import annotations

from .logging import get_logger

_log = get_logger(__name__)

# Map a normalized pitch-class name to its semitone index (0..11).
# We collapse enharmonic spellings to a single index.
_PITCH_INDEX: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}

# Camelot numbers indexed by (semitone, is_minor). Built from the wheel table
# above. _CAMELOT_NUMBER[(semitone_index, is_minor)] -> camelot number 1..12.
# Major (B ring):
_MAJOR_TO_NUMBER = {
    11: 1,   # B major
    6: 2,    # F#/Gb major
    1: 3,    # Db/C# major
    8: 4,    # Ab/G# major
    3: 5,    # Eb/D# major
    10: 6,   # Bb/A# major
    5: 7,    # F major
    0: 8,    # C major
    7: 9,    # G major
    2: 10,   # D major
    9: 11,   # A major
    4: 12,   # E major
}
# Minor (A ring):
_MINOR_TO_NUMBER = {
    8: 1,    # G#/Ab minor
    3: 2,    # D#/Eb minor
    10: 3,   # A#/Bb minor
    5: 4,    # F minor
    0: 5,    # C minor
    7: 6,    # G minor
    2: 7,    # D minor
    9: 8,    # A minor
    4: 9,    # E minor
    11: 10,  # B minor
    6: 11,   # F#/Gb minor
    1: 12,   # C#/Db minor
}

# Tokens that, when present, force a minor reading.
_MINOR_TOKENS = ("MIN", "MINOR", "-", "M")  # NOTE: bare "m" handled specially


def _normalize_accidentals(text: str) -> str:
    """Normalize unicode sharps/flats to ASCII ``#`` / ``b``."""

    return (
        text.replace("♯", "#")  # ♯
        .replace("♭", "b")  # ♭
        .replace("−", "-")  # − minus
    )


def _parse_camelot_token(token: str) -> str | None:
    """Parse a token already in Camelot form (e.g. ``8A``, ``11B``)."""

    token = token.strip().upper()
    if len(token) < 2:
        return None
    ring = token[-1]
    if ring not in ("A", "B"):
        return None
    num_part = token[:-1]
    if not num_part.isdigit():
        return None
    num = int(num_part)
    if 1 <= num <= 12:
        return f"{num}{ring}"
    return None


def to_camelot(musical_key: str | None) -> str | None:
    """Parse a musical-key string into Camelot notation (e.g. ``"8A"``).

    Accepts forms such as: ``"Am"``, ``"A min"``, ``"Amin"``, ``"A minor"``,
    ``"8A"``, ``"A#"``, ``"Bbm"``, ``"F#min"``, ``"C"``, ``"C maj"``,
    ``"Open key 1m"`` (best effort). Returns ``None`` when it cannot be parsed.
    """

    if not musical_key:
        return None

    raw = _normalize_accidentals(str(musical_key)).strip()
    if not raw:
        return None

    # 1) Already Camelot? (also handles "8A" embedded with spaces)
    compact = raw.replace(" ", "")
    direct = _parse_camelot_token(compact)
    if direct is not None:
        return direct

    # Strip a leading "open key"/"openkey" label if present.
    upper = raw.upper()
    for prefix in ("OPEN KEY", "OPENKEY", "OPEN"):
        if upper.startswith(prefix):
            upper = upper[len(prefix):].strip()
            raw = raw[-len(upper):] if upper else raw

    # Open Key notation: "1m"/"1d" etc -> map d(=major)->B, m(=minor)->A,
    # with Open Key number == Camelot number.
    ok = upper.replace(" ", "")
    if len(ok) >= 2 and ok[:-1].isdigit():
        suffix = ok[-1]
        num = int(ok[:-1])
        if 1 <= num <= 12 and suffix in ("M", "D"):
            ring = "A" if suffix == "M" else "B"
            return f"{num}{ring}"

    # 2) Pitch-class + optional accidental + optional quality.
    s = raw.strip()
    if not s:
        return None

    pitch = s[0].upper()
    if pitch not in "ABCDEFG":
        return None

    idx = 1
    # Accidental immediately after the letter.
    if idx < len(s) and s[idx] in ("#", "b"):
        pitch += "#" if s[idx] == "#" else "B"  # store flats as uppercase 'B'
        idx += 1

    semitone = _PITCH_INDEX.get(pitch.upper())
    if semitone is None:
        return None

    # Remaining text determines major vs minor. We accept ONLY a bounded
    # whitelist of quality suffixes; anything else means this was not really a
    # key string (e.g. the word "garbage" starts with a valid pitch letter)
    # and we return None rather than guessing.
    rest_raw = s[idx:].strip().replace(".", "")
    rest = rest_raw.upper()

    # Whole-string quality matches (case-insensitive).
    _MAJOR_WORDS = {"", "MAJ", "MAJOR", "MA"}
    _MINOR_WORDS = {"MIN", "MINOR", "MI", "-"}

    if rest in _MAJOR_WORDS:
        is_minor = False
    elif rest in _MINOR_WORDS:
        is_minor = True
    elif rest in ("M",):
        # Ambiguous bare "m"/"M": lowercase means minor, uppercase means major.
        is_minor = rest_raw == "m"
    else:
        # Unrecognized suffix -> not a parseable key.
        _log.debug("Unrecognized key quality in %r; treating as unknown", musical_key)
        return None

    table = _MINOR_TO_NUMBER if is_minor else _MAJOR_TO_NUMBER
    number = table.get(semitone)
    if number is None:
        return None
    ring = "A" if is_minor else "B"
    return f"{number}{ring}"


def _split_camelot(c: str) -> tuple[int, str] | None:
    """Split ``"8A"`` -> ``(8, "A")``; return None if malformed."""

    c = c.strip().upper()
    if len(c) < 2 or c[-1] not in ("A", "B") or not c[:-1].isdigit():
        return None
    return int(c[:-1]), c[-1]


def camelot_distance(a: str | None, b: str | None) -> int | None:
    """Return harmonic distance in steps on the wheel, or ``None`` if unknown.

    Conventions:
      * same key                       -> 0
      * relative major/minor (same num)-> 0  (e.g. 8A vs 8B)
      * +/- 1 on the number, same ring -> 1  (perfect-fifth neighbours)
      * otherwise -> minimal number of single-number steps around the 12-wheel,
        plus 1 if the rings differ and the numbers also differ.
    """

    ca = to_camelot(a)
    cb = to_camelot(b)
    if ca is None or cb is None:
        return None

    pa = _split_camelot(ca)
    pb = _split_camelot(cb)
    if pa is None or pb is None:
        return None

    num_a, ring_a = pa
    num_b, ring_b = pb

    # Wheel is circular over 12 numbers: minimal hop count.
    raw = abs(num_a - num_b)
    number_steps = min(raw, 12 - raw)

    if number_steps == 0:
        # Same number: 0 whether same ring (identical) or relative maj/min.
        return 0

    ring_penalty = 0 if ring_a == ring_b else 1
    return number_steps + ring_penalty


def key_compatibility(a: str | None, b: str | None) -> float:
    """Return a 0..1 harmonic-compatibility score between two keys.

    Scoring (tuned for smooth, energy-preserving day-party transitions):
      * same key                              -> 1.00
      * relative major/minor (same number)    -> 0.85
      * perfect-fifth neighbour (+/-1, ring)  -> 0.85
      * two number steps away                 -> 0.50
      * either key unknown                    -> 0.60 (neutral, don't punish)
      * everything else (clashing)            -> 0.20
    """

    ca = to_camelot(a)
    cb = to_camelot(b)
    if ca is None or cb is None:
        return 0.6  # neutral: missing key data should not dominate scoring

    if ca == cb:
        return 1.0

    pa = _split_camelot(ca)
    pb = _split_camelot(cb)
    if pa is None or pb is None:
        return 0.6

    num_a, ring_a = pa
    num_b, ring_b = pb

    raw = abs(num_a - num_b)
    number_steps = min(raw, 12 - raw)

    # Relative major/minor: same number, different ring.
    if number_steps == 0 and ring_a != ring_b:
        return 0.85

    # Perfect-fifth neighbours: one number step, same ring.
    if number_steps == 1 and ring_a == ring_b:
        return 0.85

    # Two steps on the same ring is a passable, more noticeable move.
    if number_steps == 2 and ring_a == ring_b:
        return 0.5

    # One step but ring also flips, or anything further: clashing.
    return 0.2
