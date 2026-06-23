"""Human-readable, English explanations for why a track sits where it does.

``explain_track`` turns the numeric scores produced by the planner into a short
plain-English sentence (or two) in the style of the spec examples, e.g.:

    "Opens the set as a gentle intro — low energy (0.34) eases the room in."
    "Lands the main peak — its high energy (0.76) and strong groove deliver the
    emotional high point, mixing in smoothly from the previous track."

The phrasing is chosen by the track's narrative ``role`` and by the
transition/position context (how it follows the previous track and how well it
sits inside its energy segment).

This module deliberately depends ONLY on the domain dataclasses and a
*structural* view of ``TransitionScore`` (it reads its fields by attribute).
``TransitionScore`` is owned by ``planning.transition_scorer`` (a different
phase); to avoid a hard import-time dependency on a module that may not exist
yet, we only import it under ``TYPE_CHECKING`` for annotations and access its
attributes duck-typed at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..domain.models import EventProfile, SetSegment, Track, TrackFeatures
from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .transition_scorer import TransitionScore

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Small qualitative-bucket helpers.
#
# These turn a 0..1 number into an English adjective. The thresholds are chosen
# to read naturally for a day-party context (where "high" is ~0.75, not 0.95).
# --------------------------------------------------------------------------- #


def _energy_word(energy: float) -> str:
    """Map an energy score (0..1) to a descriptive adjective."""

    if energy < 0.35:
        return "low"
    if energy < 0.50:
        return "gentle"
    if energy < 0.62:
        return "comfortable mid"
    if energy < 0.72:
        return "lifted"
    return "high"


def _quality_word(score: float) -> str:
    """Map a generic 0..1 quality score to an adjective."""

    if score >= 0.80:
        return "excellent"
    if score >= 0.65:
        return "strong"
    if score >= 0.50:
        return "solid"
    if score >= 0.35:
        return "workable"
    return "weak"


def _transition_phrase(transition_in: "TransitionScore | None") -> str:
    """Describe how this track follows the previous one.

    Reads the structural ``TransitionScore`` fields (``score``, ``bpm``,
    ``key``) by attribute so we don't hard-import the (possibly not-yet-written)
    transition_scorer module. Returns a clause without leading/trailing
    punctuation, or ``""`` when there is no incoming transition (the first
    track).
    """

    if transition_in is None:
        return ""

    # Pull fields defensively — a stand-in object may omit some of them.
    overall = float(getattr(transition_in, "score", 0.5) or 0.0)
    bpm_fit = float(getattr(transition_in, "bpm", 0.5) or 0.0)
    key_fit = float(getattr(transition_in, "key", 0.5) or 0.0)

    # Choose the headline based on overall transition quality.
    if overall >= 0.80:
        head = "mixing in smoothly from the previous track"
    elif overall >= 0.60:
        head = "blending cleanly out of the previous track"
    elif overall >= 0.40:
        head = "with a workable blend from the previous track"
    else:
        head = "an intentional shift away from the previous track"

    # Add a harmonic/tempo detail when one side is clearly the reason.
    if bpm_fit >= 0.75 and key_fit >= 0.75:
        detail = " (tempo and key both line up)"
    elif key_fit >= 0.80:
        detail = " (harmonically compatible keys)"
    elif bpm_fit >= 0.80:
        detail = " (closely matched tempo)"
    elif bpm_fit < 0.40:
        detail = " (a deliberate tempo move)"
    else:
        detail = ""

    return head + detail


# --------------------------------------------------------------------------- #
# Per-role sentence templates.
#
# Each builder returns the *opening clause* describing the track's job in the
# story. The energy / groove / transition details are appended by
# ``explain_track``. Keeping role phrasing here makes the narrative voice
# consistent and easy to tune.
# --------------------------------------------------------------------------- #


def _role_clause(role: str, f: TrackFeatures, segment: SetSegment) -> str:
    """Return the role-specific opening clause for the explanation."""

    energy = float(f.energy_score)
    e_word = _energy_word(energy)

    # Normalise the role string (it should already be a TrackRole value).
    role_key = (role or "").upper()

    if role_key == "INTRO":
        return (
            f"Opens the set as a gentle intro — its {e_word} energy "
            f"({energy:.2f}) eases the room in"
        )
    if role_key == "WARM_GROOVE":
        groove = float(f.groove_score)
        return (
            f"Settles into a warm groove — {e_word} energy ({energy:.2f}) with "
            f"a {_quality_word(groove)} groove keeps heads nodding"
        )
    if role_key == "PROGRESSIVE_BUILD":
        return (
            f"Pushes the build forward — its {e_word} energy ({energy:.2f}) "
            f"nudges the room up toward the peak"
        )
    if role_key == "SMALL_PEAK":
        return (
            f"Delivers a small peak — {e_word} energy ({energy:.2f}) lifts the "
            f"floor without spending the main moment"
        )
    if role_key == "BREATHING_SPACE":
        return (
            f"Gives the room breathing space — {e_word} energy ({energy:.2f}) "
            f"pulls back to reset before the main peak"
        )
    if role_key == "MAIN_PEAK":
        peak = float(f.peak_potential)
        return (
            f"Lands the main peak — its {e_word} energy ({energy:.2f}) and "
            f"{_quality_word(peak)} peak potential deliver the emotional high "
            f"point of the set"
        )
    if role_key == "RELEASE":
        return (
            f"Eases the release after the peak — {e_word} energy ({energy:.2f}) "
            f"lets the floor come down gently"
        )
    if role_key == "OUTRO":
        return (
            f"Closes the set as an outro — {e_word} energy ({energy:.2f}) sends "
            f"the room off softly"
        )

    # Fallback for any unrecognised role: lean on the segment name.
    seg_name = segment.name.replace("_", " ") if segment else "set"
    return (
        f"Sits in the {seg_name} section — {e_word} energy ({energy:.2f}) fits "
        f"the moment"
    )


def explain_track(
    track: Track,
    f: TrackFeatures,
    role: str,
    segment: SetSegment,
    transition_in: "TransitionScore | None",
    position_score: float,
    profile: EventProfile,
) -> str:
    """Build a human-readable English explanation for a placed track.

    Parameters mirror the planner's view of a placed slot:

    - ``track`` / ``f``: the track and its features.
    - ``role``: the narrative :class:`TrackRole` value assigned to this slot.
    - ``segment``: the energy-curve segment this position falls in.
    - ``transition_in``: the incoming :class:`TransitionScore` from the previous
      track, or ``None`` for the first track.
    - ``position_score``: the overall 0..1 placement score for this slot.
    - ``profile``: the active event profile (for context-aware phrasing).

    Returns a short, spec-style sentence. Never raises — on any unexpected
    input it logs and falls back to a minimal but valid sentence.
    """

    try:
        # 1) Role-driven opening clause (carries energy/groove/peak detail).
        clause = _role_clause(role, f, segment)

        parts: list[str] = [clause]

        # 2) Transition detail, when this isn't the first track.
        trans = _transition_phrase(transition_in)
        if trans:
            parts.append(trans)

        # 3) A restaurant-safety nod when the context is a restaurant and the
        #    track is comfortably safe — reassures the DJ it won't be harsh.
        #    restaurant_safety_score is the inverse of harshness; high == safe.
        if (
            profile is not None
            and "RESTAURANT" in (profile.venue_type or "").upper()
            and float(f.restaurant_safety_score) >= 0.70
        ):
            parts.append("comfortable for the restaurant crowd")

        sentence = parts[0]
        if len(parts) > 1:
            # Join the remaining clauses with commas; they are sub-clauses of
            # the main statement.
            sentence = sentence + ", " + ", ".join(parts[1:])

        # 4) Confidence tag from the overall position score, so the reader can
        #    gauge how strong the placement is.
        confidence = _quality_word(float(position_score))
        sentence = f"{sentence}. {confidence.capitalize()} fit for this slot."

        return sentence

    except Exception:  # never let explanation generation break the planner
        _log.exception(
            "Failed to build explanation for track_id=%s role=%s; using fallback",
            getattr(track, "id", None),
            role,
        )
        title = getattr(track, "title", None) or getattr(track, "file_path", "track")
        return f"Placed '{title}' in the {role or 'set'} role."
