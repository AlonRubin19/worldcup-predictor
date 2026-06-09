"""Lineup validation and conversion layer.

Validates a list of ExpectedLineupEntry for a given team, and converts
entries into the LineupOverride format used by the override engine.

Validation rules:
  1. Official lineups must have exactly 11 expected starters.
  2. Projected lineups with != 11 starters produce a warning (not an error).
  3. No duplicate player_id within the same team.
  4. Players with availability_status 'out' or 'suspended' must not have
     expected_starter=True.
  5. research_valid=True only when ALL entries for the team have
     source_type in {"official_lineup", "injury_report"} AND all entries
     carry research_valid=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.lineup_loader import ExpectedLineupEntry
from src.models.lineup_override import LineupOverride, PlayerOverride


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LineupValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_research_valid: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_UNAVAILABLE_STATUSES = {"out", "suspended"}
_RESEARCH_VALID_SOURCES = {"official_lineup", "injury_report"}


def validate_lineup_for_match(
    entries: list[ExpectedLineupEntry],
    team: str,
) -> LineupValidationResult:
    """Validate the lineup entries for a single team.

    Filters entries to the requested team before validating.

    Returns:
        LineupValidationResult with is_valid, errors, warnings, is_research_valid.
    """
    team_entries = [e for e in entries if e.team == team]

    errors: list[str] = []
    warnings: list[str] = []

    if not team_entries:
        return LineupValidationResult(
            is_valid=True, errors=[], warnings=[], is_research_valid=False
        )

    # ── Rule 1 / 2: starter count ────────────────────────────────────────────
    starters = [e for e in team_entries if e.expected_starter]
    has_official = any(e.lineup_status == "official" for e in team_entries)

    if has_official:
        if len(starters) != 11:
            errors.append(
                f"Official lineup must have exactly 11 expected starters; "
                f"found {len(starters)}."
            )
    else:
        if len(starters) != 11:
            warnings.append(
                f"Projected/unknown lineup has {len(starters)} expected starters "
                f"(expected 11). Predictions may be less accurate."
            )

    # ── Rule 3: no duplicate player_id ───────────────────────────────────────
    seen_ids: set[str] = set()
    for e in team_entries:
        if e.player_id in seen_ids:
            errors.append(
                f"Duplicate player_id '{e.player_id}' found in lineup for {team}."
            )
        seen_ids.add(e.player_id)

    # ── Rule 4: unavailable players cannot be starters ───────────────────────
    for e in team_entries:
        if e.expected_starter and e.availability_status in _UNAVAILABLE_STATUSES:
            errors.append(
                f"Player '{e.player_id}' ({e.player_name}) has availability_status "
                f"'{e.availability_status}' but is marked as expected_starter=True."
            )

    # ── Rule 5: research validity ─────────────────────────────────────────────
    is_research_valid = (
        len(team_entries) > 0
        and all(
            e.source_type in _RESEARCH_VALID_SOURCES and e.research_valid
            for e in team_entries
        )
    )

    return LineupValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        is_research_valid=is_research_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_lineup_entries_to_override(
    entries: list[ExpectedLineupEntry],
) -> LineupOverride:
    """Convert a list of ExpectedLineupEntry into a LineupOverride.

    All entries are assumed to belong to a single team (the team of
    entries[0] is used as the LineupOverride team name).

    Args:
        entries: Non-empty list of ExpectedLineupEntry for one team.

    Returns:
        LineupOverride ready for use with apply_lineup_override().

    Raises:
        ValueError: If entries is empty.
    """
    if not entries:
        raise ValueError(
            "Cannot convert empty entries to LineupOverride. "
            "Provide at least one ExpectedLineupEntry."
        )

    team = entries[0].team
    players = [
        PlayerOverride(
            player_id=e.player_id,
            player_name=e.player_name,
            team=e.team,
            expected_starter=e.expected_starter,
            availability_status=e.availability_status,
            availability_factor=e.availability_factor,
            form_factor=e.form_factor,
        )
        for e in entries
    ]
    return LineupOverride(team=team, players=players)
