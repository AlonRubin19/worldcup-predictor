"""team_coverage.py — build a per-team API-Football data coverage table.

Used by the global Refresh button so the UI can show, for every team in the
tournament, whether it has an API-Football id mapped, and (after a refresh)
whether live squad/injury/player-stats data was actually returned.

No team is silently dropped: every team passed to build_coverage_table()
appears exactly once in the output, with mapped=False if no id is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.refresh_pipeline import RefreshSummary


@dataclass
class CoverageRow:
    team: str
    api_team_id: int | None
    mapped: bool
    squad_available: bool = False
    injuries_available: bool = False
    player_stats_available: bool = False


def build_coverage_table(teams: list[str], api_ids: dict[str, int]) -> list[CoverageRow]:
    """Return one CoverageRow per team in `teams`, in the given order."""
    rows = []
    for team in teams:
        api_id = api_ids.get(team)
        rows.append(CoverageRow(team=team, api_team_id=api_id, mapped=api_id is not None))
    return rows


def apply_refresh_results(rows: list[CoverageRow], summary: "RefreshSummary") -> list[CoverageRow]:
    """Fill in squad/injuries/player_stats availability from a RefreshSummary.

    Returns a new list of CoverageRow (does not mutate the input).
    """
    by_team = {r.team: r for r in summary.teams}
    updated = []
    for row in rows:
        result = by_team.get(row.team)
        if result is None:
            updated.append(row)
            continue
        updated.append(CoverageRow(
            team=row.team,
            api_team_id=row.api_team_id,
            mapped=row.mapped,
            squad_available=result.squad_count > 0 and not result.squad_source.startswith("Fallback"),
            injuries_available=not result.injury_source.startswith("Fallback"),
            player_stats_available=result.stats_count > 0 and not result.stats_source.startswith("Fallback"),
        ))
    return updated
