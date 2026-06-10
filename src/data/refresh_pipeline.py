"""refresh_pipeline.py — orchestrates a "Refresh" action.

Pulls the freshest available squad, injury, and player-statistics data for a
set of teams via API-Football (with graceful fallback per team), and returns
a structured summary the UI can render (counts + per-team source labels).

This module does NOT make network calls directly -- it delegates to
src.data.live_squad_loader / live_injury_loader / live_player_stats_loader,
which can be exercised in tests via an injected ApiFootballClient _fetcher.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from src.data.api_football_client import ApiFootballClient
from src.data.live_squad_loader import load_live_squad
from src.data.live_injury_loader import load_live_injuries
from src.data.live_player_stats_loader import load_live_player_stats


@dataclass
class TeamRefreshResult:
    team: str
    api_team_id: int
    squad_count: int
    squad_source: str
    injury_count: int
    injury_source: str
    stats_count: int
    stats_source: str
    used_live_data: bool


@dataclass
class RefreshSummary:
    timestamp: str
    teams: list[TeamRefreshResult] = field(default_factory=list)

    @property
    def squads_refreshed(self) -> int:
        return sum(1 for t in self.teams if "API-Football" in t.squad_source)

    @property
    def injuries_refreshed(self) -> int:
        return sum(1 for t in self.teams if "API-Football" in t.injury_source)

    @property
    def stats_refreshed(self) -> int:
        return sum(1 for t in self.teams if "API-Football" in t.stats_source)


def refresh_team_data(
    client: ApiFootballClient,
    teams: list[tuple[str, int]],
    season: int = 2025,
) -> RefreshSummary:
    """Refresh squad/injury/player-stats data for each (team_name, api_team_id).

    Returns a RefreshSummary with one TeamRefreshResult per team. Any team
    for which live data is unavailable falls back gracefully (empty
    squad/injuries/stats + a fallback source label) -- it never raises.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    results: list[TeamRefreshResult] = []
    for team_name, api_team_id in teams:
        squad, squad_source = load_live_squad(client, api_team_id, team_name)
        injuries, injury_source = load_live_injuries(client, api_team_id)
        stats, stats_source = load_live_player_stats(client, api_team_id, season)

        used_live = "API-Football" in squad_source

        results.append(TeamRefreshResult(
            team=team_name,
            api_team_id=api_team_id,
            squad_count=len(squad),
            squad_source=squad_source,
            injury_count=len(injuries),
            injury_source=injury_source,
            stats_count=len(stats),
            stats_source=stats_source,
            used_live_data=used_live,
        ))

    return RefreshSummary(timestamp=timestamp, teams=results)
