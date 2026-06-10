"""player_squad_adapter.py — API-Football squad/player-statistics/injury adapter.

Sprint 17, Priority 2 (API-Football current squad/player data).

This module is read-only / additive: it builds player_profiles.csv-shaped
rows from API-Football data so the squad coverage can be extended from
8/48 teams to all 48 World Cup 2026 teams. It does NOT scrape -- it only
parses responses from the existing `ApiFootballClient` (API_FOOTBALL_KEY
from environment, file-based cache).

Endpoints used (API-Football v3, "Pro" plan):
  - GET /players/squads?team={id}        -> current squad list
  - GET /players?team={id}&season={year} -> per-player season statistics
  - GET /injuries?team={id}               -> current injury list

Pure parsing functions (parse_*) take raw response dicts and are fully
testable without network access. fetch_* functions wrap an
ApiFootballClient and are thin (no parsing logic of their own).

Output rows from build_player_profile_row() match the schema of
data/player_profiles.csv plus two extra columns:
  - source_type:  "api_football" | "api_football_squad_only" | "placeholder"
  - research_valid: bool -- True only when real season statistics were
    available (not just a squad listing).

If a player has no statistics, a small non-zero placeholder xG is used so
the player can still appear in Golden Boot output, but is clearly marked
research_valid=False / source_type="api_football_squad_only".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.api_football_client import ApiFootballClient, CACHE_TTL_INJURIES, CACHE_TTL_FIXTURES


# ── Position mapping (API-Football full names -> internal FW/MF/DF/GK) ───────

_POSITION_MAP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender": "DF",
    "Midfielder": "MF",
    "Attacker": "FW",
}

# Fallback xG/90 for players with squad data but no season statistics.
_PLACEHOLDER_XG_PER_90 = {
    "FW": 0.15,
    "MF": 0.08,
    "DF": 0.03,
    "GK": 0.0,
}

# Conversion used to derive an xG proxy from shots-on-target when real xG
# is not provided by the API plan.
_SHOT_ON_TARGET_XG_CONVERSION = 0.30
_SHOT_ON_TARGET_XA_CONVERSION = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SquadPlayer:
    player_id: str
    player_name: str
    team: str
    position: str  # FW / MF / DF / GK


@dataclass
class PlayerSeasonStats:
    player_id: str
    minutes: float
    appearances: int
    goals_per_90: float
    assists_per_90: float
    xg_per_90_proxy: float
    xa_per_90_proxy: float
    penalty_taker: bool


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsers
# ─────────────────────────────────────────────────────────────────────────────

def _map_position(api_position: str) -> str:
    return _POSITION_MAP.get(api_position, "MF")


def parse_squad_response(response_data: dict, team_name: str) -> list[SquadPlayer]:
    """Parse a /players/squads response into SquadPlayer rows.

    Args:
        response_data: Raw JSON dict from the API (or cache).
        team_name:      Internal team name to attach to every player
                         (the API's own team name may differ from our
                         internal naming -- caller passes the canonical name).

    Returns:
        List of SquadPlayer. Empty list if the response has no entries.
    """
    players: list[SquadPlayer] = []
    for entry in response_data.get("response", []):
        for p in entry.get("players", []):
            players.append(SquadPlayer(
                player_id=str(p.get("id", "")),
                player_name=str(p.get("name", "")),
                team=team_name,
                position=_map_position(str(p.get("position", ""))),
            ))
    return players


def parse_player_statistics_response(response_data: dict) -> dict[str, PlayerSeasonStats]:
    """Parse a /players?team=X&season=Y response into per-player stats.

    Returns:
        {player_id: PlayerSeasonStats}. Players with zero recorded minutes
        get all-zero rates (no division by zero).
    """
    out: dict[str, PlayerSeasonStats] = {}
    for entry in response_data.get("response", []):
        player = entry.get("player", {})
        player_id = str(player.get("id", ""))
        stats_list = entry.get("statistics", [])
        if not stats_list:
            continue
        stats = stats_list[0]

        games = stats.get("games", {}) or {}
        goals = stats.get("goals", {}) or {}
        shots = stats.get("shots", {}) or {}
        penalty = stats.get("penalty", {}) or {}

        minutes = float(games.get("minutes") or 0)
        appearances = int(games.get("appearences") or 0)

        if minutes > 0:
            goals_total = float(goals.get("total") or 0)
            assists_total = float(goals.get("assists") or 0)
            shots_on = float(shots.get("on") or 0)

            goals_per_90 = goals_total / minutes * 90.0
            assists_per_90 = assists_total / minutes * 90.0
            xg_per_90_proxy = shots_on * _SHOT_ON_TARGET_XG_CONVERSION / minutes * 90.0
            xa_per_90_proxy = shots_on * _SHOT_ON_TARGET_XA_CONVERSION / minutes * 90.0
        else:
            goals_per_90 = 0.0
            assists_per_90 = 0.0
            xg_per_90_proxy = 0.0
            xa_per_90_proxy = 0.0

        penalty_taker = (int(penalty.get("scored") or 0) + int(penalty.get("won") or 0)) > 0

        out[player_id] = PlayerSeasonStats(
            player_id=player_id,
            minutes=minutes,
            appearances=appearances,
            goals_per_90=goals_per_90,
            assists_per_90=assists_per_90,
            xg_per_90_proxy=xg_per_90_proxy,
            xa_per_90_proxy=xa_per_90_proxy,
            penalty_taker=penalty_taker,
        )
    return out


def parse_injuries_response(response_data: dict) -> set[str]:
    """Parse an /injuries response into a set of currently-injured player IDs."""
    injured: set[str] = set()
    for entry in response_data.get("response", []):
        player = entry.get("player", {})
        pid = player.get("id")
        if pid is not None:
            injured.add(str(pid))
    return injured


# ─────────────────────────────────────────────────────────────────────────────
# Row builder -> player_profiles.csv schema
# ─────────────────────────────────────────────────────────────────────────────

def build_player_profile_row(
    squad_player: SquadPlayer,
    stats: PlayerSeasonStats | None,
    injured: bool,
) -> dict:
    """Build one player_profiles.csv-shaped row.

    Args:
        squad_player: From parse_squad_response().
        stats:        From parse_player_statistics_response(), or None if
                       no season statistics are available for this player.
        injured:      True if the player appears in parse_injuries_response().

    Returns:
        dict with all data/player_profiles.csv columns plus
        "source_type" and "research_valid".
    """
    if stats is not None:
        goals_per_90 = stats.goals_per_90
        assists_per_90 = stats.assists_per_90
        xg_per_90 = stats.xg_per_90_proxy
        xa_per_90 = stats.xa_per_90_proxy
        penalty_taker = stats.penalty_taker
        minutes_last_90_days = stats.minutes
        source_type = "api_football"
        research_valid = True
    else:
        goals_per_90 = 0.0
        assists_per_90 = 0.0
        xg_per_90 = _PLACEHOLDER_XG_PER_90.get(squad_player.position, 0.0)
        xa_per_90 = 0.0
        penalty_taker = False
        minutes_last_90_days = 0.0
        source_type = "api_football_squad_only"
        research_valid = False

    return {
        "player_id": squad_player.player_id,
        "player_name": squad_player.player_name,
        "team": squad_player.team,
        "position": squad_player.position,
        "club": "",
        "minutes_last_90_days": minutes_last_90_days,
        "national_team_minutes_last_12_months": 0.0,
        "goals_per_90": goals_per_90,
        "assists_per_90": assists_per_90,
        "xg_per_90": xg_per_90,
        "xa_per_90": xa_per_90,
        "defensive_actions_per_90": 0.0,
        "international_caps": 0,
        "base_impact_score": 1.0,
        "penalty_taker": penalty_taker,
        "availability_factor": 0.0 if injured else 1.0,
        "source_type": source_type,
        "research_valid": research_valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live fetchers (thin wrappers -- no parsing logic)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_squad(client: ApiFootballClient, api_team_id: int, team_name: str) -> list[SquadPlayer]:
    """Fetch the current squad for a team (GET /players/squads?team=...)."""
    data = client.get("/players/squads", params={"team": str(api_team_id)}, ttl_seconds=CACHE_TTL_FIXTURES)
    return parse_squad_response(data, team_name=team_name)


def fetch_player_statistics(
    client: ApiFootballClient, api_team_id: int, season: int,
) -> dict[str, PlayerSeasonStats]:
    """Fetch per-player season statistics (GET /players?team=...&season=...)."""
    data = client.get(
        "/players", params={"team": str(api_team_id), "season": str(season)},
        ttl_seconds=CACHE_TTL_FIXTURES,
    )
    return parse_player_statistics_response(data)


def fetch_injuries(client: ApiFootballClient, api_team_id: int) -> set[str]:
    """Fetch currently-injured player IDs (GET /injuries?team=...)."""
    data = client.get("/injuries", params={"team": str(api_team_id)}, ttl_seconds=CACHE_TTL_INJURIES)
    return parse_injuries_response(data)


def build_team_player_profiles(
    client: ApiFootballClient, api_team_id: int, team_name: str, season: int,
) -> list[dict]:
    """Fetch squad + statistics + injuries for one team and build profile rows.

    If the squad endpoint returns nothing (e.g. team not found / plan
    limitation), returns an empty list -- callers should treat this as
    "no current squad data available" and surface an explicit warning
    rather than silently falling back to placeholders.
    """
    squad = fetch_squad(client, api_team_id, team_name)
    if not squad:
        return []

    stats = fetch_player_statistics(client, api_team_id, season)
    injured = fetch_injuries(client, api_team_id)

    return [
        build_player_profile_row(p, stats.get(p.player_id), injured=p.player_id in injured)
        for p in squad
    ]
