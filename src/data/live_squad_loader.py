"""live_squad_loader.py — load current squads via API-Football, with fallback.

Wraps player_squad_adapter.fetch_squad. If the API key is missing, the
request fails, or the squad is empty, returns an empty list and a
human-readable source label explaining the fallback -- callers must show
this label as a data-source warning.
"""

from __future__ import annotations

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.player_squad_adapter import SquadPlayer, fetch_squad

LIVE_SOURCE_LABEL = "API-Football live squad (/players/squads)"
FALLBACK_NO_KEY_LABEL = "Fallback: API_FOOTBALL_KEY not configured"
FALLBACK_EMPTY_LABEL = "Fallback: API-Football returned no squad data"
FALLBACK_ERROR_LABEL = "Fallback: API-Football request failed"


def load_live_squad(
    client: ApiFootballClient, api_team_id: int, team_name: str,
) -> tuple[list[SquadPlayer], str]:
    """Return (squad_players, source_label).

    squad_players is empty and source_label explains why if live data is
    unavailable for any reason.
    """
    try:
        players = fetch_squad(client, api_team_id, team_name)
    except ApiKeyMissingError:
        return [], FALLBACK_NO_KEY_LABEL
    except Exception:
        return [], FALLBACK_ERROR_LABEL

    if not players:
        return [], FALLBACK_EMPTY_LABEL

    return players, LIVE_SOURCE_LABEL
