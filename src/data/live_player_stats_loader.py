"""live_player_stats_loader.py — load current player season stats via API-Football."""

from __future__ import annotations

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.player_squad_adapter import PlayerSeasonStats, fetch_player_statistics

LIVE_SOURCE_LABEL = "API-Football live player statistics (/players)"
FALLBACK_NO_KEY_LABEL = "Fallback: API_FOOTBALL_KEY not configured"
FALLBACK_ERROR_LABEL = "Fallback: API-Football player statistics request failed"
FALLBACK_EMPTY_LABEL = "Fallback: API-Football returned no player statistics"


def load_live_player_stats(
    client: ApiFootballClient, api_team_id: int, season: int,
) -> tuple[dict[str, PlayerSeasonStats], str]:
    """Return (player_id -> PlayerSeasonStats, source_label)."""
    try:
        stats = fetch_player_statistics(client, api_team_id, season)
    except ApiKeyMissingError:
        return {}, FALLBACK_NO_KEY_LABEL
    except Exception:
        return {}, FALLBACK_ERROR_LABEL

    if not stats:
        return {}, FALLBACK_EMPTY_LABEL

    return stats, LIVE_SOURCE_LABEL
