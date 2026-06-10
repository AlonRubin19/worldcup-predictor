"""live_injury_loader.py — load current injuries via API-Football, with fallback."""

from __future__ import annotations

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.player_squad_adapter import fetch_injuries

LIVE_SOURCE_LABEL = "API-Football live injuries (/injuries)"
FALLBACK_NO_KEY_LABEL = "Fallback: API_FOOTBALL_KEY not configured"
FALLBACK_ERROR_LABEL = "Fallback: API-Football injuries request failed"
FALLBACK_NONE_LABEL = "No reported injuries (or none returned by API)"


def load_live_injuries(
    client: ApiFootballClient, api_team_id: int,
) -> tuple[set[str], str]:
    """Return (injured_player_names, source_label)."""
    try:
        injured = fetch_injuries(client, api_team_id)
    except ApiKeyMissingError:
        return set(), FALLBACK_NO_KEY_LABEL
    except Exception:
        return set(), FALLBACK_ERROR_LABEL

    if not injured:
        return set(), FALLBACK_NONE_LABEL

    return injured, LIVE_SOURCE_LABEL
