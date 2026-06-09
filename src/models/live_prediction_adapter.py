"""Live prediction adapter — orchestrates API-Football data into the prediction pipeline.

Provides:
  - LiveDataStatus: summary of API connection + data source state
  - get_live_data_status(): returns status without crashing on API errors
  - load_live_lineups_for_match(): fetches and splits lineups by team

All API failures are caught and return graceful fallbacks.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.live_lineup_loader import fetch_lineups
from src.data.lineup_loader import ExpectedLineupEntry


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class LiveDataStatus:
    """Summary of the current live-data connection and source state."""
    api_connected: bool
    last_refresh: str | None          # ISO timestamp of last successful fetch, or None
    fixture_source: str               # e.g. "API-Football (live)" or "CSV (offline)"
    lineup_source: str                # e.g. "API-Football (live)" or "Not loaded"
    lineup_status_label: str          # e.g. "Official", "Unavailable", "Not connected"


# ── Status builder ────────────────────────────────────────────────────────────

def get_live_data_status(
    client: ApiFootballClient,
    fixture_id: int | None = None,
    team_a: str | None = None,
    team_b: str | None = None,
) -> LiveDataStatus:
    """Build a LiveDataStatus reflecting the current API and lineup state.

    When fixture_id + team_a/team_b are provided, also checks whether
    lineup data is currently available for that match.

    Args:
        client:     ApiFootballClient (key may be empty → not connected).
        fixture_id: Optional fixture to check lineup availability for.
        team_a:     Home team internal name (used to detect lineup presence).
        team_b:     Away team internal name.

    Returns:
        LiveDataStatus — never raises.
    """
    # Check connectivity
    api_connected = bool(getattr(client, "_api_key", ""))
    if not api_connected:
        return LiveDataStatus(
            api_connected=False,
            last_refresh=None,
            fixture_source="CSV (offline fallback)",
            lineup_source="Not connected",
            lineup_status_label="API key not configured",
        )

    # If no fixture specified, just report connection state
    if fixture_id is None:
        return LiveDataStatus(
            api_connected=True,
            last_refresh=_now_iso(),
            fixture_source="API-Football (live)",
            lineup_source="Not loaded",
            lineup_status_label="No fixture selected",
        )

    # Try to load lineups to determine status
    entries_a, entries_b = load_live_lineups_for_match(
        client, fixture_id=fixture_id,
        team_a=team_a or "", team_b=team_b or "",
    )

    if entries_a or entries_b:
        lineup_src = "API-Football (official)"
        lineup_label = "Official lineup available"
    else:
        lineup_src = "API-Football (no lineup yet)"
        lineup_label = "Lineup unavailable — not yet published"

    return LiveDataStatus(
        api_connected=True,
        last_refresh=_now_iso(),
        fixture_source="API-Football (live)",
        lineup_source=lineup_src,
        lineup_status_label=lineup_label,
    )


# ── Lineup loader ─────────────────────────────────────────────────────────────

def load_live_lineups_for_match(
    client: ApiFootballClient,
    fixture_id: int,
    team_a: str,
    team_b: str,
) -> tuple[list[ExpectedLineupEntry], list[ExpectedLineupEntry]]:
    """Fetch and split official lineup entries by team.

    Args:
        client:     Configured ApiFootballClient.
        fixture_id: API-Football numeric fixture ID.
        team_a:     Home team internal name.
        team_b:     Away team internal name.

    Returns:
        (entries_a, entries_b) — empty lists if API unavailable or lineup
        not yet published.  Never raises.
    """
    try:
        all_entries = fetch_lineups(client, fixture_id=fixture_id)
    except (ApiKeyMissingError, Exception):
        return [], []

    entries_a = [e for e in all_entries if e.team == team_a]
    entries_b = [e for e in all_entries if e.team == team_b]
    return entries_a, entries_b


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
