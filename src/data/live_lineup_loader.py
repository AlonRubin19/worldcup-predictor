"""Live lineup loader — parses API-Football lineup responses into ExpectedLineupEntry.

Pure parsing functions are fully testable without HTTP calls.
"""

from __future__ import annotations

from src.data.api_football_client import ApiFootballClient, CACHE_TTL_LINEUPS
from src.data.lineup_loader import ExpectedLineupEntry

# Position code → full position string
_POS_MAP: dict[str, str] = {
    "G": "GK",
    "D": "CB",
    "M": "CM",
    "F": "ST",
}


def _pos(api_pos: str) -> str:
    return _POS_MAP.get(api_pos, api_pos)


# ── Pure parser ───────────────────────────────────────────────────────────────

def parse_lineup_response(
    response_data: dict,
    fixture_id: int,
    date: str,
) -> list[ExpectedLineupEntry]:
    """Parse a raw API-Football /fixtures/lineups response.

    Args:
        response_data: Full JSON dict from API or cache.
        fixture_id:    Numeric fixture ID (stored as match_id string).
        date:          Match date string (YYYY-MM-DD or ISO).

    Returns:
        List of ExpectedLineupEntry — starters + substitutes for both teams.
    """
    match_id_str = str(fixture_id)
    entries: list[ExpectedLineupEntry] = []

    for team_data in response_data.get("response", []):
        team_name = team_data.get("team", {}).get("name", "Unknown")

        # Starters
        for player_entry in team_data.get("startXI", []):
            p = player_entry.get("player", {})
            entries.append(ExpectedLineupEntry(
                match_id=match_id_str,
                date=date,
                team=team_name,
                player_id=f"api_{p.get('id', 0)}",
                player_name=p.get("name", ""),
                position=_pos(p.get("pos", "")),
                expected_starter=True,
                lineup_status="official",
                availability_status="fit",
                availability_factor=1.0,
                form_factor=1.0,
                source_type="official_lineup",
                research_valid=True,
            ))

        # Substitutes
        for player_entry in team_data.get("substitutes", []):
            p = player_entry.get("player", {})
            entries.append(ExpectedLineupEntry(
                match_id=match_id_str,
                date=date,
                team=team_name,
                player_id=f"api_{p.get('id', 0)}",
                player_name=p.get("name", ""),
                position=_pos(p.get("pos", "")),
                expected_starter=False,
                lineup_status="bench",
                availability_status="fit",
                availability_factor=1.0,
                form_factor=1.0,
                source_type="official_lineup",
                research_valid=True,
            ))

    return entries


# ── Live fetcher (uses client) ────────────────────────────────────────────────

def fetch_lineups(
    client: ApiFootballClient,
    fixture_id: int,
    date: str = "",
    ttl_seconds: int = CACHE_TTL_LINEUPS,
) -> list[ExpectedLineupEntry]:
    """Fetch official lineups from API-Football for a given fixture.

    Args:
        client:      Configured ApiFootballClient.
        fixture_id:  API-Football numeric fixture ID.
        date:        Match date for populating ExpectedLineupEntry.date.
        ttl_seconds: Cache TTL override.

    Returns:
        List of ExpectedLineupEntry (empty when lineup not yet available).
    """
    data = client.get(
        "/fixtures/lineups",
        params={"fixture": str(fixture_id)},
        ttl_seconds=ttl_seconds,
    )
    return parse_lineup_response(data, fixture_id=fixture_id, date=date)
