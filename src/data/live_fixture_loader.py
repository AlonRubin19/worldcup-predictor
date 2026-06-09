"""Live fixture loader — parses API-Football fixture responses.

Pure parsing functions are fully testable without HTTP calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.api_football_client import ApiFootballClient, CACHE_TTL_FIXTURES

# ── Team name mapping: API-Football name → internal name ─────────────────────
# Internal names come from data/teams.csv
TEAM_NAME_MAP: dict[str, str] = {
    # Standard FIFA / API-Football aliases -> internal names
    "Korea Republic":       "South Korea",
    "United States":        "USA",
    "IR Iran":              "Iran",
    "C?te d'Ivoire":        "Ivory Coast",
    "Côte d'Ivoire":        "Ivory Coast",
    "Congo DR":             "DR Congo",
    "Bosnia":               "Bosnia & Herzegovina",
    "Czech Republic":       "Czech Republic",
    "Slovak Republic":      "Slovakia",
    "Northern Ireland":     "Northern Ireland",
    # WC 2026 — country name changes / alternate spellings
    "Türkiye":              "Turkey",
    "Cape Verde Islands":   "Cape Verde",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class LiveFixture:
    """One upcoming (or live) fixture from API-Football."""
    fixture_id: int
    date: str               # ISO 8601 datetime string
    home_team_api: str      # API-Football team name (raw)
    away_team_api: str
    home_team: str          # Mapped internal name
    away_team: str
    status_short: str       # "NS", "1H", "HT", "2H", "FT", etc.
    round: str
    venue: str


# ── Pure parsers ──────────────────────────────────────────────────────────────

def map_team_name(api_name: str) -> str:
    """Translate an API-Football team name to an internal name.

    Falls back to the original name if no mapping exists.
    """
    return TEAM_NAME_MAP.get(api_name, api_name)


def parse_fixtures_response(response_data: dict) -> list[LiveFixture]:
    """Parse a raw API-Football /fixtures response into LiveFixture objects.

    Args:
        response_data: The full JSON dict from the API (or cache).

    Returns:
        List of LiveFixture, one per entry in response["response"].
    """
    fixtures: list[LiveFixture] = []
    for item in response_data.get("response", []):
        fix   = item.get("fixture", {})
        teams = item.get("teams", {})
        league = item.get("league", {})
        home_api = teams.get("home", {}).get("name", "")
        away_api = teams.get("away", {}).get("name", "")
        fixtures.append(LiveFixture(
            fixture_id=int(fix.get("id", 0)),
            date=fix.get("date", ""),
            home_team_api=home_api,
            away_team_api=away_api,
            home_team=map_team_name(home_api),
            away_team=map_team_name(away_api),
            status_short=fix.get("status", {}).get("short", ""),
            round=league.get("round", ""),
            venue=fix.get("venue", {}).get("name", ""),
        ))
    return fixtures


# ── Live fetcher (uses client) ────────────────────────────────────────────────

def fetch_upcoming_fixtures(
    client: ApiFootballClient,
    league_id: int = 1,
    season: int = 2026,
    ttl_seconds: int = CACHE_TTL_FIXTURES,
) -> list[LiveFixture]:
    """Fetch upcoming fixtures from API-Football.

    Args:
        client:      Configured ApiFootballClient.
        league_id:   API-Football league ID (1 = FIFA World Cup).
        season:      Season year (e.g. 2026).
        ttl_seconds: Cache TTL override.

    Returns:
        List of LiveFixture from the API response.
    """
    data = client.get(
        "/fixtures",
        params={"league": str(league_id), "season": str(season), "status": "NS"},
        ttl_seconds=ttl_seconds,
    )
    return parse_fixtures_response(data)
