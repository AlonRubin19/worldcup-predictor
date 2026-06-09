"""Adapter for The Odds API (https://the-odds-api.com).

INVESTIGATION RESULT (2026-06-09):
  The Odds API provides historical WC 2022 odds via a JSON REST API.
  Historical data is available from April 2022 onwards.

  Authentication: API key required ($30+/month for historical access).
  Set the environment variable ODDS_API_KEY to activate this adapter.

  API reference:
    GET /v4/historical/sports/{sport_key}/odds
    sport_key: soccer_fifa_world_cup
    Required params: apiKey, date (ISO 8601 snapshot timestamp)

  Team name differences vs our match_results.csv:
    "IR Iran"          → "Iran"
    "Korea Republic"   → "South Korea"
    "USA"              → "USA"          (matches)
    All others match directly or via this normalisation table.

  Response structure (v4 h2h market):
    {
      "data": [
        {
          "id": "...",
          "commence_time": "2022-11-21T13:00:00Z",
          "home_team": "Qatar",
          "away_team": "Ecuador",
          "bookmakers": [
            {
              "key": "bet365",
              "title": "Bet365",
              "markets": [
                {
                  "key": "h2h",
                  "outcomes": [
                    {"name": "Qatar", "price": 2.75},
                    {"name": "Ecuador", "price": 2.50},
                    {"name": "Draw", "price": 3.10}
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
"""

import os
from datetime import datetime, timezone

from src.data.market_sources.base import (
    OddsAdapter,
    MarketOddsRow,
    MissingCredentialsError,
)

# Normalisation: Odds API team name → our match_results.csv team name
_TEAM_NAME_MAP: dict[str, str] = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Kyrgyz Republic": "Kyrgyzstan",
}

_PREFERRED_BOOKMAKERS = ["bet365", "pinnacle", "draftkings", "fanduel", "betmgm"]


class TheOddsAPIAdapter(OddsAdapter):
    """Adapter for The Odds API WC 2022 historical odds."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ODDS_API_KEY", "").strip() or None

    @property
    def source_name(self) -> str:
        return "the_odds_api"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def fetch_wc2022(self) -> list[MarketOddsRow]:
        """Fetch all WC 2022 historical odds from The Odds API.

        Raises:
            MissingCredentialsError: if ODDS_API_KEY is not set.

        Note:
            This method makes real HTTP requests. It is not called during
            unit tests — parse_event() is tested separately with fixtures.
        """
        if not self.is_available():
            raise MissingCredentialsError(
                "ODDS_API_KEY environment variable is not set. "
                "A paid API key is required to fetch WC 2022 historical odds from "
                "the-odds-api.com. Plans start at $30/month. "
                "See: https://the-odds-api.com"
            )

        try:
            import requests
        except ImportError:
            raise MissingCredentialsError(
                "The 'requests' library is required for TheOddsAPIAdapter. "
                "Install it with: pip install requests"
            )

        sport_key = "soccer_fifa_world_cup"
        # WC 2022 ran 2022-11-20 to 2022-12-18; fetch a snapshot near the start
        snapshot_date = "2022-11-20T00:00:00Z"
        url = (
            f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
            f"?apiKey={self._api_key}"
            f"&regions=eu"
            f"&markets=h2h"
            f"&oddsFormat=decimal"
            f"&date={snapshot_date}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        rows = []
        for event in data.get("data", []):
            row = self.parse_event(event, match_id="")
            if row is not None:
                rows.append(row)

        return rows

    def parse_event(self, event: dict, match_id: str) -> "MarketOddsRow | None":
        """Convert a single Odds API event dict to a MarketOddsRow.

        Args:
            event: One event object from the API response "data" array.
            match_id: Pre-assigned match_id (empty string if not yet resolved).

        Returns:
            MarketOddsRow, or None if no usable bookmaker odds are present.
        """
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        # Pick first preferred bookmaker, fall back to first available
        chosen = None
        for pref in _PREFERRED_BOOKMAKERS:
            for bk in bookmakers:
                if bk["key"] == pref:
                    chosen = bk
                    break
            if chosen:
                break
        if chosen is None:
            chosen = bookmakers[0]

        # Extract h2h market
        h2h = next(
            (m for m in chosen.get("markets", []) if m["key"] == "h2h"),
            None,
        )
        if h2h is None:
            return None

        home_team = self.normalise_team_name(event["home_team"])
        away_team = self.normalise_team_name(event["away_team"])

        outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}

        # Map outcome names: home_team → home odds, away_team → away odds, Draw → draw
        home_odds = outcomes.get(event["home_team"]) or outcomes.get(home_team)
        away_odds = outcomes.get(event["away_team"]) or outcomes.get(away_team)
        draw_odds = outcomes.get("Draw")

        if home_odds is None or away_odds is None or draw_odds is None:
            return None

        date_str = event["commence_time"][:10]  # "2022-12-10T19:00:00Z" → "2022-12-10"

        return MarketOddsRow(
            match_id=match_id,
            date=date_str,
            team_a=home_team,
            team_b=away_team,
            bookmaker=chosen["title"],
            opening_home_odds=float(home_odds),
            opening_draw_odds=float(draw_odds),
            opening_away_odds=float(away_odds),
            closing_home_odds=float(home_odds),
            closing_draw_odds=float(draw_odds),
            closing_away_odds=float(away_odds),
            source_type="historical_odds",
            research_valid=True,
        )

    def normalise_team_name(self, name: str) -> str:
        """Map Odds API team names to our match_results.csv team names."""
        return _TEAM_NAME_MAP.get(name, name)
