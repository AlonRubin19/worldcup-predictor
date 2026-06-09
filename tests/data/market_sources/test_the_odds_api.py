"""Tests for The Odds API adapter.

Investigation result:
  The Odds API (the-odds-api.com) provides WC 2022 historical odds
  via a JSON API. Historical data available from April 2022 onwards.
  Requires a paid API key ($30+/month). No free tier for historical data.

These tests verify:
  1. Adapter is unavailable when no API key is set.
  2. Adapter is available when ODDS_API_KEY env var is set.
  3. fetch_wc2022() raises MissingCredentialsError when no key.
  4. parse_event() correctly converts API JSON to MarketOddsRow.
  5. parse_event() handles missing bookmakers gracefully.
  6. Team name normalisation maps Odds API names to our team names.
"""

import os
import pytest
from src.data.market_sources.the_odds_api import TheOddsAPIAdapter
from src.data.market_sources.base import MissingCredentialsError, MarketOddsRow


# Minimal fixture matching The Odds API v4 historical event structure
SAMPLE_EVENT = {
    "id": "abc123",
    "sport_key": "soccer_fifa_world_cup",
    "commence_time": "2022-12-10T19:00:00Z",
    "home_team": "England",
    "away_team": "France",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "last_update": "2022-12-10T14:00:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2022-12-10T14:00:00Z",
                    "outcomes": [
                        {"name": "England", "price": 2.45},
                        {"name": "France", "price": 2.95},
                        {"name": "Draw", "price": 3.25},
                    ]
                }
            ]
        }
    ]
}

SAMPLE_EVENT_NO_BOOKMAKERS = {
    "id": "def456",
    "sport_key": "soccer_fifa_world_cup",
    "commence_time": "2022-12-10T19:00:00Z",
    "home_team": "England",
    "away_team": "France",
    "bookmakers": []
}


class TestTheOddsAPIAdapterAvailability:
    def test_unavailable_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        adapter = TheOddsAPIAdapter()
        assert adapter.is_available() is False

    def test_available_with_api_key_in_env(self, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "test-key-12345")
        adapter = TheOddsAPIAdapter()
        assert adapter.is_available() is True

    def test_available_with_api_key_passed_directly(self):
        adapter = TheOddsAPIAdapter(api_key="test-key-12345")
        assert adapter.is_available() is True

    def test_unavailable_with_empty_string_key(self, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "")
        adapter = TheOddsAPIAdapter()
        assert adapter.is_available() is False

    def test_source_name(self):
        adapter = TheOddsAPIAdapter()
        assert adapter.source_name == "the_odds_api"


class TestTheOddsAPIFetchWithoutKey:
    def test_fetch_wc2022_raises_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        adapter = TheOddsAPIAdapter()
        with pytest.raises(MissingCredentialsError):
            adapter.fetch_wc2022()

    def test_error_message_names_env_var(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        adapter = TheOddsAPIAdapter()
        try:
            adapter.fetch_wc2022()
        except MissingCredentialsError as e:
            assert "ODDS_API_KEY" in str(e)

    def test_error_message_names_the_odds_api(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        adapter = TheOddsAPIAdapter()
        try:
            adapter.fetch_wc2022()
        except MissingCredentialsError as e:
            assert "the-odds-api.com" in str(e).lower() or "odds api" in str(e).lower()


class TestParseEvent:
    def test_parse_event_returns_market_odds_row(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        row = adapter.parse_event(SAMPLE_EVENT, match_id="45777")
        assert isinstance(row, MarketOddsRow)

    def test_parse_event_fields(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        row = adapter.parse_event(SAMPLE_EVENT, match_id="45777")
        assert row.match_id == "45777"
        assert row.team_a == "England"
        assert row.team_b == "France"
        assert row.bookmaker == "Bet365"
        # closing odds come from the event data
        assert row.closing_home_odds == pytest.approx(2.45)
        assert row.closing_away_odds == pytest.approx(2.95)
        assert row.closing_draw_odds == pytest.approx(3.25)

    def test_parse_event_source_type_and_research_valid(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        row = adapter.parse_event(SAMPLE_EVENT, match_id="45777")
        assert row.source_type == "historical_odds"
        assert row.research_valid is True

    def test_parse_event_date_extracted_from_commence_time(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        row = adapter.parse_event(SAMPLE_EVENT, match_id="45777")
        assert row.date == "2022-12-10"

    def test_parse_event_no_bookmakers_returns_none(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        result = adapter.parse_event(SAMPLE_EVENT_NO_BOOKMAKERS, match_id="45777")
        assert result is None

    def test_parse_event_opening_odds_same_as_closing_when_no_opener(self):
        """When only one snapshot available, opening == closing."""
        adapter = TheOddsAPIAdapter(api_key="test")
        row = adapter.parse_event(SAMPLE_EVENT, match_id="45777")
        # Single snapshot: opening = closing
        assert row.opening_home_odds == row.closing_home_odds
        assert row.opening_draw_odds == row.closing_draw_odds
        assert row.opening_away_odds == row.closing_away_odds


class TestTeamNameNormalisation:
    def test_normalise_known_team(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        assert adapter.normalise_team_name("Republic of Ireland") == "Republic of Ireland"
        assert adapter.normalise_team_name("IR Iran") == "Iran"
        assert adapter.normalise_team_name("Korea Republic") == "South Korea"

    def test_normalise_unknown_team_returns_unchanged(self):
        adapter = TheOddsAPIAdapter(api_key="test")
        assert adapter.normalise_team_name("Brazil") == "Brazil"
        assert adapter.normalise_team_name("France") == "France"
