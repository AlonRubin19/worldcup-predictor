"""Tests for the football-data.co.uk odds adapter.

Investigation result:
  football-data.co.uk provides domestic league odds in CSV format.
  World Cup 2022 odds are NOT available from this source.
  The WorldCup.xlsx file on the site is for WC 2026, not WC 2022.

These tests verify:
  1. The adapter is always available (no credentials needed).
  2. Calling fetch_wc2022() raises DataNotAvailableError with a clear message.
  3. The adapter can parse a domestic-league-style CSV (for future use).
  4. The adapter documents exactly why WC 2022 is unavailable.
"""

import pytest
from pathlib import Path
from src.data.market_sources.football_data_uk import FootballDataUKAdapter
from src.data.market_sources.base import DataNotAvailableError, MarketOddsRow


# Minimal domestic odds CSV in football-data.co.uk format
DOMESTIC_CSV = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A
E0,21/08/22,Arsenal,Crystal Palace,2,0,H,1.57,4.20,6.50
E0,22/08/22,Fulham,Liverpool,2,2,D,5.00,3.80,1.65
"""


class TestFootballDataUKAdapter:
    def test_is_always_available(self):
        adapter = FootballDataUKAdapter()
        assert adapter.is_available() is True

    def test_source_name(self):
        adapter = FootballDataUKAdapter()
        assert adapter.source_name == "football_data_co_uk"

    def test_fetch_wc2022_raises_data_not_available(self):
        adapter = FootballDataUKAdapter()
        with pytest.raises(DataNotAvailableError):
            adapter.fetch_wc2022()

    def test_fetch_wc2022_error_message_explains_why(self):
        adapter = FootballDataUKAdapter()
        try:
            adapter.fetch_wc2022()
        except DataNotAvailableError as e:
            msg = str(e).lower()
            # Must explain that WC 2022 is not available (not just a generic error)
            assert "world cup 2022" in msg or "wc 2022" in msg or "2022" in msg

    def test_fetch_wc2022_error_message_suggests_alternative(self):
        adapter = FootballDataUKAdapter()
        try:
            adapter.fetch_wc2022()
        except DataNotAvailableError as e:
            msg = str(e).lower()
            # Must name the real source that does have WC 2022
            assert "odds api" in msg or "the-odds-api" in msg or "the odds api" in msg

    def test_parse_domestic_csv_returns_odds_rows(self, tmp_path):
        csv_path = tmp_path / "E0.csv"
        csv_path.write_text(DOMESTIC_CSV)
        adapter = FootballDataUKAdapter()
        rows = adapter.parse_league_csv(csv_path, season="2022-23", league="E0")
        assert isinstance(rows, list)
        assert len(rows) == 2

    def test_parse_domestic_csv_fields(self, tmp_path):
        csv_path = tmp_path / "E0.csv"
        csv_path.write_text(DOMESTIC_CSV)
        adapter = FootballDataUKAdapter()
        rows = adapter.parse_league_csv(csv_path, season="2022-23", league="E0")
        r = rows[0]
        assert isinstance(r, MarketOddsRow)
        assert r.team_a == "Arsenal"
        assert r.team_b == "Crystal Palace"
        assert r.opening_home_odds == pytest.approx(1.57)
        assert r.opening_away_odds == pytest.approx(6.50)
        assert r.source_type == "football_data_co_uk"
        assert r.research_valid is True

    def test_parse_domestic_csv_date_format_converted(self, tmp_path):
        csv_path = tmp_path / "E0.csv"
        csv_path.write_text(DOMESTIC_CSV)
        adapter = FootballDataUKAdapter()
        rows = adapter.parse_league_csv(csv_path, season="2022-23", league="E0")
        # DD/MM/YY → YYYY-MM-DD
        assert rows[0].date == "2022-08-21"

    def test_parse_domestic_csv_missing_file_raises(self, tmp_path):
        adapter = FootballDataUKAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.parse_league_csv(tmp_path / "nonexistent.csv", season="2022-23", league="E0")
