"""Tests for market_loader — load bookmaker odds from CSV."""

import pytest
from pathlib import Path

from src.data.market_loader import MarketOdds, load_market_odds


MARKET_CSV = """\
match_id,date,team_a,team_b,bookmaker,opening_home_odds,opening_draw_odds,opening_away_odds,closing_home_odds,closing_draw_odds,closing_away_odds,source_type,research_valid
45777,2022-12-10,England,France,placeholder,2.50,3.20,2.90,2.45,3.25,2.95,placeholder,false
45786,2022-12-18,Argentina,France,placeholder,2.10,3.40,3.50,2.05,3.30,3.60,placeholder,false
45776,2022-12-10,Morocco,Portugal,placeholder,5.00,3.80,1.65,5.20,3.75,1.62,placeholder,false
"""


def _write(tmp_path, content=MARKET_CSV):
    p = tmp_path / "market_odds.csv"
    p.write_text(content)
    return p


class TestLoadMarketOdds:
    def test_returns_list(self, tmp_path):
        records = load_market_odds(_write(tmp_path))
        assert isinstance(records, list)
        assert len(records) == 3

    def test_fields_populated(self, tmp_path):
        records = load_market_odds(_write(tmp_path))
        r = next(x for x in records if x.match_id == "45777")
        assert r.date == "2022-12-10"
        assert r.team_a == "England"
        assert r.team_b == "France"
        assert r.bookmaker == "placeholder"
        assert r.opening_home_odds == pytest.approx(2.50)
        assert r.opening_draw_odds == pytest.approx(3.20)
        assert r.opening_away_odds == pytest.approx(2.90)
        assert r.closing_home_odds == pytest.approx(2.45)
        assert r.closing_draw_odds == pytest.approx(3.25)
        assert r.closing_away_odds == pytest.approx(2.95)

    def test_source_type_and_research_valid(self, tmp_path):
        records = load_market_odds(_write(tmp_path))
        for r in records:
            assert r.source_type == "placeholder"
            assert r.research_valid is False

    def test_research_valid_true_parsed(self, tmp_path):
        csv = (
            "match_id,date,team_a,team_b,bookmaker,"
            "opening_home_odds,opening_draw_odds,opening_away_odds,"
            "closing_home_odds,closing_draw_odds,closing_away_odds,"
            "source_type,research_valid\n"
            "45777,2022-12-10,England,France,Bet365,"
            "2.50,3.20,2.90,2.45,3.25,2.95,football_data_co_uk,true\n"
        )
        records = load_market_odds(_write(tmp_path, csv))
        assert records[0].research_valid is True
        assert records[0].source_type == "football_data_co_uk"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_market_odds(tmp_path / "nonexistent.csv")

    def test_match_id_stored_as_string(self, tmp_path):
        records = load_market_odds(_write(tmp_path))
        assert all(isinstance(r.match_id, str) for r in records)
