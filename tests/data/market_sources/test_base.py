"""Tests for the market source adapter base class and shared types."""

import pytest
from src.data.market_sources.base import (
    OddsAdapter,
    MarketOddsRow,
    DataNotAvailableError,
    MissingCredentialsError,
)


class TestMarketOddsRow:
    def test_required_fields_present(self):
        row = MarketOddsRow(
            match_id="45777",
            date="2022-12-10",
            team_a="England",
            team_b="France",
            bookmaker="Bet365",
            opening_home_odds=2.50,
            opening_draw_odds=3.20,
            opening_away_odds=2.90,
            closing_home_odds=2.45,
            closing_draw_odds=3.25,
            closing_away_odds=2.95,
            source_type="historical_odds",
            research_valid=True,
        )
        assert row.match_id == "45777"
        assert row.research_valid is True
        assert row.source_type == "historical_odds"

    def test_to_dict_has_all_csv_columns(self):
        row = MarketOddsRow(
            match_id="45777", date="2022-12-10", team_a="England", team_b="France",
            bookmaker="Bet365", opening_home_odds=2.50, opening_draw_odds=3.20,
            opening_away_odds=2.90, closing_home_odds=2.45, closing_draw_odds=3.25,
            closing_away_odds=2.95, source_type="historical_odds", research_valid=True,
        )
        d = row.to_dict()
        required_keys = {
            "match_id", "date", "team_a", "team_b", "bookmaker",
            "opening_home_odds", "opening_draw_odds", "opening_away_odds",
            "closing_home_odds", "closing_draw_odds", "closing_away_odds",
            "source_type", "research_valid",
        }
        assert required_keys.issubset(d.keys())


class TestOddsAdapterInterface:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            OddsAdapter()

    def test_concrete_subclass_must_implement_source_name(self):
        class Incomplete(OddsAdapter):
            def is_available(self): return True
            def fetch_wc2022(self): return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_is_available(self):
        class Incomplete(OddsAdapter):
            @property
            def source_name(self): return "test"
            def fetch_wc2022(self): return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_fetch_wc2022(self):
        class Incomplete(OddsAdapter):
            @property
            def source_name(self): return "test"
            def is_available(self): return True

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_concrete_subclass_works(self):
        class Concrete(OddsAdapter):
            @property
            def source_name(self): return "test_source"
            def is_available(self): return False
            def fetch_wc2022(self): return []

        adapter = Concrete()
        assert adapter.source_name == "test_source"
        assert adapter.is_available() is False
        assert adapter.fetch_wc2022() == []


class TestExceptions:
    def test_data_not_available_error_is_exception(self):
        with pytest.raises(DataNotAvailableError):
            raise DataNotAvailableError("test")

    def test_missing_credentials_error_is_exception(self):
        with pytest.raises(MissingCredentialsError):
            raise MissingCredentialsError("test")

    def test_data_not_available_carries_message(self):
        msg = "WC 2022 odds not available from this source"
        try:
            raise DataNotAvailableError(msg)
        except DataNotAvailableError as e:
            assert msg in str(e)

    def test_missing_credentials_carries_source_name(self):
        try:
            raise MissingCredentialsError("the_odds_api: ODDS_API_KEY not set")
        except MissingCredentialsError as e:
            assert "the_odds_api" in str(e)
