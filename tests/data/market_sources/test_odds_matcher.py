"""Tests for odds-to-match-id matching logic used by fetch_market_odds.

The matcher takes a list of MarketOddsRow (from any adapter) and a
match_results DataFrame, then resolves each odds row to a match_id
using bidirectional team matching (team_a/team_b may be reversed).
"""

import pytest
import pandas as pd
from src.data.market_sources.odds_matcher import (
    match_odds_to_match_ids,
    OddsMatchResult,
)
from src.data.market_sources.base import MarketOddsRow


def _odds_row(date, team_a, team_b, home_odds=2.5, draw_odds=3.2, away_odds=2.9):
    return MarketOddsRow(
        match_id="",  # not yet assigned
        date=date,
        team_a=team_a,
        team_b=team_b,
        bookmaker="Test",
        opening_home_odds=home_odds,
        opening_draw_odds=draw_odds,
        opening_away_odds=away_odds,
        closing_home_odds=home_odds,
        closing_draw_odds=draw_odds,
        closing_away_odds=away_odds,
        source_type="historical_odds",
        research_valid=True,
    )


MATCH_RESULTS_DF = pd.DataFrame([
    {
        "match_id": 45777, "date": "2022-12-10",
        "team_a": "England", "team_b": "France",
        "team_a_goals": 1, "team_b_goals": 2,
    },
    {
        "match_id": 45786, "date": "2022-12-18",
        "team_a": "Argentina", "team_b": "France",
        "team_a_goals": 3, "team_b_goals": 3,
    },
    {
        "match_id": 45773, "date": "2022-12-09",
        "team_a": "Croatia", "team_b": "Brazil",
        "team_a_goals": 1, "team_b_goals": 1,
    },
])


class TestMatchOddsToMatchIds:
    def test_exact_match_resolved(self):
        odds = [_odds_row("2022-12-10", "England", "France")]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        assert len(result.matched) == 1
        assert result.matched[0].match_id == "45777"

    def test_reversed_teams_resolved(self):
        # Odds source has France as home, England as away
        odds = [_odds_row("2022-12-10", "France", "England")]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        assert len(result.matched) == 1
        # match_id resolved correctly even though teams are reversed
        assert result.matched[0].match_id == "45777"

    def test_reversed_odds_are_swapped_back(self):
        # France is listed as home in odds → home_odds apply to France
        # But our match_results has England as team_a
        # After reversal, team_a=England, so home_odds should become away_odds
        odds = [_odds_row("2022-12-10", "France", "England",
                          home_odds=2.95, draw_odds=3.25, away_odds=2.45)]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        row = result.matched[0]
        # After swap: England (team_a) gets what was the away_odds (2.45)
        assert row.closing_home_odds == pytest.approx(2.45)
        assert row.closing_away_odds == pytest.approx(2.95)

    def test_unmatched_row_reported(self):
        odds = [_odds_row("2022-12-10", "Germany", "Brazil")]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        assert len(result.matched) == 0
        assert len(result.unmatched) == 1
        assert result.unmatched[0]["team_a"] == "Germany"

    def test_multiple_odds_partially_matched(self):
        odds = [
            _odds_row("2022-12-10", "England", "France"),
            _odds_row("2022-12-18", "Argentina", "France"),
            _odds_row("2022-12-25", "Germany", "Spain"),  # not in match_results
        ]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        assert len(result.matched) == 2
        assert len(result.unmatched) == 1

    def test_result_has_summary(self):
        odds = [
            _odds_row("2022-12-10", "England", "France"),
            _odds_row("2022-12-25", "Germany", "Spain"),
        ]
        result = match_odds_to_match_ids(odds, MATCH_RESULTS_DF)
        assert result.total_odds_rows == 2
        assert result.matched_count == 1
        assert result.unmatched_count == 1

    def test_empty_odds_returns_empty_result(self):
        result = match_odds_to_match_ids([], MATCH_RESULTS_DF)
        assert result.matched_count == 0
        assert result.unmatched_count == 0
