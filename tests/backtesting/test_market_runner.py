"""Tests for market_runner — model vs market comparison backtest."""

import pytest
from pathlib import Path

from src.backtesting.market_runner import (
    run_market_comparison,
    MarketComparisonResult,
    MarketComparisonSummary,
)


MARKET_CSV = """\
match_id,date,team_a,team_b,bookmaker,opening_home_odds,opening_draw_odds,opening_away_odds,closing_home_odds,closing_draw_odds,closing_away_odds,source_type,research_valid
45777,2022-12-10,England,France,placeholder,2.50,3.20,2.90,2.45,3.25,2.95,placeholder,false
45786,2022-12-18,Argentina,France,placeholder,2.10,3.40,3.50,2.05,3.30,3.60,placeholder,false
45776,2022-12-10,Morocco,Portugal,placeholder,5.00,3.80,1.65,5.20,3.75,1.62,placeholder,false
45723,2022-11-22,Argentina,Saudi Arabia,placeholder,1.25,6.50,12.00,1.22,6.00,13.00,placeholder,false
45773,2022-12-09,Croatia,Brazil,placeholder,4.50,3.60,1.75,4.80,3.70,1.72,placeholder,false
"""

MATCH_RESULTS_CSV = """\
match_id,date,team_a,team_b,team_a_goals,team_b_goals,team_a_elo_pre,team_b_elo_pre,team_a_goals_for_last_10,team_a_goals_against_last_10,team_b_goals_for_last_10,team_b_goals_against_last_10,team_a_points_per_game_last_10,team_b_points_per_game_last_10,team_a_matches_available,team_b_matches_available
45777,2022-12-10,England,France,1,2,2020.0,2085.0,1.8,0.9,2.1,0.7,2.1,2.4,10,10
45786,2022-12-18,Argentina,France,3,3,2134.0,2085.0,2.3,0.6,2.1,0.7,2.6,2.4,10,10
45776,2022-12-10,Morocco,Portugal,1,0,1902.0,2052.0,1.2,0.8,2.2,0.8,1.8,2.3,10,10
45723,2022-11-22,Argentina,Saudi Arabia,1,2,2134.0,1834.0,2.3,0.6,1.1,1.4,2.6,1.5,10,10
45773,2022-12-09,Croatia,Brazil,1,1,1982.0,2108.0,1.4,0.9,2.1,0.7,1.9,2.5,10,10
"""

STRENGTH_PARAMS_CSV = """\
team,alpha_attack,beta_defense,matches_used
England,1.05,0.92,50
France,1.18,0.88,50
Argentina,1.25,0.85,50
Morocco,0.92,0.98,50
Portugal,1.15,0.90,50
Saudi Arabia,0.80,1.05,50
Croatia,1.00,0.95,50
Brazil,1.22,0.86,50
"""


def _write_fixtures(tmp_path, market_csv=MARKET_CSV):
    (tmp_path / "market_odds.csv").write_text(market_csv)
    (tmp_path / "match_results.csv").write_text(MATCH_RESULTS_CSV)
    (tmp_path / "team_strength_params.csv").write_text(STRENGTH_PARAMS_CSV)
    return tmp_path


class TestRunMarketComparison:
    def test_returns_list_of_results(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        assert isinstance(results, list)
        assert len(results) == 5

    def test_result_has_required_fields(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        r = results[0]
        assert hasattr(r, "match_id")
        assert hasattr(r, "team_a")
        assert hasattr(r, "team_b")
        assert hasattr(r, "model_win_a")
        assert hasattr(r, "model_draw")
        assert hasattr(r, "model_win_b")
        assert hasattr(r, "market_home")
        assert hasattr(r, "market_draw")
        assert hasattr(r, "market_away")
        assert hasattr(r, "home_divergence")
        assert hasattr(r, "draw_divergence")
        assert hasattr(r, "away_divergence")
        assert hasattr(r, "largest_divergence_outcome")
        assert hasattr(r, "largest_divergence_value")
        assert hasattr(r, "actual_outcome")
        assert hasattr(r, "model_closer_than_market")
        assert hasattr(r, "market_research_valid")

    def test_probabilities_sum_to_one(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        for r in results:
            assert r.model_win_a + r.model_draw + r.model_win_b == pytest.approx(1.0, abs=1e-4)
            assert r.market_home + r.market_draw + r.market_away == pytest.approx(1.0, abs=1e-4)

    def test_model_closer_than_market_is_boolean(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        for r in results:
            assert isinstance(r.model_closer_than_market, bool)

    def test_divergence_sign_correct(self, tmp_path):
        """home_divergence = model_win_a - market_home."""
        d = _write_fixtures(tmp_path)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        for r in results:
            assert r.home_divergence == pytest.approx(
                r.model_win_a - r.market_home, abs=1e-5
            )

    def test_market_research_valid_false_for_placeholder(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        for r in results:
            assert r.market_research_valid is False

    def test_matches_without_market_data_excluded(self, tmp_path):
        """Only matches that have market odds should appear in results."""
        sparse_market = """\
match_id,date,team_a,team_b,bookmaker,opening_home_odds,opening_draw_odds,opening_away_odds,closing_home_odds,closing_draw_odds,closing_away_odds,source_type,research_valid
45777,2022-12-10,England,France,placeholder,2.50,3.20,2.90,2.45,3.25,2.95,placeholder,false
"""
        d = _write_fixtures(tmp_path, market_csv=sparse_market)
        results, _ = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        assert len(results) == 1
        assert results[0].match_id == "45777"


class TestMarketComparisonSummary:
    def test_summary_has_required_fields(self, tmp_path):
        d = _write_fixtures(tmp_path)
        _, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        assert hasattr(summary, "total_matches")
        assert hasattr(summary, "model_brier")
        assert hasattr(summary, "market_brier")
        assert hasattr(summary, "brier_delta")
        assert hasattr(summary, "avg_absolute_divergence")
        assert hasattr(summary, "high_divergence_count")
        assert hasattr(summary, "model_wins_high_divergence")
        assert hasattr(summary, "market_wins_high_divergence")
        assert hasattr(summary, "is_research_valid")
        assert hasattr(summary, "disclaimer")

    def test_brier_delta_is_market_minus_model(self, tmp_path):
        """brier_delta = market_brier - model_brier (positive = model is better)."""
        d = _write_fixtures(tmp_path)
        _, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        assert summary.brier_delta == pytest.approx(
            summary.market_brier - summary.model_brier, abs=1e-6
        )

    def test_placeholder_data_not_research_valid(self, tmp_path):
        d = _write_fixtures(tmp_path)
        _, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        assert summary.is_research_valid is False
        assert "engineering" in summary.disclaimer.lower()

    def test_high_divergence_threshold_5pp(self, tmp_path):
        """high_divergence_count uses 5 percentage point threshold."""
        d = _write_fixtures(tmp_path)
        results, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        expected = sum(
            1 for r in results if abs(r.largest_divergence_value) >= 0.05
        )
        assert summary.high_divergence_count == expected

    def test_avg_absolute_divergence_correct(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results, summary = run_market_comparison(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            market_odds_path=d / "market_odds.csv",
        )
        expected = sum(abs(r.largest_divergence_value) for r in results) / len(results)
        assert summary.avg_absolute_divergence == pytest.approx(expected, rel=1e-5)
