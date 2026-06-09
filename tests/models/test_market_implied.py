"""Tests for market_implied — overround removal and divergence calculation."""

import pytest
from src.models.market_implied import (
    decimal_odds_to_implied_probabilities,
    ImpliedProbabilities,
    calculate_market_divergence,
    MarketDivergence,
)


class TestDecimalOddsToImpliedProbabilities:
    def test_returns_implied_probabilities(self):
        result = decimal_odds_to_implied_probabilities(2.50, 3.20, 2.90)
        assert isinstance(result, ImpliedProbabilities)

    def test_probabilities_sum_to_one(self):
        result = decimal_odds_to_implied_probabilities(2.50, 3.20, 2.90)
        assert result.home + result.draw + result.away == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_match_roughly_equal(self):
        # Equal odds → roughly equal probabilities after overround removal
        result = decimal_odds_to_implied_probabilities(2.10, 3.20, 2.10)
        assert result.home == pytest.approx(result.away, rel=1e-4)

    def test_favourite_has_higher_probability(self):
        # Home is big favourite (short odds = high implied probability)
        result = decimal_odds_to_implied_probabilities(1.30, 5.00, 9.00)
        assert result.home > result.draw
        assert result.home > result.away

    def test_overround_is_removed(self):
        # Raw implied sum > 1.0 (bookmaker margin) → after removal should sum to 1.0
        home_raw = 1 / 2.50
        draw_raw = 1 / 3.20
        away_raw = 1 / 2.90
        raw_sum = home_raw + draw_raw + away_raw
        assert raw_sum > 1.0  # confirm there is overround

        result = decimal_odds_to_implied_probabilities(2.50, 3.20, 2.90)
        assert result.home + result.draw + result.away == pytest.approx(1.0, abs=1e-6)

    def test_known_values(self):
        # Hand-verified: raw = [0.4, 0.3125, 0.3448] → sum = 1.0573
        # normalised: [0.4/1.0573, 0.3125/1.0573, 0.3448/1.0573]
        result = decimal_odds_to_implied_probabilities(2.50, 3.20, 2.90)
        raw_sum = 1 / 2.50 + 1 / 3.20 + 1 / 2.90
        assert result.home == pytest.approx((1 / 2.50) / raw_sum, rel=1e-5)
        assert result.draw == pytest.approx((1 / 3.20) / raw_sum, rel=1e-5)
        assert result.away == pytest.approx((1 / 2.90) / raw_sum, rel=1e-5)

    def test_overround_stored(self):
        result = decimal_odds_to_implied_probabilities(2.50, 3.20, 2.90)
        expected_overround = (1 / 2.50 + 1 / 3.20 + 1 / 2.90) - 1.0
        assert result.overround == pytest.approx(expected_overround, rel=1e-5)

    def test_invalid_odds_raises(self):
        with pytest.raises(ValueError):
            decimal_odds_to_implied_probabilities(0.5, 3.20, 2.90)  # odds < 1 invalid

    def test_all_three_outcomes_positive(self):
        result = decimal_odds_to_implied_probabilities(1.40, 4.50, 7.00)
        assert result.home > 0
        assert result.draw > 0
        assert result.away > 0


class TestCalculateMarketDivergence:
    def _model(self, win_a, draw, win_b):
        return {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}

    def _market(self, home, draw, away):
        return ImpliedProbabilities(home=home, draw=draw, away=away, overround=0.05)

    def test_returns_divergence(self):
        d = calculate_market_divergence(
            self._model(0.45, 0.28, 0.27),
            self._market(0.50, 0.27, 0.23),
        )
        assert isinstance(d, MarketDivergence)

    def test_zero_divergence_when_equal(self):
        d = calculate_market_divergence(
            self._model(0.50, 0.27, 0.23),
            self._market(0.50, 0.27, 0.23),
        )
        assert d.home_divergence == pytest.approx(0.0)
        assert d.draw_divergence == pytest.approx(0.0)
        assert d.away_divergence == pytest.approx(0.0)

    def test_divergence_values_correct(self):
        d = calculate_market_divergence(
            self._model(0.45, 0.28, 0.27),
            self._market(0.50, 0.27, 0.23),
        )
        assert d.home_divergence == pytest.approx(0.45 - 0.50, abs=1e-6)
        assert d.draw_divergence == pytest.approx(0.28 - 0.27, abs=1e-6)
        assert d.away_divergence == pytest.approx(0.27 - 0.23, abs=1e-6)

    def test_largest_divergence_outcome_identified(self):
        d = calculate_market_divergence(
            self._model(0.30, 0.28, 0.42),
            self._market(0.50, 0.27, 0.23),
        )
        # home: 0.30-0.50=-0.20 (abs=0.20), draw: 0.01, away: 0.42-0.23=+0.19
        # largest by abs is home: signed value = -0.20
        assert d.largest_divergence_outcome == "team_a_win"
        assert d.largest_divergence_value == pytest.approx(-0.20, abs=1e-6)

    def test_largest_divergence_away(self):
        d = calculate_market_divergence(
            self._model(0.49, 0.28, 0.23),
            self._market(0.50, 0.28, 0.40),
        )
        assert d.largest_divergence_outcome == "team_b_win"

    def test_largest_divergence_draw(self):
        d = calculate_market_divergence(
            self._model(0.50, 0.45, 0.05),
            self._market(0.50, 0.28, 0.22),
        )
        assert d.largest_divergence_outcome == "draw"
