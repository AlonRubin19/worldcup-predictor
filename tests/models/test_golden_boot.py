"""Tests for golden_boot.py — tournament Golden Boot projection model.

TDD: written RED first.

DATA NOTE: player profiles are engineering-valid only for the 8 teams
present in player_profiles.csv (same caveat as top_scorer.py).
"""

from __future__ import annotations

import math
import pytest

from src.models.golden_boot import (
    GoldenBootPlayerResult,
    expected_team_matches,
    compute_xgt,
    poisson_at_least,
    most_likely_goals,
    predict_golden_boot,
)
from src.tournament.simulator import MonteCarloResult
from src.data.player_loader import PlayerProfile


def _mc():
    return MonteCarloResult(
        n_simulations=1000,
        win_tournament={"England": 0.10, "Brazil": 0.15},
        reach_final={"England": 0.25, "Brazil": 0.30},
        reach_sf={"England": 0.45, "Brazil": 0.50},
        reach_qf={"England": 0.65, "Brazil": 0.70},
        reach_r16={"England": 0.85, "Brazil": 0.90},
    )


def _player(player_id, name, team, xg_per_90=0.5, position="FW"):
    return PlayerProfile(
        player_id=player_id, player_name=name, team=team, position=position,
        club="Club", minutes_last_90_days=810,
        national_team_minutes_last_12_months=720,
        goals_per_90=xg_per_90, assists_per_90=0.1, xg_per_90=xg_per_90,
        xa_per_90=0.1, defensive_actions_per_90=1.0,
        international_caps=50, base_impact_score=1.2,
    )


class TestExpectedTeamMatches:
    def test_known_team(self):
        mc = _mc()
        # 3 group matches + reach probabilities for 4 knockout rounds
        expected = 3 + 0.85 + 0.65 + 0.45 + 0.25
        assert expected_team_matches("England", mc) == pytest.approx(expected)

    def test_unknown_team_defaults_to_group_stage_only(self):
        mc = _mc()
        assert expected_team_matches("Unknown", mc) == pytest.approx(3.0)


class TestComputeXgt:
    def test_basic_formula(self):
        xgt = compute_xgt(
            expected_team_matches=3.85,
            expected_minutes=90.0,
            xg_per_90=0.5,
            penalty_factor=1.0,
            starting_probability=1.0,
        )
        assert xgt == pytest.approx(3.85 * (90.0 / 90.0) * 0.5 * 1.0 * 1.0)

    def test_lower_starting_probability_reduces_xgt(self):
        full = compute_xgt(3.85, 90.0, 0.5, 1.0, 1.0)
        bench = compute_xgt(3.85, 90.0, 0.5, 1.0, 0.3)
        assert bench < full
        assert bench == pytest.approx(full * 0.3)

    def test_penalty_factor_increases_xgt(self):
        base = compute_xgt(3.85, 90.0, 0.5, 1.0, 1.0)
        with_pens = compute_xgt(3.85, 90.0, 0.5, 1.1, 1.0)
        assert with_pens > base

    def test_zero_minutes_gives_zero_xgt(self):
        assert compute_xgt(3.85, 0.0, 0.5, 1.0, 1.0) == pytest.approx(0.0)


class TestPoissonAtLeast:
    def test_zero_lambda(self):
        assert poisson_at_least(0.0, 1) == pytest.approx(0.0, abs=1e-9)

    def test_p_at_least_1_equals_1_minus_p0(self):
        lam = 1.5
        p = poisson_at_least(lam, 1)
        assert p == pytest.approx(1 - math.exp(-lam))

    def test_p_at_least_decreases_with_threshold(self):
        lam = 4.0
        assert poisson_at_least(lam, 3) > poisson_at_least(lam, 5) > poisson_at_least(lam, 7)

    def test_in_range(self):
        for lam in [0.5, 2.0, 6.0]:
            for k in [3, 5, 7]:
                p = poisson_at_least(lam, k)
                assert 0.0 <= p <= 1.0


class TestMostLikelyGoals:
    def test_mode_of_poisson_is_floor_lambda(self):
        assert most_likely_goals(4.7) == 4
        assert most_likely_goals(0.3) == 0
        assert most_likely_goals(2.0) in (1, 2)


class TestPredictGoldenBoot:
    def setup_method(self):
        self.mc = _mc()
        self.profiles = {
            "p1": _player("p1", "Star Striker", "England", xg_per_90=0.85),
            "p2": _player("p2", "Backup Striker", "England", xg_per_90=0.20),
            "p3": _player("p3", "Brazil Forward", "Brazil", xg_per_90=0.70),
            "gk": _player("gk", "Goalkeeper", "England", xg_per_90=0.0, position="GK"),
        }
        self.result = predict_golden_boot(self.profiles, self.mc, n_sims=2000, rng_seed=42)

    def test_returns_list_of_results(self):
        assert all(isinstance(r, GoldenBootPlayerResult) for r in self.result)

    def test_sorted_by_xgt_descending(self):
        xgts = [r.expected_goals for r in self.result]
        assert xgts == sorted(xgts, reverse=True)

    def test_top_scorer_is_highest_xgt_player(self):
        assert self.result[0].player_id == "p1"

    def test_probabilities_in_range(self):
        for r in self.result:
            assert 0.0 <= r.prob_top_scorer <= 1.0
            assert 0.0 <= r.prob_score_3plus <= 1.0
            assert 0.0 <= r.prob_score_5plus <= 1.0
            assert 0.0 <= r.prob_score_7plus <= 1.0

    def test_thresholds_are_monotonically_decreasing(self):
        for r in self.result:
            assert r.prob_score_3plus >= r.prob_score_5plus >= r.prob_score_7plus

    def test_top_scorer_probabilities_sum_close_to_one(self):
        total = sum(r.prob_top_scorer for r in self.result)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_zero_xg_player_has_zero_top_scorer_prob(self):
        gk = next(r for r in self.result if r.player_id == "gk")
        assert gk.prob_top_scorer == pytest.approx(0.0, abs=1e-9)

    def test_most_likely_goals_is_int(self):
        for r in self.result:
            assert isinstance(r.most_likely_goals, int)

    def test_empty_profiles_returns_empty_list(self):
        assert predict_golden_boot({}, self.mc, n_sims=100) == []
