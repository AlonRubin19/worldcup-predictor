"""Tests for top_scorer.py — match-level player goal probability model.

TDD: all tests written RED first.
No Streamlit dependency — pure computation.

Data note: player profiles are engineering-valid only for the 8 teams
that have profiles in player_profiles.csv.
"""

from __future__ import annotations

import math
import pytest

from src.models.top_scorer import (
    PlayerMatchPrediction,
    TopScorerResult,
    predict_top_scorers,
    player_score_probability,
    player_brace_probability,
)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal in-memory player rows for tests (no CSV dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _player(
    player_id: str,
    name: str,
    team: str,
    position: str = "FW",
    xg_per_90: float = 0.5,
    goals_per_90: float = 0.5,
    availability_factor: float = 1.0,
    expected_starter: bool = True,
) -> dict:
    return {
        "player_id": player_id,
        "player_name": name,
        "team": team,
        "position": position,
        "xg_per_90": xg_per_90,
        "goals_per_90": goals_per_90,
        "availability_factor": availability_factor,
        "expected_starter": expected_starter,
    }


_FRANCE_SQUAD = [
    _player("p1", "Kylian Mbappé",   "France", "FW", xg_per_90=0.85),
    _player("p2", "Olivier Giroud",  "France", "FW", xg_per_90=0.50),
    _player("p3", "Ousmane Dembélé", "France", "MF", xg_per_90=0.28),
    _player("p4", "Antoine Griezmann","France","MF", xg_per_90=0.35),
    _player("p5", "Raphaël Varane",  "France", "DF", xg_per_90=0.06),
    _player("p6", "Hugo Lloris",     "France", "GK", xg_per_90=0.00),
]

_MOROCCO_SQUAD = [
    _player("m1", "Youssef En-Nesyri","Morocco","FW", xg_per_90=0.40),
    _player("m2", "Hakim Ziyech",    "Morocco", "MF", xg_per_90=0.22),
    _player("m3", "Sofiane Boufal",  "Morocco", "MF", xg_per_90=0.18),
    _player("m4", "Achraf Hakimi",   "Morocco", "DF", xg_per_90=0.10),
    _player("m5", "Romain Saïss",    "Morocco", "DF", xg_per_90=0.05),
    _player("m6", "Yassine Bounou",  "Morocco", "GK", xg_per_90=0.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# Probability helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityHelpers:
    def test_player_score_prob_zero_lambda(self):
        assert player_score_probability(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_player_score_prob_positive(self):
        p = player_score_probability(1.0)
        assert abs(p - (1 - math.exp(-1.0))) < 1e-6

    def test_player_score_prob_in_range(self):
        for lam in [0.1, 0.5, 1.0, 2.0]:
            p = player_score_probability(lam)
            assert 0.0 < p < 1.0

    def test_player_score_prob_increases_with_lambda(self):
        assert player_score_probability(0.5) < player_score_probability(1.0)

    def test_brace_prob_less_than_score_prob(self):
        lam = 0.8
        assert player_brace_probability(lam) < player_score_probability(lam)

    def test_brace_prob_zero_lambda(self):
        assert player_brace_probability(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_brace_prob_positive(self):
        """P(scores ≥ 2) = 1 - P(0) - P(1) using Poisson."""
        lam = 1.5
        p0 = math.exp(-lam)
        p1 = lam * math.exp(-lam)
        expected = 1.0 - p0 - p1
        assert abs(player_brace_probability(lam) - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# PlayerMatchPrediction structure
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerMatchPrediction:
    def _run_one(self):
        result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
        )
        return result.team_a_players[0]

    def test_has_player_name(self):
        p = self._run_one()
        assert isinstance(p.player_name, str)
        assert len(p.player_name) > 0

    def test_has_team(self):
        p = self._run_one()
        assert p.team == "France"

    def test_has_position(self):
        p = self._run_one()
        assert p.position in ("FW", "MF", "DF", "GK")

    def test_has_expected_goals(self):
        p = self._run_one()
        assert isinstance(p.expected_goals, float)
        assert p.expected_goals >= 0.0

    def test_has_prob_scores(self):
        p = self._run_one()
        assert 0.0 <= p.prob_scores <= 1.0

    def test_has_prob_brace(self):
        p = self._run_one()
        assert 0.0 <= p.prob_brace <= p.prob_scores


# ─────────────────────────────────────────────────────────────────────────────
# TopScorerResult structure
# ─────────────────────────────────────────────────────────────────────────────

class TestTopScorerResult:
    def setup_method(self):
        self.result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
        )

    def test_returns_top_scorer_result(self):
        assert isinstance(self.result, TopScorerResult)

    def test_has_team_names(self):
        assert self.result.team_a == "France"
        assert self.result.team_b == "Morocco"

    def test_team_a_players_not_empty(self):
        assert len(self.result.team_a_players) > 0

    def test_team_b_players_not_empty(self):
        assert len(self.result.team_b_players) > 0

    def test_team_a_players_sorted_descending(self):
        """Players must be sorted by expected_goals, highest first."""
        goals = [p.expected_goals for p in self.result.team_a_players]
        assert goals == sorted(goals, reverse=True)

    def test_team_b_players_sorted_descending(self):
        goals = [p.expected_goals for p in self.result.team_b_players]
        assert goals == sorted(goals, reverse=True)

    def test_has_overall_favourite(self):
        assert self.result.overall_favourite is not None
        assert isinstance(self.result.overall_favourite, PlayerMatchPrediction)

    def test_overall_favourite_is_highest_expected_goals(self):
        """Overall favourite must be the player with the highest expected_goals."""
        fav = self.result.overall_favourite
        all_players = self.result.team_a_players + self.result.team_b_players
        max_lambda = max(p.expected_goals for p in all_players)
        assert abs(fav.expected_goals - max_lambda) < 1e-9

    def test_has_data_valid_flag(self):
        assert isinstance(self.result.data_valid, bool)

    def test_has_teams_without_data_list(self):
        assert isinstance(self.result.teams_without_data, list)

    def test_teams_with_full_data_have_empty_missing_list(self):
        assert self.result.teams_without_data == []


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestMathematicalCorrectness:
    def setup_method(self):
        self.xg_a = 1.8
        self.result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=self.xg_a, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
        )

    def test_team_a_expected_goals_sum_equals_team_xg(self):
        """Sum of all player lambdas must equal the team xG passed in."""
        total = sum(p.expected_goals for p in self.result.team_a_players)
        assert abs(total - self.xg_a) < 0.001, \
            f"Sum of player lambdas {total:.4f} != team xG {self.xg_a}"

    def test_team_b_expected_goals_sum_equals_team_xg(self):
        xg_b = 0.9
        total = sum(p.expected_goals for p in self.result.team_b_players)
        assert abs(total - xg_b) < 0.001

    def test_forward_has_higher_lambda_than_goalkeeper(self):
        # Use top_n=None to ensure GK is included (default top_n=5 may exclude 0-lambda players)
        result_all = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
            top_n=None,
        )
        fw = next(p for p in result_all.team_a_players if p.position == "FW")
        gk = next(p for p in result_all.team_a_players if p.position == "GK")
        assert fw.expected_goals > gk.expected_goals

    def test_higher_xg_player_has_higher_lambda(self):
        """Mbappé (xg 0.85) should rank above Giroud (xg 0.50)."""
        mbappe = next(p for p in self.result.team_a_players if "Mbapp" in p.player_name)
        giroud = next(p for p in self.result.team_a_players if "Giroud" in p.player_name)
        assert mbappe.expected_goals > giroud.expected_goals

    def test_higher_team_xg_means_higher_player_lambdas(self):
        """France (xg=1.8) players should have higher avg lambda than Morocco (xg=0.9) players."""
        avg_a = sum(p.expected_goals for p in self.result.team_a_players) / len(self.result.team_a_players)
        avg_b = sum(p.expected_goals for p in self.result.team_b_players) / len(self.result.team_b_players)
        assert avg_a > avg_b

    def test_prob_scores_consistent_with_expected_goals(self):
        """P(scores) must equal 1 - e^(-lambda) for every player."""
        for p in self.result.team_a_players + self.result.team_b_players:
            expected_p = 1 - math.exp(-p.expected_goals)
            assert abs(p.prob_scores - expected_p) < 1e-6, \
                f"{p.player_name}: prob_scores {p.prob_scores:.4f} != 1-e^(-{p.expected_goals:.4f})"

    def test_unavailable_player_has_zero_lambda(self):
        """A player with availability_factor=0 must contribute zero expected goals."""
        squad_with_absent = _FRANCE_SQUAD + [
            _player("p99", "Injured Star", "France", "FW",
                    xg_per_90=0.9, availability_factor=0.0, expected_starter=False)
        ]
        result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=squad_with_absent, squad_b=_MOROCCO_SQUAD,
            top_n=None,  # include all players so injured star is visible
        )
        injured = next(p for p in result.team_a_players if p.player_name == "Injured Star")
        assert injured.expected_goals == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Missing / partial data handling
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingDataHandling:
    def test_empty_squad_returns_result_with_warning(self):
        result = predict_top_scorers(
            team_a="UnknownTeam", team_b="Morocco",
            xg_a=1.5, xg_b=0.8,
            squad_a=[], squad_b=_MOROCCO_SQUAD,
        )
        assert "UnknownTeam" in result.teams_without_data

    def test_both_squads_empty_still_returns_result(self):
        result = predict_top_scorers(
            team_a="A", team_b="B",
            xg_a=1.0, xg_b=1.0,
            squad_a=[], squad_b=[],
        )
        assert result is not None
        assert result.overall_favourite is None

    def test_data_valid_false_when_squad_missing(self):
        result = predict_top_scorers(
            team_a="NoData", team_b="Morocco",
            xg_a=1.0, xg_b=0.8,
            squad_a=[], squad_b=_MOROCCO_SQUAD,
        )
        assert result.data_valid is False

    def test_data_valid_true_when_both_squads_present(self):
        result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
        )
        assert result.data_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# Top-N filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestTopNFiltering:
    def test_top_n_limits_returned_players(self):
        result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
            top_n=3,
        )
        assert len(result.team_a_players) <= 3
        assert len(result.team_b_players) <= 3

    def test_top_n_none_returns_all_players(self):
        result = predict_top_scorers(
            team_a="France", team_b="Morocco",
            xg_a=1.8, xg_b=0.9,
            squad_a=_FRANCE_SQUAD, squad_b=_MOROCCO_SQUAD,
            top_n=None,
        )
        assert len(result.team_a_players) == len(_FRANCE_SQUAD)
        assert len(result.team_b_players) == len(_MOROCCO_SQUAD)
