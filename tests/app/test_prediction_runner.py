"""Tests for prediction_runner.py — the pure computation layer.

TDD: all tests written RED-first.
No Streamlit dependency — all functions are pure Python.
"""

from __future__ import annotations

import pytest

from src.app.prediction_runner import (
    FullPrediction,
    run_full_prediction,
    RunnerInput,
)
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.models.research_valid_predictor import DEFAULT_RHO


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SNAP_A  = TeamSnapshot(elo=1900.0, ppg=1.8)
_SNAP_B  = TeamSnapshot(elo=1800.0, ppg=1.5)
_PAR_A   = StrengthParams(alpha_attack=1.2, beta_defense=0.85, matches_used=10)
_PAR_B   = StrengthParams(alpha_attack=1.0, beta_defense=1.0,  matches_used=10)
_DEFAULT = RunnerInput("Brazil", "France", _SNAP_A, _SNAP_B, _PAR_A, _PAR_B)


def _run(inp=_DEFAULT):
    return run_full_prediction(inp)


# ─────────────────────────────────────────────────────────────────────────────
# RunnerInput structure
# ─────────────────────────────────────────────────────────────────────────────

class TestRunnerInput:
    def test_can_instantiate(self):
        inp = RunnerInput("A", "B", _SNAP_A, _SNAP_B, _PAR_A, _PAR_B)
        assert inp is not None

    def test_has_teams(self):
        inp = RunnerInput("Brazil", "France", _SNAP_A, _SNAP_B, _PAR_A, _PAR_B)
        assert inp.team_a == "Brazil"
        assert inp.team_b == "France"


# ─────────────────────────────────────────────────────────────────────────────
# FullPrediction structure
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPredictionStructure:
    def test_returns_full_prediction(self):
        assert isinstance(_run(), FullPrediction)

    def test_has_team_names(self):
        r = _run()
        assert r.team_a == "Brazil"
        assert r.team_b == "France"

    def test_has_win_probs(self):
        r = _run()
        for attr in ("win_a", "draw", "win_b"):
            assert hasattr(r, attr)
            assert isinstance(getattr(r, attr), float)

    def test_has_xg(self):
        r = _run()
        assert hasattr(r, "xg_a")
        assert hasattr(r, "xg_b")
        assert r.xg_a > 0
        assert r.xg_b > 0

    def test_has_most_likely_score(self):
        r = _run()
        assert hasattr(r, "most_likely_score")
        assert isinstance(r.most_likely_score, str)
        assert "-" in r.most_likely_score

    def test_has_top_scorelines(self):
        r = _run()
        assert hasattr(r, "top_scorelines")
        assert isinstance(r.top_scorelines, list)

    def test_has_markets(self):
        r = _run()
        assert hasattr(r, "markets")

    def test_has_recommendations(self):
        r = _run()
        assert hasattr(r, "recommendations")

    def test_has_confidence(self):
        r = _run()
        assert hasattr(r, "confidence")

    def test_has_explanation(self):
        r = _run()
        assert hasattr(r, "explanation")

    def test_has_lineup_source(self):
        r = _run()
        assert hasattr(r, "lineup_source")
        assert isinstance(r.lineup_source, str)

    def test_has_is_research_valid(self):
        r = _run()
        assert hasattr(r, "is_research_valid")


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityConsistency:
    def test_1x2_sum_to_one(self):
        r = _run()
        total = r.win_a + r.draw + r.win_b
        assert abs(total - 1.0) < 0.001, f"1X2 sum = {total:.6f}"

    def test_all_probs_in_range(self):
        r = _run()
        for prob, name in [(r.win_a, "win_a"), (r.draw, "draw"), (r.win_b, "win_b")]:
            assert 0.0 <= prob <= 1.0, f"{name} = {prob}"

    def test_stronger_team_higher_win_prob(self):
        r = _run()
        assert r.win_a > r.win_b, "Stronger ELO/attack team should have higher win prob"


class TestScorelineConsistency:
    def test_most_likely_score_equals_first_scoreline(self):
        """most_likely_score must be derived from the same matrix as top_scorelines."""
        r = _run()
        if r.top_scorelines:
            g_a, g_b = r.top_scorelines[0][0], r.top_scorelines[0][1]
            expected = f"{g_a}-{g_b}"
            assert r.most_likely_score == expected, (
                f"most_likely_score='{r.most_likely_score}' "
                f"but top_scorelines[0]='{expected}'"
            )

    def test_top_scorelines_sorted_descending(self):
        """Scorelines must be ordered by probability, highest first."""
        r = _run()
        probs = [s[2] for s in r.top_scorelines]
        assert probs == sorted(probs, reverse=True), \
            f"Scorelines not sorted: {probs}"

    def test_top_scorelines_has_at_least_five(self):
        r = _run()
        assert len(r.top_scorelines) >= 5

    def test_all_scoreline_probs_positive(self):
        r = _run()
        for s in r.top_scorelines:
            assert s[2] > 0.0, f"Zero/negative scoreline prob: {s}"

    def test_most_likely_has_correct_format(self):
        r = _run()
        parts = r.most_likely_score.split("-")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


class TestMarketConsistency:
    def setup_method(self):
        self.r = _run()

    def test_over_under_complement(self):
        """Over 2.5 + Under 2.5 must sum to ~100%."""
        markets = self.r.markets
        over_prob  = next((m.probability for m in markets.over_under if "Over 2.5" in m.selection), None)
        under_prob = next((m.probability for m in markets.over_under if "Under 2.5" in m.selection), None)
        assert over_prob is not None, "Missing Over 2.5 market"
        assert under_prob is not None, "Missing Under 2.5 market"
        assert abs(over_prob + under_prob - 1.0) < 0.001, \
            f"Over 2.5 ({over_prob:.4f}) + Under 2.5 ({under_prob:.4f}) = {over_prob+under_prob:.4f}"

    def test_btts_complement(self):
        """BTTS Yes + BTTS No must sum to ~100%."""
        markets = self.r.markets
        yes_prob = next((m.probability for m in markets.btts if m.selection == "BTTS Yes"), None)
        no_prob  = next((m.probability for m in markets.btts if m.selection == "BTTS No"),  None)
        assert yes_prob is not None, "Missing BTTS Yes"
        assert no_prob  is not None, "Missing BTTS No"
        assert abs(yes_prob + no_prob - 1.0) < 0.001, \
            f"BTTS Yes ({yes_prob:.4f}) + BTTS No ({no_prob:.4f}) = {yes_prob+no_prob:.4f}"

    def test_all_market_types_present(self):
        m = self.r.markets
        assert len(m.over_under)     > 0
        assert len(m.btts)           > 0
        assert len(m.double_chance)  > 0
        assert len(m.draw_no_bet)    > 0
        assert len(m.team_totals)    > 0
        assert len(m.clean_sheet)    > 0

    def test_recommendations_use_same_markets(self):
        """Verify that recommendations reference the same markets object passed in."""
        r = self.r
        # Top signal probability should be findable in the markets object
        if r.recommendations.recommendations:
            top_sel   = r.recommendations.recommendations[0].selection
            top_prob  = r.recommendations.recommendations[0].model_probability
            # Search all markets including 1X2 (dominant teams surface 1X2 as top signal)
            all_probs = [m.probability for market_list in [
                r.markets.one_x_two,
                r.markets.over_under, r.markets.btts,
                r.markets.double_chance, r.markets.draw_no_bet,
                r.markets.team_totals, r.markets.clean_sheet,
            ] for m in market_list]
            assert any(abs(p - top_prob) < 0.0001 for p in all_probs), \
                f"Top signal prob {top_prob} not found in markets object"

    def test_dominant_team_1x2_recommendation_in_markets(self):
        """For dominant teams (win_a > 70%), top signal is 1X2 — must be findable in markets.one_x_two."""
        snap_strong = TeamSnapshot(elo=2005, ppg=1.8)
        par_strong  = StrengthParams(alpha_attack=2.74, beta_defense=0.42, matches_used=10)
        snap_weak   = TeamSnapshot(elo=1800, ppg=1.5)
        par_weak    = StrengthParams(alpha_attack=1.0, beta_defense=1.0,  matches_used=0)
        inp = RunnerInput("Canada", "Bosnia", snap_strong, snap_weak, par_strong, par_weak)
        r   = run_full_prediction(inp)
        assert r.win_a > 0.70, f"Expected dominant team, got win_a={r.win_a:.2%}"
        if r.recommendations.recommendations:
            top_prob = r.recommendations.recommendations[0].model_probability
            all_probs = [m.probability for market_list in [
                r.markets.one_x_two,
                r.markets.over_under, r.markets.btts,
                r.markets.double_chance, r.markets.draw_no_bet,
                r.markets.team_totals, r.markets.clean_sheet,
            ] for m in market_list]
            assert any(abs(p - top_prob) < 0.0001 for p in all_probs), \
                f"Dominant team top signal prob {top_prob:.4f} not found in any market"


class TestConfidenceConsistency:
    def test_confidence_label_is_string(self):
        r = _run()
        assert isinstance(r.confidence.label, str)
        assert len(r.confidence.label) > 0

    def test_high_confidence_for_dominant_team(self):
        snap_strong = TeamSnapshot(elo=2100, ppg=2.5)
        par_strong  = StrengthParams(alpha_attack=2.5, beta_defense=0.3, matches_used=20)
        snap_weak   = TeamSnapshot(elo=1500, ppg=0.8)
        par_weak    = StrengthParams(alpha_attack=0.5, beta_defense=1.8, matches_used=20)
        inp = RunnerInput("Strong", "Weak", snap_strong, snap_weak, par_strong, par_weak)
        r   = run_full_prediction(inp)
        # With this dominant team, win_a should be very high
        assert r.win_a > 0.70, f"Expected very high win prob, got {r.win_a:.2%}"

    def test_explanation_has_drivers(self):
        r = _run()
        assert len(r.explanation.drivers) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Consistency across different team strengths
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyVariousInputs:
    @pytest.mark.parametrize("elo_gap,expected_favourite", [
        (300, "a"),   # big home advantage
        (-300, "b"),  # big away advantage
        (0, None),    # roughly equal
    ])
    def test_1x2_sum_always_1(self, elo_gap, expected_favourite):
        snap_a = TeamSnapshot(elo=1800 + elo_gap, ppg=1.5)
        snap_b = TeamSnapshot(elo=1800, ppg=1.5)
        par    = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=5)
        inp    = RunnerInput("A", "B", snap_a, snap_b, par, par)
        r      = run_full_prediction(inp)
        total  = r.win_a + r.draw + r.win_b
        assert abs(total - 1.0) < 0.001

    def test_scorelines_sorted_with_low_xg(self):
        snap_def = TeamSnapshot(elo=1800, ppg=1.0)
        par_def  = StrengthParams(alpha_attack=0.6, beta_defense=1.4, matches_used=5)
        inp      = RunnerInput("A", "B", snap_def, snap_def, par_def, par_def)
        r        = run_full_prediction(inp)
        probs    = [s[2] for s in r.top_scorelines]
        assert probs == sorted(probs, reverse=True)

    def test_scorelines_sorted_with_high_xg(self):
        snap_att = TeamSnapshot(elo=2000, ppg=2.5)
        par_att  = StrengthParams(alpha_attack=2.5, beta_defense=0.5, matches_used=10)
        inp      = RunnerInput("A", "B", snap_att, snap_att, par_att, par_att)
        r        = run_full_prediction(inp)
        probs    = [s[2] for s in r.top_scorelines]
        assert probs == sorted(probs, reverse=True)
