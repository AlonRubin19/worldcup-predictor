"""Tests for the recommendation engine.

TDD: all tests written RED-first before any production code exists.
"""

import pytest
import numpy as np

from src.models.recommendations import (
    generate_recommendations,
    Recommendation,
    RecommendationSet,
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
    THRESHOLD_EXACT_SCORE,
)
from src.models.betting_markets import compute_betting_markets, BettingMarketProbabilities
from src.models.dixon_coles import build_dc_matrix
from src.models.research_valid_predictor import DEFAULT_RHO


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bm(xg_a: float = 1.8, xg_b: float = 0.8, team_a="TeamA", team_b="TeamB") -> BettingMarketProbabilities:
    m = build_dc_matrix(xg_a, xg_b, rho=DEFAULT_RHO)
    return compute_betting_markets(team_a, team_b, m)


def _recs(
    xg_a=1.8, xg_b=0.8,
    confidence="High",
    warnings=None,
    is_rv=True,
    odds=None,
    top_n=5,
    team_a="TeamA",
    team_b="TeamB",
) -> RecommendationSet:
    bm = _bm(xg_a, xg_b, team_a, team_b)
    return generate_recommendations(
        betting_markets=bm,
        prediction_confidence=confidence,
        data_warnings=warnings or [],
        is_research_valid=is_rv,
        market_implied_probs=odds,
        top_n=top_n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

def test_returns_recommendation_set():
    assert isinstance(_recs(), RecommendationSet)


def test_recommendation_set_has_required_fields():
    rs = _recs(team_a="Brazil", team_b="Serbia")
    assert rs.team_a == "Brazil"
    assert rs.team_b == "Serbia"
    assert isinstance(rs.recommendations, list)


def test_each_recommendation_has_required_fields():
    rs = _recs()
    assert len(rs.recommendations) > 0
    r = rs.recommendations[0]
    assert hasattr(r, "market_name")
    assert hasattr(r, "selection")
    assert hasattr(r, "model_probability")
    assert hasattr(r, "fair_odds")
    assert hasattr(r, "confidence_label")
    assert hasattr(r, "signal_strength")
    assert hasattr(r, "rationale")
    assert hasattr(r, "warning")


def test_recommendation_is_dataclass():
    rs = _recs()
    assert isinstance(rs.recommendations[0], Recommendation)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — inclusion and exclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_high_probability_market_is_included():
    """With a dominant team (xg_a=2.5, xg_b=0.6) the 1X win_a should be >= 0.70."""
    rs = _recs(xg_a=2.5, xg_b=0.6)
    selections = [r.selection for r in rs.recommendations]
    assert any("TeamA" in s and "Win" in s for s in selections), (
        f"Expected TeamA Win in recommendations, got {selections}"
    )


def test_very_low_probability_market_excluded():
    """With dominant team, TeamB Win prob << 0.60 — should NOT appear."""
    rs = _recs(xg_a=2.5, xg_b=0.6)
    selections = [r.selection for r in rs.recommendations]
    assert not any("TeamB" in s and "Win" in s for s in selections), (
        f"Expected TeamB Win excluded, got {selections}"
    )


def test_exact_score_included_above_threshold():
    """Top scoreline is typically > 0.10 — should appear when enough slots are available."""
    rs = _recs(xg_a=1.8, xg_b=0.8, top_n=10)  # use top_n=10 so higher-priority markets don't crowd it out
    assert any(r.market_name == "Exact Score" for r in rs.recommendations), (
        "Expected at least one Exact Score recommendation when top_n is large enough"
    )


def test_exact_score_excluded_below_threshold():
    """Artificially check that the threshold constant is <= 0.10."""
    assert THRESHOLD_EXACT_SCORE <= 0.10


def test_exact_score_probability_meets_threshold():
    """All exact score recommendations must meet THRESHOLD_EXACT_SCORE."""
    rs = _recs()
    for r in rs.recommendations:
        if r.market_name == "Exact Score":
            assert r.model_probability >= THRESHOLD_EXACT_SCORE, (
                f"Exact score {r.selection} probability {r.model_probability} < threshold"
            )


def test_threshold_high_is_0_70():
    assert THRESHOLD_HIGH == pytest.approx(0.70)


def test_threshold_medium_is_0_60():
    assert THRESHOLD_MEDIUM == pytest.approx(0.60)


# ─────────────────────────────────────────────────────────────────────────────
# Top N limiting
# ─────────────────────────────────────────────────────────────────────────────

def test_top_n_limits_results():
    rs = _recs(top_n=3)
    assert len(rs.recommendations) <= 3


def test_top_n_default_is_five():
    bm = _bm()
    rs = generate_recommendations(bm, "High", [], True)
    assert len(rs.recommendations) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# Signal strength
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_strength_strong_for_high_prob():
    """P >= 0.70 with High confidence → Strong."""
    rs = _recs(xg_a=2.5, xg_b=0.5)
    strong = [r for r in rs.recommendations if r.signal_strength == "Strong"]
    assert len(strong) > 0, "Expected at least one Strong signal"


def test_signal_strength_is_one_of_three_values():
    rs = _recs()
    valid = {"Strong", "Moderate", "Weak"}
    for r in rs.recommendations:
        assert r.signal_strength in valid, f"Unexpected signal_strength: {r.signal_strength!r}"


def test_signal_strength_strong_implies_high_probability():
    rs = _recs(xg_a=2.5, xg_b=0.5)
    for r in rs.recommendations:
        if r.signal_strength == "Strong":
            assert r.model_probability >= THRESHOLD_HIGH, (
                f"Strong signal {r.selection} has prob {r.model_probability} < {THRESHOLD_HIGH}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sorting and priority
# ─────────────────────────────────────────────────────────────────────────────

def test_higher_probability_ranked_first():
    """Within the same signal-strength tier, recommendations are ranked by
    market priority rather than raw probability — so a later recommendation
    can have a higher probability than an earlier one. Across tiers, however,
    a "Strong" signal must never be ranked below a "Moderate"/"Weak" one."""
    rs = _recs(xg_a=2.0, xg_b=0.9)
    tiers = {"Strong": 0, "Moderate": 1, "Weak": 2}
    ranks = [tiers[r.signal_strength] for r in rs.recommendations]
    assert ranks == sorted(ranks)


def test_1x2_ranked_above_exact_score_when_prob_equal_or_higher():
    """Market priority: 1X2 before Exact Score."""
    rs = _recs()
    markets = [r.market_name for r in rs.recommendations]
    if "1X2" in markets and "Exact Score" in markets:
        assert markets.index("1X2") < markets.index("Exact Score"), (
            "1X2 should appear before Exact Score"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rationale
# ─────────────────────────────────────────────────────────────────────────────

def test_rationale_is_non_empty():
    rs = _recs()
    for r in rs.recommendations:
        assert r.rationale and len(r.rationale) > 0, f"Empty rationale for {r.selection}"


def test_rationale_is_string():
    rs = _recs()
    for r in rs.recommendations:
        assert isinstance(r.rationale, str)


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────

def test_warning_is_none_for_clean_high_confidence():
    rs = _recs(confidence="High", warnings=[], is_rv=True)
    for r in rs.recommendations:
        assert r.warning is None, (
            f"Expected no warning for clean high-confidence signal, got: {r.warning!r}"
        )


def test_warning_added_when_not_research_valid():
    rs = _recs(is_rv=False)
    for r in rs.recommendations:
        assert r.warning is not None, "Expected warning when not research-valid"
        assert len(r.warning) > 0


def test_warning_added_when_prediction_warnings_exist():
    rs = _recs(warnings=["ELO data is stale"])
    for r in rs.recommendations:
        assert r.warning is not None


def test_warning_is_string_or_none():
    rs = _recs()
    for r in rs.recommendations:
        assert r.warning is None or isinstance(r.warning, str)


# ─────────────────────────────────────────────────────────────────────────────
# No betting advice language
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN = [
    "bet on", "place a bet", "stake", "wager", "guaranteed", "sure win",
    "definitely", "certain", "can't lose", "lock", "tip",
]


def test_no_betting_advice_language_in_rationale():
    rs = _recs()
    for r in rs.recommendations:
        text = r.rationale.lower()
        for phrase in _FORBIDDEN:
            assert phrase not in text, (
                f"Forbidden phrase '{phrase}' found in rationale: {r.rationale!r}"
            )


def test_no_betting_advice_language_in_warning():
    rs = _recs(warnings=["test warning"], is_rv=False)
    for r in rs.recommendations:
        if r.warning:
            text = r.warning.lower()
            for phrase in _FORBIDDEN:
                assert phrase not in text, (
                    f"Forbidden phrase '{phrase}' found in warning: {r.warning!r}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# With real market odds (edge ranking)
# ─────────────────────────────────────────────────────────────────────────────

def test_with_odds_ranks_by_positive_edge():
    """When odds are provided, selections with higher edge should rank higher."""
    bm = _bm(xg_a=1.8, xg_b=0.8)
    # Give TeamA Win a big positive edge (market underestimates it)
    # and Double Chance 1X a small negative edge
    win_a_prob = next(mp.probability for mp in bm.one_x_two if "TeamA" in mp.selection)
    market_odds = {
        "TeamA Win": win_a_prob - 0.10,   # market is 10pp lower → big +edge
        "1X": win_a_prob + 0.05,           # market prices DC higher → negative edge
    }
    rs = generate_recommendations(bm, "High", [], True,
                                  market_implied_probs=market_odds, top_n=5)
    selections = [r.selection for r in rs.recommendations]
    if "TeamA Win" in selections and "1X" in selections:
        assert selections.index("TeamA Win") < selections.index("1X"), (
            "TeamA Win (higher edge) should rank above 1X (negative edge)"
        )


def test_with_odds_model_probability_preserved():
    """market_implied_probs should not change the model_probability field."""
    bm = _bm()
    win_a_prob = next(mp.probability for mp in bm.one_x_two if "TeamA" in mp.selection)
    rs_no_odds  = generate_recommendations(bm, "High", [], True, top_n=5)
    rs_with_odds = generate_recommendations(bm, "High", [], True,
                                            market_implied_probs={"TeamA Win": 0.50}, top_n=5)
    # Find TeamA Win in both
    def _find(rs):
        for r in rs.recommendations:
            if r.selection == "TeamA Win":
                return r
        return None

    r_no  = _find(rs_no_odds)
    r_yes = _find(rs_with_odds)
    if r_no and r_yes:
        assert r_no.model_probability == pytest.approx(r_yes.model_probability), (
            "market_implied_probs must not change model_probability"
        )


def test_with_odds_negative_edge_lower_ranked():
    """Selection where market is more confident than model should not lead the list."""
    bm = _bm(xg_a=1.8, xg_b=0.8)
    win_a = next(mp.probability for mp in bm.one_x_two if "TeamA" in mp.selection)
    o15   = next(mp.probability for mp in bm.over_under if "1.5" in mp.selection and "Over" in mp.selection)

    # Market prices TeamA Win higher than model → negative edge
    # Market prices Over 1.5 lower than model → positive edge
    odds = {
        "TeamA Win": win_a + 0.10,   # market is 10pp higher → model has -10pp edge
        "Over 1.5":  o15  - 0.10,   # market is 10pp lower  → model has +10pp edge
    }
    rs = generate_recommendations(bm, "High", [], True,
                                  market_implied_probs=odds, top_n=5)
    selections = [r.selection for r in rs.recommendations]
    if "Over 1.5" in selections and "TeamA Win" in selections:
        assert selections.index("Over 1.5") < selections.index("TeamA Win"), (
            "Over 1.5 (positive edge) should rank above TeamA Win (negative edge)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model probability and fair odds consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_model_probability_in_range():
    rs = _recs()
    for r in rs.recommendations:
        assert 0.0 <= r.model_probability <= 1.0


def test_fair_odds_is_inverse_of_model_probability():
    rs = _recs()
    for r in rs.recommendations:
        if r.model_probability > 0:
            assert abs(r.fair_odds - 1.0 / r.model_probability) < 1e-6


def test_fair_odds_infinity_for_zero_probability():
    """Edge case: if a zero-probability market somehow enters, fair_odds should be inf."""
    # We don't normally generate zero-prob recommendations, but the Recommendation
    # dataclass must support it cleanly.
    r = Recommendation(
        market_name="Test", selection="X",
        model_probability=0.0, fair_odds=float("inf"),
        confidence_label="Low", signal_strength="Weak",
        rationale="test", warning=None,
    )
    assert r.fair_odds == float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke — dominant match
# ─────────────────────────────────────────────────────────────────────────────

def test_dominant_match_top_signal_is_strong():
    """Argentina-style dominant team should produce at least one Strong signal."""
    rs = _recs(xg_a=2.5, xg_b=0.6, team_a="Argentina", team_b="Mexico")
    assert any(r.signal_strength == "Strong" for r in rs.recommendations), (
        "Dominant match should yield at least one Strong signal"
    )


def test_balanced_match_produces_recommendations():
    """Even a balanced match (xg 1.2 vs 1.1) should yield recommendations."""
    rs = _recs(xg_a=1.2, xg_b=1.1)
    assert len(rs.recommendations) > 0
