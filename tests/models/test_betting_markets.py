"""Tests for the betting markets engine.

TDD: all tests written RED-first before any production code exists.

Score matrix convention: matrix[i, j] = P(team_a scores i goals, team_b scores j goals).
"""

import math
import pytest
import numpy as np

from src.models.betting_markets import (
    compute_betting_markets,
    BettingMarketProbabilities,
    MarketProbability,
    confidence_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_3x3() -> np.ndarray:
    """3×3 uniform matrix — each cell = 1/9."""
    return np.full((3, 3), 1 / 9)


def _known_matrix() -> np.ndarray:
    """Hand-crafted 3×3 matrix for exact assertions.

    Scores:  0-0  0-1  0-2
             1-0  1-1  1-2
             2-0  2-1  2-2

    Values (sum to 1):
      0-0=0.18  0-1=0.06  0-2=0.02
      1-0=0.24  1-1=0.08  1-2=0.02
      2-0=0.20  2-1=0.12  2-2=0.08

    win_a  = 1-0 + 2-0 + 2-1 = 0.24 + 0.20 + 0.12 = 0.56
    draw   = 0-0 + 1-1 + 2-2 = 0.18 + 0.08 + 0.08 = 0.34
    win_b  = 0-1 + 0-2 + 1-2 = 0.06 + 0.02 + 0.02 = 0.10
    """
    m = np.array([
        [0.18, 0.06, 0.02],
        [0.24, 0.08, 0.02],
        [0.20, 0.12, 0.08],
    ], dtype=float)
    return m


def _markets(matrix=None) -> BettingMarketProbabilities:
    if matrix is None:
        matrix = _known_matrix()
    return compute_betting_markets("TeamA", "TeamB", matrix)


def _dc_matrix(xg_a: float = 1.5, xg_b: float = 1.0) -> np.ndarray:
    """Build a real DC-corrected matrix for integration-style tests."""
    from src.models.poisson import build_score_matrix
    RHO = -0.30
    m = build_score_matrix(xg_a, xg_b)
    m[0, 0] *= 1 - (xg_a * xg_b * RHO)
    m[0, 1] *= 1 + (xg_a * RHO)
    m[1, 0] *= 1 + (xg_b * RHO)
    m[1, 1] *= 1 - RHO
    m /= m.sum()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# confidence_label
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_label_high_at_70():
    assert confidence_label(0.70) == "High"


def test_confidence_label_high_above_70():
    assert confidence_label(0.85) == "High"
    assert confidence_label(1.00) == "High"


def test_confidence_label_medium_at_55():
    assert confidence_label(0.55) == "Medium"


def test_confidence_label_medium_below_70():
    assert confidence_label(0.65) == "Medium"
    assert confidence_label(0.699) == "Medium"


def test_confidence_label_low_below_55():
    assert confidence_label(0.54) == "Low"
    assert confidence_label(0.00) == "Low"


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

def test_returns_betting_market_probabilities():
    assert isinstance(_markets(), BettingMarketProbabilities)


def test_market_probability_has_required_fields():
    m = _markets()
    mp = m.one_x_two[0]
    assert hasattr(mp, "market_name")
    assert hasattr(mp, "selection")
    assert hasattr(mp, "probability")
    assert hasattr(mp, "implied_fair_odds")
    assert hasattr(mp, "confidence_label")


def test_team_names_stored():
    m = _markets()
    assert m.team_a == "TeamA"
    assert m.team_b == "TeamB"


# ─────────────────────────────────────────────────────────────────────────────
# 1X2
# ─────────────────────────────────────────────────────────────────────────────

def test_1x2_has_three_selections():
    assert len(_markets().one_x_two) == 3


def test_1x2_sum_to_one():
    total = sum(mp.probability for mp in _markets().one_x_two)
    assert abs(total - 1.0) < 1e-9


def test_1x2_win_a_probability():
    m = _markets(_known_matrix())
    win_a = next(mp for mp in m.one_x_two if "TeamA" in mp.selection)
    assert abs(win_a.probability - 0.56) < 1e-9


def test_1x2_draw_probability():
    m = _markets(_known_matrix())
    draw = next(mp for mp in m.one_x_two if "Draw" in mp.selection)
    assert abs(draw.probability - 0.34) < 1e-9


def test_1x2_win_b_probability():
    m = _markets(_known_matrix())
    win_b = next(mp for mp in m.one_x_two if "TeamB" in mp.selection)
    assert abs(win_b.probability - 0.10) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Over / Under
# ─────────────────────────────────────────────────────────────────────────────

def test_over_under_has_five_selections():
    # O0.5, O1.5, O2.5, O3.5, U2.5
    assert len(_markets().over_under) == 5


def test_over_05_equals_1_minus_p00():
    m = _known_matrix()
    bm = compute_betting_markets("A", "B", m)
    o05 = next(mp for mp in bm.over_under if "0.5" in mp.selection and "Over" in mp.selection)
    assert abs(o05.probability - (1.0 - m[0, 0])) < 1e-9


def test_over_25_plus_under_25_equals_one():
    bm = _markets(_known_matrix())
    o25 = next(mp for mp in bm.over_under if "Over 2.5" in mp.selection)
    u25 = next(mp for mp in bm.over_under if "Under 2.5" in mp.selection)
    assert abs(o25.probability + u25.probability - 1.0) < 1e-9


def test_over_05_geq_over_15_geq_over_25_geq_over_35():
    bm = _markets(_dc_matrix(1.5, 1.0))
    ou = {mp.selection: mp.probability for mp in bm.over_under}
    assert ou["Over 0.5"] >= ou["Over 1.5"] >= ou["Over 2.5"] >= ou["Over 3.5"]


def test_over_under_all_in_range():
    for mp in _markets(_dc_matrix()).over_under:
        assert 0.0 <= mp.probability <= 1.0, f"{mp.selection}: {mp.probability}"


def test_over_25_known_matrix():
    """In the known 3x3 matrix, O2.5 = P(total >= 3) = P(2-1) + P(1-2) + P(2-2) = 0.12+0.02+0.08 = 0.22."""
    bm = _markets(_known_matrix())
    o25 = next(mp for mp in bm.over_under if "Over 2.5" in mp.selection)
    assert abs(o25.probability - 0.22) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# BTTS
# ─────────────────────────────────────────────────────────────────────────────

def test_btts_has_two_selections():
    assert len(_markets().btts) == 2


def test_btts_yes_plus_no_equals_one():
    bm = _markets()
    total = sum(mp.probability for mp in bm.btts)
    assert abs(total - 1.0) < 1e-9


def test_btts_yes_known_matrix():
    """BTTS Yes = cells where i>=1 AND j>=1.
    In known matrix: 1-1=0.08, 1-2=0.02, 2-1=0.12, 2-2=0.08 → 0.30
    """
    bm = _markets(_known_matrix())
    yes = next(mp for mp in bm.btts if mp.selection == "BTTS Yes")
    assert abs(yes.probability - 0.30) < 1e-9


def test_btts_no_known_matrix():
    bm = _markets(_known_matrix())
    no = next(mp for mp in bm.btts if mp.selection == "BTTS No")
    assert abs(no.probability - 0.70) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Double Chance
# ─────────────────────────────────────────────────────────────────────────────

def test_double_chance_has_three_selections():
    assert len(_markets().double_chance) == 3


def test_double_chance_1x_equals_win_a_plus_draw():
    bm = _markets(_known_matrix())
    dc = {mp.selection: mp.probability for mp in bm.double_chance}
    win_a = 0.56; draw = 0.34
    assert abs(dc["1X"] - (win_a + draw)) < 1e-9


def test_double_chance_x2_equals_draw_plus_win_b():
    bm = _markets(_known_matrix())
    dc = {mp.selection: mp.probability for mp in bm.double_chance}
    draw = 0.34; win_b = 0.10
    assert abs(dc["X2"] - (draw + win_b)) < 1e-9


def test_double_chance_12_equals_1_minus_draw():
    bm = _markets(_known_matrix())
    dc = {mp.selection: mp.probability for mp in bm.double_chance}
    draw = 0.34
    assert abs(dc["12"] - (1.0 - draw)) < 1e-9


def test_double_chance_all_above_half():
    """Double-chance always > 0.5 by definition (covers two outcomes)."""
    for mp in _markets(_dc_matrix()).double_chance:
        assert mp.probability > 0.50, f"{mp.selection}: {mp.probability}"


# ─────────────────────────────────────────────────────────────────────────────
# Draw No Bet
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_no_bet_has_two_selections():
    assert len(_markets().draw_no_bet) == 2


def test_dnb_team_a_plus_team_b_equals_one():
    bm = _markets()
    total = sum(mp.probability for mp in bm.draw_no_bet)
    assert abs(total - 1.0) < 1e-9


def test_dnb_excludes_draw_normalises_correctly():
    bm = _markets(_known_matrix())
    dnb = {mp.selection: mp.probability for mp in bm.draw_no_bet}
    # win_a=0.56, win_b=0.10, total=0.66
    expected_a = 0.56 / 0.66
    expected_b = 0.10 / 0.66
    assert abs(dnb["TeamA DNB"] - expected_a) < 1e-9
    assert abs(dnb["TeamB DNB"] - expected_b) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Clean Sheet
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_sheet_has_two_selections():
    assert len(_markets().clean_sheet) == 2


def test_clean_sheet_team_a_is_team_b_scores_zero():
    """Team A clean sheet = P(team_b scores 0) = sum of column 0."""
    m = _known_matrix()
    bm = compute_betting_markets("TeamA", "TeamB", m)
    cs_a = next(mp for mp in bm.clean_sheet if "TeamA" in mp.selection)
    expected = float(m[:, 0].sum())  # sum column j=0
    assert abs(cs_a.probability - expected) < 1e-9


def test_clean_sheet_team_b_is_team_a_scores_zero():
    """Team B clean sheet = P(team_a scores 0) = sum of row 0."""
    m = _known_matrix()
    bm = compute_betting_markets("TeamA", "TeamB", m)
    cs_b = next(mp for mp in bm.clean_sheet if "TeamB" in mp.selection)
    expected = float(m[0, :].sum())  # sum row i=0
    assert abs(cs_b.probability - expected) < 1e-9


def test_clean_sheet_known_matrix():
    """
    TeamA CS (TeamB scores 0): col 0 = 0.18 + 0.24 + 0.20 = 0.62
    TeamB CS (TeamA scores 0): row 0 = 0.18 + 0.06 + 0.02 = 0.26
    """
    bm = _markets(_known_matrix())
    cs = {mp.selection: mp.probability for mp in bm.clean_sheet}
    assert abs(cs["TeamA Clean Sheet"] - 0.62) < 1e-9
    assert abs(cs["TeamB Clean Sheet"] - 0.26) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Team Totals
# ─────────────────────────────────────────────────────────────────────────────

def test_team_totals_has_four_selections():
    assert len(_markets().team_totals) == 4


def test_team_a_over_05_equals_1_minus_team_a_scores_zero():
    m = _known_matrix()
    bm = compute_betting_markets("TeamA", "TeamB", m)
    tt = {mp.selection: mp.probability for mp in bm.team_totals}
    p_a_scores_zero = float(m[0, :].sum())
    assert abs(tt["TeamA Over 0.5"] - (1.0 - p_a_scores_zero)) < 1e-9


def test_team_b_over_05_equals_1_minus_team_b_scores_zero():
    m = _known_matrix()
    bm = compute_betting_markets("TeamA", "TeamB", m)
    tt = {mp.selection: mp.probability for mp in bm.team_totals}
    p_b_scores_zero = float(m[:, 0].sum())
    assert abs(tt["TeamB Over 0.5"] - (1.0 - p_b_scores_zero)) < 1e-9


def test_team_a_over_15_known_matrix():
    """TeamA Over 1.5 = P(team_a scores >= 2) = sum row 2+ = 0.20+0.12+0.08 = 0.40"""
    bm = _markets(_known_matrix())
    tt = {mp.selection: mp.probability for mp in bm.team_totals}
    assert abs(tt["TeamA Over 1.5"] - 0.40) < 1e-9


def test_team_b_over_15_known_matrix():
    """TeamB Over 1.5 = P(team_b scores >= 2) = sum col 2+ = 0.02+0.02+0.08 = 0.12"""
    bm = _markets(_known_matrix())
    tt = {mp.selection: mp.probability for mp in bm.team_totals}
    assert abs(tt["TeamB Over 1.5"] - 0.12) < 1e-9


def test_team_totals_over_05_geq_over_15():
    bm = _markets(_dc_matrix())
    tt = {mp.selection: mp.probability for mp in bm.team_totals}
    assert tt["TeamA Over 0.5"] >= tt["TeamA Over 1.5"]
    assert tt["TeamB Over 0.5"] >= tt["TeamB Over 1.5"]


# ─────────────────────────────────────────────────────────────────────────────
# Exact Score
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_score_has_five_selections():
    assert len(_markets().exact_score) == 5


def test_exact_score_sorted_descending():
    probs = [mp.probability for mp in _markets(_dc_matrix()).exact_score]
    assert probs == sorted(probs, reverse=True)


def test_exact_score_probabilities_positive():
    for mp in _markets(_dc_matrix()).exact_score:
        assert mp.probability > 0.0


def test_exact_score_selection_format():
    """Selection should be formatted as 'A-B' with team names."""
    for mp in _markets().exact_score:
        assert "-" in mp.selection


# ─────────────────────────────────────────────────────────────────────────────
# Implied fair odds
# ─────────────────────────────────────────────────────────────────────────────

def test_implied_fair_odds_is_inverse_of_probability():
    bm = _markets(_dc_matrix())
    for mp in bm.one_x_two + bm.over_under + bm.btts:
        if mp.probability > 0:
            assert abs(mp.implied_fair_odds - 1.0 / mp.probability) < 1e-9


def test_implied_fair_odds_zero_probability_is_infinity():
    """P=0 should yield inf, not divide by zero."""
    # Construct a matrix where team_b win probability is 0
    m = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=float)
    bm = compute_betting_markets("A", "B", m)
    win_b = next(mp for mp in bm.one_x_two if "B" in mp.selection and "Win" in mp.selection)
    assert win_b.probability == pytest.approx(0.0, abs=1e-6)
    assert win_b.implied_fair_odds == float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# All probabilities in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_all_probabilities_in_range():
    bm = _markets(_dc_matrix(1.8, 0.9))
    all_markets = (
        bm.one_x_two + bm.over_under + bm.btts +
        bm.double_chance + bm.draw_no_bet + bm.clean_sheet +
        bm.team_totals + bm.exact_score
    )
    for mp in all_markets:
        assert 0.0 <= mp.probability <= 1.0, (
            f"{mp.market_name}/{mp.selection}: {mp.probability}"
        )


def test_all_confidence_labels_valid():
    bm = _markets(_dc_matrix())
    all_markets = (
        bm.one_x_two + bm.over_under + bm.btts +
        bm.double_chance + bm.draw_no_bet + bm.clean_sheet +
        bm.team_totals + bm.exact_score
    )
    for mp in all_markets:
        assert mp.confidence_label in ("High", "Medium", "Low"), (
            f"{mp.market_name}/{mp.selection}: {mp.confidence_label!r}"
        )
