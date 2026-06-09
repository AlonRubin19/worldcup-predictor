"""Betting market probability engine.

Derives all standard football betting markets from an existing
DC-corrected, normalised score probability matrix.

Matrix convention: matrix[i, j] = P(team_a scores i goals, team_b scores j goals).
Matrix must be normalised (sum == 1) before passing in.

No re-computation of model probabilities occurs here — the matrix
produced by the Dixon-Coles predictor is passed directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketProbability:
    market_name: str
    selection: str
    probability: float
    implied_fair_odds: float
    confidence_label: str


@dataclass
class BettingMarketProbabilities:
    team_a: str
    team_b: str
    one_x_two: list[MarketProbability]
    exact_score: list[MarketProbability]
    over_under: list[MarketProbability]
    btts: list[MarketProbability]
    double_chance: list[MarketProbability]
    draw_no_bet: list[MarketProbability]
    clean_sheet: list[MarketProbability]
    team_totals: list[MarketProbability]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def confidence_label(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.55:
        return "Medium"
    return "Low"


def _mp(market_name: str, selection: str, prob: float) -> MarketProbability:
    prob = float(np.clip(prob, 0.0, 1.0))
    odds = 1.0 / prob if prob > 0.0 else float("inf")
    return MarketProbability(
        market_name=market_name,
        selection=selection,
        probability=prob,
        implied_fair_odds=odds,
        confidence_label=confidence_label(prob),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Market computations
# ─────────────────────────────────────────────────────────────────────────────

def _compute_1x2(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    win_a = float(np.sum(np.tril(matrix, k=-1)))
    draw  = float(np.sum(np.diag(matrix)))
    win_b = float(np.sum(np.triu(matrix, k=1)))
    return [
        _mp("1X2", f"{team_a} Win", win_a),
        _mp("1X2", "Draw",          draw),
        _mp("1X2", f"{team_b} Win", win_b),
    ]


def _compute_exact_score(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    n = matrix.shape[0]
    scorelines = [
        (i, j, float(matrix[i, j]))
        for i in range(n)
        for j in range(n)
    ]
    scorelines.sort(key=lambda x: -x[2])
    return [
        _mp("Exact Score", f"{i}-{j}", p)
        for i, j, p in scorelines[:5]
    ]


def _compute_over_under(matrix: np.ndarray) -> list[MarketProbability]:
    n = matrix.shape[0]
    results = []
    for line in (0.5, 1.5, 2.5, 3.5):
        threshold = int(line + 0.5)   # 1, 2, 3, 4
        over_prob = float(sum(
            matrix[i, j]
            for i in range(n)
            for j in range(n)
            if i + j >= threshold
        ))
        results.append(_mp("Over/Under", f"Over {line}", over_prob))

    # Under 2.5 only
    u25 = float(sum(
        matrix[i, j]
        for i in range(n)
        for j in range(n)
        if i + j <= 2
    ))
    results.append(_mp("Over/Under", "Under 2.5", u25))
    return results


def _compute_btts(matrix: np.ndarray) -> list[MarketProbability]:
    n = matrix.shape[0]
    yes = float(sum(
        matrix[i, j]
        for i in range(1, n)
        for j in range(1, n)
    ))
    return [
        _mp("BTTS", "BTTS Yes", yes),
        _mp("BTTS", "BTTS No",  1.0 - yes),
    ]


def _compute_double_chance(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    win_a = float(np.sum(np.tril(matrix, k=-1)))
    draw  = float(np.sum(np.diag(matrix)))
    win_b = float(np.sum(np.triu(matrix, k=1)))
    return [
        _mp("Double Chance", "1X",  win_a + draw),
        _mp("Double Chance", "X2",  draw  + win_b),
        _mp("Double Chance", "12",  win_a + win_b),
    ]


def _compute_draw_no_bet(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    win_a = float(np.sum(np.tril(matrix, k=-1)))
    win_b = float(np.sum(np.triu(matrix, k=1)))
    total = win_a + win_b
    if total > 0:
        p_a = win_a / total
        p_b = win_b / total
    else:
        p_a = p_b = 0.5
    return [
        _mp("Draw No Bet", f"{team_a} DNB", p_a),
        _mp("Draw No Bet", f"{team_b} DNB", p_b),
    ]


def _compute_clean_sheet(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    # Team A clean sheet = team_b scores 0 goals = column 0
    cs_a = float(matrix[:, 0].sum())
    # Team B clean sheet = team_a scores 0 goals = row 0
    cs_b = float(matrix[0, :].sum())
    return [
        _mp("Clean Sheet", f"{team_a} Clean Sheet", cs_a),
        _mp("Clean Sheet", f"{team_b} Clean Sheet", cs_b),
    ]


def _compute_team_totals(
    team_a: str, team_b: str, matrix: np.ndarray
) -> list[MarketProbability]:
    n = matrix.shape[0]
    # Team A scores
    a_o05 = float(1.0 - matrix[0, :].sum())         # team_a >= 1
    a_o15 = float(matrix[2:, :].sum())               # team_a >= 2
    # Team B scores
    b_o05 = float(1.0 - matrix[:, 0].sum())          # team_b >= 1
    b_o15 = float(matrix[:, 2:].sum())               # team_b >= 2
    return [
        _mp("Team Totals", f"{team_a} Over 0.5", a_o05),
        _mp("Team Totals", f"{team_a} Over 1.5", a_o15),
        _mp("Team Totals", f"{team_b} Over 0.5", b_o05),
        _mp("Team Totals", f"{team_b} Over 1.5", b_o15),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_betting_markets(
    team_a: str,
    team_b: str,
    score_matrix: np.ndarray,
) -> BettingMarketProbabilities:
    """Compute all standard betting market probabilities from a score matrix.

    Args:
        team_a: Name of the home/first team.
        team_b: Name of the away/second team.
        score_matrix: DC-corrected, normalised joint probability matrix.
            shape (N, N) where matrix[i, j] = P(team_a scores i, team_b scores j).

    Returns:
        BettingMarketProbabilities with all markets populated.
    """
    m = np.asarray(score_matrix, dtype=float)

    return BettingMarketProbabilities(
        team_a=team_a,
        team_b=team_b,
        one_x_two     = _compute_1x2(team_a, team_b, m),
        exact_score   = _compute_exact_score(team_a, team_b, m),
        over_under    = _compute_over_under(m),
        btts          = _compute_btts(m),
        double_chance = _compute_double_chance(team_a, team_b, m),
        draw_no_bet   = _compute_draw_no_bet(team_a, team_b, m),
        clean_sheet   = _compute_clean_sheet(team_a, team_b, m),
        team_totals   = _compute_team_totals(team_a, team_b, m),
    )
