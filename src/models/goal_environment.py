"""goal_environment.py — total-goals / match-profile layer.

Derives over/under, BTTS, clean-sheet probabilities and an overall
match_goal_profile classification directly from the Dixon-Coles score
matrix (which already encodes both teams' xG). This is the layer the
exact-score recommendation consults so it doesn't default to low-scoring
scorelines (1-0/0-0/1-1) when the goal environment points to an open game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GoalEnvironment:
    expected_total_goals: float
    over_1_5_probability: float
    over_2_5_probability: float
    over_3_5_probability: float
    under_2_5_probability: float
    btts_probability: float
    clean_sheet_probability_team1: float
    clean_sheet_probability_team2: float
    match_goal_profile: str


def _profile_for(expected_total_goals: float, over_2_5: float, btts: float) -> str:
    if expected_total_goals <= 1.60:
        return "very_low_scoring"
    if expected_total_goals <= 2.10:
        return "low_scoring"
    if expected_total_goals <= 2.70:
        return "balanced"
    if expected_total_goals <= 3.20:
        return "open" if over_2_5 >= 0.50 else "balanced"
    if expected_total_goals <= 4.00:
        return "high_scoring"
    return "chaotic" if btts >= 0.55 else "high_scoring"


def compute_goal_environment(matrix: np.ndarray, xg_a: float, xg_b: float) -> GoalEnvironment:
    """Compute total-goals/profile metrics from the score matrix + xG.

    The matrix is normalized over (goals_a, goals_b) in [0, n-1]^2; totals
    above the matrix's max are negligible for realistic xG ranges.
    """
    n = matrix.shape[0]
    total_prob = float(matrix.sum())
    if total_prob <= 0:
        total_prob = 1.0

    over_1_5 = over_2_5 = over_3_5 = 0.0
    btts = 0.0
    cs_a = cs_b = 0.0
    expected_total = 0.0
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j]) / total_prob
            total_goals = i + j
            expected_total += total_goals * p
            if total_goals >= 2:
                over_1_5 += p
            if total_goals >= 3:
                over_2_5 += p
            if total_goals >= 4:
                over_3_5 += p
            if i >= 1 and j >= 1:
                btts += p
            if j == 0:
                cs_a += p
            if i == 0:
                cs_b += p

    under_2_5 = 1.0 - over_2_5
    profile = _profile_for(expected_total, over_2_5, btts)

    return GoalEnvironment(
        expected_total_goals=expected_total,
        over_1_5_probability=over_1_5,
        over_2_5_probability=over_2_5,
        over_3_5_probability=over_3_5,
        under_2_5_probability=under_2_5,
        btts_probability=btts,
        clean_sheet_probability_team1=cs_a,
        clean_sheet_probability_team2=cs_b,
        match_goal_profile=profile,
    )
