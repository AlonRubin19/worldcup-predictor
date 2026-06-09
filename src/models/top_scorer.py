"""top_scorer.py — match-level player goal probability model.

Given team xG values (from Dixon-Coles) and a squad of players
(from player_profiles.csv), distributes expected goals proportionally
to each player's xg_per_90 share and computes Poisson probabilities.

DATA LABEL: Engineering validation only.
Player profiles currently exist for 8 teams. For other teams the
result will list them in `teams_without_data` and return no players.

Usage:
    from src.models.top_scorer import predict_top_scorers
    result = predict_top_scorers(
        team_a="France", team_b="Morocco",
        xg_a=1.8, xg_b=0.9,
        squad_a=france_players, squad_b=morocco_players,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerMatchPrediction:
    """Goal probability model for one player in one match.

    Attributes:
        player_id:      Raw player id from profiles CSV.
        player_name:    Display name.
        team:           Team name (internal).
        position:       FW / MF / DF / GK.
        expected_goals: Poisson lambda — player's expected goals in this match.
        prob_scores:    P(player scores ≥ 1) = 1 − e^(−lambda).
        prob_brace:     P(player scores ≥ 2).
        is_starter:     Whether player is an expected starter.
    """
    player_id:      str
    player_name:    str
    team:           str
    position:       str
    expected_goals: float
    prob_scores:    float
    prob_brace:     float
    is_starter:     bool


@dataclass
class TopScorerResult:
    """Full top-scorer prediction for a match.

    Attributes:
        team_a, team_b:       Team names.
        team_a_players:       Players sorted by expected_goals descending.
        team_b_players:       Players sorted by expected_goals descending.
        overall_favourite:    Player with the single highest expected_goals
                              across both teams. None if no player data.
        data_valid:           True when both squads have player data.
        teams_without_data:   Teams for which no squad was supplied.
    """
    team_a:             str
    team_b:             str
    team_a_players:     list[PlayerMatchPrediction]
    team_b_players:     list[PlayerMatchPrediction]
    overall_favourite:  PlayerMatchPrediction | None
    data_valid:         bool
    teams_without_data: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Probability helpers (pure functions, easily testable)
# ─────────────────────────────────────────────────────────────────────────────

def player_score_probability(lam: float) -> float:
    """P(Poisson(lambda) ≥ 1) = 1 − e^(−lambda)."""
    if lam <= 0.0:
        return 0.0
    return 1.0 - math.exp(-lam)


def player_brace_probability(lam: float) -> float:
    """P(Poisson(lambda) ≥ 2) = 1 − P(0) − P(1) = 1 − e^(−λ) − λe^(−λ)."""
    if lam <= 0.0:
        return 0.0
    e_neg_lam = math.exp(-lam)
    return 1.0 - e_neg_lam - lam * e_neg_lam


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _build_player_predictions(
    team_name: str,
    team_xg: float,
    squad: list[dict[str, Any]],
    top_n: int | None,
) -> list[PlayerMatchPrediction]:
    """Distribute team_xg across squad proportional to xg_per_90 share.

    Steps:
      1. Zero out unavailable players (availability_factor = 0).
      2. Compute effective xG-per-90 for each player:
             effective_xg = xg_per_90 * availability_factor
      3. Total effective xG across squad.
      4. If total == 0 (e.g. all GKs with xg=0), fall back to goals_per_90.
      5. player_lambda = team_xg * (effective_xg / total_effective_xg)
    """
    if not squad:
        return []

    # Effective xG for each player (zero if unavailable)
    eff = []
    for p in squad:
        avail = float(p.get("availability_factor", 1.0))
        xg90  = float(p.get("xg_per_90", 0.0))
        eff.append(xg90 * avail)

    total_eff = sum(eff)

    # Fallback: if all xg_per_90 are 0, use goals_per_90
    if total_eff <= 0.0:
        eff = []
        for p in squad:
            avail = float(p.get("availability_factor", 1.0))
            g90   = float(p.get("goals_per_90", 0.0))
            eff.append(g90 * avail)
        total_eff = sum(eff)

    predictions: list[PlayerMatchPrediction] = []
    for i, p in enumerate(squad):
        if total_eff > 0.0:
            lam = team_xg * (eff[i] / total_eff)
        else:
            lam = 0.0

        predictions.append(PlayerMatchPrediction(
            player_id      = str(p.get("player_id", "")),
            player_name    = str(p.get("player_name", "")),
            team           = team_name,
            position       = str(p.get("position", "")),
            expected_goals = lam,
            prob_scores    = player_score_probability(lam),
            prob_brace     = player_brace_probability(lam),
            is_starter     = bool(p.get("expected_starter", True)),
        ))

    # Sort descending by expected goals
    predictions.sort(key=lambda x: x.expected_goals, reverse=True)

    if top_n is not None:
        predictions = predictions[:top_n]

    return predictions


def predict_top_scorers(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    squad_a: list[dict[str, Any]],
    squad_b: list[dict[str, Any]],
    top_n: int | None = 5,
) -> TopScorerResult:
    """Predict the most likely goal scorer(s) for a match.

    Args:
        team_a, team_b: Internal team names.
        xg_a, xg_b:    Expected goals from the Dixon-Coles model.
        squad_a, b:    List of player dicts (from player_profiles.csv rows).
                       Pass [] if no data is available for that team.
        top_n:         Max players to return per team. None = all players.

    Returns:
        TopScorerResult with players sorted by expected_goals descending.
    """
    teams_without_data: list[str] = []

    if not squad_a:
        teams_without_data.append(team_a)
    if not squad_b:
        teams_without_data.append(team_b)

    players_a = _build_player_predictions(team_a, xg_a, squad_a, top_n)
    players_b = _build_player_predictions(team_b, xg_b, squad_b, top_n)

    # Overall favourite: highest expected_goals across both squads
    all_players = players_a + players_b
    favourite = max(all_players, key=lambda p: p.expected_goals) if all_players else None

    data_valid = len(teams_without_data) == 0

    return TopScorerResult(
        team_a             = team_a,
        team_b             = team_b,
        team_a_players     = players_a,
        team_b_players     = players_b,
        overall_favourite  = favourite,
        data_valid         = data_valid,
        teams_without_data = teams_without_data,
    )
