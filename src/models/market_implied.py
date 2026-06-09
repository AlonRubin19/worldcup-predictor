"""Convert decimal bookmaker odds to implied probabilities and measure divergence.

Overround (bookmaker margin) is removed by normalising raw implied probabilities
so they sum to exactly 1.0. This gives a fair comparison against model probabilities.
"""

from dataclasses import dataclass


@dataclass
class ImpliedProbabilities:
    home: float
    draw: float
    away: float
    overround: float  # raw implied sum minus 1 — the bookmaker's margin


@dataclass
class MarketDivergence:
    home_divergence: float   # model_win_a - market_home  (signed)
    draw_divergence: float   # model_draw  - market_draw  (signed)
    away_divergence: float   # model_win_b - market_away  (signed)
    largest_divergence_outcome: str   # "team_a_win" | "draw" | "team_b_win"
    largest_divergence_value: float   # signed value of the largest absolute divergence


def decimal_odds_to_implied_probabilities(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> ImpliedProbabilities:
    """Convert decimal odds to normalised implied probabilities.

    Args:
        home_odds: decimal odds for home win (must be >= 1.0)
        draw_odds: decimal odds for draw
        away_odds: decimal odds for away win

    Returns:
        ImpliedProbabilities with home+draw+away summing to 1.0.

    Raises:
        ValueError: if any odds value is less than 1.0.
    """
    for name, val in [("home", home_odds), ("draw", draw_odds), ("away", away_odds)]:
        if val < 1.0:
            raise ValueError(
                f"Invalid {name} odds {val!r}: decimal odds must be >= 1.0"
            )

    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    raw_sum = raw_home + raw_draw + raw_away
    overround = raw_sum - 1.0

    return ImpliedProbabilities(
        home=raw_home / raw_sum,
        draw=raw_draw / raw_sum,
        away=raw_away / raw_sum,
        overround=overround,
    )


def calculate_market_divergence(
    model_probs: dict[str, float],
    market_probs: ImpliedProbabilities,
) -> MarketDivergence:
    """Calculate signed divergence between model and market probabilities.

    Args:
        model_probs: dict with keys "team_a_win", "draw", "team_b_win"
        market_probs: normalised market implied probabilities

    Returns:
        MarketDivergence with per-outcome signed differences and the
        largest absolute divergence identified by outcome name.
    """
    home_div = model_probs["team_a_win"] - market_probs.home
    draw_div = model_probs["draw"] - market_probs.draw
    away_div = model_probs["team_b_win"] - market_probs.away

    candidates = {
        "team_a_win": home_div,
        "draw": draw_div,
        "team_b_win": away_div,
    }
    largest_outcome = max(candidates, key=lambda k: abs(candidates[k]))

    return MarketDivergence(
        home_divergence=home_div,
        draw_divergence=draw_div,
        away_divergence=away_div,
        largest_divergence_outcome=largest_outcome,
        largest_divergence_value=candidates[largest_outcome],
    )
