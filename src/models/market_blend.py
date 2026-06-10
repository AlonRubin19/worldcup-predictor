"""market_blend.py — blend model 1X2 probabilities with bookmaker market.

final_probability = 0.85 * model_probability + 0.15 * market_implied_probability

Only applied when real, research-valid bookmaker odds are available. Otherwise
the model probability is used unchanged (used_market=False).
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_WEIGHT = 0.85
MARKET_WEIGHT = 0.15

BLEND_LABEL = "Model + Market Blend: 85% model / 15% bookmaker market"
MODEL_ONLY_LABEL = "Model only (no valid bookmaker market available)"


@dataclass
class BlendedProbabilities:
    win_a: float
    draw: float
    win_b: float
    used_market: bool
    label: str


def blend_probabilities(
    model_win_a: float,
    model_draw: float,
    model_win_b: float,
    market_win_a: float | None,
    market_draw: float | None,
    market_win_b: float | None,
    market_research_valid: bool,
) -> BlendedProbabilities:
    """Blend model and market 1X2 probabilities, normalized to sum to 1."""
    has_market = (
        market_research_valid
        and market_win_a is not None
        and market_draw is not None
        and market_win_b is not None
    )

    if not has_market:
        return BlendedProbabilities(
            win_a=model_win_a,
            draw=model_draw,
            win_b=model_win_b,
            used_market=False,
            label=MODEL_ONLY_LABEL,
        )

    win_a = MODEL_WEIGHT * model_win_a + MARKET_WEIGHT * market_win_a
    draw = MODEL_WEIGHT * model_draw + MARKET_WEIGHT * market_draw
    win_b = MODEL_WEIGHT * model_win_b + MARKET_WEIGHT * market_win_b

    total = win_a + draw + win_b
    if total > 0:
        win_a, draw, win_b = win_a / total, draw / total, win_b / total

    return BlendedProbabilities(
        win_a=win_a, draw=draw, win_b=win_b,
        used_market=True, label=BLEND_LABEL,
    )
