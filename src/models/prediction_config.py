"""prediction_config.py — central, tunable configuration for the prediction engine.

All blend weights, xG caps, and recommendation thresholds live here so the
model has no scattered magic numbers. Scenario modes are expressed as
config presets, so "market-heavy" or "fm-heavy" simulations reuse exactly
the same engine with different weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PredictionConfig:
    # Blend weights when research-valid bookmaker odds exist.
    odds_weight: float = 0.55
    fm_weight_with_odds: float = 0.30
    form_weight_with_odds: float = 0.15

    # Blend weights when odds are missing.
    fm_weight_no_odds: float = 0.60
    form_weight_no_odds: float = 0.40

    # xG bounds.
    min_xg: float = 0.25
    max_xg: float = 3.20
    fm_base_xg: float = 1.25
    fm_adjustment_cap: float = 0.45
    fm_adjustment_cap_extreme: float = 0.65
    extreme_overall_gap: float = 15.0

    # Scales the xG gap around its mean (1.0 = unchanged; <1 compresses the
    # gap, making upsets/draws likelier; >1 widens it).
    xg_gap_scale: float = 1.0

    # Score matrix.
    score_matrix_max_goals: int = 6
    rho: float = -0.13

    # Recommendation thresholds (see select_score_recommendations).
    favourite_min_win_prob: float = 0.50
    favourite_draw_gap: float = 0.10
    # Modest favourite: below 50% but clearly the most likely outcome.
    modest_favourite_prob: float = 0.42
    modest_favourite_gap: float = 0.15
    strong_favourite_prob: float = 0.55
    dominant_favourite_prob: float = 0.60
    draw_block_prob: float = 0.30
    draw_block_prob_soft: float = 0.31
    draw_score_ratio_block: float = 1.50
    draw_score_ratio_block_soft: float = 1.50
    favourite_score_ratio: float = 0.70

    # Simulation.
    simulation_runs_default: int = 10_000


DEFAULT_CONFIG = PredictionConfig()

# Scenario presets — same engine, different emphasis.
SCENARIOS: dict[str, PredictionConfig] = {
    "balanced": DEFAULT_CONFIG,
    "market-heavy": replace(
        DEFAULT_CONFIG,
        odds_weight=0.70, fm_weight_with_odds=0.20, form_weight_with_odds=0.10,
    ),
    "fm-heavy": replace(
        DEFAULT_CONFIG,
        odds_weight=0.35, fm_weight_with_odds=0.50, form_weight_with_odds=0.15,
        fm_weight_no_odds=0.75, form_weight_no_odds=0.25,
    ),
    "conservative": replace(
        DEFAULT_CONFIG,
        xg_gap_scale=0.85,
        favourite_min_win_prob=0.55,
        strong_favourite_prob=0.60,
        dominant_favourite_prob=0.65,
    ),
    "upset-sensitive": replace(
        DEFAULT_CONFIG,
        xg_gap_scale=0.75,
        favourite_min_win_prob=0.55,
        draw_block_prob=0.27,
        draw_block_prob_soft=0.28,
    ),
}


def get_scenario_config(name: str | None) -> PredictionConfig:
    """Return the config for a scenario name (case-insensitive); unknown or
    None falls back to the balanced default."""
    if not name:
        return DEFAULT_CONFIG
    return SCENARIOS.get(name.strip().lower(), DEFAULT_CONFIG)
