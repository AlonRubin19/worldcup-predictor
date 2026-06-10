"""prediction_runner.py — pure Python prediction computation layer.

Extracts all model computation out of the Streamlit render layer so it can be:
  - tested without Streamlit
  - called from Tournament Overview (inline) and Match Analyzer (full page)
  - audited for mathematical consistency

No Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.models.research_valid_predictor import (
    predict_research_valid, ResearchValidInput, DEFAULT_RHO,
)
from src.models.dixon_coles import build_dc_matrix
from src.models.betting_markets import compute_betting_markets, BettingMarketProbabilities as BettingMarkets
from src.models.recommendations import generate_recommendations, RecommendationSet
from src.explainability.driver import build_explanation, ExplanationInput, PredictionExplanation
from src.app.components.prediction_cards import compute_confidence, ConfidenceResult
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.market_blend import blend_probabilities, BlendedProbabilities


# ─────────────────────────────────────────────────────────────────────────────
# Input / output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerInput:
    """All inputs needed to compute a full match prediction.

    Attributes:
        team_a, team_b:  Internal team names.
        snapshot_a/b:    ELO + PPG snapshots.
        params_a/b:      MLE attack/defence strength params.
        is_research_valid: Whether the input data is research-valid.
        lineup_source:   Human-readable label describing the lineup data source.
        data_warnings:   Extra warning strings to pass to the recommendation engine.
    """
    team_a:            str
    team_b:            str
    snapshot_a:        TeamSnapshot
    snapshot_b:        TeamSnapshot
    params_a:          StrengthParams
    params_b:          StrengthParams
    is_research_valid: bool       = True
    lineup_source:     str        = "Manual (placeholder)"
    data_warnings:     list[str]  = field(default_factory=list)
    market_win_a:           float | None = None
    market_draw:            float | None = None
    market_win_b:           float | None = None
    market_research_valid:  bool         = False


@dataclass
class FullPrediction:
    """All outputs of one complete match prediction run.

    Every field here maps 1-to-1 to something rendered in the UI so the
    render layer can be a thin wrapper with no computation of its own.
    """
    # Teams
    team_a:            str
    team_b:            str
    # 1X2
    win_a:             float
    draw:              float
    win_b:             float
    # xG
    xg_a:              float
    xg_b:              float
    # Scorelines
    most_likely_score: str
    top_scorelines:    list          # list of (g_a, g_b, prob)
    # Markets (full object — lets callers drill in)
    markets:           BettingMarkets
    # Signals
    recommendations:   RecommendationSet
    # Confidence
    confidence:        ConfidenceResult
    # Explanation
    explanation:       PredictionExplanation
    # Meta
    is_research_valid: bool
    lineup_source:     str
    model_label:       str = "ELO + MLE + Dixon-Coles (rho=-0.30)"
    # Market blend (1X2 only -- scoreline matrix is unaffected)
    blend:             BlendedProbabilities | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_prediction(inp: RunnerInput) -> FullPrediction:
    """Compute a complete match prediction from a RunnerInput.

    This is the single source of truth for all prediction computation.
    It is called from:
      - Tournament Overview (inline analysis)
      - Match Analyzer tab (full-page analysis)
      - Consistency audit script
      - Tests

    Args:
        inp: RunnerInput with teams, snapshots, strength params, and flags.

    Returns:
        FullPrediction with all outputs needed by the UI.
    """
    # ── xG ───────────────────────────────────────────────────────────────────
    raw_a, raw_b = calculate_strength_adjusted_xg(
        inp.snapshot_a.elo, inp.snapshot_b.elo,
        inp.params_a, inp.params_b,
        inp.snapshot_a.ppg, inp.snapshot_b.ppg,
    )
    xg_a, xg_b = calibrate_xg(raw_a), calibrate_xg(raw_b)

    # ── Probabilities (DC matrix) ─────────────────────────────────────────────
    result = predict_research_valid(ResearchValidInput(
        team_a=inp.team_a, team_b=inp.team_b,
        snapshot_a=inp.snapshot_a, snapshot_b=inp.snapshot_b,
        params_a=inp.params_a, params_b=inp.params_b,
    ))

    # ── Scorelines ────────────────────────────────────────────────────────────
    # top_scorelines is already sorted descending by the predictor
    top_scorelines = result.top_scorelines
    most_likely = (
        f"{top_scorelines[0][0]}-{top_scorelines[0][1]}"
        if top_scorelines else "?-?"
    )

    # ── Markets ───────────────────────────────────────────────────────────────
    # IMPORTANT: build_dc_matrix and compute_betting_markets use the SAME xg_a/xg_b
    # so market probabilities are consistent with the scoreline table.
    matrix  = build_dc_matrix(xg_a, xg_b, rho=DEFAULT_RHO)
    markets = compute_betting_markets(inp.team_a, inp.team_b, matrix)

    # ── Confidence ────────────────────────────────────────────────────────────
    all_warnings = list(inp.data_warnings)
    confidence   = compute_confidence(result.win_a, result.draw, result.win_b, all_warnings)

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = generate_recommendations(
        betting_markets=markets,
        prediction_confidence=confidence.label,
        data_warnings=all_warnings,
        is_research_valid=inp.is_research_valid,
        top_n=5,
    )

    # ── Explanation ───────────────────────────────────────────────────────────
    expl_inp = ExplanationInput(
        match_id="live",
        team_a=inp.team_a, team_b=inp.team_b,
        model_type="ELO + MLE + Dixon-Coles (rho=-0.30)",
        elo_a=inp.snapshot_a.elo, elo_b=inp.snapshot_b.elo,
        alpha_attack_a=inp.params_a.alpha_attack, alpha_attack_b=inp.params_b.alpha_attack,
        beta_defense_a=inp.params_a.beta_defense, beta_defense_b=inp.params_b.beta_defense,
        xg_a_base=xg_a, xg_b_base=xg_b,
        squad_factor_a=1.0, squad_factor_b=1.0,
        xg_a_final=xg_a, xg_b_final=xg_b,
        win_a=result.win_a, draw=result.draw, win_b=result.win_b,
        top_scorelines=top_scorelines,
        player_data_research_valid=False,
        market_home_prob=None, market_draw_prob=None, market_away_prob=None,
        market_research_valid=False,
    )
    explanation = build_explanation(expl_inp)

    blend = blend_probabilities(
        model_win_a=result.win_a, model_draw=result.draw, model_win_b=result.win_b,
        market_win_a=inp.market_win_a, market_draw=inp.market_draw, market_win_b=inp.market_win_b,
        market_research_valid=inp.market_research_valid,
    )

    return FullPrediction(
        team_a=inp.team_a,
        team_b=inp.team_b,
        win_a=result.win_a,
        draw=result.draw,
        win_b=result.win_b,
        xg_a=round(xg_a, 3),
        xg_b=round(xg_b, 3),
        most_likely_score=most_likely,
        top_scorelines=top_scorelines,
        markets=markets,
        recommendations=recommendations,
        confidence=confidence,
        explanation=explanation,
        is_research_valid=inp.is_research_valid,
        lineup_source=inp.lineup_source,
        blend=blend,
    )
