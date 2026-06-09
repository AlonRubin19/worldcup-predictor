"""Compare model predictions against bookmaker market odds.

Market odds are used for comparison and calibration only.
They are NOT used to change model predictions.

Engineering validation only — market data is placeholder until
real odds are sourced from football-data.co.uk or The Odds API.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data.strength_loader import load_strength_params
from src.data.market_loader import load_market_odds
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.dixon_coles import predict_dixon_coles
from src.models.market_implied import (
    decimal_odds_to_implied_probabilities,
    calculate_market_divergence,
    ImpliedProbabilities,
)

_DEFAULT_MATCH_RESULTS = Path(__file__).parent.parent.parent / "data" / "match_results.csv"
_DEFAULT_STRENGTH_PARAMS = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"
_DEFAULT_MARKET_ODDS = Path(__file__).parent.parent.parent / "data" / "market_odds.csv"

_DEFAULT_RHO = -0.10
_HIGH_DIVERGENCE_THRESHOLD = 0.05  # 5 percentage points


@dataclass
class MarketComparisonResult:
    match_id: str
    date: str
    team_a: str
    team_b: str
    actual_goals_a: int
    actual_goals_b: int
    actual_outcome: str
    # Model probabilities (MLE + Dixon-Coles)
    model_win_a: float
    model_draw: float
    model_win_b: float
    # Market implied probabilities (overround removed)
    market_home: float
    market_draw: float
    market_away: float
    market_overround: float
    # Divergence
    home_divergence: float
    draw_divergence: float
    away_divergence: float
    largest_divergence_outcome: str
    largest_divergence_value: float
    # Which was closer?
    model_closer_than_market: bool
    model_brier_contribution: float
    market_brier_contribution: float
    # Provenance
    bookmaker: str
    market_source_type: str
    market_research_valid: bool


@dataclass
class MarketComparisonSummary:
    total_matches: int
    model_brier: float
    market_brier: float
    brier_delta: float           # market_brier - model_brier (positive = model better)
    avg_absolute_divergence: float
    high_divergence_count: int   # matches where |largest_divergence| >= 5pp
    model_wins_high_divergence: int
    market_wins_high_divergence: int
    is_research_valid: bool
    disclaimer: str


def _brier_contribution(win_a: float, draw: float, win_b: float, outcome: str) -> float:
    o_a = 1.0 if outcome == "team_a_win" else 0.0
    o_d = 1.0 if outcome == "draw" else 0.0
    o_b = 1.0 if outcome == "team_b_win" else 0.0
    return (win_a - o_a) ** 2 + (draw - o_d) ** 2 + (win_b - o_b) ** 2


def run_market_comparison(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    market_odds_path: Path | None = None,
    rho: float = _DEFAULT_RHO,
) -> tuple[list[MarketComparisonResult], MarketComparisonSummary]:
    """Compare model predictions against bookmaker market odds.

    Only matches present in both match_results and market_odds are included.
    Market data is used for comparison only — not to modify predictions.

    Returns:
        (list[MarketComparisonResult], MarketComparisonSummary)
    """
    mr_path = match_results_path or _DEFAULT_MATCH_RESULTS
    sp_path = strength_params_path or _DEFAULT_STRENGTH_PARAMS
    mo_path = market_odds_path or _DEFAULT_MARKET_ODDS

    df = pd.read_csv(mr_path)
    strength = load_strength_params(sp_path)
    market_records = load_market_odds(mo_path)

    # Index market odds by match_id (take first record if multiple bookmakers)
    market_by_id: dict[str, object] = {}
    for m in market_records:
        if m.match_id not in market_by_id:
            market_by_id[m.match_id] = m

    results = []
    for _, row in df.iterrows():
        match_id = str(row["match_id"])
        if match_id not in market_by_id:
            continue

        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if team_a not in strength or team_b not in strength:
            continue

        odds = market_by_id[match_id]

        xg_a, xg_b = calculate_strength_adjusted_xg(
            elo_a=float(row["team_a_elo_pre"]),
            elo_b=float(row["team_b_elo_pre"]),
            params_a=strength[team_a],
            params_b=strength[team_b],
            ppg_a=float(row["team_a_points_per_game_last_10"]),
            ppg_b=float(row["team_b_points_per_game_last_10"]),
        )
        pred = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)

        market_implied = decimal_odds_to_implied_probabilities(
            odds.closing_home_odds,
            odds.closing_draw_odds,
            odds.closing_away_odds,
        )

        model_probs = {
            "team_a_win": pred.win_a,
            "draw": pred.draw,
            "team_b_win": pred.win_b,
        }
        divergence = calculate_market_divergence(model_probs, market_implied)

        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])
        if goals_a > goals_b:
            actual_outcome = "team_a_win"
        elif goals_a == goals_b:
            actual_outcome = "draw"
        else:
            actual_outcome = "team_b_win"

        model_bc = _brier_contribution(pred.win_a, pred.draw, pred.win_b, actual_outcome)
        market_bc = _brier_contribution(
            market_implied.home, market_implied.draw, market_implied.away, actual_outcome
        )

        results.append(MarketComparisonResult(
            match_id=match_id,
            date=str(row["date"]),
            team_a=team_a,
            team_b=team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            model_win_a=pred.win_a,
            model_draw=pred.draw,
            model_win_b=pred.win_b,
            market_home=market_implied.home,
            market_draw=market_implied.draw,
            market_away=market_implied.away,
            market_overround=market_implied.overround,
            home_divergence=divergence.home_divergence,
            draw_divergence=divergence.draw_divergence,
            away_divergence=divergence.away_divergence,
            largest_divergence_outcome=divergence.largest_divergence_outcome,
            largest_divergence_value=divergence.largest_divergence_value,
            model_closer_than_market=model_bc < market_bc,
            model_brier_contribution=model_bc,
            market_brier_contribution=market_bc,
            bookmaker=odds.bookmaker,
            market_source_type=odds.source_type,
            market_research_valid=odds.research_valid,
        ))

    summary = _build_summary(results)
    return results, summary


def _build_summary(results: list[MarketComparisonResult]) -> MarketComparisonSummary:
    if not results:
        return MarketComparisonSummary(
            total_matches=0,
            model_brier=0.0,
            market_brier=0.0,
            brier_delta=0.0,
            avg_absolute_divergence=0.0,
            high_divergence_count=0,
            model_wins_high_divergence=0,
            market_wins_high_divergence=0,
            is_research_valid=False,
            disclaimer=(
                "Market Intelligence is currently engineering-valid only "
                "until sourced odds are loaded."
            ),
        )

    n = len(results)
    model_brier = sum(r.model_brier_contribution for r in results) / n
    market_brier = sum(r.market_brier_contribution for r in results) / n
    avg_div = sum(abs(r.largest_divergence_value) for r in results) / n

    high_div = [r for r in results if abs(r.largest_divergence_value) >= _HIGH_DIVERGENCE_THRESHOLD]
    model_wins_hd = sum(1 for r in high_div if r.model_closer_than_market)
    market_wins_hd = sum(1 for r in high_div if not r.model_closer_than_market)

    is_valid = any(r.market_research_valid for r in results)
    if is_valid:
        disclaimer = "Market Intelligence is partially research-valid."
    else:
        disclaimer = (
            "Market Intelligence is currently engineering-valid only "
            "until sourced odds are loaded."
        )

    return MarketComparisonSummary(
        total_matches=n,
        model_brier=model_brier,
        market_brier=market_brier,
        brier_delta=market_brier - model_brier,
        avg_absolute_divergence=avg_div,
        high_divergence_count=len(high_div),
        model_wins_high_divergence=model_wins_hd,
        market_wins_high_divergence=market_wins_hd,
        is_research_valid=is_valid,
        disclaimer=disclaimer,
    )
