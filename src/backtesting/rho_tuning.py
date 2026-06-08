from dataclasses import dataclass
from pathlib import Path

from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

DEFAULT_RHO_GRID = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]


@dataclass
class RhoResult:
    """Backtesting metrics for one rho value."""
    rho: float
    accuracy_1x2: float
    exact_score_accuracy: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    brier_score: float
    avg_prob_actual_result: float


def tune_rho(
    ratings: dict,
    rho_grid: list[float] | None = None,
    matches_path: Path | None = None,
) -> list[RhoResult]:
    """Run Dixon-Coles backtest for each rho in the grid.

    Args:
        ratings: Dict from load_team_ratings().
        rho_grid: List of rho values to test. Defaults to DEFAULT_RHO_GRID.
        matches_path: Path to historical_matches.csv. Defaults to the project data file.

    Returns:
        One RhoResult per rho value, in the same order as rho_grid.
    """
    grid = rho_grid if rho_grid is not None else DEFAULT_RHO_GRID

    results = []
    for rho in grid:
        match_results = run_backtest(
            matches_path=matches_path,
            ratings=ratings,
            model_type="dixon_coles",
            rho=rho,
        )
        m = compute_metrics(match_results)
        results.append(RhoResult(
            rho=rho,
            accuracy_1x2=m.accuracy_1x2,
            exact_score_accuracy=m.exact_score_accuracy,
            top_3_hit_rate=m.top_3_hit_rate,
            top_5_hit_rate=m.top_5_hit_rate,
            brier_score=m.brier_score,
            avg_prob_actual_result=m.avg_prob_actual_result,
        ))

    return results


def select_best_rho(results: list[RhoResult]) -> RhoResult:
    """Select the best rho from a list of RhoResult objects.

    Selection criteria (in priority order):
    1. Lowest brier_score (primary)
    2. Highest top_3_hit_rate (tie-break)
    3. Highest exact_score_accuracy (second tie-break)

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        raise ValueError("Cannot select best rho from empty results list")

    return min(
        results,
        key=lambda r: (r.brier_score, -r.top_3_hit_rate, -r.exact_score_accuracy),
    )
