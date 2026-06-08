from dataclasses import dataclass
from src.backtesting.runner import MatchResult


@dataclass
class BacktestMetrics:
    """Aggregate accuracy metrics for a backtesting run."""
    total_matches: int
    accuracy_1x2: float           # fraction where predicted_outcome == actual_outcome
    exact_score_accuracy: float   # fraction where exact_score_hit is True
    top_3_hit_rate: float         # fraction where in_top_3 is True
    top_5_hit_rate: float         # fraction where in_top_5 is True
    brier_score: float            # multi-class Brier score for 1X2 probabilities
    avg_prob_actual_result: float # mean predicted probability of the actual 1X2 outcome


def compute_metrics(results: list[MatchResult]) -> BacktestMetrics:
    """Compute aggregate metrics from a list of MatchResult objects.

    Brier score (multi-class, 3 outcomes):
        BS = (1/N) * sum_i [ (p_win_a - o_win_a)^2
                           + (p_draw  - o_draw )^2
                           + (p_win_b - o_win_b)^2 ]
    where o_* is 1.0 if that outcome occurred, 0.0 otherwise.

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        raise ValueError("Cannot compute metrics on empty results list")

    n = len(results)

    correct_1x2  = sum(1 for r in results if r.predicted_outcome == r.actual_outcome)
    exact_hits   = sum(1 for r in results if r.exact_score_hit)
    top3_hits    = sum(1 for r in results if r.in_top_3)
    top5_hits    = sum(1 for r in results if r.in_top_5)
    total_prob   = sum(r.prob_of_actual_result for r in results)

    brier_total = 0.0
    for r in results:
        o_win_a = 1.0 if r.actual_outcome == "team_a_win" else 0.0
        o_draw  = 1.0 if r.actual_outcome == "draw"       else 0.0
        o_win_b = 1.0 if r.actual_outcome == "team_b_win" else 0.0

        brier_total += (
            (r.win_a_prob - o_win_a) ** 2
            + (r.draw_prob  - o_draw)  ** 2
            + (r.win_b_prob - o_win_b) ** 2
        )

    return BacktestMetrics(
        total_matches=n,
        accuracy_1x2=correct_1x2 / n,
        exact_score_accuracy=exact_hits / n,
        top_3_hit_rate=top3_hits / n,
        top_5_hit_rate=top5_hits / n,
        brier_score=brier_total / n,
        avg_prob_actual_result=total_prob / n,
    )
