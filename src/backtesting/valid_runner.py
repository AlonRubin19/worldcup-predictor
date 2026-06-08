from pathlib import Path

from src.data.pre_match_loader import load_pre_match_stats
from src.models.pre_match_xg import calculate_pre_match_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import MatchResult   # reuse same result dataclass


def run_valid_backtest(
    path: Path | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[MatchResult]:
    """Run backtest using ONLY pre-match statistics — no manually estimated ratings.

    Unlike run_backtest() (which reads team_ratings.csv), this function derives
    xG exclusively from per-match pre-game statistics in pre_match_team_stats.csv.

    Args:
        path: Path to pre_match_team_stats.csv. Defaults to the project data file.
        model_type: "poisson" (default) or "dixon_coles".
        rho: Dixon-Coles parameter. Only used when model_type == "dixon_coles".
        min_matches: Minimum historical matches required to be considered reliable.
        exclude_insufficient: If True, skip rows where either team has fewer than
                              min_matches available.

    Returns:
        list[MatchResult] — same structure as run_backtest(), compatible with
        compute_metrics() and all downstream analysis.

    Raises:
        FileNotFoundError: if CSV is missing.
        ValueError: if model_type is invalid.
    """
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")

    match_stats = load_pre_match_stats(
        path=path,
        min_matches=min_matches,
        exclude_insufficient=exclude_insufficient,
    )

    results = []
    for m in match_stats:
        xg_a, xg_b = calculate_pre_match_xg(m)

        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(m.team_a, m.team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(m.team_a, m.team_b, xg_a, xg_b)

        goals_a = m.team_a_goals
        goals_b = m.team_b_goals

        if goals_a > goals_b:
            actual_outcome = "team_a_win"
        elif goals_a == goals_b:
            actual_outcome = "draw"
        else:
            actual_outcome = "team_b_win"

        probs = {
            "team_a_win": prediction.win_a,
            "draw": prediction.draw,
            "team_b_win": prediction.win_b,
        }
        predicted_outcome = max(probs, key=probs.get)

        top5 = [(g_a, g_b) for g_a, g_b, _ in prediction.top_scorelines]
        exact_score_hit = len(top5) > 0 and top5[0] == (goals_a, goals_b)
        in_top_3 = (goals_a, goals_b) in top5[:3]
        in_top_5 = (goals_a, goals_b) in top5

        results.append(MatchResult(
            date=m.date,
            team_a=m.team_a,
            team_b=m.team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            win_a_prob=prediction.win_a,
            draw_prob=prediction.draw,
            win_b_prob=prediction.win_b,
            predicted_outcome=predicted_outcome,
            top_scorelines=prediction.top_scorelines,
            exact_score_hit=exact_score_hit,
            in_top_3=in_top_3,
            in_top_5=in_top_5,
            prob_of_actual_result=probs[actual_outcome],
        ))

    return results
