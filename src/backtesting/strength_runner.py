"""Run backtesting using real match data and MLE-fitted strength parameters.

Unlike run_valid_backtest() (which uses raw goal averages), this runner
uses strength-adjusted xG derived from Dixon-Coles MLE parameters.

Data sources:
    data/match_results.csv        -- real historical pre-match stats
    data/team_strength_params.csv -- MLE α/β for each team
"""

from pathlib import Path
import pandas as pd

from src.data.strength_loader import load_strength_params
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import MatchResult

_DEFAULT_MATCH_RESULTS = Path(__file__).parent.parent.parent / "data" / "match_results.csv"
_DEFAULT_STRENGTH_PARAMS = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"


def run_strength_backtest(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
) -> list[MatchResult]:
    """Run backtest using real data and MLE strength parameters.

    Args:
        match_results_path: Path to match_results.csv. Default: data/match_results.csv.
        strength_params_path: Path to team_strength_params.csv. Default: data/team_strength_params.csv.
        model_type: "poisson" or "dixon_coles".
        rho: Dixon-Coles rho parameter (ignored for Poisson).
        filter_date_from: Optional ISO date string — only include matches from this date.
        filter_date_to: Optional ISO date string — only include matches up to this date.

    Returns:
        list[MatchResult] — same structure as run_backtest() and run_valid_backtest().

    Raises:
        FileNotFoundError: if either CSV is missing.
        ValueError: if a team in match_results is not in strength_params, or invalid model_type.
    """
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")

    mr_path = match_results_path or _DEFAULT_MATCH_RESULTS
    sp_path = strength_params_path or _DEFAULT_STRENGTH_PARAMS

    df = pd.read_csv(mr_path)
    strength = load_strength_params(sp_path)

    if filter_date_from:
        df = df[df["date"] >= filter_date_from]
    if filter_date_to:
        df = df[df["date"] <= filter_date_to]

    results = []
    for _, row in df.iterrows():
        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if team_a not in strength:
            raise ValueError(f"Team '{team_a}' not found in strength parameters")
        if team_b not in strength:
            raise ValueError(f"Team '{team_b}' not found in strength parameters")

        xg_a, xg_b = calculate_strength_adjusted_xg(
            elo_a=float(row["team_a_elo_pre"]),
            elo_b=float(row["team_b_elo_pre"]),
            params_a=strength[team_a],
            params_b=strength[team_b],
            ppg_a=float(row["team_a_points_per_game_last_10"]),
            ppg_b=float(row["team_b_points_per_game_last_10"]),
        )

        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)

        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])

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

        results.append(MatchResult(
            date=str(row["date"]),
            team_a=team_a,
            team_b=team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            win_a_prob=prediction.win_a,
            draw_prob=prediction.draw,
            win_b_prob=prediction.win_b,
            predicted_outcome=predicted_outcome,
            top_scorelines=prediction.top_scorelines,
            exact_score_hit=len(top5) > 0 and top5[0] == (goals_a, goals_b),
            in_top_3=(goals_a, goals_b) in top5[:3],
            in_top_5=(goals_a, goals_b) in top5,
            prob_of_actual_result=probs[actual_outcome],
        ))

    return results
