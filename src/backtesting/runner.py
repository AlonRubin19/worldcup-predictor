from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data.loader import load_team_ratings
from src.models.xg_calculator import calculate_xg
from src.models.poisson import predict

_MATCHES_CSV = Path(__file__).parent.parent.parent / "data" / "historical_matches.csv"

_REQUIRED_COLUMNS = {"date", "team_a", "team_b", "team_a_goals", "team_b_goals"}


@dataclass
class MatchResult:
    """Predicted vs actual outcome for one historical match."""
    date: str
    team_a: str
    team_b: str
    actual_goals_a: int
    actual_goals_b: int
    actual_outcome: str          # "team_a_win" | "draw" | "team_b_win"
    win_a_prob: float
    draw_prob: float
    win_b_prob: float
    predicted_outcome: str       # outcome with highest predicted probability
    top_scorelines: list[tuple[int, int, float]]
    exact_score_hit: bool        # actual score is #1 predicted scoreline
    in_top_3: bool               # actual score in top 3 predicted scorelines
    in_top_5: bool               # actual score in top 5 predicted scorelines
    prob_of_actual_result: float # predicted probability of the actual 1X2 outcome


def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
) -> list[MatchResult]:
    """Run model predictions for all historical matches and return per-match results.

    Args:
        matches_path: Path to historical_matches.csv. Defaults to data/historical_matches.csv.
        ratings: Dict from load_team_ratings(). Loaded from default CSV if not provided.

    Raises:
        FileNotFoundError: if matches CSV is missing.
        ValueError: if a team in the CSV is not found in ratings.
    """
    path = matches_path if matches_path is not None else _MATCHES_CSV

    if not path.exists():
        raise FileNotFoundError(f"Historical matches file not found: {path}")

    if ratings is None:
        ratings = load_team_ratings()

    df = pd.read_csv(path)
    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"historical_matches.csv missing columns: {sorted(missing_cols)}")

    results = []
    for _, row in df.iterrows():
        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if team_a not in ratings:
            raise ValueError(f"Team '{team_a}' not found in ratings")
        if team_b not in ratings:
            raise ValueError(f"Team '{team_b}' not found in ratings")

        xg_a, xg_b = calculate_xg(ratings[team_a], ratings[team_b])
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
        exact_score_hit = len(top5) > 0 and top5[0] == (goals_a, goals_b)
        in_top_3 = (goals_a, goals_b) in top5[:3]
        in_top_5 = (goals_a, goals_b) in top5

        prob_of_actual_result = probs[actual_outcome]

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
            exact_score_hit=exact_score_hit,
            in_top_3=in_top_3,
            in_top_5=in_top_5,
            prob_of_actual_result=prob_of_actual_result,
        ))

    return results
