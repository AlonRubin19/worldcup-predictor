from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson


@dataclass
class PredictionResult:
    """Holds the output of a single match prediction."""
    team_a: str
    team_b: str
    win_a: float   # probability Team A wins
    draw: float    # probability of a draw
    win_b: float   # probability Team B wins
    # Each entry is (goals_a, goals_b, probability), sorted by probability descending.
    top_scorelines: list[tuple[int, int, float]]


def build_score_matrix(xg_a: float, xg_b: float) -> np.ndarray:
    """Build a joint Poisson probability matrix for goals 0 to N-1.

    N = max(11, int(max(xg_a, xg_b) * 3) + 1) — adaptive for high xG.

    Returns an (N x N) array where cell [i][j] =
    poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b).
    """
    max_goals = max(11, int(max(xg_a, xg_b) * 3) + 1)
    goals_range = np.arange(max_goals)
    prob_a = poisson.pmf(goals_range, xg_a)  # shape: (max_goals,)
    prob_b = poisson.pmf(goals_range, xg_b)  # shape: (max_goals,)
    return np.outer(prob_a, prob_b)           # shape: (max_goals, max_goals)


def _extract_result(team_a: str, team_b: str, matrix: np.ndarray) -> "PredictionResult":
    """Derive win/draw/win probabilities and top 5 scorelines from a score matrix."""
    max_goals = matrix.shape[0]

    win_a = float(np.sum(np.tril(matrix, k=-1)))  # Team A scores more (below diagonal)
    draw  = float(np.sum(np.diag(matrix)))         # Equal scores (diagonal)
    win_b = float(np.sum(np.triu(matrix, k=1)))    # Team B scores more (above diagonal)

    scorelines = [
        (i, j, matrix[i, j])
        for i in range(max_goals)
        for j in range(max_goals)
    ]
    scorelines.sort(key=lambda x: (-x[2], x[0], x[1]))
    top_scorelines = [(int(g_a), int(g_b), float(p)) for g_a, g_b, p in scorelines[:5]]

    return PredictionResult(
        team_a=team_a,
        team_b=team_b,
        win_a=win_a,
        draw=draw,
        win_b=win_b,
        top_scorelines=top_scorelines,
    )


def predict(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
) -> PredictionResult:
    """Predict match outcome probabilities using independent Poisson distributions.

    Args:
        team_a: Name of the first team.
        team_b: Name of the second team.
        xg_a: Expected goals for Team A (must be > 0).
        xg_b: Expected goals for Team B (must be > 0).

    Returns:
        PredictionResult with win/draw/loss probabilities and top 5 scorelines.

    Raises:
        ValueError: If either xG value is <= 0.
    """
    if xg_a <= 0 or xg_b <= 0:
        raise ValueError(f"Expected goals must be > 0, got xg_a={xg_a}, xg_b={xg_b}")

    matrix = build_score_matrix(xg_a, xg_b)
    return _extract_result(team_a, team_b, matrix)
