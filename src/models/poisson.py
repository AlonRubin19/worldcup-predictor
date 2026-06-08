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


def predict(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
) -> PredictionResult:
    """Predict match outcome probabilities using independent Poisson distributions.

    Models each team's goals as a Poisson random variable parameterised by their
    expected goals (xG). The score matrix covers 0–10 goals per team, which
    captures >99.9% of real-match probability mass for typical xG values.

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

    max_goals = max(11, int(max(xg_a, xg_b) * 3) + 1)  # expands range for high xG

    # Build probability vectors for each team using the Poisson PMF.
    goals_range = np.arange(max_goals)
    prob_a = poisson.pmf(goals_range, xg_a)  # shape: (max_goals,)
    prob_b = poisson.pmf(goals_range, xg_b)  # shape: (max_goals,)

    # Outer product gives the joint probability matrix.
    # matrix[i][j] = P(Team A scores i) * P(Team B scores j)
    matrix = np.outer(prob_a, prob_b)  # shape: (max_goals, max_goals)

    # Win/draw probabilities from the score matrix.
    win_a = float(np.sum(np.tril(matrix, k=-1)))  # Team A scores more (below diagonal)
    draw  = float(np.sum(np.diag(matrix)))         # Equal scores (diagonal)
    win_b = float(np.sum(np.triu(matrix, k=1)))    # Team B scores more (above diagonal)

    # Top 5 scorelines: flatten to (goals_a, goals_b, probability) tuples,
    # sort by probability descending, break ties by (goals_a, goals_b) ascending.
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
