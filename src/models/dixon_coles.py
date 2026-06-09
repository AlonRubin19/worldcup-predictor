import numpy as np
from src.models.poisson import build_score_matrix, _extract_result, PredictionResult


def predict_dixon_coles(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    rho: float = -0.10,
) -> PredictionResult:
    """Predict match outcomes using Poisson + Dixon-Coles tau correction.

    Applies a tau correction factor to the four low-score cells of the Poisson
    score matrix to better capture the statistical dependence between low scores.
    The matrix is then normalized to ensure probabilities sum to 1.

    Tau correction (Dixon & Coles, 1997):
        0-0:  tau = 1 - (xg_a * xg_b * rho)
        0-1:  tau = 1 + (xg_a * rho)
        1-0:  tau = 1 + (xg_b * rho)
        1-1:  tau = 1 - rho
        else: tau = 1  (unchanged)

    When rho=0, all tau values equal 1 and the result matches pure Poisson.
    Default rho=-0.10 is the empirical value from Dixon & Coles (1997).

    Args:
        team_a: Name of the first team.
        team_b: Name of the second team.
        xg_a: Expected goals for Team A (must be > 0).
        xg_b: Expected goals for Team B (must be > 0).
        rho: Correction parameter (default -0.10). Negative values reduce
             0-0 and 1-1 probabilities relative to pure Poisson.

    Returns:
        PredictionResult with win/draw/loss probabilities and top 5 scorelines.

    Raises:
        ValueError: If either xG value is <= 0.
    """
    if xg_a <= 0 or xg_b <= 0:
        raise ValueError(f"Expected goals must be > 0, got xg_a={xg_a}, xg_b={xg_b}")

    matrix = build_score_matrix(xg_a, xg_b)

    # Apply Dixon-Coles tau correction to the four low-score cells.
    # All other cells are multiplied by tau=1 (no change).
    matrix[0, 0] *= 1 - (xg_a * xg_b * rho)  # 0-0
    matrix[0, 1] *= 1 + (xg_a * rho)           # 0-1
    matrix[1, 0] *= 1 + (xg_b * rho)           # 1-0
    matrix[1, 1] *= 1 - rho                     # 1-1

    # Normalize so all probabilities sum to 1.
    matrix /= matrix.sum()

    return _extract_result(team_a, team_b, matrix)


def build_dc_matrix(xg_a: float, xg_b: float, rho: float = -0.10) -> np.ndarray:
    """Return the DC-corrected, normalised score probability matrix.

    Same math as predict_dixon_coles but returns the raw matrix instead of
    deriving win/draw/lose probabilities. Useful for downstream market engines.
    """
    if xg_a <= 0 or xg_b <= 0:
        raise ValueError(f"Expected goals must be > 0, got xg_a={xg_a}, xg_b={xg_b}")

    matrix = build_score_matrix(xg_a, xg_b)
    matrix[0, 0] *= 1 - (xg_a * xg_b * rho)
    matrix[0, 1] *= 1 + (xg_a * rho)
    matrix[1, 0] *= 1 + (xg_b * rho)
    matrix[1, 1] *= 1 - rho
    np.clip(matrix, 0.0, None, out=matrix)
    matrix /= matrix.sum()
    return matrix
