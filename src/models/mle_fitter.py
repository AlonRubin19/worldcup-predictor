"""Maximum Likelihood Estimation of team attack and defense strength parameters.

Implements the Maher (1982) / Dixon-Coles (1997) Poisson model:
    lambda_a = alpha_a * beta_b   (neutral venue — no home advantage)
    lambda_b = alpha_b * beta_a

Parameters are fitted by minimizing negative weighted log-likelihood using
scipy.optimize.minimize with L-BFGS-B and bounds > 0.1.

Usage:
    from src.models.mle_fitter import fit_team_params
    params = fit_team_params(match_records, as_of_date="2022-11-19")
"""

import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class TeamStrengthParams:
    """MLE-fitted attack and defense parameters for one team."""
    team: str
    alpha_attack: float   # Attack strength (1.0 = average)
    beta_defense: float   # Defense multiplier (lower = harder to score against)
    matches_used: int
    log_likelihood: float  # Weighted log-likelihood at the fitted parameters


def _negative_log_likelihood(
    params: np.ndarray,
    team_list: list[str],
    matches: list[dict],
) -> float:
    """Compute negative weighted log-likelihood for all matches.

    params: 1D array of [alpha_A, alpha_B, ..., beta_A, beta_B, ...]
    team_list: ordered list of teams (defines param array layout)
    matches: list of dicts with team_a, team_b, team_a_goals, team_b_goals, weight
    """
    n = len(team_list)
    idx = {team: i for i, team in enumerate(team_list)}
    alpha = params[:n]   # Attack parameters
    beta = params[n:]    # Defense parameters

    total = 0.0
    for m in matches:
        i_a = idx[m["team_a"]]
        i_b = idx[m["team_b"]]
        lam_a = alpha[i_a] * beta[i_b]  # Expected goals for team A
        lam_b = alpha[i_b] * beta[i_a]  # Expected goals for team B
        g_a = m["team_a_goals"]
        g_b = m["team_b_goals"]
        w = m.get("weight", 1.0)

        # Poisson log-likelihood: goals * log(lambda) - lambda  (constant terms dropped)
        # Guard against lam = 0 or negative
        if lam_a <= 0 or lam_b <= 0:
            return 1e10

        ll = w * (
            g_a * math.log(lam_a) - lam_a +
            g_b * math.log(lam_b) - lam_b
        )
        total += ll

    return -total  # Negative because we minimize


def fit_team_params(
    matches: list[dict],
    min_matches: int = 5,
    decay_halflife_days: int = 180,
) -> dict[str, TeamStrengthParams]:
    """Fit Poisson attack/defense parameters via MLE for all teams.

    Args:
        matches: List of dicts with keys:
                 date (str ISO), team_a (str), team_b (str),
                 team_a_goals (int), team_b_goals (int).
                 Weights are computed from date (most recent = highest weight).
                 Optional "weight" key overrides computed weight.
        min_matches: Minimum total appearances required to include a team.
        decay_halflife_days: Half-life for exponential time decay.
                             Matches played decay_halflife_days ago have weight 0.5.

    Returns:
        Dict of {team_name: TeamStrengthParams} for all teams with enough data.
    """
    import datetime

    if not matches:
        return {}

    # Parse dates and compute time-decay weights
    max_date_str = max(m["date"] for m in matches)
    max_date = datetime.date.fromisoformat(max_date_str)
    lambda_decay = math.log(2) / decay_halflife_days

    weighted = []
    for m in matches:
        if "weight" not in m:
            match_date = datetime.date.fromisoformat(m["date"])
            days_ago = (max_date - match_date).days
            w = math.exp(-lambda_decay * days_ago)
        else:
            w = m["weight"]

        weighted.append({**m, "weight": w})

    # Count appearances per team
    appearances: dict[str, int] = {}
    for m in weighted:
        appearances[m["team_a"]] = appearances.get(m["team_a"], 0) + 1
        appearances[m["team_b"]] = appearances.get(m["team_b"], 0) + 1

    # Filter to teams with sufficient data; exclude others from optimization
    eligible = {t for t, cnt in appearances.items() if cnt >= min_matches}
    filtered = [m for m in weighted if m["team_a"] in eligible and m["team_b"] in eligible]

    if not filtered:
        return {}

    team_list = sorted(eligible)
    n = len(team_list)

    # Initial guess: all alphas = 1.0 (average attack), all betas = 1.0 (average defense)
    x0 = np.ones(2 * n)
    # Bounds: all parameters must be > 0.1
    bounds = [(0.1, None)] * (2 * n)

    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(team_list, filtered),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    final_params = result.x
    alpha_vals = final_params[:n]
    beta_vals = final_params[n:]

    # Build per-team output
    output = {}
    for i, team in enumerate(team_list):
        output[team] = TeamStrengthParams(
            team=team,
            alpha_attack=float(alpha_vals[i]),
            beta_defense=float(beta_vals[i]),
            matches_used=appearances[team],
            log_likelihood=-result.fun / len(filtered),  # Per-match average
        )

    return output
