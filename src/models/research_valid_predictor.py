"""Research-valid match prediction: ELO + MLE strength params + Dixon-Coles.

This is the primary prediction pipeline, aligned with the validated backtest.
It does not use the legacy `team_ratings.csv` or `calculate_xg()`.
"""

from dataclasses import dataclass

from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.dixon_coles import predict_dixon_coles
from src.models.poisson import PredictionResult

DEFAULT_RHO = -0.30


@dataclass
class ResearchValidInput:
    team_a: str
    team_b: str
    snapshot_a: TeamSnapshot
    snapshot_b: TeamSnapshot
    params_a: StrengthParams
    params_b: StrengthParams
    rho: float = DEFAULT_RHO


@dataclass
class ResearchValidResult:
    team_a: str
    team_b: str
    xg_a: float
    xg_b: float
    win_a: float
    draw: float
    win_b: float
    top_scorelines: list
    # Inputs stored for explainability
    elo_a: float
    elo_b: float
    alpha_attack_a: float
    alpha_attack_b: float
    beta_defense_a: float
    beta_defense_b: float


def predict_research_valid(inp: ResearchValidInput) -> ResearchValidResult:
    """Run the full research-valid prediction pipeline.

    Steps:
        1. calculate_strength_adjusted_xg (ELO + MLE alpha/beta + PPG form)
        2. predict_dixon_coles (Poisson + tau correction at given rho)

    Returns:
        ResearchValidResult with all inputs stored for explainability.
    """
    xg_a_raw, xg_b_raw = calculate_strength_adjusted_xg(
        elo_a=inp.snapshot_a.elo,
        elo_b=inp.snapshot_b.elo,
        params_a=inp.params_a,
        params_b=inp.params_b,
        ppg_a=inp.snapshot_a.ppg,
        ppg_b=inp.snapshot_b.ppg,
    )
    xg_a = calibrate_xg(xg_a_raw)
    xg_b = calibrate_xg(xg_b_raw)

    prediction: PredictionResult = predict_dixon_coles(
        team_a=inp.team_a,
        team_b=inp.team_b,
        xg_a=xg_a,
        xg_b=xg_b,
        rho=inp.rho,
    )

    return ResearchValidResult(
        team_a=inp.team_a,
        team_b=inp.team_b,
        xg_a=xg_a,
        xg_b=xg_b,
        win_a=prediction.win_a,
        draw=prediction.draw,
        win_b=prediction.win_b,
        top_scorelines=prediction.top_scorelines,
        elo_a=inp.snapshot_a.elo,
        elo_b=inp.snapshot_b.elo,
        alpha_attack_a=inp.params_a.alpha_attack,
        alpha_attack_b=inp.params_b.alpha_attack,
        beta_defense_a=inp.params_a.beta_defense,
        beta_defense_b=inp.params_b.beta_defense,
    )
