"""Calculate expected goals using MLE-fitted opponent strength parameters.

This replaces the raw goals-average approach in pre_match_xg.py.
Formula: xg_a = BASE_XG * alpha_a * beta_b * form_a * elo_factor_a

The same BASE_XG, XG_MIN, XG_MAX, FORM_BASE, FORM_SCALE constants are used
so both xG calculators are directly comparable.
"""

from src.data.strength_loader import StrengthParams

BASE_XG    = 1.35
XG_MIN     = 0.2
XG_MAX     = 4.5
FORM_BASE  = 0.85
FORM_SCALE = 0.30


def calculate_strength_adjusted_xg(
    elo_a: float,
    elo_b: float,
    params_a: StrengthParams,
    params_b: StrengthParams,
    ppg_a: float,
    ppg_b: float,
) -> tuple[float, float]:
    """Calculate expected goals from MLE strength parameters.

    No manually estimated ratings. No raw goal averages.
    All signal comes from:
    - α: how prolific the team is at scoring (MLE-fitted)
    - β: how easy the opponent is to score against (MLE-fitted)
    - ELO: relative strength for this specific matchup
    - Form: recent PPG as a dynamic form multiplier

    Formula:
        xg_a = BASE_XG * alpha_a * beta_b * form_a * elo_factor_a
        xg_b = BASE_XG * alpha_b * beta_a * form_b * elo_factor_b

        form   = FORM_BASE + (ppg / 3) * FORM_SCALE
        elo_factor_a = 1 + (elo_a - elo_b) / 4000

    Both clamped to [XG_MIN, XG_MAX].
    """
    form_a = FORM_BASE + (ppg_a / 3) * FORM_SCALE
    form_b = FORM_BASE + (ppg_b / 3) * FORM_SCALE

    elo_factor_a = 1 + (elo_a - elo_b) / 4000
    elo_factor_b = 1 + (elo_b - elo_a) / 4000

    xg_a = BASE_XG * params_a.alpha_attack * params_b.beta_defense * form_a * elo_factor_a
    xg_b = BASE_XG * params_b.alpha_attack * params_a.beta_defense * form_b * elo_factor_b

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
