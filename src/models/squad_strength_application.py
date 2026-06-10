"""squad_strength_application.py — apply current-squad-strength factors to a
match's xG and re-derive 1X2 probabilities, for transparency in the UI.

Pure presentation/transparency layer: shows "before" (baseline ELO+MLE xG)
vs "after" (squad-adjusted xG) so users can see whether live squad data
changed the prediction. Falls back to "live_data_available=False" (no
change) when no research-valid player profiles exist for either team.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.player_loader import PlayerProfile
from src.models.current_squad_strength import compute_squad_strength_factor
from src.models.dixon_coles import predict_dixon_coles

DEFAULT_RHO = -0.10


@dataclass
class SquadStrengthMatchResult:
    xg_a_before: float
    xg_b_before: float
    xg_a_after: float
    xg_b_after: float
    factor_a: float
    factor_b: float
    win_a_before: float
    draw_before: float
    win_b_before: float
    win_a_after: float
    draw_after: float
    win_b_after: float
    live_data_available: bool


def apply_squad_strength_to_match(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    profiles: dict[str, PlayerProfile],
    injured_player_names: set[str] | None = None,
    rho: float = DEFAULT_RHO,
) -> SquadStrengthMatchResult:
    """Compute before/after xG and 1X2 probabilities using current squad data.

    If neither team has research-valid player profiles, the factors are
    1.0 (neutral) and live_data_available is False -- the UI should show
    "Live squad data unavailable -- using baseline team model."
    """
    res_a = compute_squad_strength_factor(team_a, profiles, injured_player_names)
    res_b = compute_squad_strength_factor(team_b, profiles, injured_player_names)

    before = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)

    xg_a_after = xg_a * res_a.factor
    xg_b_after = xg_b * res_b.factor
    after = predict_dixon_coles(team_a, team_b, xg_a_after, xg_b_after, rho=rho)

    return SquadStrengthMatchResult(
        xg_a_before=xg_a, xg_b_before=xg_b,
        xg_a_after=xg_a_after, xg_b_after=xg_b_after,
        factor_a=res_a.factor, factor_b=res_b.factor,
        win_a_before=before.win_a, draw_before=before.draw, win_b_before=before.win_b,
        win_a_after=after.win_a, draw_after=after.draw, win_b_after=after.win_b,
        live_data_available=(res_a.research_valid or res_b.research_valid),
    )
