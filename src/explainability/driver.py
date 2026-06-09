"""Build a PredictionExplanation from model inputs.

Computes DriverContribution entries from before/after deltas and
parameter comparisons. No model re-running — all inputs are pre-computed.
"""

from dataclasses import dataclass, field

_ELO_THRESHOLD = 10.0      # minimum ELO gap to emit driver
_ALPHA_THRESHOLD = 0.05    # minimum alpha difference to emit driver
_BETA_THRESHOLD = 0.05     # minimum beta difference to emit driver
_SQUAD_THRESHOLD = 0.005   # minimum xG delta from squad factor to emit driver


@dataclass
class DriverContribution:
    name: str
    team: str
    direction: str       # "positive" | "negative" | "neutral"
    magnitude: float
    description: str


@dataclass
class PredictionExplanation:
    match_id: str
    team_a: str
    team_b: str
    model_type: str
    final_xg_a: float
    final_xg_b: float
    win_a: float
    draw: float
    win_b: float
    top_scorelines: list
    drivers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class ExplanationInput:
    match_id: str
    team_a: str
    team_b: str
    model_type: str
    elo_a: float
    elo_b: float
    alpha_attack_a: float
    alpha_attack_b: float
    beta_defense_a: float
    beta_defense_b: float
    xg_a_base: float
    xg_b_base: float
    squad_factor_a: float
    squad_factor_b: float
    xg_a_final: float
    xg_b_final: float
    win_a: float
    draw: float
    win_b: float
    top_scorelines: list
    player_data_research_valid: bool
    market_home_prob: float | None
    market_draw_prob: float | None
    market_away_prob: float | None
    market_research_valid: bool


def build_explanation(inp: ExplanationInput) -> PredictionExplanation:
    """Build a PredictionExplanation from all model inputs."""
    drivers: list[DriverContribution] = []
    warnings: list[str] = []

    # ELO advantage
    elo_gap = inp.elo_a - inp.elo_b
    if abs(elo_gap) >= _ELO_THRESHOLD:
        fav = inp.team_a if elo_gap > 0 else inp.team_b
        direction = "positive" if elo_gap > 0 else "negative"
        # magnitude: contribution to xG via elo_factor = gap / 4000
        mag = abs(elo_gap) / 4000.0
        drivers.append(DriverContribution(
            name="ELO advantage",
            team=fav,
            direction=direction,
            magnitude=round(mag, 4),
            description=(
                f"{fav} has a {abs(elo_gap):.0f}-point ELO advantage "
                f"(ELO {inp.elo_a:.0f} vs {inp.elo_b:.0f})."
            ),
        ))

    # Attack strength
    alpha_diff = inp.alpha_attack_a - inp.alpha_attack_b
    if abs(alpha_diff) >= _ALPHA_THRESHOLD:
        fav = inp.team_a if alpha_diff > 0 else inp.team_b
        drivers.append(DriverContribution(
            name="Attack strength",
            team=fav,
            direction="positive",
            magnitude=round(abs(alpha_diff), 4),
            description=(
                f"{fav} has higher attacking strength "
                f"(alpha {inp.alpha_attack_a:.2f} vs {inp.alpha_attack_b:.2f})."
            ),
        ))

    # Defensive weakness (high beta = easy to score against)
    beta_diff = inp.beta_defense_b - inp.beta_defense_a  # positive = team_b weaker defender
    if abs(beta_diff) >= _BETA_THRESHOLD:
        weak_defender = inp.team_b if beta_diff > 0 else inp.team_a
        drivers.append(DriverContribution(
            name="Defensive weakness",
            team=weak_defender,
            direction="negative",
            magnitude=round(abs(beta_diff), 4),
            description=(
                f"{weak_defender} has higher defensive vulnerability "
                f"(beta {inp.beta_defense_b:.2f} vs {inp.beta_defense_a:.2f})."
            ),
        ))

    # Player impact — team A
    xg_delta_a = inp.xg_a_final - inp.xg_a_base
    if abs(xg_delta_a) >= _SQUAD_THRESHOLD and abs(inp.squad_factor_a - 1.0) >= 0.001:
        direction = "positive" if xg_delta_a > 0 else "negative"
        drivers.append(DriverContribution(
            name="Player impact",
            team=inp.team_a,
            direction=direction,
            magnitude=round(abs(xg_delta_a), 4),
            description=(
                f"{inp.team_a} squad factor {inp.squad_factor_a:.2f} "
                f"adjusts xG by {xg_delta_a:+.3f} "
                f"({inp.xg_a_base:.2f} -> {inp.xg_a_final:.2f})."
            ),
        ))
        if not inp.player_data_research_valid:
            warnings.append(
                f"Player impact for {inp.team_a} is based on engineering-valid "
                "data only — not yet research-valid."
            )

    # Player impact — team B
    xg_delta_b = inp.xg_b_final - inp.xg_b_base
    if abs(xg_delta_b) >= _SQUAD_THRESHOLD and abs(inp.squad_factor_b - 1.0) >= 0.001:
        direction = "positive" if xg_delta_b > 0 else "negative"
        drivers.append(DriverContribution(
            name="Player impact",
            team=inp.team_b,
            direction=direction,
            magnitude=round(abs(xg_delta_b), 4),
            description=(
                f"{inp.team_b} squad factor {inp.squad_factor_b:.2f} "
                f"adjusts xG by {xg_delta_b:+.3f} "
                f"({inp.xg_b_base:.2f} -> {inp.xg_b_final:.2f})."
            ),
        ))
        if not inp.player_data_research_valid:
            team_b_warning = (
                f"Player impact for {inp.team_b} is based on engineering-valid "
                "data only — not yet research-valid."
            )
            if team_b_warning not in warnings:
                warnings.append(team_b_warning)

    # Dixon-Coles adjustment — always present as context
    drivers.append(DriverContribution(
        name="Dixon-Coles adjustment",
        team="both",
        direction="neutral",
        magnitude=0.0,
        description=(
            "Dixon-Coles tau correction applied to low-score cells "
            "(0-0, 1-0, 0-1, 1-1), shifting probability mass away from "
            "these scorelines relative to pure Poisson."
        ),
    ))

    # Market divergence — only when research-valid real odds are available
    if (
        inp.market_research_valid
        and inp.market_home_prob is not None
        and inp.market_draw_prob is not None
        and inp.market_away_prob is not None
    ):
        home_div = inp.win_a - inp.market_home_prob
        away_div = inp.win_b - inp.market_away_prob
        draw_div = inp.draw - inp.market_draw_prob
        largest = max(
            [("team_a_win", home_div), ("draw", draw_div), ("team_b_win", away_div)],
            key=lambda x: abs(x[1]),
        )
        direction = "positive" if largest[1] > 0 else "negative"
        drivers.append(DriverContribution(
            name="Market divergence",
            team=inp.team_a if largest[0] == "team_a_win" else (
                inp.team_b if largest[0] == "team_b_win" else "both"
            ),
            direction=direction,
            magnitude=round(abs(largest[1]), 4),
            description=(
                f"Model vs market: home {home_div:+.3f}, "
                f"draw {draw_div:+.3f}, away {away_div:+.3f}. "
                f"Largest divergence on {largest[0]} ({largest[1]:+.3f})."
            ),
        ))

    return PredictionExplanation(
        match_id=inp.match_id,
        team_a=inp.team_a,
        team_b=inp.team_b,
        model_type=inp.model_type,
        final_xg_a=inp.xg_a_final,
        final_xg_b=inp.xg_b_final,
        win_a=inp.win_a,
        draw=inp.draw,
        win_b=inp.win_b,
        top_scorelines=inp.top_scorelines,
        drivers=drivers,
        warnings=warnings,
    )
