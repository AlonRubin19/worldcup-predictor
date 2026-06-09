"""Backtest comparing MLE+Dixon-Coles with and without player impact.

Engineering validation only — player data not yet research-valid.

Adds squad availability factor on top of the strength-adjusted xG pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import pandas as pd

from src.data.strength_loader import load_strength_params
from src.data.player_loader import load_player_profiles, load_match_availability, PlayerAvailability
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.player_impact import calculate_squad_factor, apply_player_impact, PlayerImpactInput
from src.models.dixon_coles import predict_dixon_coles

_DEFAULT_MATCH_RESULTS = Path(__file__).parent.parent.parent / "data" / "match_results.csv"
_DEFAULT_STRENGTH_PARAMS = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"
_DEFAULT_PROFILES = Path(__file__).parent.parent.parent / "data" / "player_profiles.csv"
_DEFAULT_AVAILABILITY = Path(__file__).parent.parent.parent / "data" / "match_player_availability.csv"

_DEFAULT_RHO = -0.10


@dataclass
class PlayerImpactResult:
    match_id: str
    date: str
    team_a: str
    team_b: str
    actual_goals_a: int
    actual_goals_b: int
    actual_outcome: str
    # Base model (no player impact)
    xg_a_base: float
    xg_b_base: float
    win_a_prob_base: float
    draw_prob_base: float
    win_b_prob_base: float
    predicted_outcome_base: str
    # Player-impact-adjusted model
    squad_factor_a: float
    squad_factor_b: float
    xg_a_adjusted: float
    xg_b_adjusted: float
    win_a_prob_adjusted: float
    draw_prob_adjusted: float
    win_b_prob_adjusted: float
    predicted_outcome_adjusted: str
    # Audit metadata
    source_types_a: set = field(default_factory=set)
    source_types_b: set = field(default_factory=set)
    any_research_valid_a: bool = False
    any_research_valid_b: bool = False


@dataclass
class AuditSummary:
    total_matches: int
    engineering_valid_matches: int
    research_valid_matches: int
    is_research_valid: bool
    disclaimer: str
    source_type_counts: dict


def _collect_audit(team: str, avail: list[PlayerAvailability]) -> tuple[set, bool]:
    """Return (set of source_types, any_research_valid) for a team's availability rows."""
    team_rows = [a for a in avail if a.team == team]
    source_types = {a.source_type for a in team_rows}
    any_valid = any(a.research_valid for a in team_rows)
    return source_types, any_valid


def audit_research_validity(results: list[PlayerImpactResult]) -> AuditSummary:
    """Summarise research validity across all backtest results."""
    source_counter: Counter = Counter()
    research_valid_matches = 0

    for r in results:
        all_sources = r.source_types_a | r.source_types_b
        source_counter.update(all_sources)
        if r.any_research_valid_a or r.any_research_valid_b:
            research_valid_matches += 1

    engineering_valid = len(results) - research_valid_matches
    is_valid = research_valid_matches > 0

    if is_valid:
        disclaimer = (
            f"Player Impact is partially research-valid "
            f"({research_valid_matches}/{len(results)} matches)."
        )
    else:
        disclaimer = (
            "Player Impact is engineering-valid only. "
            "It is not yet research-valid."
        )

    return AuditSummary(
        total_matches=len(results),
        engineering_valid_matches=engineering_valid,
        research_valid_matches=research_valid_matches,
        is_research_valid=is_valid,
        disclaimer=disclaimer,
        source_type_counts=dict(source_counter),
    )


def run_player_impact_backtest(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    player_profiles_path: Path | None = None,
    match_availability_path: Path | None = None,
    rho: float = _DEFAULT_RHO,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
    skip_missing_teams: bool = False,
) -> list[PlayerImpactResult]:
    """Run backtest with and without player impact for each match.

    For each match:
    1. Compute base xG from MLE strength parameters + ELO + form.
    2. Run Dixon-Coles on base xG → base probabilities.
    3. Compute squad factors from player availability.
    4. Apply squad factors → adjusted xG.
    5. Run Dixon-Coles on adjusted xG → adjusted probabilities.

    Returns:
        list[PlayerImpactResult] with both base and adjusted predictions,
        plus audit metadata (source_types, research_valid flags).
    """
    mr_path = match_results_path or _DEFAULT_MATCH_RESULTS
    sp_path = strength_params_path or _DEFAULT_STRENGTH_PARAMS
    pp_path = player_profiles_path or _DEFAULT_PROFILES
    av_path = match_availability_path or _DEFAULT_AVAILABILITY

    df = pd.read_csv(mr_path)
    strength = load_strength_params(sp_path)
    profiles = load_player_profiles(pp_path)
    all_availability = load_match_availability(av_path)

    if filter_date_from:
        df = df[df["date"] >= filter_date_from]
    if filter_date_to:
        df = df[df["date"] <= filter_date_to]

    results = []
    for _, row in df.iterrows():
        match_id = str(row["match_id"])
        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if skip_missing_teams and (team_a not in strength or team_b not in strength):
            continue

        xg_a_base, xg_b_base = calculate_strength_adjusted_xg(
            elo_a=float(row["team_a_elo_pre"]),
            elo_b=float(row["team_b_elo_pre"]),
            params_a=strength[team_a],
            params_b=strength[team_b],
            ppg_a=float(row["team_a_points_per_game_last_10"]),
            ppg_b=float(row["team_b_points_per_game_last_10"]),
        )

        pred_base = predict_dixon_coles(team_a, team_b, xg_a_base, xg_b_base, rho=rho)

        match_avail = [a for a in all_availability if a.match_id == match_id]
        squad_factor_a = calculate_squad_factor(team_a, profiles, match_avail)
        squad_factor_b = calculate_squad_factor(team_b, profiles, match_avail)

        xg_a_adj, xg_b_adj = apply_player_impact(PlayerImpactInput(
            xg_a=xg_a_base,
            xg_b=xg_b_base,
            squad_factor_a=squad_factor_a,
            squad_factor_b=squad_factor_b,
        ))

        pred_adj = predict_dixon_coles(team_a, team_b, xg_a_adj, xg_b_adj, rho=rho)

        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])

        if goals_a > goals_b:
            actual_outcome = "team_a_win"
        elif goals_a == goals_b:
            actual_outcome = "draw"
        else:
            actual_outcome = "team_b_win"

        def _best(win_a, draw, win_b):
            probs = {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}
            return max(probs, key=probs.get)

        source_types_a, any_rv_a = _collect_audit(team_a, match_avail)
        source_types_b, any_rv_b = _collect_audit(team_b, match_avail)

        results.append(PlayerImpactResult(
            match_id=match_id,
            date=str(row["date"]),
            team_a=team_a,
            team_b=team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            xg_a_base=xg_a_base,
            xg_b_base=xg_b_base,
            win_a_prob_base=pred_base.win_a,
            draw_prob_base=pred_base.draw,
            win_b_prob_base=pred_base.win_b,
            predicted_outcome_base=_best(pred_base.win_a, pred_base.draw, pred_base.win_b),
            squad_factor_a=squad_factor_a,
            squad_factor_b=squad_factor_b,
            xg_a_adjusted=xg_a_adj,
            xg_b_adjusted=xg_b_adj,
            win_a_prob_adjusted=pred_adj.win_a,
            draw_prob_adjusted=pred_adj.draw,
            win_b_prob_adjusted=pred_adj.win_b,
            predicted_outcome_adjusted=_best(pred_adj.win_a, pred_adj.draw, pred_adj.win_b),
            source_types_a=source_types_a,
            source_types_b=source_types_b,
            any_research_valid_a=any_rv_a,
            any_research_valid_b=any_rv_b,
        ))

    return results
