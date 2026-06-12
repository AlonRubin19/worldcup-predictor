"""match_simulator.py — coherent match simulation engine.

Combines, in one shared xG/score-matrix framework:
  1. bookmaker odds (when research-valid odds are available)
  2. existing ELO + MLE + calibrated-xG model (research_valid_predictor)
  3. Football Manager squad/team strength (data/fm_team_strength.csv)
  4. Dixon-Coles score matrix -> W/D/L + exact-score probabilities
  5. Monte Carlo simulation over the same final xG values

The key design goal is *coherence*: win/draw/loss probabilities and the
exact-score table are both derived from the same final score matrix, so
the recommended exact score is never "1-1 by default" while the model
says one team is a 58% favourite.

Blend weights:
  - With research-valid bookmaker odds:  55% market / 30% FM / 15% existing model
  - Without bookmaker odds:               60% FM / 40% existing model

If FM data is unavailable for one or both teams, the FM weight is
redistributed to the existing model (FM layer degrades gracefully).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.stats import poisson as _poisson

from src.data.team_snapshot_loader import TeamSnapshot, load_team_snapshots
from src.data.strength_loader import StrengthParams, load_strength_params
from src.data.fm_strength_loader import FMTeamStrength, load_fm_team_strength, get_fm_strength
from src.data.market_odds_loader import get_market_odds_for_match, MarketOddsResult
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.dixon_coles import predict_dixon_coles, build_dc_matrix

DEFAULT_RHO = -0.13

# Base xG range (per team, before adjustments).
FM_BASE_XG = 1.25

# xG caps.
MIN_XG = 0.25
MAX_XG_NORMAL = 3.20
FM_ADJ_CAP_NORMAL = 0.45
FM_ADJ_CAP_EXTREME = 0.65
EXTREME_OVERALL_GAP = 15.0  # FM overall-rating gap above which the wider cap applies

# Blend weights.
WEIGHT_WITH_ODDS = {"market": 0.55, "fm": 0.30, "existing": 0.15}
WEIGHT_NO_ODDS = {"fm": 0.60, "existing": 0.40}

_DEFAULT_SNAP = TeamSnapshot(elo=1800.0, ppg=1.5)
_DEFAULT_PAR = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _cached_snapshots() -> dict[str, TeamSnapshot]:
    try:
        return load_team_snapshots()
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def _cached_params() -> dict[str, StrengthParams]:
    try:
        return load_strength_params()
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def _cached_fm() -> dict[str, FMTeamStrength]:
    return load_fm_team_strength()


# ─────────────────────────────────────────────────────────────────────────────
# FM matchup -> xG adjustment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FMEdges:
    attack_edge: float
    midfield_edge: float
    opponent_gk_rating: float
    depth_edge: float
    overall_a: float
    overall_b: float
    adjustment: float


def _fm_adjustment(team_fm: FMTeamStrength, opp_fm: FMTeamStrength) -> FMEdges:
    """Convert an FM component matchup into an xG adjustment for `team_fm`."""
    attack_edge = team_fm.attack - opp_fm.defense
    midfield_edge = team_fm.midfield - opp_fm.midfield
    depth_edge = team_fm.depth - opp_fm.depth

    adjustment = (
        0.022 * attack_edge
        + 0.010 * midfield_edge
        - 0.010 * (opp_fm.goalkeeper - 80.0)
        + 0.006 * depth_edge
    )

    overall_gap = abs(team_fm.overall - opp_fm.overall)
    cap = FM_ADJ_CAP_EXTREME if overall_gap > EXTREME_OVERALL_GAP else FM_ADJ_CAP_NORMAL
    adjustment = float(np.clip(adjustment, -cap, cap))

    return FMEdges(
        attack_edge=attack_edge,
        midfield_edge=midfield_edge,
        opponent_gk_rating=opp_fm.goalkeeper,
        depth_edge=depth_edge,
        overall_a=team_fm.overall,
        overall_b=opp_fm.overall,
        adjustment=adjustment,
    )


# ─────────────────────────────────────────────────────────────────────────────
# xG calibration to a target W/D/L (e.g. market-implied)
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_xg_to_target(
    xg_a: float,
    xg_b: float,
    target_win_a: float,
    target_draw: float,
    target_win_b: float,
    rho: float = DEFAULT_RHO,
    iters: int = 30,
    step: float = 0.12,
) -> tuple[float, float]:
    """Nudge (xg_a, xg_b) so predict_dixon_coles(xg_a, xg_b) gets closer to
    the target W/D/L distribution. Pure hill-climbing — no closed-form
    inversion exists for the DC-corrected Poisson matrix.
    """
    for _ in range(iters):
        res = predict_dixon_coles("a", "b", xg_a, xg_b, rho=rho)
        diff_a = target_win_a - res.win_a
        diff_b = target_win_b - res.win_b
        xg_a *= (1.0 + step * diff_a)
        xg_b *= (1.0 + step * diff_b)
        xg_a = float(np.clip(xg_a, MIN_XG, MAX_XG_NORMAL))
        xg_b = float(np.clip(xg_b, MIN_XG, MAX_XG_NORMAL))
    return xg_a, xg_b


# ─────────────────────────────────────────────────────────────────────────────
# Exact-score helpers
# ─────────────────────────────────────────────────────────────────────────────

def _full_scorelines(matrix: np.ndarray, max_score: int = 5) -> list[tuple[int, int, float]]:
    """All (goals_a, goals_b, probability) for 0..max_score, sorted descending."""
    n = min(max_score + 1, matrix.shape[0])
    lines = [(i, j, float(matrix[i, j])) for i in range(n) for j in range(n)]
    lines.sort(key=lambda x: (-x[2], x[0], x[1]))
    return lines


def _select_recommended_score(
    team_a: str,
    team_b: str,
    scorelines: list[tuple[int, int, float]],
    win_a: float,
    draw: float,
    win_b: float,
) -> tuple[str, str, str]:
    """Return (raw_top_score, recommended_exact_score, reason)."""
    raw_top = scorelines[0]
    raw_top_score = f"{raw_top[0]}-{raw_top[1]}"

    recommended = raw_top
    reason = "Highest-probability scoreline in the calibrated score matrix."

    if raw_top[0] == raw_top[1]:
        draw_p = raw_top[2]
        favourite, threshold_team = None, None
        if win_a >= 0.55:
            favourite, threshold_team = "a", team_a
        elif win_b >= 0.55:
            favourite, threshold_team = "b", team_b

        if favourite == "a":
            candidates = [s for s in scorelines if s[0] > s[1]]
        elif favourite == "b":
            candidates = [s for s in scorelines if s[1] > s[0]]
        else:
            candidates = []

        if candidates:
            best_win = max(candidates, key=lambda s: s[2])
            # A team with >=55% win probability is, by definition, more likely
            # to win than to draw -- never recommend the draw scoreline here,
            # regardless of how close the top winning scoreline's probability
            # is to the top draw scoreline's probability.
            recommended = best_win
            reason = (
                f"{threshold_team} win probability is "
                f"{(win_a if favourite == 'a' else win_b):.0%}, which exceeds the draw "
                f"probability ({draw:.0%}); the best winning scoreline "
                f"({best_win[0]}-{best_win[1]}) is recommended instead of the top "
                f"draw scoreline ({raw_top_score})."
            )

        if recommended is raw_top:
            reason = "Draw is genuinely the most likely outcome in the calibrated score matrix."
    else:
        winner = team_a if raw_top[0] > raw_top[1] else team_b
        reason = (
            f"{winner} is favoured to win and the top-probability scoreline "
            f"({raw_top_score}) is consistent with that outcome."
        )

    recommended_score = f"{recommended[0]}-{recommended[1]}"
    return raw_top_score, recommended_score, reason


def _confidence_label(win_a: float, draw: float, win_b: float) -> str:
    m = max(win_a, draw, win_b)
    if m >= 0.60:
        return "high"
    if m >= 0.45:
        return "medium-high"
    if m >= 0.35:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchSimulationResult:
    team1: str
    team2: str
    team1_win_probability: float
    draw_probability: float
    team2_win_probability: float
    expected_goals_team1: float
    expected_goals_team2: float
    raw_top_score: str
    recommended_exact_score: str
    top_5_exact_scores: list[dict]
    confidence: str
    explanation: str
    fm_used: bool
    fm_edges: dict | None
    odds_used: bool
    market_odds: dict | None
    top_players_team1: str
    top_players_team2: str


@dataclass
class MonteCarloMatchResult(MatchSimulationResult):
    n_simulations: int = 0
    simulated_team1_win_probability: float = 0.0
    simulated_draw_probability: float = 0.0
    simulated_team2_win_probability: float = 0.0
    simulated_top_5_exact_scores: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute final xG + W/D/L + exact scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_match_xg(
    team_a: str,
    team_b: str,
    snaps: dict[str, TeamSnapshot] | None = None,
    params: dict[str, StrengthParams] | None = None,
    fm_data: dict[str, FMTeamStrength] | None = None,
    odds: MarketOddsResult | None = None,
    rho: float = DEFAULT_RHO,
) -> dict:
    """Compute the final, blended/calibrated xG pair plus all supporting data.

    Returns a dict with keys: xg_a, xg_b, win_a, draw, win_b, matrix,
    fm_used, fm_edges_a, fm_edges_b, odds_used, market_odds,
    existing_xg_a, existing_xg_b, fm_xg_a, fm_xg_b.
    """
    snaps = snaps if snaps is not None else _cached_snapshots()
    params = params if params is not None else _cached_params()
    fm_data = fm_data if fm_data is not None else _cached_fm()
    if odds is None:
        odds = get_market_odds_for_match(team_a, team_b)

    # 1. Existing ELO + MLE + calibrated-xG signal.
    snap_a = snaps.get(team_a, _DEFAULT_SNAP)
    snap_b = snaps.get(team_b, _DEFAULT_SNAP)
    par_a = params.get(team_a, _DEFAULT_PAR)
    par_b = params.get(team_b, _DEFAULT_PAR)
    raw_a, raw_b = calculate_strength_adjusted_xg(
        elo_a=snap_a.elo, elo_b=snap_b.elo,
        params_a=par_a, params_b=par_b,
        ppg_a=snap_a.ppg, ppg_b=snap_b.ppg,
    )
    existing_xg_a, existing_xg_b = calibrate_xg(raw_a), calibrate_xg(raw_b)

    # 2. FM squad/team strength signal.
    fm_a = get_fm_strength(team_a, fm_data)
    fm_b = get_fm_strength(team_b, fm_data)
    fm_used = fm_a is not None and fm_b is not None

    fm_edges_a = fm_edges_b = None
    if fm_used:
        fm_edges_a = _fm_adjustment(fm_a, fm_b)
        fm_edges_b = _fm_adjustment(fm_b, fm_a)
        fm_xg_a = FM_BASE_XG + fm_edges_a.adjustment
        fm_xg_b = FM_BASE_XG + fm_edges_b.adjustment
    else:
        fm_xg_a, fm_xg_b = existing_xg_a, existing_xg_b

    # 3. Market odds signal.
    odds_used = bool(odds.research_valid and odds.win_a is not None)

    # 4. Blend FM + existing (pre-market).
    if fm_used:
        if odds_used:
            w_fm, w_exist = WEIGHT_WITH_ODDS["fm"], WEIGHT_WITH_ODDS["existing"]
        else:
            w_fm, w_exist = WEIGHT_NO_ODDS["fm"], WEIGHT_NO_ODDS["existing"]
        total = w_fm + w_exist
        pre_xg_a = (w_fm * fm_xg_a + w_exist * existing_xg_a) / total
        pre_xg_b = (w_fm * fm_xg_b + w_exist * existing_xg_b) / total
    else:
        pre_xg_a, pre_xg_b = existing_xg_a, existing_xg_b

    pre_xg_a = float(np.clip(pre_xg_a, MIN_XG, MAX_XG_NORMAL))
    pre_xg_b = float(np.clip(pre_xg_b, MIN_XG, MAX_XG_NORMAL))

    # 5. Calibrate against market W/D/L, if available.
    if odds_used:
        model_result = predict_dixon_coles(team_a, team_b, pre_xg_a, pre_xg_b, rho=rho)
        w_mkt = WEIGHT_WITH_ODDS["market"]
        w_model = WEIGHT_WITH_ODDS["fm"] + WEIGHT_WITH_ODDS["existing"]
        target_win_a = w_mkt * odds.win_a + w_model * model_result.win_a
        target_draw = w_mkt * odds.draw + w_model * model_result.draw
        target_win_b = w_mkt * odds.win_b + w_model * model_result.win_b
        final_xg_a, final_xg_b = _calibrate_xg_to_target(
            pre_xg_a, pre_xg_b, target_win_a, target_draw, target_win_b, rho=rho,
        )
    else:
        final_xg_a, final_xg_b = pre_xg_a, pre_xg_b

    final_xg_a = float(np.clip(final_xg_a, MIN_XG, MAX_XG_NORMAL))
    final_xg_b = float(np.clip(final_xg_b, MIN_XG, MAX_XG_NORMAL))

    matrix = build_dc_matrix(final_xg_a, final_xg_b, rho=rho)
    prediction = predict_dixon_coles(team_a, team_b, final_xg_a, final_xg_b, rho=rho)

    return {
        "xg_a": final_xg_a,
        "xg_b": final_xg_b,
        "win_a": prediction.win_a,
        "draw": prediction.draw,
        "win_b": prediction.win_b,
        "matrix": matrix,
        "fm_used": fm_used,
        "fm_a": fm_a,
        "fm_b": fm_b,
        "fm_edges_a": fm_edges_a,
        "fm_edges_b": fm_edges_b,
        "odds_used": odds_used,
        "odds": odds,
        "existing_xg_a": existing_xg_a,
        "existing_xg_b": existing_xg_b,
        "fm_xg_a": fm_xg_a,
        "fm_xg_b": fm_xg_b,
    }


def _build_explanation(team_a: str, team_b: str, data: dict, reason: str) -> str:
    parts = []
    if data["fm_used"]:
        fa, fb = data["fm_a"], data["fm_b"]
        gap = fa.overall - fb.overall
        if abs(gap) >= 1.0:
            leader = team_a if gap > 0 else team_b
            parts.append(
                f"{leader} has the FM squad-strength edge "
                f"({fa.overall:.1f} vs {fb.overall:.1f} overall)."
            )
        ea = data["fm_edges_a"]
        if ea.attack_edge >= 3:
            parts.append(f"{team_a}'s attack rates well above {team_b}'s defense.")
        elif ea.attack_edge <= -3:
            parts.append(f"{team_b}'s defense rates well above {team_a}'s attack.")
    else:
        parts.append("FM squad data unavailable for one or both teams — using the existing model only.")

    if data["odds_used"]:
        parts.append("Bookmaker odds were blended into the final probabilities and used to calibrate the score matrix.")
    else:
        parts.append("No research-valid bookmaker odds available — prediction is model-based.")

    parts.append(reason)
    return " ".join(parts)


def predict_match(
    team_a: str,
    team_b: str,
    snaps: dict[str, TeamSnapshot] | None = None,
    params: dict[str, StrengthParams] | None = None,
    fm_data: dict[str, FMTeamStrength] | None = None,
    odds: MarketOddsResult | None = None,
    rho: float = DEFAULT_RHO,
) -> MatchSimulationResult:
    """Full coherent prediction: xG, W/D/L, exact-score table, recommendation."""
    data = compute_match_xg(team_a, team_b, snaps, params, fm_data, odds, rho)

    scorelines = _full_scorelines(data["matrix"], max_score=5)
    top5 = scorelines[:5]
    raw_top, recommended, reason = _select_recommended_score(
        team_a, team_b, scorelines, data["win_a"], data["draw"], data["win_b"],
    )

    explanation = _build_explanation(team_a, team_b, data, reason)

    fm_edges_dict = None
    if data["fm_used"]:
        ea, eb = data["fm_edges_a"], data["fm_edges_b"]
        fm_edges_dict = {
            "team1_overall": data["fm_a"].overall,
            "team2_overall": data["fm_b"].overall,
            "attack_edge_team1": ea.attack_edge,
            "attack_edge_team2": eb.attack_edge,
            "midfield_edge_team1": ea.midfield_edge,
            "midfield_edge_team2": eb.midfield_edge,
            "depth_edge_team1": ea.depth_edge,
            "depth_edge_team2": eb.depth_edge,
            "team1_xg_adjustment": ea.adjustment,
            "team2_xg_adjustment": eb.adjustment,
        }

    market_odds_dict = None
    if data["odds_used"]:
        market_odds_dict = {
            "win_a": data["odds"].win_a,
            "draw": data["odds"].draw,
            "win_b": data["odds"].win_b,
            "bookmaker": data["odds"].bookmaker,
        }

    return MatchSimulationResult(
        team1=team_a,
        team2=team_b,
        team1_win_probability=data["win_a"],
        draw_probability=data["draw"],
        team2_win_probability=data["win_b"],
        expected_goals_team1=data["xg_a"],
        expected_goals_team2=data["xg_b"],
        raw_top_score=raw_top,
        recommended_exact_score=recommended,
        top_5_exact_scores=[{"score": f"{a}-{b}", "probability": p} for a, b, p in top5],
        confidence=_confidence_label(data["win_a"], data["draw"], data["win_b"]),
        explanation=explanation,
        fm_used=data["fm_used"],
        fm_edges=fm_edges_dict,
        odds_used=data["odds_used"],
        market_odds=market_odds_dict,
        top_players_team1=data["fm_a"].top_players if data["fm_a"] else "",
        top_players_team2=data["fm_b"].top_players if data["fm_b"] else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_match(
    team_a: str,
    team_b: str,
    n: int = 10_000,
    snaps: dict[str, TeamSnapshot] | None = None,
    params: dict[str, StrengthParams] | None = None,
    fm_data: dict[str, FMTeamStrength] | None = None,
    odds: MarketOddsResult | None = None,
    rho: float = DEFAULT_RHO,
    rng_seed: int | None = None,
) -> MonteCarloMatchResult:
    """Run n Monte Carlo simulations on top of the final calibrated xG.

    Goals for each side are drawn independently from Poisson(final_xg);
    the resulting win/draw/loss and exact-score frequencies should track
    the analytic Dixon-Coles score matrix (the DC tau correction is a small
    perturbation of independent Poisson, so close-but-not-identical results
    are expected).
    """
    base = predict_match(team_a, team_b, snaps, params, fm_data, odds, rho)

    rng = np.random.default_rng(rng_seed)
    goals_a = rng.poisson(base.expected_goals_team1, size=n)
    goals_b = rng.poisson(base.expected_goals_team2, size=n)

    win_a = float(np.mean(goals_a > goals_b))
    draw = float(np.mean(goals_a == goals_b))
    win_b = float(np.mean(goals_a < goals_b))

    capped_a = np.minimum(goals_a, 5)
    capped_b = np.minimum(goals_b, 5)
    pairs, counts = np.unique(np.stack([capped_a, capped_b], axis=1), axis=0, return_counts=True)
    sim_scores = sorted(
        ((int(a), int(b), float(c) / n) for (a, b), c in zip(pairs, counts)),
        key=lambda x: (-x[2], x[0], x[1]),
    )[:5]

    return MonteCarloMatchResult(
        **base.__dict__,
        n_simulations=n,
        simulated_team1_win_probability=win_a,
        simulated_draw_probability=draw,
        simulated_team2_win_probability=win_b,
        simulated_top_5_exact_scores=[{"score": f"{a}-{b}", "probability": p} for a, b, p in sim_scores],
    )
