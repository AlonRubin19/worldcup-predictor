"""Tournament simulator: group stage + knockout + Monte Carlo.

Simulation pipeline:
  1. Load fixtures from CSV.
  2. For each group-stage match, sample a scoreline from the Poisson/DC
     score matrix. Update group standings.
  3. Qualify top-2 from each group.
  4. Simulate knockout bracket (no draws — penalties resolved via model probs).
  5. Repeat N times for Monte Carlo probabilities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.tournament.fixtures import load_fixtures, Fixture
from src.tournament.standings import TeamStanding, update_standing, qualify_from_group, rank_group
from src.tournament.bracket_2026 import qualify_2026, build_r32_bracket, GROUPS_2026
from src.tournament.calibration import CalibrationParams, apply_temperature, apply_xg_noise, apply_upset_factor
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.models.research_valid_predictor import ResearchValidInput
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.dixon_coles import predict_dixon_coles
from src.models.poisson import build_score_matrix

_DEFAULT_SNAP = TeamSnapshot(elo=1800.0, ppg=1.5)
_DEFAULT_PAR  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)
_RHO = -0.13
_MAX_GOALS = 10   # cap scoreline sampling range


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchOutcome:
    goals_a: int
    goals_b: int
    winner: str   # "team_a" | "draw" | "team_b"


@dataclass
class TournamentResult:
    champion: str
    advancement: dict[str, str]   # team → furthest stage reached


@dataclass
class MonteCarloResult:
    n_simulations: int
    win_tournament: dict[str, float]
    reach_final: dict[str, float]
    reach_sf: dict[str, float]
    reach_qf: dict[str, float]
    reach_r16: dict[str, float]
    reach_r32: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core match simulation
# ─────────────────────────────────────────────────────────────────────────────

def _get_xg(
    team_a: str,
    team_b: str,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
) -> tuple[float, float]:
    snap_a = snaps.get(team_a, _DEFAULT_SNAP)
    snap_b = snaps.get(team_b, _DEFAULT_SNAP)
    par_a  = params.get(team_a, _DEFAULT_PAR)
    par_b  = params.get(team_b, _DEFAULT_PAR)
    raw_a, raw_b = calculate_strength_adjusted_xg(
        snap_a.elo, snap_b.elo, par_a, par_b, snap_a.ppg, snap_b.ppg,
    )
    return calibrate_xg(raw_a), calibrate_xg(raw_b)


def _sample_scoreline(
    xg_a: float,
    xg_b: float,
    rng: np.random.Generator,
    calib: CalibrationParams | None = None,
) -> tuple[int, int]:
    """Sample (goals_a, goals_b) from the joint Poisson/DC score matrix."""
    if calib is not None and calib.xg_noise_sigma > 0.0:
        xg_a = apply_xg_noise(xg_a, calib.xg_noise_sigma, rng)
        xg_b = apply_xg_noise(xg_b, calib.xg_noise_sigma, rng)
        # Keep noisy xG in a sensible range so DC corrections stay stable
        xg_a = float(max(0.1, min(4.0, xg_a)))
        xg_b = float(max(0.1, min(4.0, xg_b)))

    matrix = build_score_matrix(xg_a, xg_b)

    # Apply DC tau corrections
    matrix[0, 0] *= 1 - (xg_a * xg_b * _RHO)
    matrix[0, 1] *= 1 + (xg_a * _RHO)
    matrix[1, 0] *= 1 + (xg_b * _RHO)
    matrix[1, 1] *= 1 - _RHO

    # Clip any floating-point negatives produced by DC corrections, then normalise
    np.clip(matrix, 0.0, None, out=matrix)
    matrix /= matrix.sum()

    if calib is not None and calib.temperature != 1.0:
        # Partition cells into win_a (i>j), draw (i==j), win_b (i<j) regions,
        # apply temperature to those three probabilities, then redistribute
        # each region's mass proportionally.
        n_mat = matrix.shape[0]
        mask_a    = np.array([[i > j for j in range(n_mat)] for i in range(n_mat)])
        mask_draw = np.eye(n_mat, dtype=bool)
        mask_b    = np.array([[i < j for j in range(n_mat)] for i in range(n_mat)])

        w_a   = float(matrix[mask_a].sum())
        draw  = float(matrix[mask_draw].sum())
        w_b   = float(matrix[mask_b].sum())

        w_a_t, draw_t, w_b_t = apply_temperature((w_a, draw, w_b), calib.temperature)

        if w_a   > 1e-12: matrix[mask_a]    *= w_a_t   / w_a
        if draw  > 1e-12: matrix[mask_draw] *= draw_t  / draw
        if w_b   > 1e-12: matrix[mask_b]    *= w_b_t   / w_b

        # Clip tiny floating-point negatives and renormalise
        np.clip(matrix, 0.0, None, out=matrix)
        matrix /= matrix.sum()

    # Flatten and sample
    n = matrix.shape[0]
    flat = matrix.flatten()
    idx = rng.choice(len(flat), p=flat)
    goals_a, goals_b = divmod(idx, n)
    return int(goals_a), int(goals_b)


def _get_calibrated_xg(
    inp: ResearchValidInput,
    rng: np.random.Generator,
    calib: CalibrationParams | None,
) -> tuple[float, float]:
    raw_a, raw_b = calculate_strength_adjusted_xg(
        inp.snapshot_a.elo, inp.snapshot_b.elo,
        inp.params_a, inp.params_b,
        inp.snapshot_a.ppg, inp.snapshot_b.ppg,
    )
    return calibrate_xg(raw_a), calibrate_xg(raw_b)


def simulate_match(
    inp: ResearchValidInput,
    rng_seed: int | None = None,
    calib: CalibrationParams | None = None,
) -> MatchOutcome:
    """Simulate one match by sampling from the score probability matrix.

    Returns goals and winner (draw is allowed).
    """
    rng = np.random.default_rng(rng_seed)
    xg_a, xg_b = _get_calibrated_xg(inp, rng, calib)
    goals_a, goals_b = _sample_scoreline(xg_a, xg_b, rng, calib)

    if goals_a > goals_b:
        winner = "team_a"
    elif goals_b > goals_a:
        winner = "team_b"
    else:
        winner = "draw"

    return MatchOutcome(goals_a=goals_a, goals_b=goals_b, winner=winner)


def simulate_knockout_match(
    inp: ResearchValidInput,
    rng_seed: int | None = None,
    calib: CalibrationParams | None = None,
) -> MatchOutcome:
    """Simulate a knockout match — no draws.

    If 90-minute result is a draw, resolve via penalty shootout simulation:
    probability of team_a winning penalties = win_a / (win_a + win_b),
    optionally mixed toward 0.5 by upset_factor.
    """
    outcome = simulate_match(inp, rng_seed=rng_seed, calib=calib)

    if outcome.winner != "draw":
        return outcome

    # Penalty resolution: use model win probabilities (excluding draw)
    xg_a, xg_b = _get_calibrated_xg(inp, np.random.default_rng(rng_seed), calib)
    pred = predict_dixon_coles(inp.team_a, inp.team_b, xg_a, xg_b, rho=_RHO)
    pen_prob_a = pred.win_a / (pred.win_a + pred.win_b)

    if calib is not None and calib.upset_factor > 0.0:
        pen_prob_a = apply_upset_factor(pen_prob_a, calib.upset_factor)

    pen_rng = np.random.default_rng(
        None if rng_seed is None else rng_seed + 999_999
    )
    winner = "team_a" if pen_rng.random() < pen_prob_a else "team_b"

    return MatchOutcome(goals_a=outcome.goals_a, goals_b=outcome.goals_b, winner=winner)


# ─────────────────────────────────────────────────────────────────────────────
# Full tournament simulation
# ─────────────────────────────────────────────────────────────────────────────

# WC 2022 R16 bracket: (A1 vs B2, B1 vs A2, C1 vs D2, D1 vs C2,
#                        E1 vs F2, F1 vs E2, G1 vs H2, H1 vs G2)
_R16_BRACKET = [
    ("A", 0, "B", 1),  # A1 vs B2
    ("B", 0, "A", 1),  # B1 vs A2
    ("C", 0, "D", 1),  # C1 vs D2
    ("D", 0, "C", 1),  # D1 vs C2
    ("E", 0, "F", 1),  # E1 vs F2
    ("F", 0, "E", 1),  # F1 vs E2
    ("G", 0, "H", 1),  # G1 vs H2
    ("H", 0, "G", 1),  # H1 vs G2
]


def _make_inp(
    team_a: str,
    team_b: str,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
) -> ResearchValidInput:
    return ResearchValidInput(
        team_a=team_a, team_b=team_b,
        snapshot_a=snaps.get(team_a, _DEFAULT_SNAP),
        snapshot_b=snaps.get(team_b, _DEFAULT_SNAP),
        params_a=params.get(team_a, _DEFAULT_PAR),
        params_b=params.get(team_b, _DEFAULT_PAR),
        rho=_RHO,
    )


def run_tournament(
    fixture_path: Path,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
    rng_seed: int | None = None,
    calibration: CalibrationParams | None = None,
) -> TournamentResult:
    """Run one full tournament simulation. Returns champion and advancement map."""
    rng = np.random.default_rng(rng_seed)

    def _seed() -> int:
        return int(rng.integers(0, 2**31))

    fixtures = load_fixtures(fixture_path)
    advancement: dict[str, str] = {}

    # ── Group stage ───────────────────────────────────────────────────────────
    group_standings: dict[str, dict[str, TeamStanding]] = {}
    group_fixtures = [f for f in fixtures if f.stage == "group"]

    for f in group_fixtures:
        g = f.group
        if g not in group_standings:
            group_standings[g] = {}
        for team in (f.team_a, f.team_b):
            if team not in group_standings[g]:
                group_standings[g][team] = TeamStanding(
                    team=team, points=0, goals_for=0, goals_against=0,
                    goal_diff=0, played=0,
                )

    for f in group_fixtures:
        inp = _make_inp(f.team_a, f.team_b, snaps, params)
        outcome = simulate_match(inp, rng_seed=_seed(), calib=calibration)
        group_standings[f.group][f.team_a] = update_standing(
            group_standings[f.group][f.team_a], outcome.goals_a, outcome.goals_b
        )
        group_standings[f.group][f.team_b] = update_standing(
            group_standings[f.group][f.team_b], outcome.goals_b, outcome.goals_a
        )

    # All group teams mark advancement as "group"
    for g_teams in group_standings.values():
        for team in g_teams:
            advancement[team] = "group"

    # Qualify top-2 from each group
    qualifiers: dict[str, list[str]] = {}  # group → [winner, runner_up]
    for g, standings in group_standings.items():
        w, ru = qualify_from_group(standings)
        qualifiers[g] = [w, ru]
        advancement[w]  = "round_of_16"
        advancement[ru] = "round_of_16"

    # ── Knockout stages ───────────────────────────────────────────────────────
    stage_order = ["round_of_16", "quarter_final", "semi_final", "final"]

    # Build R16 bracket
    r16_matchups = [
        (qualifiers[g_a][idx_a], qualifiers[g_b][idx_b])
        for g_a, idx_a, g_b, idx_b in _R16_BRACKET
    ]
    current_round = r16_matchups

    for stage in stage_order:
        next_round = []
        for team_a, team_b in current_round:
            inp = _make_inp(team_a, team_b, snaps, params)
            outcome = simulate_knockout_match(inp, rng_seed=_seed(), calib=calibration)
            winner = team_a if outcome.winner == "team_a" else team_b
            loser  = team_b if winner == team_a else team_a

            if stage == "final":
                advancement[winner] = "final"
                advancement[loser]  = "final"
            else:
                next_stage = stage_order[stage_order.index(stage) + 1]
                advancement[winner] = next_stage

            next_round.append(winner)
        current_round = _pair_winners(next_round)

    champion = current_round[0] if current_round else next_round[0]
    advancement[champion] = "final"

    return TournamentResult(champion=champion, advancement=advancement)


def _pair_winners(winners: list[str]) -> list[tuple[str, str]]:
    """Pair consecutive winners into next-round matchups."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners) - 1, 2)]


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_RANK = {
    "group": 0,
    "round_of_16": 1,
    "quarter_final": 2,
    "semi_final": 3,
    "final": 4,
}


def run_monte_carlo(
    fixture_path: Path,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
    n: int = 10_000,
    rng_seed: int | None = None,
    calibration: CalibrationParams | None = None,
) -> MonteCarloResult:
    """Run N tournament simulations. Return probability distributions."""
    master_rng = np.random.default_rng(rng_seed)

    win_counts: dict[str, int] = defaultdict(int)
    stage_counts: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in _STAGE_RANK
    }

    for sim_i in range(n):
        seed_i = int(master_rng.integers(0, 2**31))
        result = run_tournament(fixture_path, snaps, params, rng_seed=seed_i, calibration=calibration)

        win_counts[result.champion] += 1

        for team, stage in result.advancement.items():
            rank = _STAGE_RANK.get(stage, 0)
            for s, s_rank in _STAGE_RANK.items():
                if rank >= s_rank:
                    stage_counts[s][team] += 1

    def _probs(counts: dict[str, int]) -> dict[str, float]:
        return {team: count / n for team, count in counts.items()}

    # Collect all teams
    all_teams = set()
    for team in win_counts:
        all_teams.add(team)
    for s_counts in stage_counts.values():
        all_teams.update(s_counts.keys())

    return MonteCarloResult(
        n_simulations=n,
        win_tournament=_probs(win_counts),
        reach_final=_probs(stage_counts["final"]),
        reach_sf=_probs(stage_counts["semi_final"]),
        reach_qf=_probs(stage_counts["quarter_final"]),
        reach_r16=_probs(stage_counts["round_of_16"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WC2026: 48 teams, 12 groups, top-2 + best-8-thirds -> Round of 32
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_ORDER_2026 = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]

_STAGE_RANK_2026 = {
    "group": 0,
    "round_of_32": 1,
    "round_of_16": 2,
    "quarter_final": 3,
    "semi_final": 4,
    "final": 5,
}


def run_tournament_2026(
    fixture_path: Path,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
    rng_seed: int | None = None,
    calibration: CalibrationParams | None = None,
) -> TournamentResult:
    """Run one full WC2026-format tournament simulation (48 teams, 12 groups,
    Round of 32 onward). Returns champion and advancement map."""
    rng = np.random.default_rng(rng_seed)

    def _seed() -> int:
        return int(rng.integers(0, 2**31))

    fixtures = load_fixtures(fixture_path)
    advancement: dict[str, str] = {}

    # ── Group stage (12 groups of 4) ───────────────────────────────────────
    group_standings: dict[str, dict[str, TeamStanding]] = {}
    group_fixtures = [f for f in fixtures if f.stage == "group"]

    for f in group_fixtures:
        g = f.group
        if g not in group_standings:
            group_standings[g] = {}
        for team in (f.team_a, f.team_b):
            if team not in group_standings[g]:
                group_standings[g][team] = TeamStanding(
                    team=team, points=0, goals_for=0, goals_against=0,
                    goal_diff=0, played=0,
                )

    for f in group_fixtures:
        inp = _make_inp(f.team_a, f.team_b, snaps, params)
        outcome = simulate_match(inp, rng_seed=_seed(), calib=calibration)
        group_standings[f.group][f.team_a] = update_standing(
            group_standings[f.group][f.team_a], outcome.goals_a, outcome.goals_b
        )
        group_standings[f.group][f.team_b] = update_standing(
            group_standings[f.group][f.team_b], outcome.goals_b, outcome.goals_a
        )

    for g_teams in group_standings.values():
        for team in g_teams:
            advancement[team] = "group"

    # ── Qualification: top-2 + best-8-thirds = 32 teams ────────────────────
    qualified = qualify_2026(group_standings, snaps)
    for q in qualified:
        advancement[q.team] = "round_of_32"

    # ── Round of 32 bracket ─────────────────────────────────────────────────
    r32_matchups = build_r32_bracket(qualified)
    current_round = r32_matchups

    for stage in _STAGE_ORDER_2026:
        next_round = []
        for team_a, team_b in current_round:
            inp = _make_inp(team_a, team_b, snaps, params)
            outcome = simulate_knockout_match(inp, rng_seed=_seed(), calib=calibration)
            winner = team_a if outcome.winner == "team_a" else team_b
            loser  = team_b if winner == team_a else team_a

            if stage == "final":
                advancement[winner] = "final"
                advancement[loser]  = "final"
            else:
                next_stage = _STAGE_ORDER_2026[_STAGE_ORDER_2026.index(stage) + 1]
                advancement[winner] = next_stage

            next_round.append(winner)
        current_round = _pair_winners(next_round)

    champion = current_round[0] if current_round else next_round[0]
    advancement[champion] = "final"

    return TournamentResult(champion=champion, advancement=advancement)


def run_monte_carlo_2026(
    fixture_path: Path,
    snaps: dict[str, TeamSnapshot],
    params: dict[str, StrengthParams],
    n: int = 10_000,
    rng_seed: int | None = None,
    calibration: CalibrationParams | None = None,
) -> MonteCarloResult:
    """Run N WC2026-format tournament simulations. Return probability distributions."""
    master_rng = np.random.default_rng(rng_seed)

    win_counts: dict[str, int] = defaultdict(int)
    stage_counts: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in _STAGE_RANK_2026
    }

    for sim_i in range(n):
        seed_i = int(master_rng.integers(0, 2**31))
        result = run_tournament_2026(fixture_path, snaps, params, rng_seed=seed_i, calibration=calibration)

        win_counts[result.champion] += 1

        for team, stage in result.advancement.items():
            rank = _STAGE_RANK_2026.get(stage, 0)
            for s, s_rank in _STAGE_RANK_2026.items():
                if rank >= s_rank:
                    stage_counts[s][team] += 1

    def _probs(counts: dict[str, int]) -> dict[str, float]:
        return {team: count / n for team, count in counts.items()}

    return MonteCarloResult(
        n_simulations=n,
        win_tournament=_probs(win_counts),
        reach_final=_probs(stage_counts["final"]),
        reach_sf=_probs(stage_counts["semi_final"]),
        reach_qf=_probs(stage_counts["quarter_final"]),
        reach_r16=_probs(stage_counts["round_of_16"]),
        reach_r32=_probs(stage_counts["round_of_32"]),
    )
