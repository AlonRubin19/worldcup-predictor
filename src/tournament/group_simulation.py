"""group_simulation.py — simulate all WC2026 group-stage fixtures using the
coherent match_simulator engine and produce predicted group tables.

For each fixture, the "predicted result" is the simulator's
`recommended_exact_score` (the same coherent W/D/L + exact-score output used
by the Match Analyzer / Run Simulation flow) — not an independent toy model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.tournament.fixtures import load_fixtures
from src.tournament.standings import TeamStanding, update_standing, rank_group
from src.tournament.bracket_2026 import qualify_2026, GROUPS_2026
from src.models.match_simulator import predict_match, compute_match_xg, MatchSimulationResult

_FIXTURE_PATH = Path(__file__).parent.parent.parent / "data" / "world_cup_2026_fixtures.csv"


@dataclass
class FixtureSimResult:
    match_id: int
    group: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    prediction: MatchSimulationResult


@dataclass
class GroupSimulationResult:
    fixture_results: list[FixtureSimResult]
    group_tables: dict[str, list[TeamStanding]] = field(default_factory=dict)
    qualified: list[str] = field(default_factory=list)


def simulate_group_stage(fixture_path: Path | None = None) -> GroupSimulationResult:
    """Simulate every group-stage fixture and build predicted group tables.

    Each fixture's predicted scoreline is its `recommended_exact_score` from
    `predict_match` — the same coherent xG/score-matrix output shown in the
    Match Analyzer. Standings are then ranked with the same tiebreakers used
    by the tournament simulator (points -> goal_diff -> goals_for -> name),
    and qualification (top-2 + best-8-thirds) reuses `qualify_2026`.
    """
    fixture_path = fixture_path or _FIXTURE_PATH
    fixtures = [f for f in load_fixtures(fixture_path) if f.stage == "group"]

    standings: dict[str, dict[str, TeamStanding]] = {g: {} for g in GROUPS_2026}
    fixture_results: list[FixtureSimResult] = []

    for f in fixtures:
        for team in (f.team_a, f.team_b):
            standings[f.group].setdefault(
                team, TeamStanding(team=team, points=0, goals_for=0, goals_against=0, goal_diff=0, played=0)
            )

        pred = predict_match(f.team_a, f.team_b)
        ga_str, gb_str = pred.recommended_exact_score.split("-")
        goals_a, goals_b = int(ga_str), int(gb_str)

        standings[f.group][f.team_a] = update_standing(standings[f.group][f.team_a], goals_a, goals_b)
        standings[f.group][f.team_b] = update_standing(standings[f.group][f.team_b], goals_b, goals_a)

        fixture_results.append(FixtureSimResult(
            match_id=f.match_id, group=f.group, team_a=f.team_a, team_b=f.team_b,
            goals_a=goals_a, goals_b=goals_b, prediction=pred,
        ))

    group_tables = {g: rank_group(s) for g, s in standings.items() if s}

    qualified_teams: list[str] = []
    if all(len(group_tables.get(g, [])) == 4 for g in GROUPS_2026):
        qualified_teams = [q.team for q in qualify_2026(standings, snaps={})]

    return GroupSimulationResult(
        fixture_results=fixture_results,
        group_tables=group_tables,
        qualified=qualified_teams,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo group-stage simulation (qualification probabilities)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamGroupOutlook:
    team: str
    group: str
    avg_points: float
    avg_goals_for: float
    avg_goals_against: float
    avg_goal_diff: float
    group_winner_probability: float
    second_place_probability: float
    qualification_probability: float


@dataclass
class MonteCarloGroupResult:
    n_runs: int
    outlooks: dict[str, TeamGroupOutlook] = field(default_factory=dict)


def simulate_group_stage_mc(
    n_runs: int = 1000,
    fixture_path: Path | None = None,
    rng_seed: int | None = None,
    scenario: str = "balanced",
) -> MonteCarloGroupResult:
    """Monte Carlo group-stage simulation.

    Each run samples every fixture's scoreline from its Dixon-Coles score
    matrix (the same matrix behind the analytic prediction), then ranks each
    group and applies WC2026 qualification (top-2 + best-8 thirds). Returns
    per-team averages and probabilities across runs.
    """
    from src.models.prediction_config import get_scenario_config

    fixture_path = fixture_path or _FIXTURE_PATH
    fixtures = [f for f in load_fixtures(fixture_path) if f.stage == "group"]
    config = get_scenario_config(scenario)
    rng = np.random.default_rng(rng_seed)

    # Pre-sample n_runs scorelines per fixture from each match's score matrix.
    sampled: list[tuple[object, np.ndarray, np.ndarray]] = []
    for f in fixtures:
        data = compute_match_xg(f.team_a, f.team_b, config=config)
        matrix = data["matrix"]
        flat = matrix.flatten()
        flat = flat / flat.sum()
        idx = rng.choice(len(flat), size=n_runs, p=flat)
        goals_a, goals_b = np.divmod(idx, matrix.shape[1])
        sampled.append((f, goals_a, goals_b))

    teams_by_group: dict[str, set[str]] = {g: set() for g in GROUPS_2026}
    for f in fixtures:
        teams_by_group[f.group].update((f.team_a, f.team_b))

    acc: dict[str, dict[str, float]] = {
        t: {"pts": 0.0, "gf": 0.0, "ga": 0.0, "win_group": 0.0, "second": 0.0, "qualify": 0.0}
        for g in teams_by_group for t in teams_by_group[g]
    }
    team_group = {t: g for g, ts in teams_by_group.items() for t in ts}

    for run in range(n_runs):
        standings: dict[str, dict[str, TeamStanding]] = {
            g: {t: TeamStanding(team=t, points=0, goals_for=0, goals_against=0,
                                goal_diff=0, played=0) for t in ts}
            for g, ts in teams_by_group.items()
        }
        for f, goals_a, goals_b in sampled:
            ga, gb = int(goals_a[run]), int(goals_b[run])
            standings[f.group][f.team_a] = update_standing(standings[f.group][f.team_a], ga, gb)
            standings[f.group][f.team_b] = update_standing(standings[f.group][f.team_b], gb, ga)

        for g, s in standings.items():
            ranked = rank_group(s)
            acc[ranked[0].team]["win_group"] += 1
            acc[ranked[1].team]["second"] += 1
            for st in ranked:
                acc[st.team]["pts"] += st.points
                acc[st.team]["gf"] += st.goals_for
                acc[st.team]["ga"] += st.goals_against

        for q in qualify_2026(standings, snaps={}):
            acc[q.team]["qualify"] += 1

    outlooks = {
        t: TeamGroupOutlook(
            team=t,
            group=team_group[t],
            avg_points=v["pts"] / n_runs,
            avg_goals_for=v["gf"] / n_runs,
            avg_goals_against=v["ga"] / n_runs,
            avg_goal_diff=(v["gf"] - v["ga"]) / n_runs,
            group_winner_probability=v["win_group"] / n_runs,
            second_place_probability=v["second"] / n_runs,
            qualification_probability=v["qualify"] / n_runs,
        )
        for t, v in acc.items()
    }
    return MonteCarloGroupResult(n_runs=n_runs, outlooks=outlooks)
