"""group_simulation.py — simulate all WC2026 group-stage fixtures using the
coherent match_simulator engine and produce predicted group tables.

For each fixture, the "predicted result" is the simulator's
`recommended_exact_score` (the same coherent W/D/L + exact-score output used
by the Match Analyzer / Run Simulation flow) — not an independent toy model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.tournament.fixtures import load_fixtures
from src.tournament.standings import TeamStanding, update_standing, rank_group
from src.tournament.bracket_2026 import qualify_2026, GROUPS_2026
from src.models.match_simulator import predict_match, MatchSimulationResult

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
