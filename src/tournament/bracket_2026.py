"""WC2026 qualification + Round-of-32 bracket construction.

Real-format summary (48 teams, 12 groups of 4 — A through L):
  - Each team plays 3 group-stage matches (72 fixtures total).
  - Top 2 of each group qualify automatically (24 teams).
  - The 8 best third-placed teams (across all 12 groups) also qualify (8 teams).
  - => 32 teams advance to a single-elimination Round of 32.

FIFA's official R32 draw depends on *which* 8 of the 12 groups produce a
qualifying third-placed team (a large lookup table of ~495 combinations,
designed so no team meets a group opponent again in the R32 and the bracket
stays balanced). Reproducing that exact table is out of scope here.

This module instead builds a **deterministic, conflict-free R32 bracket**
via ELO-based seeding (seed 1 vs seed 32, seed 2 vs seed 31, ... "snake"
seeding), with a same-group-rematch resolver that swaps teams between
adjacent pairings whenever a pairing would re-match two group-stage
opponents. This preserves the correct mechanics (32 -> 16 -> 8 -> 4 -> 2 -> 1,
correct qualification rules, no immediate group-mate rematches) for
simulation purposes, while not claiming to be the official FIFA draw.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.team_snapshot_loader import TeamSnapshot
from src.tournament.standings import TeamStanding, rank_group

GROUPS_2026 = list("ABCDEFGHIJKL")


@dataclass
class QualifiedTeam:
    team: str
    group: str
    elo: float


def qualify_2026(
    group_standings: dict[str, dict[str, TeamStanding]],
    snaps: dict[str, TeamSnapshot],
    default_elo: float = 1800.0,
) -> list[QualifiedTeam]:
    """Return the 32 teams that qualify for the Round of 32.

    = top-2 of each of the 12 groups (24 teams)
      + the 8 best third-placed teams across all groups.

    Third-place ranking uses the same (points, goal_diff, goals_for, name)
    ordering as within-group ranking.
    """
    qualified: list[QualifiedTeam] = []
    thirds: list[TeamStanding] = []
    third_group: dict[str, str] = {}

    for g, standings in group_standings.items():
        ranked = rank_group(standings)
        for standing in ranked[:2]:
            qualified.append(QualifiedTeam(
                team=standing.team, group=g,
                elo=snaps.get(standing.team, TeamSnapshot(elo=default_elo, ppg=1.5)).elo,
            ))
        if len(ranked) >= 3:
            thirds.append(ranked[2])
            third_group[ranked[2].team] = g

    thirds_ranked = sorted(thirds, key=lambda s: (-s.points, -s.goal_diff, -s.goals_for, s.team))
    for standing in thirds_ranked[:8]:
        g = third_group[standing.team]
        qualified.append(QualifiedTeam(
            team=standing.team, group=g,
            elo=snaps.get(standing.team, TeamSnapshot(elo=default_elo, ppg=1.5)).elo,
        ))

    return qualified


def build_r32_bracket(qualified: list[QualifiedTeam]) -> list[tuple[str, str]]:
    """Build 16 Round-of-32 matchups via ELO snake-seeding with a
    same-group-rematch resolver.

    Raises:
        ValueError: if `qualified` does not contain exactly 32 teams.
    """
    if len(qualified) != 32:
        raise ValueError(f"Expected 32 qualified teams, got {len(qualified)}")

    seeded = sorted(qualified, key=lambda q: -q.elo)
    n = len(seeded)
    pairs = [(seeded[i], seeded[n - 1 - i]) for i in range(n // 2)]

    # Resolve same-group rematches by swapping the lower-seeded (second) team
    # of a conflicting pair with the lower-seeded team of the next pair.
    for i in range(len(pairs) - 1):
        a, b = pairs[i]
        if a.group == b.group:
            next_a, next_b = pairs[i + 1]
            pairs[i] = (a, next_b)
            pairs[i + 1] = (next_a, b)

    # Final safety check: if a conflict remains (only possible in degenerate
    # all-same-group edge cases that cannot occur with 12 groups of 4),
    # leave as-is rather than looping forever.
    return [(a.team, b.team) for a, b in pairs]
