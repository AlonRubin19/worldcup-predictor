"""Group standings logic: points, goal difference, goals scored."""

from dataclasses import dataclass, replace


@dataclass
class GroupResult:
    goals_a: int
    goals_b: int


@dataclass
class TeamStanding:
    team: str
    points: int
    goals_for: int
    goals_against: int
    goal_diff: int
    played: int


def update_standing(standing: TeamStanding, goals_for: int, goals_against: int) -> TeamStanding:
    """Return updated standing after one match result.

    Awards 3 pts for win, 1 for draw, 0 for loss.
    """
    if goals_for > goals_against:
        pts = 3
    elif goals_for == goals_against:
        pts = 1
    else:
        pts = 0

    new_gf = standing.goals_for + goals_for
    new_ga = standing.goals_against + goals_against
    return replace(
        standing,
        points=standing.points + pts,
        goals_for=new_gf,
        goals_against=new_ga,
        goal_diff=new_gf - new_ga,
        played=standing.played + 1,
    )


def rank_group(standings: dict[str, TeamStanding]) -> list[TeamStanding]:
    """Sort teams: points → goal_diff → goals_for → alphabetical (stable tiebreaker)."""
    return sorted(
        standings.values(),
        key=lambda s: (-s.points, -s.goal_diff, -s.goals_for, s.team),
    )


def qualify_from_group(standings: dict[str, TeamStanding]) -> tuple[str, str]:
    """Return (group_winner, runner_up) from a group's final standings."""
    ranked = rank_group(standings)
    return ranked[0].team, ranked[1].team
