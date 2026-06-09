"""Tests for group standings logic."""

import pytest
from src.tournament.standings import (
    GroupResult,
    TeamStanding,
    update_standing,
    rank_group,
    qualify_from_group,
)


def _standing(team, pts=0, gf=0, ga=0, gd=None) -> TeamStanding:
    if gd is None:
        gd = gf - ga
    return TeamStanding(team=team, points=pts, goals_for=gf, goals_against=ga, goal_diff=gd, played=0)


def _result(winner=None, goals_a=0, goals_b=0) -> GroupResult:
    return GroupResult(goals_a=goals_a, goals_b=goals_b)


# ── update_standing ───────────────────────────────────────────────────────────

def test_win_adds_three_points():
    s = _standing("England")
    s = update_standing(s, goals_for=2, goals_against=0)
    assert s.points == 3


def test_draw_adds_one_point():
    s = _standing("England")
    s = update_standing(s, goals_for=1, goals_against=1)
    assert s.points == 1


def test_loss_adds_zero_points():
    s = _standing("England")
    s = update_standing(s, goals_for=0, goals_against=2)
    assert s.points == 0


def test_update_increments_goals_for_and_against():
    s = _standing("England", pts=3, gf=2, ga=0)
    s = update_standing(s, goals_for=1, goals_against=1)
    assert s.goals_for == 3
    assert s.goals_against == 1


def test_update_increments_played():
    s = _standing("England")
    s = update_standing(s, goals_for=2, goals_against=1)
    assert s.played == 1


def test_update_maintains_goal_diff():
    s = _standing("England")
    s = update_standing(s, goals_for=3, goals_against=1)
    assert s.goal_diff == 2


# ── rank_group ────────────────────────────────────────────────────────────────

def test_rank_group_by_points():
    standings = {
        "A": _standing("A", pts=9),
        "B": _standing("B", pts=6),
        "C": _standing("C", pts=3),
        "D": _standing("D", pts=0),
    }
    ranked = rank_group(standings)
    assert [s.team for s in ranked] == ["A", "B", "C", "D"]


def test_rank_group_by_goal_diff_on_equal_points():
    standings = {
        "A": _standing("A", pts=4, gf=4, ga=1),
        "B": _standing("B", pts=4, gf=3, ga=2),
        "C": _standing("C", pts=4, gf=2, ga=3),
        "D": _standing("D", pts=0),
    }
    ranked = rank_group(standings)
    # A has gd=+3, B has gd=+1, C has gd=-1
    assert ranked[0].team == "A"
    assert ranked[1].team == "B"
    assert ranked[2].team == "C"


def test_rank_group_by_goals_scored_on_equal_gd():
    standings = {
        "A": _standing("A", pts=4, gf=4, ga=2),  # gd=+2, gf=4
        "B": _standing("B", pts=4, gf=3, ga=1),  # gd=+2, gf=3
    }
    ranked = rank_group(standings)
    assert ranked[0].team == "A"  # higher goals for


# ── qualify_from_group ────────────────────────────────────────────────────────

def test_qualify_returns_top_two():
    standings = {
        "A": _standing("A", pts=9),
        "B": _standing("B", pts=6),
        "C": _standing("C", pts=3),
        "D": _standing("D", pts=0),
    }
    winner, runner_up = qualify_from_group(standings)
    assert winner == "A"
    assert runner_up == "B"


def test_qualify_winner_is_first_in_ranked():
    standings = {
        "X": _standing("X", pts=7, gf=5, ga=1),
        "Y": _standing("Y", pts=7, gf=3, ga=2),
        "Z": _standing("Z", pts=2),
        "W": _standing("W", pts=1),
    }
    winner, runner_up = qualify_from_group(standings)
    assert winner == "X"
    assert runner_up == "Y"
