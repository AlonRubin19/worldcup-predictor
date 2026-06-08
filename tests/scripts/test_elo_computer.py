import pytest
from scripts.elo_computer import (
    compute_expected_score,
    update_elo,
    compute_elo_history,
)


def test_expected_score_equal_ratings():
    assert abs(compute_expected_score(1600, 1600) - 0.5) < 1e-9


def test_expected_score_higher_rated_favoured():
    e = compute_expected_score(1800, 1600)
    assert e > 0.5


def test_expected_scores_sum_to_one():
    ea = compute_expected_score(1750, 1650)
    eb = compute_expected_score(1650, 1750)
    assert abs(ea + eb - 1.0) < 1e-9


def test_elo_increases_on_win():
    new_a, new_b = update_elo(1600, 1600, win_a=True, draw=False)
    assert new_a > 1600
    assert new_b < 1600


def test_elo_decreases_on_loss():
    new_a, new_b = update_elo(1600, 1600, win_a=False, draw=False)
    assert new_a < 1600
    assert new_b > 1600


def test_elo_unchanged_sum_on_draw():
    new_a, new_b = update_elo(1600, 1600, win_a=False, draw=True)
    # Both at 1600 → expect no change on draw
    assert abs(new_a - 1600) < 1e-9
    assert abs(new_b - 1600) < 1e-9


def test_elo_is_zero_sum():
    """Total ELO in the system is conserved."""
    new_a, new_b = update_elo(1800, 1650, win_a=True, draw=False)
    assert abs((new_a + new_b) - (1800 + 1650)) < 1e-6


def test_compute_elo_history_no_leakage():
    """ELO used for match N must NOT include match N's result."""
    matches = [
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "home_score": 3, "away_score": 0},
        {"date": "2020-02-01", "home_team": "A", "away_team": "B",
         "home_score": 0, "away_score": 2},
    ]
    history = compute_elo_history(matches)
    # Row for first match: both teams at starting ELO 1600
    first = [r for r in history if r["date"] == "2020-01-01" and r["team"] == "A"][0]
    assert first["elo_pre"] == 1600.0

    # Row for second match: A's ELO should reflect first match WIN (> 1600)
    second = [r for r in history if r["date"] == "2020-02-01" and r["team"] == "A"][0]
    assert second["elo_pre"] > 1600.0


def test_new_team_starts_at_1600():
    matches = [
        {"date": "2020-01-01", "home_team": "NewTeam", "away_team": "OtherTeam",
         "home_score": 1, "away_score": 0},
    ]
    history = compute_elo_history(matches)
    row = [r for r in history if r["team"] == "NewTeam"][0]
    assert row["elo_pre"] == 1600.0
