import pytest
from src.data.loader import load_teams


def test_returns_list_of_strings():
    teams = load_teams()
    assert isinstance(teams, list)
    assert all(isinstance(t, str) for t in teams)


def test_returns_expected_teams():
    teams = load_teams()
    assert "Argentina" in teams
    assert "Brazil" in teams
    assert "France" in teams


def test_returns_sorted_list():
    teams = load_teams()
    assert teams == sorted(teams)


def test_no_duplicates():
    teams = load_teams()
    assert len(teams) == len(set(teams))


def test_minimum_team_count():
    teams = load_teams()
    assert len(teams) >= 40
