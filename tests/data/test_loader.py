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


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_teams(tmp_path / "missing.csv")


def test_missing_column_raises(tmp_path):
    f = tmp_path / "teams.csv"
    f.write_text("name\nArgentina\n")
    with pytest.raises(ValueError, match="'team' column"):
        load_teams(f)


def test_empty_file_raises(tmp_path):
    f = tmp_path / "teams.csv"
    f.write_text("team\n")
    with pytest.raises(ValueError, match="no team entries"):
        load_teams(f)
