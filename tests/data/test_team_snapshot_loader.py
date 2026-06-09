"""Tests for team snapshot loader — latest ELO + PPG from match_results.csv."""

import pytest
import pandas as pd
from pathlib import Path
from src.data.team_snapshot_loader import load_team_snapshots, TeamSnapshot

_FIXTURE_ROWS = [
    # match_id, date, team_a, team_b, goals_a, goals_b,
    # elo_pre_a, elo_pre_b, gf10_a, ga10_a, gf10_b, ga10_b, ppg_a, ppg_b, avail_a, avail_b
    "1,2022-11-01,England,France,1,0,1900.0,1950.0,1.5,1.0,1.8,1.1,2.0,1.8,8,9",
    "2,2022-11-10,Germany,England,2,1,1850.0,1910.0,1.4,1.1,1.6,1.2,1.9,2.1,9,9",
    "3,2022-11-20,France,Germany,0,0,1960.0,1840.0,1.7,1.0,1.3,1.3,1.7,1.5,10,9",
]

_COLS = (
    "match_id,date,team_a,team_b,team_a_goals,team_b_goals,"
    "team_a_elo_pre,team_b_elo_pre,"
    "team_a_goals_for_last_10,team_a_goals_against_last_10,"
    "team_b_goals_for_last_10,team_b_goals_against_last_10,"
    "team_a_points_per_game_last_10,team_b_points_per_game_last_10,"
    "team_a_matches_available,team_b_matches_available"
)


def _make_csv(tmp_path: Path) -> Path:
    p = tmp_path / "match_results.csv"
    lines = [_COLS] + _FIXTURE_ROWS
    p.write_text("\n".join(lines))
    return p


def test_load_returns_dict_of_team_snapshots(tmp_path):
    path = _make_csv(tmp_path)
    snapshots = load_team_snapshots(path)
    assert isinstance(snapshots, dict)
    assert all(isinstance(v, TeamSnapshot) for v in snapshots.values())


def test_snapshot_has_elo_and_ppg_fields(tmp_path):
    path = _make_csv(tmp_path)
    snapshots = load_team_snapshots(path)
    snap = snapshots["England"]
    assert hasattr(snap, "elo")
    assert hasattr(snap, "ppg")
    assert snap.elo > 0
    assert snap.ppg >= 0


def test_snapshot_uses_latest_match_for_team(tmp_path):
    path = _make_csv(tmp_path)
    snapshots = load_team_snapshots(path)
    # England appears as team_a in match 1 (elo=1900) and team_b in match 2 (elo=1910)
    # Latest date for England is 2022-11-10 (match 2 as team_b) — elo=1910, ppg=2.1
    assert abs(snapshots["England"].elo - 1910.0) < 0.1
    assert abs(snapshots["England"].ppg - 2.1) < 0.01


def test_snapshot_covers_all_teams_in_file(tmp_path):
    path = _make_csv(tmp_path)
    snapshots = load_team_snapshots(path)
    assert "England" in snapshots
    assert "France" in snapshots
    assert "Germany" in snapshots


def test_raises_file_not_found_for_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_team_snapshots(tmp_path / "nonexistent.csv")
