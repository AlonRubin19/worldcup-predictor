import pytest
import pandas as pd
from pathlib import Path
from src.data.pre_match_loader import load_pre_match_stats, PreMatchStats


def _make_csv(tmp_path, rows, header=True):
    """Write a minimal pre-match CSV for testing."""
    f = tmp_path / "stats.csv"
    cols = [
        "match_id","date","team_a","team_b",
        "team_a_elo_pre","team_b_elo_pre",
        "team_a_goals_for_last_10","team_a_goals_against_last_10",
        "team_b_goals_for_last_10","team_b_goals_against_last_10",
        "team_a_points_per_game_last_10","team_b_points_per_game_last_10",
        "team_a_matches_available","team_b_matches_available",
        "team_a_goals","team_b_goals",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f, index=False)
    return f


def _row(match_id=1, ma=10, mb=10):
    return [match_id,"2022-11-20","France","Brazil",2015,2044,
            1.8,0.7,2.1,0.7,2.5,2.6,ma,mb,2,0]


def test_returns_list_of_pre_match_stats(tmp_path):
    csv = _make_csv(tmp_path, [_row()])
    result = load_pre_match_stats(csv)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], PreMatchStats)


def test_comment_lines_are_skipped(tmp_path):
    f = tmp_path / "stats.csv"
    f.write_text(
        "# WARNING: PLACEHOLDER\n"
        "match_id,date,team_a,team_b,team_a_elo_pre,team_b_elo_pre,"
        "team_a_goals_for_last_10,team_a_goals_against_last_10,"
        "team_b_goals_for_last_10,team_b_goals_against_last_10,"
        "team_a_points_per_game_last_10,team_b_points_per_game_last_10,"
        "team_a_matches_available,team_b_matches_available,"
        "team_a_goals,team_b_goals\n"
        "1,2022-11-20,A,B,2000,1900,1.5,0.8,1.2,1.0,2.2,1.8,10,10,2,1\n"
    )
    result = load_pre_match_stats(f)
    assert len(result) == 1


def test_exclude_insufficient_removes_low_match_rows(tmp_path):
    csv = _make_csv(tmp_path, [_row(1, ma=10, mb=10), _row(2, ma=3, mb=10), _row(3, ma=10, mb=4)])
    full = load_pre_match_stats(csv, exclude_insufficient=False)
    filtered = load_pre_match_stats(csv, exclude_insufficient=True, min_matches=5)
    assert len(full) == 3
    assert len(filtered) == 1  # only the row with both >= 5


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pre_match_stats(tmp_path / "missing.csv")


def test_missing_column_raises(tmp_path):
    f = tmp_path / "stats.csv"
    f.write_text("match_id,date\n1,2022-11-20\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_pre_match_stats(f)
