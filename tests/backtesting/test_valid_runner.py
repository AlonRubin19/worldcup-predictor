import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.runner import MatchResult


def _make_stats_csv(tmp_path, rows):
    """Write a minimal pre-match CSV."""
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


def _row(match_id=1, ma=10, mb=10, ga=2, gb=1):
    return [match_id,"2022-11-20","France","Brazil",2015,2044,
            1.8,0.7,2.1,0.7,2.5,2.6,ma,mb,ga,gb]


def test_returns_list_of_match_results(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_works_with_poisson_model(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv, model_type="poisson")
    assert results[0].win_a_prob > 0


def test_works_with_dixon_coles_model(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv, model_type="dixon_coles", rho=-0.20)
    assert results[0].win_a_prob > 0


def test_exclude_insufficient_reduces_results(tmp_path):
    rows = [_row(1, ma=10, mb=10), _row(2, ma=3, mb=10), _row(3, ma=10, mb=4)]
    csv = _make_stats_csv(tmp_path, rows)
    full = run_valid_backtest(csv, exclude_insufficient=False)
    filtered = run_valid_backtest(csv, exclude_insufficient=True, min_matches=5)
    assert len(full) == 3
    assert len(filtered) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_valid_backtest(tmp_path / "missing.csv")


def test_does_not_use_team_ratings_csv(tmp_path, monkeypatch):
    """Verify valid_runner never reads team_ratings.csv."""
    import src.data.loader as loader_module
    call_count = [0]
    original = loader_module.load_team_ratings

    def patched(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "load_team_ratings", patched)
    csv = _make_stats_csv(tmp_path, [_row()])
    run_valid_backtest(csv)
    assert call_count[0] == 0, "valid_runner must not call load_team_ratings()"
