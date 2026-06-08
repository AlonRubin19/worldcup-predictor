import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.strength_runner import run_strength_backtest
from src.backtesting.runner import MatchResult


def _make_match_results_csv(tmp_path, rows):
    f = tmp_path / "match_results.csv"
    cols = [
        "match_id","date","team_a","team_b","team_a_goals","team_b_goals",
        "team_a_elo_pre","team_b_elo_pre",
        "team_a_goals_for_last_10","team_a_goals_against_last_10",
        "team_b_goals_for_last_10","team_b_goals_against_last_10",
        "team_a_points_per_game_last_10","team_b_points_per_game_last_10",
        "team_a_matches_available","team_b_matches_available",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(f, index=False)
    return f


def _make_strength_params_csv(tmp_path, rows):
    f = tmp_path / "params.csv"
    pd.DataFrame(rows, columns=["team","alpha_attack","beta_defense","matches_used","as_of_date"]).to_csv(f, index=False)
    return f


def _row(team_a="France", team_b="Brazil", ga=2, gb=1):
    return [1,"2022-11-23",team_a,team_b,ga,gb,2015,2044,1.8,0.7,2.1,0.7,2.5,2.6,10,10]


def _params(teams=("France","Brazil")):
    return [[t, 1.4, 0.8, 20, "2022-11-19"] for t in teams]


def test_returns_list_of_match_result(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_works_with_poisson(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp, model_type="poisson")
    assert results[0].win_a_prob > 0


def test_works_with_dixon_coles(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp, model_type="dixon_coles", rho=-0.20)
    assert results[0].win_a_prob > 0


def test_missing_team_in_params_raises(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row("France", "Brazil")])
    sp = _make_strength_params_csv(tmp_path, [["France", 1.4, 0.8, 20, "2022-11-19"]])  # Brazil missing
    with pytest.raises(ValueError, match="Brazil"):
        run_strength_backtest(mr, sp)
