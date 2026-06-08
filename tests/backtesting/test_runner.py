import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.runner import run_backtest, MatchResult


def _make_csv(tmp_path, rows):
    """Write a minimal matches CSV for testing."""
    f = tmp_path / "matches.csv"
    df = pd.DataFrame(rows, columns=["date","team_a","team_b","team_a_goals","team_b_goals"])
    df.to_csv(f, index=False)
    return f


def _minimal_ratings():
    """Ratings dict covering the test teams."""
    from src.data.loader import load_team_ratings
    return load_team_ratings()


def test_returns_list_of_match_results(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_actual_outcome_team_a_win(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",3,0]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "team_a_win"


def test_actual_outcome_draw(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","Germany","Spain",1,1]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "draw"


def test_actual_outcome_team_b_win(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","Qatar","France",0,3]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "team_b_win"


def test_predicted_outcome_is_max_probability(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",2,0]])
    result = run_backtest(csv, ratings)[0]
    probs = {
        "team_a_win": result.win_a_prob,
        "draw": result.draw_prob,
        "team_b_win": result.win_b_prob,
    }
    assert result.predicted_outcome == max(probs, key=probs.get)


def test_in_top_5_when_score_present(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",1,0]])
    result = run_backtest(csv, ratings)[0]
    top5_scores = [(g_a, g_b) for g_a, g_b, _ in result.top_scorelines]
    assert result.in_top_5 == ((result.actual_goals_a, result.actual_goals_b) in top5_scores)


def test_missing_file_raises(tmp_path):
    ratings = _minimal_ratings()
    with pytest.raises(FileNotFoundError):
        run_backtest(tmp_path / "nonexistent.csv", ratings)


def test_missing_team_in_ratings_raises(tmp_path):
    csv = _make_csv(tmp_path, [["2022-11-20","France","UnknownTeam",1,0]])
    ratings = {"France": {"elo":2020,"attack_rating":1.18,"defense_rating":0.85,
                          "form_rating":1.08,"squad_rating":1.12}}
    with pytest.raises(ValueError, match="not found in ratings"):
        run_backtest(csv, ratings)


def test_run_backtest_with_poisson_model_type(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings, model_type="poisson")
    assert len(results) == 1


def test_run_backtest_with_dixon_coles_model_type(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings, model_type="dixon_coles")
    assert len(results) == 1
    assert isinstance(results[0].win_a_prob, float)


def test_run_backtest_invalid_model_type_raises(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    with pytest.raises(ValueError, match="model_type"):
        run_backtest(csv, ratings, model_type="invalid")
