import pytest
from src.backtesting.rho_tuning import tune_rho, select_best_rho, RhoResult, DEFAULT_RHO_GRID
from src.data.loader import load_team_ratings


def _ratings():
    return load_team_ratings()


def test_tune_rho_returns_one_result_per_rho():
    ratings = _ratings()
    grid = [-0.10, -0.05, 0.00]
    results = tune_rho(ratings, rho_grid=grid)
    assert len(results) == 3


def test_tune_rho_returns_rho_result_objects():
    ratings = _ratings()
    results = tune_rho(ratings, rho_grid=[-0.10])
    assert isinstance(results[0], RhoResult)
    assert results[0].rho == -0.10


def test_tune_rho_preserves_grid_order():
    ratings = _ratings()
    grid = [-0.30, -0.10, 0.10]
    results = tune_rho(ratings, rho_grid=grid)
    assert [r.rho for r in results] == grid


def test_select_best_rho_picks_lowest_brier():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.60,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.05, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.65,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20


def test_select_best_rho_tiebreak_top3():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.35, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20  # higher top_3_hit_rate wins tie


def test_select_best_rho_tiebreak_exact_score():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.12,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.15,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20  # higher exact_score wins second tie


def test_select_best_rho_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        select_best_rho([])
