import pytest
from src.backtesting.metrics import compute_metrics, BacktestMetrics
from src.backtesting.runner import MatchResult


def _result(actual, win_a, draw, win_b, exact=False, top3=False, top5=False):
    """Build a minimal MatchResult for metric testing."""
    prob = {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}[actual]
    predicted = max({"team_a_win": win_a, "draw": draw, "team_b_win": win_b},
                    key=lambda k: {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}[k])
    return MatchResult(
        date="2022-11-20", team_a="A", team_b="B",
        actual_goals_a=1, actual_goals_b=0,
        actual_outcome=actual,
        win_a_prob=win_a, draw_prob=draw, win_b_prob=win_b,
        predicted_outcome=predicted,
        top_scorelines=[(1,0,0.10),(2,0,0.08),(0,0,0.07)] if top5 else [(9,9,0.01)]*5,
        exact_score_hit=exact,
        in_top_3=top3,
        in_top_5=top5,
        prob_of_actual_result=prob,
    )


def test_returns_backtest_metrics():
    results = [_result("team_a_win", 0.6, 0.25, 0.15)]
    m = compute_metrics(results)
    assert isinstance(m, BacktestMetrics)


def test_accuracy_1x2_all_correct():
    results = [
        _result("team_a_win", 0.6, 0.25, 0.15),  # predicted: team_a_win ✓
        _result("draw",       0.2, 0.50, 0.30),   # predicted: draw ✓
    ]
    m = compute_metrics(results)
    assert m.accuracy_1x2 == pytest.approx(1.0)


def test_accuracy_1x2_none_correct():
    results = [
        _result("team_b_win", 0.6, 0.25, 0.15),  # predicted: team_a_win ✗
        _result("team_a_win", 0.1, 0.30, 0.60),  # predicted: team_b_win ✗
    ]
    m = compute_metrics(results)
    assert m.accuracy_1x2 == pytest.approx(0.0)


def test_brier_score_perfect_prediction():
    results = [_result("team_a_win", 1.0, 0.0, 0.0)]
    m = compute_metrics(results)
    assert m.brier_score == pytest.approx(0.0, abs=1e-9)


def test_brier_score_worst_prediction():
    results = [_result("team_a_win", 0.0, 0.0, 1.0)]
    m = compute_metrics(results)
    # BS = (0-1)^2 + (0-0)^2 + (1-0)^2 = 1 + 0 + 1 = 2.0
    assert m.brier_score == pytest.approx(2.0, abs=1e-9)


def test_hit_rates_computed_correctly():
    results = [
        _result("team_a_win", 0.6, 0.25, 0.15, exact=True,  top3=True,  top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=True,  top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=False, top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=False, top5=False),
    ]
    m = compute_metrics(results)
    assert m.exact_score_accuracy == pytest.approx(0.25)
    assert m.top_3_hit_rate       == pytest.approx(0.5)
    assert m.top_5_hit_rate       == pytest.approx(0.75)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])
