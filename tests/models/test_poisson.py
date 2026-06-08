import pytest
from src.models.poisson import predict, PredictionResult, build_score_matrix


def test_returns_prediction_result():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    assert isinstance(result, PredictionResult)


def test_probabilities_sum_to_one():
    result = predict("France", "Germany", 1.3, 1.1)
    total = result.win_a + result.draw + result.win_b
    assert abs(total - 1.0) < 1e-4  # adaptive matrix ensures tight coverage


def test_probabilities_are_between_0_and_1():
    result = predict("Spain", "England", 1.4, 1.0)
    assert 0.0 <= result.win_a <= 1.0
    assert 0.0 <= result.draw <= 1.0
    assert 0.0 <= result.win_b <= 1.0


def test_higher_xg_team_has_higher_win_probability():
    result = predict("Brazil", "Qatar", 2.5, 0.5)
    assert result.win_a > result.win_b


def test_equal_xg_gives_symmetric_probabilities():
    result = predict("A", "B", 1.5, 1.5)
    assert abs(result.win_a - result.win_b) < 0.001


def test_top_scorelines_returns_five():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    assert len(result.top_scorelines) == 5


def test_top_scorelines_are_sorted_descending():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    probs = [p for _, _, p in result.top_scorelines]
    assert probs == sorted(probs, reverse=True)


def test_top_scorelines_format():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    for goals_a, goals_b, prob in result.top_scorelines:
        assert isinstance(goals_a, int)
        assert isinstance(goals_b, int)
        assert 0.0 < prob <= 1.0


def test_xg_validation_raises_on_zero():
    with pytest.raises(ValueError):
        predict("A", "B", 0.0, 1.0)


def test_xg_validation_raises_on_negative():
    with pytest.raises(ValueError):
        predict("A", "B", 1.0, -0.5)


def test_team_names_stored_in_result():
    result = predict("France", "Brazil", 1.2, 1.5)
    assert result.team_a == "France"
    assert result.team_b == "Brazil"


def test_build_score_matrix_shape_and_sum():
    matrix = build_score_matrix(1.5, 1.2)
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    # Matrix should sum to approximately 1 (small gap due to truncation)
    assert abs(matrix.sum() - 1.0) < 1e-4


def test_build_score_matrix_cell_values():
    from scipy.stats import poisson as _poisson
    matrix = build_score_matrix(1.5, 1.2)
    # Cell [2][1] should equal poisson.pmf(2, 1.5) * poisson.pmf(1, 1.2)
    expected = _poisson.pmf(2, 1.5) * _poisson.pmf(1, 1.2)
    assert abs(matrix[2, 1] - expected) < 1e-12
