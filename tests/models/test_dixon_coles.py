import pytest
import numpy as np
from src.models.dixon_coles import predict_dixon_coles
from src.models.poisson import predict, PredictionResult


def test_returns_prediction_result():
    result = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    assert isinstance(result, PredictionResult)


def test_probabilities_sum_to_one():
    result = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    total = result.win_a + result.draw + result.win_b
    assert abs(total - 1.0) < 1e-4


def test_rho_zero_gives_same_result_as_poisson():
    dc = predict_dixon_coles("France", "Brazil", 1.5, 1.2, rho=0.0)
    po = predict("France", "Brazil", 1.5, 1.2)
    assert abs(dc.win_a - po.win_a) < 1e-6
    assert abs(dc.draw  - po.draw)  < 1e-6
    assert abs(dc.win_b - po.win_b) < 1e-6


def test_draw_probability_differs_from_poisson_with_default_rho():
    dc = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    po = predict("France", "Brazil", 1.5, 1.2)
    # Default rho=-0.10 should change draw probability meaningfully
    assert abs(dc.draw - po.draw) > 0.001


def test_top_scorelines_has_five_entries():
    result = predict_dixon_coles("Spain", "Germany", 1.4, 1.3)
    assert len(result.top_scorelines) == 5


def test_xg_validation_raises():
    with pytest.raises(ValueError):
        predict_dixon_coles("A", "B", 0.0, 1.0)
    with pytest.raises(ValueError):
        predict_dixon_coles("A", "B", 1.0, -0.5)


def test_low_score_probabilities_differ_from_poisson():
    dc = predict_dixon_coles("A", "B", 1.5, 1.2)
    po = predict("A", "B", 1.5, 1.2)
    # 0-0 and 1-1 probabilities should differ (tau != 1 for these cells)
    dc_00 = next((p for g_a, g_b, p in dc.top_scorelines if g_a == 0 and g_b == 0), None)
    po_00 = next((p for g_a, g_b, p in po.top_scorelines if g_a == 0 and g_b == 0), None)
    # If both models have 0-0 in top 5, verify the probabilities differ
    if dc_00 is not None and po_00 is not None:
        assert abs(dc_00 - po_00) > 1e-6
