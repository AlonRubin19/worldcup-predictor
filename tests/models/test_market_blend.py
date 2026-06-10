import pytest
from src.models.market_blend import blend_probabilities, BlendedProbabilities


def test_blend_85_15_with_valid_market():
    result = blend_probabilities(
        model_win_a=0.50, model_draw=0.25, model_win_b=0.25,
        market_win_a=0.40, market_draw=0.30, market_win_b=0.30,
        market_research_valid=True,
    )
    assert result.win_a == pytest.approx(0.85 * 0.50 + 0.15 * 0.40)
    assert result.draw == pytest.approx(0.85 * 0.25 + 0.15 * 0.30)
    assert result.win_b == pytest.approx(0.85 * 0.25 + 0.15 * 0.30)
    assert result.used_market is True


def test_blend_falls_back_to_model_when_not_research_valid():
    result = blend_probabilities(
        model_win_a=0.50, model_draw=0.25, model_win_b=0.25,
        market_win_a=0.40, market_draw=0.30, market_win_b=0.30,
        market_research_valid=False,
    )
    assert result.win_a == pytest.approx(0.50)
    assert result.draw == pytest.approx(0.25)
    assert result.win_b == pytest.approx(0.25)
    assert result.used_market is False


def test_blend_falls_back_when_market_probs_missing():
    result = blend_probabilities(
        model_win_a=0.50, model_draw=0.25, model_win_b=0.25,
        market_win_a=None, market_draw=None, market_win_b=None,
        market_research_valid=True,
    )
    assert result.win_a == pytest.approx(0.50)
    assert result.used_market is False


def test_blend_normalizes_to_one():
    result = blend_probabilities(
        model_win_a=0.50, model_draw=0.25, model_win_b=0.25,
        market_win_a=0.40, market_draw=0.30, market_win_b=0.30,
        market_research_valid=True,
    )
    assert result.win_a + result.draw + result.win_b == pytest.approx(1.0)


def test_blend_label():
    result = blend_probabilities(
        model_win_a=0.50, model_draw=0.25, model_win_b=0.25,
        market_win_a=0.40, market_draw=0.30, market_win_b=0.30,
        market_research_valid=True,
    )
    assert "85%" in result.label and "15%" in result.label
