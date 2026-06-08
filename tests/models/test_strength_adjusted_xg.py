import pytest
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg, XG_MIN, XG_MAX
from src.data.strength_loader import StrengthParams


def _params(alpha=1.0, beta=1.0):
    return StrengthParams(alpha_attack=alpha, beta_defense=beta, matches_used=20)


def test_returns_two_floats():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=1900, elo_b=1900,
        params_a=_params(), params_b=_params(),
        ppg_a=1.5, ppg_b=1.5,
    )
    assert isinstance(xg_a, float) and isinstance(xg_b, float)


def test_equal_teams_produce_equal_xg():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=1900, elo_b=1900,
        params_a=_params(1.0, 1.0), params_b=_params(1.0, 1.0),
        ppg_a=1.5, ppg_b=1.5,
    )
    assert abs(xg_a - xg_b) < 1e-9


def test_stronger_attack_increases_xg():
    low, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0), _params(1.0), 1.5, 1.5)
    high, _ = calculate_strength_adjusted_xg(1900, 1900, _params(2.0), _params(1.0), 1.5, 1.5)
    assert high > low


def test_stronger_opponent_defense_decreases_xg():
    base, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0, 1.0), _params(1.0, 1.0), 1.5, 1.5)
    vs_solid, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0, 1.0), _params(1.0, 0.5), 1.5, 1.5)
    assert vs_solid < base


def test_higher_elo_increases_xg():
    base, _ = calculate_strength_adjusted_xg(1900, 1900, _params(), _params(), 1.5, 1.5)
    high, _ = calculate_strength_adjusted_xg(2100, 1700, _params(), _params(), 1.5, 1.5)
    assert high > base


def test_output_clamped():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=2500, elo_b=1000,
        params_a=_params(10.0, 0.1), params_b=_params(0.1, 10.0),
        ppg_a=3.0, ppg_b=0.0,
    )
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN
