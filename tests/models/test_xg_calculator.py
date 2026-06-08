import pytest
from src.models.xg_calculator import calculate_xg, BASE_XG, XG_MIN, XG_MAX


def _ratings(elo=1900, attack=1.0, defense=1.0, form=1.0, squad=1.0):
    """Helper: build a ratings dict with sensible defaults."""
    return {
        "elo": elo,
        "attack_rating": attack,
        "defense_rating": defense,
        "form_rating": form,
        "squad_rating": squad,
    }


def test_returns_tuple_of_two_floats():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert isinstance(xg_a, float)
    assert isinstance(xg_b, float)


def test_equal_ratings_give_equal_xg():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert abs(xg_a - xg_b) < 1e-9


def test_higher_elo_team_gets_higher_xg():
    xg_a, xg_b = calculate_xg(_ratings(elo=2000), _ratings(elo=1700))
    assert xg_a > xg_b


def test_lower_elo_team_gets_lower_xg():
    xg_a, xg_b = calculate_xg(_ratings(elo=1700), _ratings(elo=2000))
    assert xg_a < xg_b


def test_strong_attack_increases_xg():
    base_a, _ = calculate_xg(_ratings(attack=1.0), _ratings())
    high_a, _ = calculate_xg(_ratings(attack=1.3), _ratings())
    assert high_a > base_a


def test_strong_defense_decreases_opponent_xg():
    # When team B has strong defense (low multiplier 0.7), xg_a is lower
    # because xg_a uses defense_b as a factor
    xg_a_strong, _ = calculate_xg(_ratings(), _ratings(defense=0.7))
    xg_a_weak,   _ = calculate_xg(_ratings(), _ratings(defense=1.3))
    assert xg_a_strong < xg_a_weak


def test_extreme_elo_gap_clamps_xg():
    # Extreme values that would push xG outside [XG_MIN, XG_MAX] without clamping
    xg_a, xg_b = calculate_xg(_ratings(elo=3000, attack=1.3, form=1.2, squad=1.2),
                               _ratings(elo=1000, attack=0.7, form=0.8, squad=0.8))
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN


def test_xg_values_always_within_bounds():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert XG_MIN <= xg_a <= XG_MAX
    assert XG_MIN <= xg_b <= XG_MAX
