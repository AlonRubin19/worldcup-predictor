import pytest
from src.models.pre_match_xg import calculate_pre_match_xg, BASE_XG, XG_MIN, XG_MAX
from src.data.pre_match_loader import PreMatchStats


def _match(
    elo_a=1900, elo_b=1900,
    gf_a=1.35, ga_a=1.35,
    gf_b=1.35, ga_b=1.35,
    ppg_a=1.5, ppg_b=1.5,
):
    """Build a test PreMatchStats with sensible defaults (1.0 multipliers)."""
    return PreMatchStats(
        match_id=1, date="2022-11-20", team_a="A", team_b="B",
        team_a_elo_pre=elo_a, team_b_elo_pre=elo_b,
        team_a_goals_for_last_10=gf_a, team_a_goals_against_last_10=ga_a,
        team_b_goals_for_last_10=gf_b, team_b_goals_against_last_10=ga_b,
        team_a_points_per_game_last_10=ppg_a, team_b_points_per_game_last_10=ppg_b,
        team_a_matches_available=10, team_b_matches_available=10,
        team_a_goals=1, team_b_goals=0,
    )


def test_returns_two_floats():
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert isinstance(xg_a, float)
    assert isinstance(xg_b, float)


def test_equal_teams_produce_equal_xg():
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert abs(xg_a - xg_b) < 1e-9


def test_average_team_produces_xg_near_base():
    # All multipliers = 1.0 -> xg should be close to BASE_XG
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert abs(xg_a - BASE_XG) < 0.05


def test_higher_elo_gets_higher_xg():
    xg_a, xg_b = calculate_pre_match_xg(_match(elo_a=2100, elo_b=1700))
    assert xg_a > xg_b


def test_higher_attack_increases_xg():
    base_a, _ = calculate_pre_match_xg(_match(gf_a=1.35))
    high_a, _ = calculate_pre_match_xg(_match(gf_a=2.00))
    assert high_a > base_a


def test_strong_defense_decreases_opponent_xg():
    # Team B with low goals_against -> lower xg_a
    normal, _ = calculate_pre_match_xg(_match(ga_b=1.35))
    tight, _ = calculate_pre_match_xg(_match(ga_b=0.70))
    assert tight < normal


def test_higher_form_increases_xg():
    low_form_a, _ = calculate_pre_match_xg(_match(ppg_a=0.5))
    high_form_a, _ = calculate_pre_match_xg(_match(ppg_a=2.8))
    assert high_form_a > low_form_a


def test_output_clamped_to_bounds():
    xg_a, xg_b = calculate_pre_match_xg(_match(
        elo_a=2500, elo_b=1000,
        gf_a=3.0, ga_b=3.0, ppg_a=3.0,
        gf_b=0.3, ga_a=0.3, ppg_b=0.0,
    ))
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN
