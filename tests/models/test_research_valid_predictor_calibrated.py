"""Tests that research_valid_predictor uses calibrated xG by default."""

import pytest
from src.models.research_valid_predictor import predict_research_valid, ResearchValidInput
from src.models.xg_calibration import XG_FLOOR, XG_CEIL
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams


def _extreme_mismatch() -> ResearchValidInput:
    """Input that previously produced extreme xG (like Argentina vs Saudi Arabia)."""
    return ResearchValidInput(
        team_a="Strong",
        team_b="Weak",
        snapshot_a=TeamSnapshot(elo=2200.0, ppg=2.5),
        snapshot_b=TeamSnapshot(elo=1600.0, ppg=0.8),
        params_a=StrengthParams(alpha_attack=5.0, beta_defense=0.10, matches_used=1000),
        params_b=StrengthParams(alpha_attack=0.4, beta_defense=3.0, matches_used=200),
        rho=-0.30,
    )


def _balanced() -> ResearchValidInput:
    return ResearchValidInput(
        team_a="TeamA",
        team_b="TeamB",
        snapshot_a=TeamSnapshot(elo=1900.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1900.0, ppg=2.0),
        params_a=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        rho=-0.30,
    )


# --- Calibration applied ---

def test_extreme_input_produces_bounded_xg():
    result = predict_research_valid(_extreme_mismatch())
    assert result.xg_a <= XG_CEIL, f"xg_a {result.xg_a} exceeds ceiling {XG_CEIL}"
    assert result.xg_b >= XG_FLOOR, f"xg_b {result.xg_b} below floor {XG_FLOOR}"


def test_xg_never_exceeds_ceiling_for_any_input():
    inputs = [
        ResearchValidInput(
            team_a="T", team_b="T2",
            snapshot_a=TeamSnapshot(elo=2300.0, ppg=3.0),
            snapshot_b=TeamSnapshot(elo=1500.0, ppg=0.5),
            params_a=StrengthParams(alpha_attack=alpha, beta_defense=0.1, matches_used=100),
            params_b=StrengthParams(alpha_attack=0.3, beta_defense=3.0, matches_used=100),
            rho=-0.30,
        )
        for alpha in [2.0, 5.0, 8.0, 10.0]
    ]
    for inp in inputs:
        result = predict_research_valid(inp)
        assert result.xg_a <= XG_CEIL
        assert result.xg_b <= XG_CEIL


def test_xg_never_below_floor_for_any_input():
    inp = ResearchValidInput(
        team_a="Weak", team_b="Strong",
        snapshot_a=TeamSnapshot(elo=1500.0, ppg=0.5),
        snapshot_b=TeamSnapshot(elo=2200.0, ppg=2.5),
        params_a=StrengthParams(alpha_attack=0.2, beta_defense=0.1, matches_used=50),
        params_b=StrengthParams(alpha_attack=5.0, beta_defense=3.0, matches_used=1000),
        rho=-0.30,
    )
    result = predict_research_valid(inp)
    assert result.xg_a >= XG_FLOOR
    assert result.xg_b >= XG_FLOOR


# --- Ordering preserved ---

def test_stronger_attack_still_produces_higher_xg_after_calibration():
    inp_strong = ResearchValidInput(
        team_a="Strong", team_b="Avg",
        snapshot_a=TeamSnapshot(elo=2000.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1900.0, ppg=2.0),
        params_a=StrengthParams(alpha_attack=4.0, beta_defense=1.0, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        rho=-0.30,
    )
    inp_weak = ResearchValidInput(
        team_a="Weak", team_b="Avg",
        snapshot_a=TeamSnapshot(elo=2000.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1900.0, ppg=2.0),
        params_a=StrengthParams(alpha_attack=0.6, beta_defense=1.0, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        rho=-0.30,
    )
    assert predict_research_valid(inp_strong).xg_a > predict_research_valid(inp_weak).xg_a


# --- Mid-range unaffected ---

def test_balanced_teams_produce_realistic_xg():
    result = predict_research_valid(_balanced())
    # Near-baseline teams should have xG in a reasonable range
    assert 0.8 <= result.xg_a <= 2.5
    assert 0.8 <= result.xg_b <= 2.5


# --- Probabilities still valid ---

def test_probabilities_sum_to_one_after_calibration():
    result = predict_research_valid(_extreme_mismatch())
    assert abs(result.win_a + result.draw + result.win_b - 1.0) < 1e-6


def test_strong_team_wins_more_often_even_after_calibration():
    result = predict_research_valid(_extreme_mismatch())
    assert result.win_a > result.win_b
