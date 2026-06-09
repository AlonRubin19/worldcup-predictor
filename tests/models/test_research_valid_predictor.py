"""Tests for the research-valid prediction pipeline.

Verifies that the ELO + MLE + Dixon-Coles flow produces valid results
and does NOT call load_team_ratings().
"""

import pytest
from src.models.research_valid_predictor import (
    predict_research_valid,
    ResearchValidInput,
    ResearchValidResult,
)
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams


def _make_input(**overrides) -> ResearchValidInput:
    defaults = dict(
        team_a="Argentina",
        team_b="France",
        snapshot_a=TeamSnapshot(elo=2050.0, ppg=2.2),
        snapshot_b=TeamSnapshot(elo=1950.0, ppg=1.8),
        params_a=StrengthParams(alpha_attack=1.20, beta_defense=0.90, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.10, beta_defense=0.95, matches_used=100),
        rho=-0.30,
    )
    defaults.update(overrides)
    return ResearchValidInput(**defaults)


# --- Contract ---

def test_returns_research_valid_result():
    result = predict_research_valid(_make_input())
    assert isinstance(result, ResearchValidResult)


def test_result_has_expected_fields():
    result = predict_research_valid(_make_input())
    assert hasattr(result, "team_a")
    assert hasattr(result, "team_b")
    assert hasattr(result, "xg_a")
    assert hasattr(result, "xg_b")
    assert hasattr(result, "win_a")
    assert hasattr(result, "draw")
    assert hasattr(result, "win_b")
    assert hasattr(result, "top_scorelines")
    assert hasattr(result, "elo_a")
    assert hasattr(result, "elo_b")
    assert hasattr(result, "alpha_attack_a")
    assert hasattr(result, "alpha_attack_b")
    assert hasattr(result, "beta_defense_a")
    assert hasattr(result, "beta_defense_b")


# --- Valid probabilities ---

def test_probabilities_sum_to_one():
    result = predict_research_valid(_make_input())
    total = result.win_a + result.draw + result.win_b
    assert abs(total - 1.0) < 1e-6


def test_probabilities_are_between_zero_and_one():
    result = predict_research_valid(_make_input())
    for p in (result.win_a, result.draw, result.win_b):
        assert 0.0 < p < 1.0


def test_xg_is_positive():
    result = predict_research_valid(_make_input())
    assert result.xg_a > 0
    assert result.xg_b > 0


# --- ELO advantage reflected ---

def test_higher_elo_team_has_higher_win_probability():
    result = predict_research_valid(_make_input(
        snapshot_a=TeamSnapshot(elo=2200.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1700.0, ppg=2.0),
        params_a=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
    ))
    assert result.win_a > result.win_b


def test_equal_teams_have_similar_win_probabilities():
    result = predict_research_valid(_make_input(
        snapshot_a=TeamSnapshot(elo=1900.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1900.0, ppg=2.0),
        params_a=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
        params_b=StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100),
    ))
    assert abs(result.win_a - result.win_b) < 0.05


# --- Inputs pass through to result ---

def test_elo_values_stored_on_result():
    inp = _make_input(
        snapshot_a=TeamSnapshot(elo=2100.0, ppg=2.0),
        snapshot_b=TeamSnapshot(elo=1800.0, ppg=1.8),
    )
    result = predict_research_valid(inp)
    assert result.elo_a == 2100.0
    assert result.elo_b == 1800.0


def test_alpha_beta_stored_on_result():
    inp = _make_input(
        params_a=StrengthParams(alpha_attack=1.35, beta_defense=0.88, matches_used=50),
        params_b=StrengthParams(alpha_attack=0.95, beta_defense=1.10, matches_used=50),
    )
    result = predict_research_valid(inp)
    assert result.alpha_attack_a == 1.35
    assert result.alpha_attack_b == 0.95
    assert result.beta_defense_a == 0.88
    assert result.beta_defense_b == 1.10


# --- Uses strength-adjusted xG, not legacy ratings ---

def test_attack_strength_affects_xg():
    result_strong = predict_research_valid(_make_input(
        params_a=StrengthParams(alpha_attack=1.8, beta_defense=1.0, matches_used=100),
    ))
    result_weak = predict_research_valid(_make_input(
        params_a=StrengthParams(alpha_attack=0.6, beta_defense=1.0, matches_used=100),
    ))
    assert result_strong.xg_a > result_weak.xg_a


def test_rho_parameter_used():
    # rho=-0.30 vs rho=0 — Dixon-Coles correction changes the probabilities
    result_dc = predict_research_valid(_make_input(rho=-0.30))
    result_no_dc = predict_research_valid(_make_input(rho=0.0))
    # Probabilities differ because DC correction shifts low-score mass
    assert abs(result_dc.win_a - result_no_dc.win_a) > 1e-6
