"""Tests for PredictionExplanation driver calculation logic."""

import pytest
from src.explainability.driver import (
    build_explanation,
    ExplanationInput,
    PredictionExplanation,
    DriverContribution,
)


def _base_input(**overrides) -> ExplanationInput:
    defaults = dict(
        match_id="99999",
        team_a="Argentina",
        team_b="France",
        model_type="MLE+DC",
        elo_a=1900.0,
        elo_b=1900.0,
        alpha_attack_a=1.0,
        alpha_attack_b=1.0,
        beta_defense_a=1.0,
        beta_defense_b=1.0,
        xg_a_base=1.4,
        xg_b_base=1.4,
        squad_factor_a=1.0,
        squad_factor_b=1.0,
        xg_a_final=1.4,
        xg_b_final=1.4,
        win_a=0.40,
        draw=0.25,
        win_b=0.35,
        top_scorelines=[(1, 1, 0.10), (1, 0, 0.09), (0, 1, 0.08)],
        player_data_research_valid=False,
        market_home_prob=None,
        market_draw_prob=None,
        market_away_prob=None,
        market_research_valid=False,
    )
    defaults.update(overrides)
    return ExplanationInput(**defaults)


# --- Dataclass contracts ---

def test_explanation_has_required_fields():
    inp = _base_input()
    result = build_explanation(inp)
    assert isinstance(result, PredictionExplanation)
    assert result.match_id == "99999"
    assert result.team_a == "Argentina"
    assert result.team_b == "France"
    assert result.model_type == "MLE+DC"
    assert isinstance(result.drivers, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.top_scorelines, list)


def test_driver_contribution_fields():
    inp = _base_input(elo_a=2000.0, elo_b=1800.0)
    result = build_explanation(inp)
    elo_drivers = [d for d in result.drivers if d.name == "ELO advantage"]
    assert len(elo_drivers) == 1
    d = elo_drivers[0]
    assert hasattr(d, "name")
    assert hasattr(d, "team")
    assert hasattr(d, "direction")
    assert hasattr(d, "magnitude")
    assert hasattr(d, "description")


# --- ELO driver ---

def test_elo_driver_present_when_gap_exists():
    inp = _base_input(elo_a=2000.0, elo_b=1800.0)
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "ELO advantage" in names


def test_elo_driver_favours_higher_elo_team():
    inp = _base_input(elo_a=2000.0, elo_b=1800.0)
    result = build_explanation(inp)
    elo = next(d for d in result.drivers if d.name == "ELO advantage")
    assert elo.team == "Argentina"
    assert elo.direction == "positive"


def test_elo_driver_magnitude_scales_with_gap():
    inp_small = _base_input(elo_a=1950.0, elo_b=1900.0)
    inp_large = _base_input(elo_a=2100.0, elo_b=1800.0)
    r_small = build_explanation(inp_small)
    r_large = build_explanation(inp_large)
    mag_small = next(d.magnitude for d in r_small.drivers if d.name == "ELO advantage")
    mag_large = next(d.magnitude for d in r_large.drivers if d.name == "ELO advantage")
    assert mag_large > mag_small


def test_elo_driver_absent_when_equal():
    inp = _base_input(elo_a=1900.0, elo_b=1900.0)
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "ELO advantage" not in names


# --- Attack strength driver ---

def test_attack_driver_present_when_alpha_differs():
    inp = _base_input(alpha_attack_a=1.4, alpha_attack_b=0.8)
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Attack strength" in names


def test_attack_driver_favours_higher_alpha_team():
    inp = _base_input(alpha_attack_a=1.4, alpha_attack_b=0.8)
    result = build_explanation(inp)
    drv = next(d for d in result.drivers if d.name == "Attack strength")
    assert drv.team == "Argentina"
    assert drv.direction == "positive"


# --- Defensive weakness driver ---

def test_defense_driver_present_when_beta_differs():
    # High beta_defense = easier to score against = opponent defensive weakness
    inp = _base_input(beta_defense_a=1.0, beta_defense_b=1.3)
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Defensive weakness" in names


def test_defense_driver_team_is_weaker_defender():
    # beta_defense_b=1.3 means team_b defends poorly → Argentina benefits
    inp = _base_input(beta_defense_a=1.0, beta_defense_b=1.3)
    result = build_explanation(inp)
    drv = next(d for d in result.drivers if d.name == "Defensive weakness")
    assert drv.team == "France"  # France is the weak defender


# --- Player impact driver ---

def test_player_impact_driver_present_when_squad_factor_not_one():
    inp = _base_input(
        squad_factor_a=0.92,
        xg_a_base=1.5,
        xg_a_final=1.38,  # 1.5 * 0.92
    )
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Player impact" in names


def test_player_impact_driver_direction_negative_for_weakened_squad():
    inp = _base_input(
        squad_factor_a=0.92,
        xg_a_base=1.5,
        xg_a_final=1.38,
    )
    result = build_explanation(inp)
    drv = next(d for d in result.drivers if d.name == "Player impact")
    assert drv.team == "Argentina"
    assert drv.direction == "negative"


def test_player_impact_driver_absent_when_squad_factor_one():
    inp = _base_input(squad_factor_a=1.0, squad_factor_b=1.0)
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Player impact" not in names


def test_player_impact_magnitude_is_xg_delta():
    inp = _base_input(
        squad_factor_a=0.90,
        xg_a_base=1.5,
        xg_a_final=1.35,  # delta = 0.15
    )
    result = build_explanation(inp)
    drv = next(d for d in result.drivers if d.name == "Player impact")
    assert abs(drv.magnitude - 0.15) < 0.001


# --- Dixon-Coles adjustment driver ---

def test_dc_adjustment_driver_present():
    inp = _base_input()
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Dixon-Coles adjustment" in names


# --- Market divergence driver ---

def test_market_divergence_driver_excluded_when_placeholder():
    inp = _base_input(
        market_home_prob=0.45,
        market_draw_prob=0.25,
        market_away_prob=0.30,
        market_research_valid=False,
    )
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Market divergence" not in names


def test_market_divergence_driver_included_when_research_valid():
    inp = _base_input(
        win_a=0.55,
        win_b=0.25,
        draw=0.20,
        market_home_prob=0.40,
        market_draw_prob=0.25,
        market_away_prob=0.35,
        market_research_valid=True,
    )
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Market divergence" in names


def test_market_divergence_driver_excluded_when_no_market_data():
    inp = _base_input(
        market_home_prob=None,
        market_research_valid=False,
    )
    result = build_explanation(inp)
    names = [d.name for d in result.drivers]
    assert "Market divergence" not in names


# --- Warnings ---

def test_warning_added_when_player_data_not_research_valid():
    inp = _base_input(
        squad_factor_a=0.95,
        xg_a_base=1.5,
        xg_a_final=1.425,
        player_data_research_valid=False,
    )
    result = build_explanation(inp)
    assert any("not research-valid" in w.lower() or "engineering" in w.lower()
               for w in result.warnings)


def test_no_player_validity_warning_when_squad_factor_is_one():
    inp = _base_input(squad_factor_a=1.0, squad_factor_b=1.0,
                      player_data_research_valid=False)
    result = build_explanation(inp)
    # squad factor = 1 means player data not used; no warning needed
    player_warnings = [w for w in result.warnings
                       if "player" in w.lower() or "squad" in w.lower()]
    assert len(player_warnings) == 0


# --- Probabilities pass-through ---

def test_final_probabilities_stored_on_explanation():
    inp = _base_input(win_a=0.55, draw=0.22, win_b=0.23)
    result = build_explanation(inp)
    assert abs(result.win_a - 0.55) < 1e-6
    assert abs(result.draw - 0.22) < 1e-6
    assert abs(result.win_b - 0.23) < 1e-6


def test_top_scorelines_stored_on_explanation():
    scorelines = [(1, 0, 0.12), (0, 0, 0.10), (1, 1, 0.09)]
    inp = _base_input(top_scorelines=scorelines)
    result = build_explanation(inp)
    assert result.top_scorelines == scorelines
