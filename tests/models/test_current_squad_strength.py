import pytest
from src.models.current_squad_strength import (
    compute_squad_strength_factor,
    SquadStrengthResult,
)
from src.data.player_loader import PlayerProfile


def _profile(pid, team, xg, position="FW", research_valid=True, source_type="api_football"):
    return PlayerProfile(
        player_id=pid, player_name=pid, team=team, position=position, club="",
        minutes_last_90_days=900, national_team_minutes_last_12_months=900,
        goals_per_90=xg, assists_per_90=0.0, xg_per_90=xg, xa_per_90=0.0,
        defensive_actions_per_90=0.0, international_caps=10, base_impact_score=1.0,
        source_type=source_type, research_valid=research_valid,
    )


def test_no_profiles_returns_neutral_factor():
    result = compute_squad_strength_factor("Spain", {})
    assert result.factor == pytest.approx(1.0)
    assert result.research_valid is False


def test_above_baseline_xg_gives_factor_above_one():
    profiles = {
        "p1": _profile("p1", "Spain", 0.6),
        "p2": _profile("p2", "Spain", 0.5),
    }
    result = compute_squad_strength_factor("Spain", profiles)
    assert result.factor > 1.0
    assert result.research_valid is True


def test_only_considers_matching_team():
    profiles = {
        "p1": _profile("p1", "Spain", 0.6),
        "p2": _profile("p2", "France", 0.05),
    }
    result = compute_squad_strength_factor("France", profiles)
    assert result.team_player_count == 1


def test_injured_players_excluded():
    profiles = {
        "p1": _profile("p1", "Spain", 0.6),
        "p2": _profile("p2", "Spain", 0.05),
    }
    result = compute_squad_strength_factor("Spain", profiles, injured_player_names={"p1"})
    assert result.team_player_count == 1


def test_placeholder_only_data_marks_not_research_valid():
    profiles = {
        "p1": _profile("p1", "Spain", 0.2, source_type="placeholder", research_valid=False),
    }
    result = compute_squad_strength_factor("Spain", profiles)
    assert result.research_valid is False
    assert result.factor == pytest.approx(1.0)
