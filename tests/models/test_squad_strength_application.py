from __future__ import annotations

import pytest

from src.models.squad_strength_application import apply_squad_strength_to_match
from src.data.player_loader import PlayerProfile


def _profile(pid, team, xg, research_valid=True, source_type="api_football"):
    return PlayerProfile(
        player_id=pid, player_name=pid, team=team, position="FW", club="",
        minutes_last_90_days=900, national_team_minutes_last_12_months=900,
        goals_per_90=xg, assists_per_90=0.0, xg_per_90=xg, xa_per_90=0.0,
        defensive_actions_per_90=0.0, international_caps=10, base_impact_score=1.0,
        source_type=source_type, research_valid=research_valid,
    )


def test_no_research_valid_profiles_returns_unchanged_xg_and_unavailable():
    result = apply_squad_strength_to_match("Spain", "France", 1.5, 1.2, profiles={})

    assert result.xg_a_before == pytest.approx(1.5)
    assert result.xg_b_before == pytest.approx(1.2)
    assert result.xg_a_after == pytest.approx(1.5)
    assert result.xg_b_after == pytest.approx(1.2)
    assert result.live_data_available is False


def test_research_valid_profiles_adjust_xg_and_win_probabilities():
    profiles = {
        "p1": _profile("p1", "Spain", 0.6),
        "p2": _profile("p2", "France", 0.1),
    }
    result = apply_squad_strength_to_match("Spain", "France", 1.5, 1.2, profiles=profiles)

    assert result.live_data_available is True
    assert result.xg_a_after > result.xg_a_before
    assert result.xg_b_after < result.xg_b_before
    # win probabilities recomputed and sum to ~1
    assert abs(result.win_a_after + result.draw_after + result.win_b_after - 1.0) < 1e-6
    # squad strength favors Spain -> Spain's win prob should rise vs before
    assert result.win_a_after > result.win_a_before


def test_injured_players_excluded_from_factor():
    profiles = {
        "p1": _profile("p1", "Spain", 0.6),
        "p2": _profile("p2", "Spain", 0.05),
    }
    result = apply_squad_strength_to_match(
        "Spain", "France", 1.5, 1.2, profiles=profiles, injured_player_names={"p1"},
    )
    assert result.live_data_available is True
    # only p2 (low xG) remains -> factor below 1 -> xG_a decreases
    assert result.xg_a_after < result.xg_a_before
