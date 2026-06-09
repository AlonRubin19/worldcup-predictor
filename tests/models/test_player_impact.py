"""Tests for player_impact — squad factor calculation and xG modification."""

import pytest
from src.models.player_impact import (
    calculate_player_match_impact,
    calculate_squad_factor,
    apply_player_impact,
    PlayerImpactInput,
)
from src.data.player_loader import PlayerProfile, PlayerAvailability


def _profile(player_id, team, base_impact_score):
    return PlayerProfile(
        player_id=player_id,
        player_name=f"Player {player_id}",
        team=team,
        position="MF",
        club="Club",
        minutes_last_90_days=810,
        national_team_minutes_last_12_months=720,
        goals_per_90=0.3,
        assists_per_90=0.2,
        xg_per_90=0.28,
        xa_per_90=0.18,
        defensive_actions_per_90=3.0,
        international_caps=40,
        base_impact_score=base_impact_score,
    )


def _avail(player_id, team, expected_starter, availability_factor, form_factor):
    return PlayerAvailability(
        match_id="m001",
        date="2022-11-21",
        team=team,
        player_id=player_id,
        expected_starter=expected_starter,
        availability_status="fit",
        availability_factor=availability_factor,
        form_factor=form_factor,
    )


class TestCalculatePlayerMatchImpact:
    def test_fit_player_full_impact(self):
        profile = _profile("p1", "England", 1.20)
        avail = _avail("p1", "England", True, 1.0, 1.0)
        impact = calculate_player_match_impact(profile, avail)
        assert impact == pytest.approx(1.20)

    def test_doubtful_player_reduced_impact(self):
        profile = _profile("p1", "England", 1.20)
        avail = _avail("p1", "England", True, 0.6, 1.0)
        impact = calculate_player_match_impact(profile, avail)
        assert impact == pytest.approx(0.72)

    def test_out_player_zero_impact(self):
        profile = _profile("p1", "England", 1.20)
        avail = _avail("p1", "England", True, 0.0, 1.0)
        impact = calculate_player_match_impact(profile, avail)
        assert impact == pytest.approx(0.0)

    def test_form_factor_multiplies_impact(self):
        profile = _profile("p1", "England", 1.0)
        avail = _avail("p1", "England", True, 1.0, 1.10)
        impact = calculate_player_match_impact(profile, avail)
        assert impact == pytest.approx(1.10)


class TestCalculateSquadFactor:
    def _build_full_squad(self, team, base_score=1.0, avail_factor=1.0, form_factor=1.0):
        """Build 11 fit starters all with the same base_impact_score."""
        profiles = {f"p{i}": _profile(f"p{i}", team, base_score) for i in range(11)}
        availability = [
            _avail(f"p{i}", team, True, avail_factor, form_factor)
            for i in range(11)
        ]
        return profiles, availability

    def test_all_fit_starters_gives_factor_one(self):
        profiles, availability = self._build_full_squad("England", base_score=1.0)
        factor = calculate_squad_factor("England", profiles, availability)
        assert factor == pytest.approx(1.0)

    def test_all_starters_out_gives_minimum_factor(self):
        profiles, availability = self._build_full_squad("England", avail_factor=0.0)
        factor = calculate_squad_factor("England", profiles, availability)
        assert factor == pytest.approx(0.85)  # clamped at minimum

    def test_strong_form_gives_maximum_factor(self):
        # form_factor high enough to push above 1.15 ceiling
        profiles, availability = self._build_full_squad("England", form_factor=2.0)
        factor = calculate_squad_factor("England", profiles, availability)
        assert factor == pytest.approx(1.15)  # clamped at maximum

    def test_key_player_out_reduces_factor(self):
        """Losing a star player (high base_impact_score) should reduce the squad factor."""
        profiles = {
            "star": _profile("star", "England", 2.0),
            **{f"p{i}": _profile(f"p{i}", "England", 1.0) for i in range(10)},
        }
        # All fit baseline
        avail_all_fit = [_avail("star", "England", True, 1.0, 1.0)] + [
            _avail(f"p{i}", "England", True, 1.0, 1.0) for i in range(10)
        ]
        # Star is out
        avail_star_out = [_avail("star", "England", True, 0.0, 1.0)] + [
            _avail(f"p{i}", "England", True, 1.0, 1.0) for i in range(10)
        ]

        factor_all_fit = calculate_squad_factor("England", profiles, avail_all_fit)
        factor_star_out = calculate_squad_factor("England", profiles, avail_star_out)

        assert factor_all_fit == pytest.approx(1.0)
        assert factor_star_out < factor_all_fit

    def test_no_availability_data_for_team_gives_factor_one(self):
        """If no availability data for a team, squad factor defaults to 1.0."""
        profiles = {}
        availability = []
        factor = calculate_squad_factor("England", profiles, availability)
        assert factor == pytest.approx(1.0)


class TestApplyPlayerImpact:
    def test_factor_one_leaves_xg_unchanged(self):
        inp = PlayerImpactInput(xg_a=1.5, xg_b=1.2, squad_factor_a=1.0, squad_factor_b=1.0)
        xg_a, xg_b = apply_player_impact(inp)
        assert xg_a == pytest.approx(1.5)
        assert xg_b == pytest.approx(1.2)

    def test_reduced_squad_lowers_xg(self):
        inp = PlayerImpactInput(xg_a=1.5, xg_b=1.2, squad_factor_a=0.90, squad_factor_b=1.0)
        xg_a, xg_b = apply_player_impact(inp)
        assert xg_a == pytest.approx(1.35)
        assert xg_b == pytest.approx(1.2)

    def test_strong_squad_raises_xg(self):
        inp = PlayerImpactInput(xg_a=1.5, xg_b=1.2, squad_factor_a=1.0, squad_factor_b=1.10)
        xg_a, xg_b = apply_player_impact(inp)
        assert xg_a == pytest.approx(1.5)
        assert xg_b == pytest.approx(1.32)

    def test_both_teams_adjusted_independently(self):
        inp = PlayerImpactInput(xg_a=1.0, xg_b=1.0, squad_factor_a=0.90, squad_factor_b=1.10)
        xg_a, xg_b = apply_player_impact(inp)
        assert xg_a == pytest.approx(0.90)
        assert xg_b == pytest.approx(1.10)
