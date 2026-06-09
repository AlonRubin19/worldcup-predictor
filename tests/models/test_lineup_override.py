"""Tests for the lineup override engine.

TDD: all tests written RED-first before any production code exists.

Coverage:
  - PlayerOverride / LineupOverride dataclasses
  - compute_override_squad_factor
  - apply_lineup_override
  - create_default_lineup
  - Clamping, edge cases, immutability
"""

from __future__ import annotations

import pytest

from src.models.lineup_override import (
    PlayerOverride,
    LineupOverride,
    LineupOverrideResult,
    compute_override_squad_factor,
    apply_lineup_override,
    create_default_lineup,
    SQUAD_FACTOR_MIN,
    SQUAD_FACTOR_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _player(
    name: str = "Player",
    team: str = "Brazil",
    expected_starter: bool = True,
    availability_status: str = "fit",
    availability_factor: float = 1.0,
    form_factor: float = 1.0,
    player_id: str | None = None,
) -> PlayerOverride:
    return PlayerOverride(
        player_id=player_id or name.lower().replace(" ", "_"),
        player_name=name,
        team=team,
        expected_starter=expected_starter,
        availability_status=availability_status,
        availability_factor=availability_factor,
        form_factor=form_factor,
    )


def _default_lineup(team: str = "Brazil") -> LineupOverride:
    return create_default_lineup(team)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass structure
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerOverrideStructure:
    def test_has_required_fields(self):
        p = _player()
        assert hasattr(p, "player_id")
        assert hasattr(p, "player_name")
        assert hasattr(p, "team")
        assert hasattr(p, "expected_starter")
        assert hasattr(p, "availability_status")
        assert hasattr(p, "availability_factor")
        assert hasattr(p, "form_factor")

    def test_availability_status_fit(self):
        p = _player(availability_status="fit")
        assert p.availability_status == "fit"

    def test_availability_status_out(self):
        p = _player(availability_status="out", availability_factor=0.0)
        assert p.availability_status == "out"
        assert p.availability_factor == 0.0

    def test_availability_status_doubtful(self):
        p = _player(availability_status="doubtful", availability_factor=0.7)
        assert p.availability_status == "doubtful"
        assert p.availability_factor == pytest.approx(0.7)

    def test_availability_status_suspended(self):
        p = _player(availability_status="suspended", availability_factor=0.0)
        assert p.availability_status == "suspended"

    def test_form_factor_above_one_is_allowed(self):
        p = _player(form_factor=1.2)
        assert p.form_factor == pytest.approx(1.2)


class TestLineupOverrideStructure:
    def test_has_required_fields(self):
        lo = LineupOverride(team="Brazil", players=[])
        assert hasattr(lo, "team")
        assert hasattr(lo, "players")

    def test_players_list(self):
        players = [_player("Neymar"), _player("Vini")]
        lo = LineupOverride(team="Brazil", players=players)
        assert len(lo.players) == 2


class TestLineupOverrideResultStructure:
    def test_has_required_fields(self):
        result = apply_lineup_override(
            team_a="Brazil", team_b="Serbia",
            xg_a_base=1.8, xg_b_base=0.8,
        )
        for field in [
            "team_a", "team_b",
            "squad_factor_a", "squad_factor_b",
            "xg_a_base", "xg_b_base",
            "xg_a_adjusted", "xg_b_adjusted",
            "win_a_base", "draw_base", "win_b_base",
            "win_a_adjusted", "draw_adjusted", "win_b_adjusted",
            "delta_win_a", "delta_draw", "delta_win_b",
            "delta_xg_a", "delta_xg_b",
            "is_research_valid",
        ]:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_is_always_engineering_valid(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        assert result.is_research_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# compute_override_squad_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOverrideSquadFactor:
    def test_all_fit_returns_one(self):
        players = [_player(f"P{i}", expected_starter=True) for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players)
        assert compute_override_squad_factor(lo) == pytest.approx(1.0)

    def test_no_starters_returns_one(self):
        players = [_player(f"P{i}", expected_starter=False) for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players)
        assert compute_override_squad_factor(lo) == pytest.approx(1.0)

    def test_empty_players_returns_one(self):
        lo = LineupOverride(team="Brazil", players=[])
        assert compute_override_squad_factor(lo) == pytest.approx(1.0)

    def test_one_starter_out_lowers_factor(self):
        """10 fit + 1 out (availability_factor=0.0, still expected_starter) → factor < 1.0."""
        players = [_player(f"P{i}", expected_starter=True) for i in range(10)]
        players.append(_player("Missing", expected_starter=True,
                               availability_status="out", availability_factor=0.0))
        lo = LineupOverride(team="Brazil", players=players)
        factor = compute_override_squad_factor(lo)
        assert factor < 1.0

    def test_factor_below_min_is_clamped(self):
        """All starters out → raw factor 0.0 → should clamp to SQUAD_FACTOR_MIN."""
        players = [_player(f"P{i}", expected_starter=True,
                           availability_status="out", availability_factor=0.0)
                   for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players)
        factor = compute_override_squad_factor(lo)
        assert factor == pytest.approx(SQUAD_FACTOR_MIN)

    def test_above_one_form_increases_factor(self):
        """All starters with form_factor=1.1 → raw factor 1.1 → clamped to SQUAD_FACTOR_MAX at most."""
        players = [_player(f"P{i}", expected_starter=True, form_factor=1.2)
                   for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players)
        factor = compute_override_squad_factor(lo)
        assert factor > 1.0
        assert factor <= SQUAD_FACTOR_MAX

    def test_factor_above_max_is_clamped(self):
        """Extreme form_factor → clamped to SQUAD_FACTOR_MAX."""
        players = [_player(f"P{i}", expected_starter=True, form_factor=5.0)
                   for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players)
        factor = compute_override_squad_factor(lo)
        assert factor == pytest.approx(SQUAD_FACTOR_MAX)

    def test_bench_players_excluded_from_factor(self):
        """Non-starters should not affect squad factor."""
        starters = [_player(f"P{i}", expected_starter=True) for i in range(11)]
        bench = [_player(f"B{i}", expected_starter=False, availability_factor=0.0)
                 for i in range(4)]
        lo = LineupOverride(team="Brazil", players=starters + bench)
        factor = compute_override_squad_factor(lo)
        assert factor == pytest.approx(1.0)

    def test_replace_starter_with_bench_changes_factor(self):
        """Mark original starter as not expected, bench player as expected starter."""
        # 10 fit starters + 1 bench replacement (availability_factor=0.8) as starter
        players = [_player(f"P{i}", expected_starter=True) for i in range(10)]
        replacement = _player("Bench", expected_starter=True,
                               availability_status="fit", availability_factor=0.8, form_factor=0.9)
        players.append(replacement)
        lo = LineupOverride(team="Brazil", players=players)
        factor = compute_override_squad_factor(lo)
        # mean([1.0]*10 + [0.8*0.9]) / 11 < 1.0
        assert factor < 1.0

    def test_squad_factor_constants(self):
        assert SQUAD_FACTOR_MIN == pytest.approx(0.85)
        assert SQUAD_FACTOR_MAX == pytest.approx(1.15)


# ─────────────────────────────────────────────────────────────────────────────
# apply_lineup_override — no overrides (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyLineupOverrideBaseline:
    def test_no_override_squad_factors_are_one(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        assert result.squad_factor_a == pytest.approx(1.0)
        assert result.squad_factor_b == pytest.approx(1.0)

    def test_no_override_xg_unchanged(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        assert result.xg_a_adjusted == pytest.approx(result.xg_a_base)
        assert result.xg_b_adjusted == pytest.approx(result.xg_b_base)

    def test_no_override_deltas_are_zero(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        assert result.delta_win_a == pytest.approx(0.0, abs=1e-9)
        assert result.delta_win_b == pytest.approx(0.0, abs=1e-9)
        assert result.delta_xg_a == pytest.approx(0.0, abs=1e-9)
        assert result.delta_xg_b == pytest.approx(0.0, abs=1e-9)

    def test_no_override_base_probs_sum_to_one(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        total = result.win_a_base + result.draw_base + result.win_b_base
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_no_override_adjusted_probs_sum_to_one(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        total = result.win_a_adjusted + result.draw_adjusted + result.win_b_adjusted
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_team_names_preserved(self):
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8)
        assert result.team_a == "Brazil"
        assert result.team_b == "Serbia"


# ─────────────────────────────────────────────────────────────────────────────
# apply_lineup_override — with overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyLineupOverrideWithOverrides:
    def _starter_out_override(self, team: str) -> LineupOverride:
        players = [_player(f"P{i}", team=team, expected_starter=True) for i in range(10)]
        players.append(_player("Missing", team=team, expected_starter=True,
                               availability_status="out", availability_factor=0.0))
        return LineupOverride(team=team, players=players)

    def test_starter_out_reduces_adjusted_xg_a(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.xg_a_adjusted < result.xg_a_base

    def test_starter_out_reduces_win_a_probability(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.win_a_adjusted < result.win_a_base

    def test_starter_out_increases_squad_factor_reflected_in_xg(self):
        override_b = self._starter_out_override("Serbia")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_b=override_b)
        assert result.xg_b_adjusted < result.xg_b_base
        assert result.squad_factor_b < 1.0

    def test_delta_win_a_negative_when_a_weakened(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.delta_win_a < 0.0

    def test_delta_win_b_positive_when_a_weakened(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.delta_win_b > 0.0

    def test_delta_xg_a_correct(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.delta_xg_a == pytest.approx(
            result.xg_a_adjusted - result.xg_a_base, abs=1e-9
        )

    def test_delta_xg_b_zero_when_only_a_overridden(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        assert result.delta_xg_b == pytest.approx(0.0, abs=1e-9)

    def test_adjusted_probs_sum_to_one_with_override(self):
        override_a = self._starter_out_override("Brazil")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a)
        total = result.win_a_adjusted + result.draw_adjusted + result.win_b_adjusted
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_both_overrides_independent(self):
        override_a = self._starter_out_override("Brazil")
        override_b = self._starter_out_override("Serbia")
        result = apply_lineup_override("Brazil", "Serbia", 1.8, 0.8,
                                       override_a=override_a, override_b=override_b)
        assert result.squad_factor_a < 1.0
        assert result.squad_factor_b < 1.0

    def test_does_not_mutate_input_override(self):
        """Applying the override must not change the input LineupOverride."""
        players = [_player(f"P{i}", team="Brazil", expected_starter=True)
                   for i in range(11)]
        lo = LineupOverride(team="Brazil", players=players[:])
        original_count = len(lo.players)
        apply_lineup_override("Brazil", "Serbia", 1.8, 0.8, override_a=lo)
        assert len(lo.players) == original_count


# ─────────────────────────────────────────────────────────────────────────────
# create_default_lineup
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateDefaultLineup:
    def test_returns_lineup_override(self):
        lo = create_default_lineup("Brazil")
        assert isinstance(lo, LineupOverride)

    def test_team_name_correct(self):
        lo = create_default_lineup("Argentina")
        assert lo.team == "Argentina"

    def test_default_has_eleven_starters(self):
        lo = create_default_lineup("Brazil")
        starters = [p for p in lo.players if p.expected_starter]
        assert len(starters) == 11

    def test_all_default_players_fit(self):
        lo = create_default_lineup("Brazil")
        for p in lo.players:
            assert p.availability_status == "fit"
            assert p.availability_factor == pytest.approx(1.0)

    def test_all_default_form_factors_one(self):
        lo = create_default_lineup("Brazil")
        for p in lo.players:
            assert p.form_factor == pytest.approx(1.0)

    def test_default_squad_factor_is_one(self):
        lo = create_default_lineup("Brazil")
        assert compute_override_squad_factor(lo) == pytest.approx(1.0)

    def test_custom_n_starters(self):
        lo = create_default_lineup("Brazil", n_starters=8)
        assert len(lo.players) == 8
        assert all(p.expected_starter for p in lo.players)

    def test_all_players_belong_to_team(self):
        lo = create_default_lineup("Brazil")
        for p in lo.players:
            assert p.team == "Brazil"

    def test_player_ids_are_unique(self):
        lo = create_default_lineup("Brazil")
        ids = [p.player_id for p in lo.players]
        assert len(set(ids)) == len(ids)
