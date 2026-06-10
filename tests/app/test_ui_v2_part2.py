"""Smoke tests for UI/UX V2 Part 2 (Match Analyzer + Golden Boot redesign).

These tests exercise the underlying data/computation that the redesigned
Streamlit UI renders, without requiring a running Streamlit server. They
verify that the building blocks the new layout depends on behave correctly:
selected-match persistence, market-blend "no odds" fallback, tournament/
golden-boot result persistence shape, and golden boot top-N rendering data.
"""

from __future__ import annotations

from src.app.selected_fixture import SelectedFixture, is_valid_selected_fixture
from src.models.market_blend import blend_probabilities, MODEL_ONLY_LABEL, BLEND_LABEL
from src.data.team_snapshot_loader import load_team_snapshots
from src.data.strength_loader import load_strength_params
from src.data.player_loader import load_player_profiles
from src.tournament.fixtures import _DEFAULT as FIXTURE_PATH
from src.tournament.simulator import run_monte_carlo
from src.models.golden_boot import predict_golden_boot, expected_team_matches


def test_selected_match_persists_as_valid_fixture():
    fix = SelectedFixture(
        fixture_id="123", source_type="api",
        team_a="Spain", team_b="France", date="2026-06-15",
        stage="group", group="A",
    )
    assert is_valid_selected_fixture(fix)
    assert fix.team_a == "Spain" and fix.team_b == "France"


def test_market_blend_inactive_badge_when_no_odds():
    result = blend_probabilities(
        model_win_a=0.5, model_draw=0.3, model_win_b=0.2,
        market_win_a=None, market_draw=None, market_win_b=None,
        market_research_valid=False,
    )
    assert result.used_market is False
    assert result.label == MODEL_ONLY_LABEL
    # raw model probabilities are preserved unchanged
    assert (result.win_a, result.draw, result.win_b) == (0.5, 0.3, 0.2)


def test_market_blend_active_label_when_odds_present():
    result = blend_probabilities(
        model_win_a=0.5, model_draw=0.3, model_win_b=0.2,
        market_win_a=0.45, market_draw=0.30, market_win_b=0.25,
        market_research_valid=True,
    )
    assert result.used_market is True
    assert result.label == BLEND_LABEL


def test_match_analyzer_summary_can_render_for_two_teams():
    """The hero card needs win/draw/win + xG + most-likely-score for any
    two-team matchup -- verify the underlying prediction pipeline produces
    these fields without error for a sample WC2026 matchup."""
    from src.app.prediction_runner import run_full_prediction, RunnerInput

    snaps = load_team_snapshots()
    params = load_strength_params()
    inp = RunnerInput(
        team_a="Spain", team_b="France",
        snapshot_a=snaps.get("Spain"), snapshot_b=snaps.get("France"),
        params_a=params.get("Spain"), params_b=params.get("France"),
    )
    full = run_full_prediction(inp)

    assert 0.0 <= full.win_a <= 1.0
    assert 0.0 <= full.draw <= 1.0
    assert 0.0 <= full.win_b <= 1.0
    assert abs(full.win_a + full.draw + full.win_b - 1.0) < 1e-6
    assert full.most_likely_score
    assert full.confidence.label in ("High", "Medium", "Low")
    # blend defaults to model-only when no market odds supplied
    assert full.blend.used_market is False


def test_golden_boot_top_cards_can_render():
    """Top-5 card data: player name, team, prob_top_scorer, expected_goals,
    and expected_team_matches must all be available for the cached MC result."""
    snaps = load_team_snapshots()
    params = load_strength_params()
    profiles = load_player_profiles()
    mc = run_monte_carlo(FIXTURE_PATH, snaps, params, n=200, rng_seed=42)

    results = predict_golden_boot(profiles, mc, n_sims=500, rng_seed=42)
    assert results

    top5 = results[:5]
    for r in top5:
        assert r.player_name
        assert r.team
        assert r.expected_goals >= 0
        assert 0.0 <= r.prob_top_scorer <= 1.0
        # expected_team_matches must be computable for every top-5 player's team
        matches = expected_team_matches(r.team, mc)
        assert matches >= 3.0  # at least the 3 group-stage matches


def test_golden_boot_validity_flag_distinguishes_research_valid_players():
    profiles = load_player_profiles()
    valid = [p for p in profiles.values() if p.research_valid]
    placeholder = [p for p in profiles.values() if not p.research_valid]
    # both kinds exist in the current dataset -- the UI must show both badges
    assert valid
    assert placeholder
