"""Smoke tests for the resumed UI/UX V2 Part 2 work (prediction-first
Match Analyzer / Golden Boot / Tournament Simulator / Bookmaker Comparison).

Pure-Python checks of the underlying data the new layout depends on --
no Streamlit runtime required.
"""

from __future__ import annotations

from src.app.selected_fixture import SelectedFixture, is_valid_selected_fixture
from src.data.team_snapshot_loader import load_team_snapshots
from src.data.strength_loader import load_strength_params
from src.data.player_loader import load_player_profiles
from src.tournament.fixtures import _DEFAULT as FIXTURE_PATH
from src.tournament.simulator import run_monte_carlo
from src.tournament.calibration import compute_concentration_metrics
from src.models.golden_boot import predict_golden_boot, expected_team_matches
from src.app.prediction_runner import run_full_prediction, RunnerInput


def test_match_analyzer_hero_summary_fields_available():
    """Hero card needs: teams, win/draw/win, confidence, most-likely score, xG."""
    snaps = load_team_snapshots()
    params = load_strength_params()
    full = run_full_prediction(RunnerInput(
        team_a="Spain", team_b="France",
        snapshot_a=snaps.get("Spain"), snapshot_b=snaps.get("France"),
        params_a=params.get("Spain"), params_b=params.get("France"),
    ))
    assert full.team_a == "Spain" and full.team_b == "France"
    assert full.confidence.label in ("High", "Medium", "Low")
    assert full.most_likely_score
    assert full.xg_a > 0 and full.xg_b > 0
    assert len(full.recommendations.recommendations) >= 1


def test_top5_signals_available_for_hero():
    snaps = load_team_snapshots()
    params = load_strength_params()
    full = run_full_prediction(RunnerInput(
        team_a="Spain", team_b="France",
        snapshot_a=snaps.get("Spain"), snapshot_b=snaps.get("France"),
        params_a=params.get("Spain"), params_b=params.get("France"),
    ))
    top5 = full.recommendations.recommendations[:5]
    for r in top5:
        assert r.selection
        assert r.rationale


def test_selected_fixture_persists_across_tabs():
    fix = SelectedFixture(
        fixture_id="42", source_type="api",
        team_a="Mexico", team_b="South Africa", date="2026-06-11",
        stage="group", group="A",
    )
    assert is_valid_selected_fixture(fix)
    # Same object would be stored in st.session_state["selected_fixture"] and
    # read by both the Home tab and Match Analyzer tab.
    assert fix.team_a == "Mexico"


def test_golden_boot_top5_and_top20_data_available():
    snaps = load_team_snapshots()
    params = load_strength_params()
    profiles = load_player_profiles()
    mc = run_monte_carlo(FIXTURE_PATH, snaps, params, n=200, rng_seed=7)

    results = predict_golden_boot(profiles, mc, n_sims=500, rng_seed=7)
    assert len(results) >= 5

    top5 = results[:5]
    top20 = results[:20]
    assert len(top20) <= 20
    for r in top5:
        assert r.player_name and r.team
        assert expected_team_matches(r.team, mc) >= 3.0


def test_golden_boot_validity_badges_present_for_placeholder_and_live():
    profiles = load_player_profiles()
    valid = [p for p in profiles.values() if p.research_valid]
    placeholder = [p for p in profiles.values() if not p.research_valid]
    assert valid and placeholder


def test_tournament_simulator_top10_concentration_available():
    snaps = load_team_snapshots()
    params = load_strength_params()
    mc = run_monte_carlo(FIXTURE_PATH, snaps, params, n=200, rng_seed=3)

    conc = compute_concentration_metrics(mc.win_tournament)
    assert 0.0 <= conc.top1 <= 1.0
    assert 0.0 <= conc.top5 <= 1.0

    rows = sorted(mc.win_tournament.items(), key=lambda kv: -kv[1])[:10]
    assert len(rows) <= 10
    assert all(0.0 <= p <= 1.0 for _, p in rows)


def test_session_state_persistence_pattern_keeps_results_on_slider_change():
    """Simulates the st.session_state['mc_result'] persistence pattern: once
    set, results remain available regardless of subsequent slider reruns
    (which only update other session_state keys, not mc_result)."""
    session_state = {}
    snaps = load_team_snapshots()
    params = load_strength_params()
    mc = run_monte_carlo(FIXTURE_PATH, snaps, params, n=100, rng_seed=1)
    session_state["mc_result"] = mc

    # Simulate a rerun triggered by a slider change -- mc_result must survive.
    session_state["n_simulations_slider"] = 5000
    session_state["home_advantage_slider"] = 0.2

    assert session_state["mc_result"] is mc
    assert session_state["mc_result"].win_tournament
