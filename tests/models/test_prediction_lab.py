"""Tests for the Prediction Lab upgrade: config/scenarios, alternative-score
recommendations with reason codes, debug output, and Monte Carlo group sim."""

from src.models.prediction_config import (
    PredictionConfig, DEFAULT_CONFIG, SCENARIOS, get_scenario_config,
)
from src.models.match_simulator import (
    predict_match,
    simulate_match,
    select_score_recommendations,
)
from src.tournament.group_simulation import simulate_group_stage_mc


# ── Config / scenarios ───────────────────────────────────────────────────────

def test_default_config_weights_sum_to_one():
    c = DEFAULT_CONFIG
    assert abs(c.odds_weight + c.fm_weight_with_odds + c.form_weight_with_odds - 1.0) < 1e-9
    assert abs(c.fm_weight_no_odds + c.form_weight_no_odds - 1.0) < 1e-9


def test_get_scenario_config_known_and_unknown():
    assert get_scenario_config("market-heavy").odds_weight > DEFAULT_CONFIG.odds_weight
    assert get_scenario_config("FM-Heavy").fm_weight_with_odds > DEFAULT_CONFIG.fm_weight_with_odds
    assert get_scenario_config("nonsense") is DEFAULT_CONFIG
    assert get_scenario_config(None) is DEFAULT_CONFIG


def test_all_scenarios_run_without_error():
    for name in SCENARIOS:
        r = predict_match("Brazil", "Morocco", scenario=name)
        assert r.scenario == name
        total = r.team1_win_probability + r.draw_probability + r.team2_win_probability
        assert abs(total - 1.0) < 0.02


def test_upset_sensitive_narrows_favourite_edge():
    base = predict_match("Brazil", "Morocco", scenario="balanced")
    upset = predict_match("Brazil", "Morocco", scenario="upset-sensitive")
    assert upset.team1_win_probability < base.team1_win_probability


# ── Recommendation layer (Brazil regression from the spec) ──────────────────

_BRAZIL_SCORES = [
    (1, 1, 0.127), (1, 0, 0.098), (2, 0, 0.098), (2, 1, 0.098), (0, 0, 0.080),
    (0, 1, 0.045), (2, 2, 0.030), (3, 0, 0.028), (3, 1, 0.025), (1, 2, 0.024),
]


def test_brazil_morocco_regression_spec_numbers():
    rec = select_score_recommendations(
        "Brazil", "Morocco", _BRAZIL_SCORES,
        win_a=0.534, draw=0.266, win_b=0.200, xg_a=1.74, xg_b=0.99,
    )
    assert rec.raw_top_score == "1-1"
    a, b = (int(x) for x in rec.recommended_exact_score.split("-"))
    assert a > b  # must be a Brazil-win score
    assert rec.recommended_exact_score == "1-0"
    assert "top5_cluster_favors_favorite" in rec.reason_codes
    assert "best_favorite_score_close_to_raw_top" in rec.reason_codes
    assert "xg_gap_favors_favorite" in rec.reason_codes


def test_true_draw_case_keeps_one_one():
    scores = [(1, 1, 0.15), (1, 0, 0.05), (0, 1, 0.05), (0, 0, 0.08), (2, 2, 0.03)]
    rec = select_score_recommendations(
        "A", "B", scores, win_a=0.36, draw=0.32, win_b=0.32, xg_a=1.12, xg_b=1.05,
    )
    assert rec.recommended_exact_score == "1-1"
    assert "draw_genuinely_likely" in rec.reason_codes


def test_strong_favourite_never_gets_draw():
    scores = [(1, 1, 0.12), (1, 0, 0.08), (2, 0, 0.07), (2, 1, 0.07), (0, 0, 0.06)]
    rec = select_score_recommendations(
        "A", "B", scores, win_a=0.62, draw=0.22, win_b=0.16, xg_a=1.9, xg_b=0.8,
    )
    a, b = (int(x) for x in rec.recommended_exact_score.split("-"))
    assert a > b
    assert "dominant_favorite" in rec.reason_codes


def test_underdog_favourite_symmetric():
    scores = [(1, 1, 0.127), (0, 1, 0.098), (0, 2, 0.098), (1, 2, 0.098), (0, 0, 0.080)]
    rec = select_score_recommendations(
        "A", "B", scores, win_a=0.200, draw=0.266, win_b=0.534, xg_a=0.99, xg_b=1.74,
    )
    a, b = (int(x) for x in rec.recommended_exact_score.split("-"))
    assert b > a  # team B win score
    assert rec.recommended_exact_score == "0-1"


def test_alternative_scores_returned_and_distinct_outcomes():
    rec = select_score_recommendations(
        "Brazil", "Morocco", _BRAZIL_SCORES,
        win_a=0.534, draw=0.266, win_b=0.200, xg_a=1.74, xg_b=0.99,
    )
    assert rec.conservative_exact_score == "1-0"
    # Expressive should match total xG (~2.7) better: a 2-1 type score.
    assert rec.expressive_exact_score == "2-1"
    # Riskier: higher-margin win or upset.
    assert rec.riskier_exact_score in ("2-0", "3-0", "3-1", "1-2", "0-1")


def test_predict_match_exposes_alternatives_and_debug():
    r = predict_match("Brazil", "Morocco")
    assert r.conservative_exact_score
    assert r.expressive_exact_score
    assert r.riskier_exact_score
    assert isinstance(r.reason_codes, list)
    assert "final_xg" in r.debug
    assert "weights" in r.debug
    assert len(r.debug["final_xg"]) == 2


# ── Simulation engine ────────────────────────────────────────────────────────

def test_simulate_match_mismatch_flag_and_scenario():
    r = simulate_match("Brazil", "Morocco", n=20_000, rng_seed=3, scenario="balanced")
    assert r.simulation_matrix_mismatch is False
    assert r.n_simulations == 20_000


def test_simulate_match_seed_reproducible():
    r1 = simulate_match("Brazil", "Morocco", n=5_000, rng_seed=11)
    r2 = simulate_match("Brazil", "Morocco", n=5_000, rng_seed=11)
    assert r1.simulated_team1_win_probability == r2.simulated_team1_win_probability


# ── Monte Carlo group simulation ─────────────────────────────────────────────

def test_simulate_group_stage_mc_structure():
    res = simulate_group_stage_mc(n_runs=20, rng_seed=1)
    assert res.n_runs == 20
    assert len(res.outlooks) == 48
    for o in res.outlooks.values():
        assert 0.0 <= o.qualification_probability <= 1.0
        assert 0.0 <= o.group_winner_probability <= 1.0
        assert 0.0 <= o.avg_points <= 9.0


def test_simulate_group_stage_mc_probabilities_consistent():
    res = simulate_group_stage_mc(n_runs=20, rng_seed=2)
    # Each group has exactly one winner and one runner-up per run.
    by_group: dict[str, float] = {}
    for o in res.outlooks.values():
        by_group[o.group] = by_group.get(o.group, 0.0) + o.group_winner_probability
    for g, total in by_group.items():
        assert abs(total - 1.0) < 1e-9, g
    # 32 of 48 qualify each run.
    total_qual = sum(o.qualification_probability for o in res.outlooks.values())
    assert abs(total_qual - 32.0) < 1e-6
