"""Tests for tournament-level calibration functions.

TDD cycle: all tests written RED-first before any production code exists.
"""

import math
import pytest
import numpy as np

from src.tournament.calibration import (
    apply_temperature,
    apply_xg_noise,
    apply_upset_factor,
    compute_concentration_metrics,
    CalibrationParams,
    ConcentrationMetrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# apply_temperature
# ─────────────────────────────────────────────────────────────────────────────

def test_temperature_identity_at_one():
    probs = (0.55, 0.25, 0.20)
    result = apply_temperature(probs, tau=1.0)
    assert abs(result[0] - 0.55) < 1e-9
    assert abs(result[1] - 0.25) < 1e-9
    assert abs(result[2] - 0.20) < 1e-9


def test_temperature_output_sums_to_one():
    for tau in (0.5, 1.0, 1.5, 2.0, 3.0):
        result = apply_temperature((0.60, 0.25, 0.15), tau=tau)
        assert abs(sum(result) - 1.0) < 1e-9, f"failed at tau={tau}"


def test_temperature_above_one_reduces_max_prob():
    probs = (0.70, 0.20, 0.10)
    result = apply_temperature(probs, tau=2.0)
    assert result[0] < probs[0], "tau>1 must flatten — max prob should decrease"


def test_temperature_above_one_increases_min_prob():
    probs = (0.70, 0.20, 0.10)
    result = apply_temperature(probs, tau=2.0)
    assert result[2] > probs[2], "tau>1 must flatten — min prob should increase"


def test_temperature_preserves_ordering():
    probs = (0.60, 0.25, 0.15)
    result = apply_temperature(probs, tau=2.0)
    assert result[0] > result[1] > result[2], "relative order must be preserved"


def test_temperature_below_one_sharpens_distribution():
    probs = (0.60, 0.25, 0.15)
    result = apply_temperature(probs, tau=0.5)
    assert result[0] > probs[0], "tau<1 should sharpen — max prob increases"


def test_temperature_equal_probs_stays_equal():
    probs = (1/3, 1/3, 1/3)
    result = apply_temperature(probs, tau=2.0)
    assert abs(result[0] - 1/3) < 1e-9
    assert abs(result[1] - 1/3) < 1e-9
    assert abs(result[2] - 1/3) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# apply_xg_noise
# ─────────────────────────────────────────────────────────────────────────────

def test_xg_noise_identity_at_zero_sigma():
    rng = np.random.default_rng(0)
    assert apply_xg_noise(1.50, sigma=0.0, rng=rng) == 1.50
    assert apply_xg_noise(0.80, sigma=0.0, rng=rng) == 0.80


def test_xg_noise_produces_positive_values():
    rng = np.random.default_rng(42)
    for _ in range(100):
        result = apply_xg_noise(1.0, sigma=0.4, rng=rng)
        assert result > 0.0, "xG must remain positive after noise"


def test_xg_noise_varies_across_samples():
    rng = np.random.default_rng(7)
    results = [apply_xg_noise(1.0, sigma=0.3, rng=rng) for _ in range(20)]
    assert len(set(results)) > 1, "noise should produce varying outputs"


def test_xg_noise_mean_near_original():
    """Log-normal noise is mean-preserving on expectation."""
    rng = np.random.default_rng(0)
    samples = [apply_xg_noise(1.5, sigma=0.3, rng=rng) for _ in range(2000)]
    mean = sum(samples) / len(samples)
    # E[X * exp(N(0, sigma))] = X * exp(sigma^2/2), so mean will be slightly above
    # but for sigma=0.3 this is exp(0.045)≈1.046, within 5% of original
    assert 1.4 < mean < 1.7, f"mean {mean:.3f} should be near 1.5"


def test_xg_noise_deterministic_with_rng_state():
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    assert apply_xg_noise(1.2, sigma=0.3, rng=rng1) == apply_xg_noise(1.2, sigma=0.3, rng=rng2)


# ─────────────────────────────────────────────────────────────────────────────
# apply_upset_factor
# ─────────────────────────────────────────────────────────────────────────────

def test_upset_factor_identity_at_zero():
    assert apply_upset_factor(0.75, epsilon=0.0) == pytest.approx(0.75)
    assert apply_upset_factor(0.40, epsilon=0.0) == pytest.approx(0.40)


def test_upset_factor_shifts_weak_team_toward_half():
    p = 0.30   # weak team prob
    result = apply_upset_factor(p, epsilon=0.3)
    assert result > p, "upset factor should increase weak team's probability"


def test_upset_factor_reduces_strong_team_prob():
    p = 0.80
    result = apply_upset_factor(p, epsilon=0.3)
    assert result < p, "upset factor should reduce dominant team's probability"


def test_upset_factor_strong_team_still_favoured():
    p = 0.70
    result = apply_upset_factor(p, epsilon=0.5)
    assert result > 0.50, "even with large upset factor, favoured team stays above 50%"


def test_upset_factor_output_is_valid_prob():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        for eps in (0.0, 0.2, 0.5, 1.0):
            result = apply_upset_factor(p, epsilon=eps)
            assert 0.0 <= result <= 1.0, f"p={p} eps={eps} → {result}"


def test_upset_factor_at_one_gives_half():
    result = apply_upset_factor(0.90, epsilon=1.0)
    assert result == pytest.approx(0.50)


# ─────────────────────────────────────────────────────────────────────────────
# compute_concentration_metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_concentration_metrics_returns_correct_type():
    probs = {"A": 0.50, "B": 0.30, "C": 0.15, "D": 0.05}
    m = compute_concentration_metrics(probs)
    assert isinstance(m, ConcentrationMetrics)


def test_concentration_top1_is_highest_prob():
    probs = {"X": 0.10, "Y": 0.60, "Z": 0.30}
    m = compute_concentration_metrics(probs)
    assert m.top1 == pytest.approx(0.60)


def test_concentration_top2_cumulative():
    probs = {"A": 0.50, "B": 0.30, "C": 0.20}
    m = compute_concentration_metrics(probs)
    assert m.top2 == pytest.approx(0.80)


def test_concentration_top5_with_fewer_teams():
    probs = {"A": 0.50, "B": 0.30, "C": 0.20}
    m = compute_concentration_metrics(probs)
    assert m.top5 == pytest.approx(1.0)  # only 3 teams — sums to all


def test_concentration_entropy_uniform_is_max():
    n = 32
    probs = {str(i): 1/n for i in range(n)}
    m = compute_concentration_metrics(probs)
    assert m.entropy == pytest.approx(math.log2(n), rel=1e-6)


def test_concentration_entropy_single_team_is_zero():
    probs = {"only": 1.0}
    m = compute_concentration_metrics(probs)
    assert m.entropy == pytest.approx(0.0, abs=1e-9)


def test_concentration_entropy_decreases_with_concentration():
    spread = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    concentrated = {"A": 0.70, "B": 0.20, "C": 0.07, "D": 0.03}
    m_spread = compute_concentration_metrics(spread)
    m_conc   = compute_concentration_metrics(concentrated)
    assert m_spread.entropy > m_conc.entropy


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationParams defaults
# ─────────────────────────────────────────────────────────────────────────────

def test_calibration_params_defaults_are_identity():
    p = CalibrationParams()
    assert p.temperature == 1.0
    assert p.xg_noise_sigma == 0.0
    assert p.upset_factor == 0.0


def test_calibration_params_default_is_no_op():
    """Default params should produce identity behaviour in all three functions."""
    probs = (0.60, 0.25, 0.15)
    t_result = apply_temperature(probs, tau=CalibrationParams().temperature)
    assert abs(t_result[0] - 0.60) < 1e-9

    rng = np.random.default_rng(0)
    n_result = apply_xg_noise(1.5, sigma=CalibrationParams().xg_noise_sigma, rng=rng)
    assert n_result == 1.5

    u_result = apply_upset_factor(0.75, epsilon=CalibrationParams().upset_factor)
    assert u_result == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: calibrated simulator conservation laws
# ─────────────────────────────────────────────────────────────────────────────

def _run_mc(calib: CalibrationParams, n: int = 200):
    from pathlib import Path
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    from src.tournament.simulator import run_monte_carlo

    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    snaps  = load_team_snapshots()
    params = load_strength_params()
    return run_monte_carlo(fixture_path, snaps, params, n=n, rng_seed=0, calibration=calib)


def test_calibrated_simulator_win_probs_sum_to_one():
    mc = _run_mc(CalibrationParams(temperature=1.5, upset_factor=0.2))
    assert abs(sum(mc.win_tournament.values()) - 1.0) < 0.01


def test_calibrated_simulator_reach_final_sums_to_two():
    mc = _run_mc(CalibrationParams(temperature=1.5))
    assert abs(sum(mc.reach_final.values()) - 2.0) < 0.1


def test_calibrated_simulator_reach_r16_sums_to_sixteen():
    mc = _run_mc(CalibrationParams(upset_factor=0.3))
    assert abs(sum(mc.reach_r16.values()) - 16.0) < 1.0


def test_calibrated_simulator_deterministic_with_seed():
    calib = CalibrationParams(temperature=1.5, xg_noise_sigma=0.2, upset_factor=0.2)
    mc1 = _run_mc(calib, n=100)
    mc2 = _run_mc(calib, n=100)
    assert mc1.win_tournament == mc2.win_tournament


def test_calibrated_simulator_default_params_identical_to_raw():
    """CalibrationParams() must be a strict no-op — same results as no calibration arg."""
    from pathlib import Path
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    from src.tournament.simulator import run_monte_carlo

    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    snaps  = load_team_snapshots()
    params = load_strength_params()

    mc_raw    = run_monte_carlo(fixture_path, snaps, params, n=100, rng_seed=5)
    mc_default = run_monte_carlo(fixture_path, snaps, params, n=100, rng_seed=5,
                                  calibration=CalibrationParams())
    assert mc_raw.win_tournament == mc_default.win_tournament


def test_calibrated_favourites_rank_above_weak_teams():
    """Even with calibration, strong teams (Arg/Brazil) should still lead."""
    mc = _run_mc(CalibrationParams(temperature=1.5, upset_factor=0.2), n=500)
    win_sorted = sorted(mc.win_tournament.items(), key=lambda x: -x[1])
    top3 = {t for t, _ in win_sorted[:3]}
    assert "Argentina" in top3 or "Brazil" in top3, (
        f"Expected at least one of Argentina/Brazil in top-3, got {top3}"
    )


def test_temperature_reduces_top2_concentration():
    """Tournament top-2 concentration should drop with temperature > 1."""
    mc_raw  = _run_mc(CalibrationParams(temperature=1.0), n=1000)
    mc_hot  = _run_mc(CalibrationParams(temperature=2.0), n=1000)

    raw_top2 = sum(sorted(mc_raw.win_tournament.values(), reverse=True)[:2])
    hot_top2 = sum(sorted(mc_hot.win_tournament.values(), reverse=True)[:2])
    assert hot_top2 < raw_top2, (
        f"temperature=2.0 top-2={hot_top2:.1%} should be < raw top-2={raw_top2:.1%}"
    )
