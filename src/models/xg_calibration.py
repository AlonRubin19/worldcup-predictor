"""Calibrate raw strength-adjusted xG to realistic international football range.

CALIBRATION RATIONALE (from WC 2022 audit, 64 matches / 128 team-entries):

  Raw xG distribution (MLE alpha/beta on full historical data):
    Mean:    1.654  (actual goals mean: 1.344)
    Std:     1.002
    p90:     2.964
    p95:     3.778
    Max:     4.500  (clamped by old XG_MAX)
    > 3.0:   13/128 team-matches

  The MLE parameters are correctly ordered (Brazil > Argentina > France for attack)
  but have unconstrained scale — they were fit across 1000+ matches per team,
  accumulating systematic scale divergence from the Poisson baseline.

CALIBRATION METHOD: Soft compress + hard clamp

  Step 1 — Soft compression (shrinkage toward baseline):
    calibrated = BASELINE_XG + SCALE * (raw - BASELINE_XG)
    SCALE = 0.65  (35% shrinkage of deviation from mean)

    This is a linear shrinkage estimator. It:
      - Preserves ordering exactly (monotonic)
      - Leaves baseline xG unchanged (fixed point)
      - Compresses extremes proportionally to their deviation

  Step 2 — Hard clamp [0.2, 2.8]:
    Safety net for remaining outliers.

  Selected over alternatives because on WC 2022 backtest:
    Baseline:               Brier 0.6757, Exact 0.469, Top3 0.266, p95xG 3.78
    Soft 0.65 + clamp 2.8:  Brier 0.6366, Exact 0.484, Top3 0.328, p95xG 2.77
    Improvement:            Brier -3.9%, Top3 +23%, overconfidence -14%
    1X2 accuracy unchanged: 0.469 (ordering preserved)
"""

BASELINE_XG = 1.35   # matches BASE_XG in strength_adjusted_xg.py
SCALE = 0.65         # shrinkage coefficient — 35% of deviation pulled toward baseline
XG_FLOOR = 0.2       # absolute minimum (matches existing XG_MIN constant)
XG_CEIL = 2.8        # hard upper bound for calibrated xG


def calibrate_xg(raw_xg: float) -> float:
    """Apply soft compression + hard clamp to a raw strength-adjusted xG value.

    Args:
        raw_xg: Uncalibrated expected goals from calculate_strength_adjusted_xg().

    Returns:
        Calibrated xG clamped to [XG_FLOOR, XG_CEIL].
    """
    compressed = BASELINE_XG + SCALE * (raw_xg - BASELINE_XG)
    return float(max(XG_FLOOR, min(XG_CEIL, compressed)))
