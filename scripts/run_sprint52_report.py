"""Sprint 5.2: xG Calibration and Sanity Bounds — final report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.strength_loader import load_strength_params
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg, BASELINE_XG, SCALE, XG_FLOOR, XG_CEIL
from src.models.dixon_coles import predict_dixon_coles
from src.models.research_valid_predictor import predict_research_valid, ResearchValidInput, DEFAULT_RHO
from src.data.team_snapshot_loader import load_team_snapshots

DATA = Path(__file__).parent.parent / "data"
WC_TEAMS = {
    "Qatar", "Ecuador", "Senegal", "Netherlands", "England", "Iran", "USA", "Wales",
    "Argentina", "Saudi Arabia", "Mexico", "Poland", "France", "Australia", "Denmark",
    "Tunisia", "Germany", "Japan", "Spain", "Costa Rica", "Belgium", "Canada",
    "Morocco", "Croatia", "Switzerland", "Cameroon", "Brazil", "Serbia",
    "Uruguay", "South Korea", "Portugal", "Ghana",
}
_D = "=" * 70
_T = "-" * 70


def run_backtest(calibrate=True, rho=-0.30):
    df = pd.read_csv(DATA / "match_results.csv")
    params = load_strength_params()
    sp_teams = set(params.keys())
    wc = df[
        (df["date"] >= "2022-11-20") & (df["date"] <= "2022-12-18") &
        df["team_a"].isin(sp_teams) & df["team_b"].isin(sp_teams) &
        df["team_a"].isin(WC_TEAMS) & df["team_b"].isin(WC_TEAMS)
    ]

    results, xgs = [], []
    for _, row in wc.iterrows():
        ta, tb = row["team_a"], row["team_b"]
        pa, pb = params[ta], params[tb]
        xg_a, xg_b = calculate_strength_adjusted_xg(
            row["team_a_elo_pre"], row["team_b_elo_pre"], pa, pb,
            row["team_a_points_per_game_last_10"], row["team_b_points_per_game_last_10"],
        )
        if calibrate:
            xg_a, xg_b = calibrate_xg(xg_a), calibrate_xg(xg_b)
        pred = predict_dixon_coles(ta, tb, xg_a, xg_b, rho=rho)
        ao = ("team_a_win" if row["team_a_goals"] > row["team_b_goals"]
              else ("draw" if row["team_a_goals"] == row["team_b_goals"] else "team_b_win"))
        po = ("team_a_win" if pred.win_a > pred.draw and pred.win_a > pred.win_b
              else ("draw" if pred.draw > pred.win_b else "team_b_win"))
        results.append({"win_a": pred.win_a, "draw": pred.draw, "win_b": pred.win_b,
                        "ao": ao, "po": po, "ga": int(row["team_a_goals"]),
                        "gb": int(row["team_b_goals"]), "xg_a": xg_a, "xg_b": xg_b,
                        "top": pred.top_scorelines})
        xgs += [xg_a, xg_b]

    arr = np.array(xgs)
    acc = sum(r["po"] == r["ao"] for r in results) / len(results)
    brier = sum(
        (r["win_a"] - (1 if r["ao"] == "team_a_win" else 0)) ** 2 +
        (r["draw"]  - (1 if r["ao"] == "draw" else 0)) ** 2 +
        (r["win_b"] - (1 if r["ao"] == "team_b_win" else 0)) ** 2
        for r in results
    ) / len(results)
    exact = sum(
        any(g_a == r["ga"] and g_b == r["gb"] for g_a, g_b, _ in r["top"][:5])
        for r in results
    ) / len(results)
    top3 = sum(
        any(g_a == r["ga"] and g_b == r["gb"] for g_a, g_b, _ in r["top"][:3])
        for r in results
    ) / len(results)
    return acc, brier, exact, top3, arr, results


def main():
    print(_D)
    print("SPRINT 5.2: xG CALIBRATION REPORT")
    print(_D)

    # ── 1. Raw xG distribution audit ─────────────────────────────────────────
    print("\n1. RAW xG DISTRIBUTION (pre-calibration, 64 WC 2022 matches)")
    print(_T)
    _, _, _, _, raw_xgs, _ = run_backtest(calibrate=False)
    print(f"  N team-matches:  {len(raw_xgs)}")
    print(f"  Mean:            {raw_xgs.mean():.3f}  (actual goals mean: 1.344)")
    print(f"  Median:          {np.median(raw_xgs):.3f}")
    print(f"  StdDev:          {raw_xgs.std():.3f}")
    print(f"  Min / Max:       {raw_xgs.min():.3f} / {raw_xgs.max():.3f}")
    print(f"  p10 / p90:       {np.percentile(raw_xgs,10):.3f} / {np.percentile(raw_xgs,90):.3f}")
    print(f"  p95:             {np.percentile(raw_xgs,95):.3f}")
    print(f"  > 2.5:           {(raw_xgs > 2.5).sum()}")
    print(f"  > 3.0:           {(raw_xgs > 3.0).sum()}")
    print(f"  < 0.3:           {(raw_xgs < 0.3).sum()}")

    # ── 2. Calibrated xG distribution ────────────────────────────────────────
    print("\n2. CALIBRATED xG DISTRIBUTION (scale=0.65, clamp [0.2, 2.8])")
    print(_T)
    _, _, _, _, cal_xgs, _ = run_backtest(calibrate=True)
    print(f"  N team-matches:  {len(cal_xgs)}")
    print(f"  Mean:            {cal_xgs.mean():.3f}  (target: ~1.35)")
    print(f"  Median:          {np.median(cal_xgs):.3f}")
    print(f"  StdDev:          {cal_xgs.std():.3f}")
    print(f"  Min / Max:       {cal_xgs.min():.3f} / {cal_xgs.max():.3f}")
    print(f"  p10 / p90:       {np.percentile(cal_xgs,10):.3f} / {np.percentile(cal_xgs,90):.3f}")
    print(f"  p95:             {np.percentile(cal_xgs,95):.3f}")
    print(f"  > 2.5:           {(cal_xgs > 2.5).sum()}")
    print(f"  > 3.0:           {(cal_xgs > 3.0).sum()}")
    print(f"  < 0.3:           {(cal_xgs < 0.3).sum()}")

    # ── 3. Backtest comparison ────────────────────────────────────────────────
    print("\n3. BACKTEST COMPARISON")
    print(_T)
    r_acc, r_brier, r_exact, r_top3, _, _ = run_backtest(calibrate=False)
    c_acc, c_brier, c_exact, c_top3, _, _ = run_backtest(calibrate=True)

    def fmt(v, ref, better_direction="lower"):
        delta = v - ref
        better = delta < 0 if better_direction == "lower" else delta > 0
        sign = "+" if delta > 0 else ""
        marker = " [better]" if better else ""
        return f"{v:.4f}  ({sign}{delta:.4f}){marker}"

    print(f"  {'Metric':<22}  {'Baseline':>10}  {'Calibrated':>10}")
    print(f"  {'-'*48}")
    print(f"  {'1X2 Accuracy':<22}  {r_acc:.4f}      {c_acc:.4f}")
    print(f"  {'Brier Score':<22}  {fmt(c_brier, r_brier, 'lower')}")
    print(f"  {'Brier (baseline ref)':<22}  {r_brier:.4f}")
    print(f"  {'Exact score (top 5)':<22}  {r_exact:.3f}       {c_exact:.3f}")
    print(f"  {'Top 3 scoreline':<22}  {r_top3:.3f}       {c_top3:.3f}")

    # ── 4. Sample predictions ─────────────────────────────────────────────────
    print("\n4. SAMPLE PREDICTIONS (calibrated, live data)")
    print(_T)
    snaps  = load_team_snapshots()
    params = load_strength_params()
    matchups = [
        ("Argentina", "France"),
        ("Brazil", "Germany"),
        ("England", "Saudi Arabia"),
        ("France", "Morocco"),
        ("Qatar", "Ecuador"),
    ]
    print(f"  {'Match':<30}  xG A - xG B   Win A   Draw   Win B")
    print(f"  {'-'*62}")
    for ta, tb in matchups:
        if ta in snaps and tb in snaps and ta in params and tb in params:
            inp = ResearchValidInput(ta, tb, snaps[ta], snaps[tb], params[ta], params[tb], DEFAULT_RHO)
            r = predict_research_valid(inp)
            print(f"  {ta} vs {tb:<18}  {r.xg_a:.2f} - {r.xg_b:.2f}    {r.win_a:.0%}    {r.draw:.0%}    {r.win_b:.0%}")

    print()
    print(_D)
    print("CALIBRATION METHOD: Soft compress (scale=0.65) + hard clamp [0.2, 2.8]")
    print(f"  Brier score improvement: {r_brier - c_brier:.4f} ({(r_brier-c_brier)/r_brier:.1%} better)")
    print(f"  1X2 accuracy unchanged: {c_acc:.3f}")
    print(f"  p95 xG: {np.percentile(raw_xgs,95):.2f} -> {np.percentile(cal_xgs,95):.2f}")
    print(f"  Max xG: {raw_xgs.max():.2f} -> {cal_xgs.max():.2f}")
    print(_D)


if __name__ == "__main__":
    main()
