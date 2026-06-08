"""Sprint 2 validation: 40-match apples-to-apples comparison across 4 models.

Run from project root:
    python scripts/run_sprint2_report.py

Prints a comparison table and a recommendation.
"""

import sys, math
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.match_resolver import resolve_all_matches, ResolvedMatchStats
from src.data.strength_loader import load_strength_params
from src.models.pre_match_xg import calculate_pre_match_xg, BASE_XG as PRE_MATCH_BASE
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import run_backtest, MatchResult
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import DEFAULT_RHO_GRID, RhoResult, select_best_rho
from src.data.loader import load_team_ratings
from src.data.pre_match_loader import PreMatchStats

_ROOT = Path(__file__).parent.parent


def _resolve_wc2022():
    """Return list of ResolvedMatchStats for all 40 WC 2022 matches."""
    hist = pd.read_csv(_ROOT / "data" / "historical_matches.csv")
    mr   = pd.read_csv(_ROOT / "data" / "match_results.csv")
    resolved, unresolved = resolve_all_matches(hist, mr)
    if unresolved:
        print(f"WARNING: {len(unresolved)} matches unresolved:")
        for u in unresolved:
            print(f"  {u}")
    return resolved, hist


def _make_pre_match_stats(r: ResolvedMatchStats, ga: int, gb: int) -> PreMatchStats:
    """Convert a ResolvedMatchStats into a PreMatchStats for calculate_pre_match_xg()."""
    return PreMatchStats(
        match_id=0, date=r.date, team_a=r.team_a, team_b=r.team_b,
        team_a_elo_pre=r.team_a_elo_pre,
        team_b_elo_pre=r.team_b_elo_pre,
        team_a_goals_for_last_10=r.team_a_goals_for_last_10,
        team_a_goals_against_last_10=r.team_a_goals_against_last_10,
        team_b_goals_for_last_10=r.team_b_goals_for_last_10,
        team_b_goals_against_last_10=r.team_b_goals_against_last_10,
        team_a_points_per_game_last_10=r.team_a_points_per_game_last_10,
        team_b_points_per_game_last_10=r.team_b_points_per_game_last_10,
        team_a_matches_available=r.team_a_matches_available,
        team_b_matches_available=r.team_b_matches_available,
        team_a_goals=ga, team_b_goals=gb,
    )


def _make_match_result(r: ResolvedMatchStats, pred, ga: int, gb: int) -> MatchResult:
    """Build a MatchResult from prediction and actuals."""
    if ga > gb: actual = "team_a_win"
    elif ga == gb: actual = "draw"
    else: actual = "team_b_win"
    probs = {"team_a_win": pred.win_a, "draw": pred.draw, "team_b_win": pred.win_b}
    top5 = [(g_a, g_b) for g_a, g_b, _ in pred.top_scorelines]
    return MatchResult(
        date=r.date, team_a=r.team_a, team_b=r.team_b,
        actual_goals_a=ga, actual_goals_b=gb, actual_outcome=actual,
        win_a_prob=pred.win_a, draw_prob=pred.draw, win_b_prob=pred.win_b,
        predicted_outcome=max(probs, key=probs.get),
        top_scorelines=pred.top_scorelines,
        exact_score_hit=len(top5) > 0 and top5[0] == (ga, gb),
        in_top_3=(ga, gb) in top5[:3],
        in_top_5=(ga, gb) in top5,
        prob_of_actual_result=probs[actual],
    )


def run_model2_rolling_stats(resolved, hist):
    """Model 2: Real rolling stats → calculate_pre_match_xg() → Poisson."""
    results = []
    hist_idx = hist.set_index(["date", "team_a", "team_b"])
    for r in resolved:
        key = (r.date, r.team_a, r.team_b)
        row = hist_idx.loc[key]
        ga, gb = int(row["team_a_goals"]), int(row["team_b_goals"])
        pms = _make_pre_match_stats(r, ga, gb)
        xg_a, xg_b = calculate_pre_match_xg(pms)
        pred = predict(r.team_a, r.team_b, xg_a, xg_b)
        results.append(_make_match_result(r, pred, ga, gb))
    return results


def run_model3_mle(resolved, hist, strength_params, model_type="poisson", rho=-0.10):
    """Model 3/4: Real data + MLE strength params → Poisson or Dixon-Coles."""
    results = []
    hist_idx = hist.set_index(["date", "team_a", "team_b"])
    for r in resolved:
        if r.team_a not in strength_params or r.team_b not in strength_params:
            continue
        key = (r.date, r.team_a, r.team_b)
        row = hist_idx.loc[key]
        ga, gb = int(row["team_a_goals"]), int(row["team_b_goals"])
        xg_a, xg_b = calculate_strength_adjusted_xg(
            r.team_a_elo_pre, r.team_b_elo_pre,
            strength_params[r.team_a], strength_params[r.team_b],
            r.team_a_points_per_game_last_10, r.team_b_points_per_game_last_10,
        )
        if model_type == "dixon_coles":
            pred = predict_dixon_coles(r.team_a, r.team_b, xg_a, xg_b, rho=rho)
        else:
            pred = predict(r.team_a, r.team_b, xg_a, xg_b)
        results.append(_make_match_result(r, pred, ga, gb))
    return results


def print_row(label, m, n):
    print(f"  {label:<35s}  {n:>4}  {m.accuracy_1x2:>6.1%}  {m.brier_score:>7.4f}  "
          f"{m.exact_score_accuracy:>6.1%}  {m.top_3_hit_rate:>6.1%}  "
          f"{m.top_5_hit_rate:>6.1%}  {m.avg_prob_actual_result:>6.1%}")


def main():
    print("Loading data...")
    resolved, hist = _resolve_wc2022()
    strength_params = load_strength_params()
    all_ratings = load_team_ratings()

    print(f"  {len(resolved)} / 40 WC 2022 matches resolved")
    print()

    # ── Model 1: Illustrative (team_ratings.csv) ─────────────────────────
    m1_results = run_backtest(ratings=all_ratings)
    m1 = compute_metrics(m1_results)

    # ── Model 2: Real rolling stats → pre_match_xg ───────────────────────
    m2_results = run_model2_rolling_stats(resolved, hist)
    m2 = compute_metrics(m2_results)

    # ── Model 3: Real data + MLE strength → Poisson ───────────────────────
    m3_results = run_model3_mle(resolved, hist, strength_params)
    m3 = compute_metrics(m3_results)

    # ── Model 4: MLE + Dixon-Coles (rho grid) ────────────────────────────
    print("Running rho grid for Model 4...")
    rho_results = []
    for rho in DEFAULT_RHO_GRID:
        dc_results = run_model3_mle(resolved, hist, strength_params, "dixon_coles", rho)
        dc_m = compute_metrics(dc_results)
        rho_results.append(RhoResult(
            rho=rho, accuracy_1x2=dc_m.accuracy_1x2,
            exact_score_accuracy=dc_m.exact_score_accuracy,
            top_3_hit_rate=dc_m.top_3_hit_rate,
            top_5_hit_rate=dc_m.top_5_hit_rate,
            brier_score=dc_m.brier_score,
            avg_prob_actual_result=dc_m.avg_prob_actual_result,
        ))
    best_rho = select_best_rho(rho_results)
    m4_results = run_model3_mle(resolved, hist, strength_params, "dixon_coles", best_rho.rho)
    m4 = compute_metrics(m4_results)

    # ── Print comparison table ────────────────────────────────────────────
    print()
    print("=" * 95)
    print("  SPRINT 2 BACKTEST — WC 2022 (apples-to-apples)")
    print("=" * 95)
    print(f"  {'Model':<35s}  {'N':>4}  {'1X2':>6}  {'Brier':>7}  {'Exact':>6}  "
          f"{'Top3':>6}  {'Top5':>6}  {'AvgP':>6}")
    print("-" * 95)
    print_row("1. Illustrative (AI ratings)", m1, m1.total_matches)
    print_row("2. Real rolling stats (Poisson)", m2, m2.total_matches)
    print_row("3. Real data + MLE (Poisson)", m3, m3.total_matches)
    print_row(f"4. MLE + Dixon-Coles (rho={best_rho.rho:.2f})", m4, m4.total_matches)
    print("=" * 95)
    print()

    # ── Full rho grid ─────────────────────────────────────────────────────
    print("Dixon-Coles rho grid (Model 4 sweep):")
    print(f"  {'rho':>6}  {'Brier':>7}  {'1X2':>6}  {'Top3':>6}  {'Exact':>6}")
    for r in rho_results:
        marker = " <-- best" if r.rho == best_rho.rho else ""
        print(f"  {r.rho:>+6.2f}  {r.brier_score:>7.4f}  {r.accuracy_1x2:>6.1%}  "
              f"{r.top_3_hit_rate:>6.1%}  {r.exact_score_accuracy:>6.1%}{marker}")
    print()

    # ── Recommendation ────────────────────────────────────────────────────
    best_brier = min(m1.brier_score, m2.brier_score, m3.brier_score, m4.brier_score)
    best_label = {
        m1.brier_score: "Model 1 (Illustrative)",
        m2.brier_score: "Model 2 (Real rolling stats)",
        m3.brier_score: "Model 3 (MLE Poisson)",
        m4.brier_score: f"Model 4 (MLE + DC rho={best_rho.rho:.2f})",
    }[best_brier]

    mle_vs_rolling = m2.brier_score - m3.brier_score
    print("Recommendation:")
    print(f"  Best Brier overall: {best_label} ({best_brier:.4f})")
    print(f"  MLE vs rolling stats: {mle_vs_rolling:+.4f} "
          f"({'MLE improves' if mle_vs_rolling > 0 else 'Rolling stats better or equal'})")
    if mle_vs_rolling > 0.005:
        print("  RECOMMEND: Use MLE opponent strength for future development.")
    elif mle_vs_rolling > 0:
        print("  MARGINAL: MLE helps slightly. Continue MLE but investigate calibration.")
    else:
        print("  INCONCLUSIVE: MLE does not improve over rolling stats on this sample.")
        print("    Possible causes: small sample (40 matches), scale of normalization,")
        print("    or xG formula design. Investigate before concluding.")


if __name__ == "__main__":
    main()
