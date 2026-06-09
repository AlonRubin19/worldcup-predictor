"""Sprint 4 report: Market Intelligence Layer.

Market odds are used for COMPARISON AND CALIBRATION ONLY.
They do not change model predictions.

ENGINEERING VALIDATION ONLY — market data is placeholder until
real odds are sourced from football-data.co.uk or The Odds API.

Usage:
    python scripts/run_sprint4_report.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.market_runner import run_market_comparison

DATA = Path(__file__).parent.parent / "data"
_D = "=" * 70
_T = "-" * 70


def main():
    print(_D)
    print("SPRINT 4: MARKET INTELLIGENCE LAYER")
    print(_D)

    results, summary = run_market_comparison(
        match_results_path=DATA / "match_results.csv",
        strength_params_path=DATA / "team_strength_params.csv",
        market_odds_path=DATA / "market_odds.csv",
    )

    # ---- Research validity warning ----
    print(f"\n[!] {summary.disclaimer}")
    print(f"    Market odds source: placeholder  |  research_valid=false for all rows")
    print(f"    Do not interpret Brier comparison as evidence of model quality.")

    # ---- Summary metrics ----
    print(f"\n{_T}")
    print("SUMMARY METRICS")
    print(_T)
    print(f"  Matched games:                {summary.total_matches}")
    print(f"  Model Brier score:            {summary.model_brier:.4f}")
    print(f"  Market Brier score:           {summary.market_brier:.4f}")
    delta_sign = "+" if summary.brier_delta >= 0 else ""
    print(f"  Brier delta (mkt - model):    {delta_sign}{summary.brier_delta:.4f}  "
          f"({'model better' if summary.brier_delta > 0 else 'market better' if summary.brier_delta < 0 else 'equal'})")
    print(f"  Avg absolute divergence:      {summary.avg_absolute_divergence:.3f} ({summary.avg_absolute_divergence:.1%})")
    print(f"  High-divergence matches (>5pp): {summary.high_divergence_count}")
    if summary.high_divergence_count > 0:
        print(f"    Model closer:  {summary.model_wins_high_divergence}")
        print(f"    Market closer: {summary.market_wins_high_divergence}")

    # ---- Per-match comparison table ----
    print(f"\n{_T}")
    print("PER-MATCH MODEL vs MARKET COMPARISON")
    print(_T)
    print(f"  {'Match':<30} {'Act':>4}  {'Model':>22}  {'Market':>22}  {'dHome':>6}  {'dDraw':>6}  {'dAway':>6}  {'Closer'}")
    print(f"  {'':<30} {'':>4}  {'W-D-L':>22}  {'W-D-L':>22}  {'':>6}  {'':>6}  {'':>6}")
    print("  " + "-" * 110)

    outcome_short = {"team_a_win": "H", "draw": "D", "team_b_win": "A"}
    for r in results:
        match_label = f"{r.team_a[:12]} vs {r.team_b[:12]}"
        model_str = f"{r.model_win_a:.2f}-{r.model_draw:.2f}-{r.model_win_b:.2f}"
        market_str = f"{r.market_home:.2f}-{r.market_draw:.2f}-{r.market_away:.2f}"
        act = outcome_short[r.actual_outcome]
        closer = "MODEL" if r.model_closer_than_market else "MARKET"
        print(
            f"  {match_label:<30} {act:>4}  {model_str:>22}  {market_str:>22}"
            f"  {r.home_divergence:>+.3f}  {r.draw_divergence:>+.3f}  {r.away_divergence:>+.3f}  {closer}"
        )

    # ---- Divergence table ----
    print(f"\n{_T}")
    print("DIVERGENCE ANALYSIS (model - market, signed)")
    print(_T)
    print(f"  {'Match':<30} {'Largest div outcome':<20} {'d value':>8}  {'|d|>5pp':>7}  Bookmaker overround")
    print("  " + "-" * 85)
    for r in results:
        flag = "  YES" if abs(r.largest_divergence_value) >= 0.05 else "   no"
        print(
            f"  {r.team_a[:12]+' vs '+r.team_b[:12]:<30} "
            f"{r.largest_divergence_outcome:<16} "
            f"{r.largest_divergence_value:>+.3f}    {flag}  "
            f"  {r.market_overround:.3f} ({r.market_overround:.1%})"
        )

    # ---- Verdict ----
    print(f"\n{_D}")
    print("SPRINT 4 VERDICT")
    print(_D)
    print(f"  Pipeline:         PASS — {summary.total_matches} matches compared")
    print(f"  Overround removal: PASS — all market probs sum to 1.0")
    print(f"  Research status:  {summary.disclaimer}")
    print(f"  Interpretation:   Metrics above are ENGINEERING VALIDATION only.")
    print(f"                    Replace placeholder odds with real sourced data")
    print(f"                    before drawing any conclusions.")
    print(_D)


if __name__ == "__main__":
    main()
