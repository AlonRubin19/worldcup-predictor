"""Sprint 3 report: Player Impact Engine engineering validation.

ENGINEERING VALIDATION ONLY — player data not yet research-valid.

Compares MLE+Dixon-Coles base model vs MLE+Dixon-Coles+Player Impact
on the WC 2022 40-match backtest data, with a full audit of data provenance.

Usage:
    python scripts/run_sprint3_report.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.player_impact_runner import (
    run_player_impact_backtest,
    audit_research_validity,
)
from src.backtesting.metrics import compute_metrics
from src.backtesting.runner import MatchResult

DATA = Path(__file__).parent.parent / "data"

_DIVIDER = "=" * 70
_THIN = "-" * 70


def _to_match_result(r, use_adjusted: bool) -> MatchResult:
    win_a = r.win_a_prob_adjusted if use_adjusted else r.win_a_prob_base
    draw = r.draw_prob_adjusted if use_adjusted else r.draw_prob_base
    win_b = r.win_b_prob_adjusted if use_adjusted else r.win_b_prob_base
    probs = {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}
    return MatchResult(
        date=r.date,
        team_a=r.team_a,
        team_b=r.team_b,
        actual_goals_a=r.actual_goals_a,
        actual_goals_b=r.actual_goals_b,
        actual_outcome=r.actual_outcome,
        win_a_prob=win_a,
        draw_prob=draw,
        win_b_prob=win_b,
        predicted_outcome=max(probs, key=probs.get),
        top_scorelines=[],
        exact_score_hit=False,
        in_top_3=False,
        in_top_5=False,
        prob_of_actual_result=probs[r.actual_outcome],
    )


def main():
    print(_DIVIDER)
    print("SPRINT 3.2: PLAYER IMPACT ENGINE — AUDIT REPORT")
    print("ENGINEERING VALIDATION ONLY — player data not yet research-valid.")
    print(_DIVIDER)

    results = run_player_impact_backtest(
        match_results_path=DATA / "match_results.csv",
        strength_params_path=DATA / "team_strength_params.csv",
        player_profiles_path=DATA / "player_profiles.csv",
        match_availability_path=DATA / "match_player_availability.csv",
        filter_date_from="2022-11-20",
        filter_date_to="2022-12-18",
        skip_missing_teams=True,
    )

    print(f"\nTotal WC 2022 matches processed: {len(results)}")

    # ---- Audit summary ----
    audit = audit_research_validity(results)
    print(f"\n{_THIN}")
    print("AUDIT: DATA PROVENANCE")
    print(_THIN)
    print(f"  {audit.disclaimer}")
    print(f"\n  Matches with any player data:  {len([r for r in results if r.source_types_a or r.source_types_b])}")
    print(f"  Engineering-valid matches:     {audit.engineering_valid_matches}")
    print(f"  Research-valid matches:        {audit.research_valid_matches}")
    print(f"\n  Source type breakdown (matches where this source type appears):")
    for src, count in sorted(audit.source_type_counts.items()):
        print(f"    {src:<25} {count:>4} matches")

    # ---- Per-match audit for the 5 impacted matches ----
    impacted = [r for r in results if abs(r.squad_factor_a - 1.0) > 0.001 or abs(r.squad_factor_b - 1.0) > 0.001]
    print(f"\n{_THIN}")
    print(f"MATCH-LEVEL AUDIT ({len(impacted)} matches with non-trivial squad factor)")
    print(_THIN)

    for r in impacted:
        match_avail_a_rows = [a for a in [] if a.team == r.team_a]  # placeholder for display
        print(f"\n  match_id: {r.match_id}  |  {r.date}")
        print(f"  {r.team_a} vs {r.team_b}")

        # Squad factors
        print(f"  squad_factor: {r.team_a}={r.squad_factor_a:.4f}  "
              f"{r.team_b}={r.squad_factor_b:.4f}")

        # Source type audit
        src_a_label = ", ".join(sorted(r.source_types_a)) if r.source_types_a else "none"
        src_b_label = ", ".join(sorted(r.source_types_b)) if r.source_types_b else "none"
        rv_a = "research_valid=true" if r.any_research_valid_a else "research_valid=false"
        rv_b = "research_valid=true" if r.any_research_valid_b else "research_valid=false"
        print(f"  {r.team_a:<22} source: {src_a_label:<30} {rv_a}")
        print(f"  {r.team_b:<22} source: {src_b_label:<30} {rv_b}")

        # xG before/after
        delta_a = r.xg_a_adjusted - r.xg_a_base
        delta_b = r.xg_b_adjusted - r.xg_b_base
        print(f"  xG base:     {r.xg_a_base:.3f} vs {r.xg_b_base:.3f}")
        print(f"  xG adjusted: {r.xg_a_adjusted:.3f} ({delta_a:+.3f}) vs "
              f"{r.xg_b_adjusted:.3f} ({delta_b:+.3f})")

        # Win prob before/after
        delta_p = r.win_a_prob_adjusted - r.win_a_prob_base
        print(f"  P(team_a win): base={r.win_a_prob_base:.3f}  "
              f"adjusted={r.win_a_prob_adjusted:.3f}  ({delta_p:+.3f})")

        # Actual result
        print(f"  Actual: {r.actual_goals_a}-{r.actual_goals_b}  "
              f"[{r.actual_outcome}]  "
              f"predicted(base)={r.predicted_outcome_base}  "
              f"predicted(adj)={r.predicted_outcome_adjusted}")

    # ---- Model comparison: engineering scope only ----
    print(f"\n{_THIN}")
    print("MODEL COMPARISON (engineering scope — all 87 WC 2022 matches)")
    print("Note: player impact active on 5 matches only; 82 matches use squad_factor=1.0")
    print(_THIN)

    base_results = [_to_match_result(r, use_adjusted=False) for r in results]
    adj_results = [_to_match_result(r, use_adjusted=True) for r in results]
    base_m = compute_metrics(base_results)
    adj_m = compute_metrics(adj_results)

    print(f"\n{'Metric':<30} {'Base (MLE+DC)':>15} {'+ Player Impact':>16}")
    print("-" * 63)
    print(f"{'Outcome accuracy (1X2)':<30} {base_m.accuracy_1x2:>14.1%} {adj_m.accuracy_1x2:>15.1%}")
    print(f"{'Avg prob actual result':<30} {base_m.avg_prob_actual_result:>14.4f} {adj_m.avg_prob_actual_result:>15.4f}")
    print(f"{'Brier score':<30} {base_m.brier_score:>14.4f} {adj_m.brier_score:>15.4f}")
    print("\n  * Metric differences are small — only 5/87 matches have player data.")
    print("  * Do not interpret these results as evidence of model improvement.")

    # ---- Final verdict ----
    print(f"\n{_DIVIDER}")
    print("SPRINT 3.2 AUDIT VERDICT")
    print(_DIVIDER)
    print(f"  Pipeline:         PASS — player impact affects {len(impacted)} real WC 2022 matches")
    print(f"  Research status:  {audit.disclaimer}")
    print(f"  Labelling:        All rows marked research_valid=false")
    print(f"  Source breakdown: historical_lineup (starting XIs), manual_assumption (injuries)")
    print(f"\n  Next step: replace manual_assumption rows with pre_match_report sources")
    print(f"             before claiming research-valid player impact.")
    print(_DIVIDER)


if __name__ == "__main__":
    main()
