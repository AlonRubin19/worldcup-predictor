#!/usr/bin/env python
"""Prediction Consistency Audit.

Loads live WC 2026 fixtures and verifies that every prediction is
internally consistent:
  - 1X2 probabilities sum to 100%
  - Most likely score == top_scorelines[0]
  - Top 5 scorelines sorted descending by probability
  - Over 2.5 + Under 2.5 == 100%
  - BTTS Yes + BTTS No == 100%
  - Top signal is drawn from the same markets object shown in UI
  - Anomaly flags (high BTTS but top score 2-0, draw most likely but >60% win, etc.)

Usage:
    python scripts/run_prediction_consistency_audit.py
    python scripts/run_prediction_consistency_audit.py --n 72 --skip-api
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.live_fixture_loader import fetch_upcoming_fixtures
from src.data.team_snapshot_loader import load_team_snapshots, TeamSnapshot
from src.data.strength_loader import load_strength_params, StrengthParams
from src.app.prediction_runner import RunnerInput, run_full_prediction


_SNAP_DEF = TeamSnapshot(elo=1800.0, ppg=1.5)
_PAR_DEF  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)

TOLERANCE = 0.001


def sep(n=70): print("=" * n)


def check_one(team_a, team_b, fixture_id, snaps, params, issues):
    """Run prediction for one match and collect any consistency issues."""
    snap_a = snaps.get(team_a, _SNAP_DEF)
    snap_b = snaps.get(team_b, _SNAP_DEF)
    par_a  = params.get(team_a, _PAR_DEF)
    par_b  = params.get(team_b, _PAR_DEF)

    inp = RunnerInput(team_a, team_b, snap_a, snap_b, par_a, par_b)
    r   = run_full_prediction(inp)

    local_issues = []

    # 1. 1X2 sum
    total = r.win_a + r.draw + r.win_b
    if abs(total - 1.0) > TOLERANCE:
        local_issues.append(f"1X2 sum = {total:.6f} (expected 1.0)")

    # 2. Most likely score matches top_scorelines[0]
    if r.top_scorelines:
        expected = f"{r.top_scorelines[0][0]}-{r.top_scorelines[0][1]}"
        if r.most_likely_score != expected:
            local_issues.append(
                f"most_likely_score='{r.most_likely_score}' but top_scorelines[0]='{expected}'"
            )

    # 3. Scorelines sorted descending
    probs = [s[2] for s in r.top_scorelines]
    if probs != sorted(probs, reverse=True):
        local_issues.append(f"Scorelines not sorted: {probs[:5]}")

    # 4. Over/Under complement
    over_prob  = next((m.probability for m in r.markets.over_under if "Over 2.5" in m.selection), None)
    under_prob = next((m.probability for m in r.markets.over_under if "Under 2.5" in m.selection), None)
    if over_prob is not None and under_prob is not None:
        if abs(over_prob + under_prob - 1.0) > TOLERANCE:
            local_issues.append(f"Over+Under = {over_prob+under_prob:.6f}")

    # 5. BTTS complement
    btts_yes = next((m.probability for m in r.markets.btts if m.selection == "BTTS Yes"), None)
    btts_no  = next((m.probability for m in r.markets.btts if m.selection == "BTTS No"),  None)
    if btts_yes is not None and btts_no is not None:
        if abs(btts_yes + btts_no - 1.0) > TOLERANCE:
            local_issues.append(f"BTTS Yes+No = {btts_yes+btts_no:.6f}")

    # 6. Top signal is from same markets object
    # NOTE: includes one_x_two — dominant teams (>70% win) surface 1X2 as top signal
    if r.recommendations.recommendations:
        top_prob = r.recommendations.recommendations[0].model_probability
        all_market_probs = {
            m.probability
            for lst in [
                r.markets.one_x_two,
                r.markets.over_under, r.markets.btts,
                r.markets.double_chance, r.markets.draw_no_bet,
                r.markets.team_totals, r.markets.clean_sheet,
            ]
            for m in lst
        }
        if not any(abs(p - top_prob) < 0.0001 for p in all_market_probs):
            local_issues.append(f"Top signal prob {top_prob:.4f} not found in markets")

    # ── Anomaly flags (not bugs, but worth logging) ───────────────────────────
    anomalies = []

    # Draw most likely but big win-prob gap
    if r.top_scorelines and r.top_scorelines[0][0] == r.top_scorelines[0][1]:
        if max(r.win_a, r.win_b) > 0.60:
            winner = team_a if r.win_a > r.win_b else team_b
            anomalies.append(
                f"Draw is most likely score ({r.most_likely_score}) "
                f"but {winner} has {max(r.win_a, r.win_b):.1%} win prob "
                f"(DC tau correction boosting draw scorelines — mathematically valid)"
            )

    # High BTTS but top score is a clean sheet
    top_score = r.most_likely_score
    if btts_yes and btts_yes > 0.65:
        g_a, g_b = [int(x) for x in top_score.split("-")]
        if g_a == 0 or g_b == 0:
            anomalies.append(
                f"BTTS Yes={btts_yes:.1%} but top score is {top_score} (clean sheet)"
            )

    # Over 2.5 high but top score has ≤ 2 goals
    if over_prob and over_prob > 0.65:
        g_a, g_b = [int(x) for x in top_score.split("-")]
        if g_a + g_b <= 2:
            anomalies.append(
                f"Over 2.5={over_prob:.1%} but top score {top_score} has {g_a+g_b} goals"
            )

    return r, local_issues, anomalies


def main():
    parser = argparse.ArgumentParser(description="Prediction consistency audit")
    parser.add_argument("--n",        type=int, default=20, help="Max fixtures to audit")
    parser.add_argument("--skip-api", action="store_true", help="Use CSV fixtures only")
    parser.add_argument("--season",   type=int, default=2026)
    parser.add_argument("--league",   type=int, default=1)
    args = parser.parse_args()

    sep()
    print("  World Cup 2026 — Prediction Consistency Audit")
    sep()

    # Load model data
    print("\nLoading model data...")
    try:
        snaps  = load_team_snapshots()
        params = load_strength_params()
        print(f"  ELO snapshots: {len(snaps)} teams")
        print(f"  Strength params: {len(params)} teams")
    except FileNotFoundError as e:
        print(f"  WARN: {e} — using default params")
        snaps, params = {}, {}

    # Load fixtures
    fixtures = []
    if not args.skip_api:
        api_key = os.environ.get("API_FOOTBALL_KEY", "")
        if api_key:
            try:
                client = ApiFootballClient(
                    api_key=api_key,
                    cache_dir=str(_ROOT / "data" / "api_cache"),
                )
                fixtures = fetch_upcoming_fixtures(
                    client, league_id=args.league, season=args.season, ttl_seconds=0
                )
                print(f"  Live API: {len(fixtures)} upcoming fixtures")
            except Exception as e:
                print(f"  API error: {e}")
        else:
            print("  No API_FOOTBALL_KEY — using CSV")

    if not fixtures:
        from src.data.fixture_provider import get_fixtures, FixtureSource
        prov = get_fixtures(FixtureSource.CSV)
        # Convert to minimal fixture-like objects for audit
        class _F:
            def __init__(self, f):
                self.fixture_id  = f.match_id
                self.home_team   = f.team_a
                self.away_team   = f.team_b
                self.date        = f.date
                self.round       = f.stage
                self.status_short = f.status
        fixtures = [_F(f) for f in prov.fixtures]
        print(f"  CSV fallback: {len(fixtures)} fixtures")

    audit_set = fixtures[:args.n]
    print(f"\nAuditing {len(audit_set)} fixtures...\n")

    total         = 0
    passed        = 0
    failed        = 0
    anomaly_count = 0
    all_issues    = []
    all_anomalies = []

    for f in audit_set:
        total += 1
        try:
            r, issues, anomalies = check_one(
                f.home_team, f.away_team, getattr(f, "fixture_id", "?"),
                snaps, params, all_issues,
            )
            status = "PASS" if not issues else "FAIL"
            if issues:
                failed += 1
                all_issues.append((f, issues))
            else:
                passed += 1
            if anomalies:
                anomaly_count += len(anomalies)
                all_anomalies.append((f, anomalies))

            # Print compact summary
            date_str = getattr(f, "date", "?")[:10]
            score_str = r.most_likely_score
            probs_str = f"{r.win_a:.0%}/{r.draw:.0%}/{r.win_b:.0%}"
            print(
                f"  [{status}] {f.home_team:20s} vs {f.away_team:20s} "
                f"| {date_str} | {probs_str} | MLS: {score_str}"
            )
            if issues:
                for issue in issues:
                    print(f"         ISSUE: {issue}")
            if anomalies:
                for a in anomalies:
                    print(f"         NOTE:  {a}")

        except Exception as e:
            failed += 1
            print(f"  [ERROR] {f.home_team} vs {f.away_team}: {e}")

    # Summary
    print()
    sep()
    print(f"  Consistency audit complete")
    print(f"  Fixtures audited: {total}")
    print(f"  PASSED: {passed}  |  FAILED: {failed}  |  Anomalies noted: {anomaly_count}")

    if failed == 0:
        print("  RESULT: All predictions are internally consistent.")
    else:
        print("  RESULT: Consistency issues found — review above.")

    if all_anomalies:
        print()
        print("  Anomaly summary (mathematically valid but worth noting):")
        seen = set()
        for _, anomalies in all_anomalies:
            for a in anomalies:
                # Deduplicate by first 80 chars
                key = a[:80]
                if key not in seen:
                    print(f"    - {a[:120]}")
                    seen.add(key)

    sep()


if __name__ == "__main__":
    main()
