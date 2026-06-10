#!/usr/bin/env python
"""Model Sanity / Calibration Audit — WC 2026 Group Stage.

Sprint 15 extension (pre Golden Boot work).

For every WC 2026 fixture, compute:
  - Win/Draw/Loss probabilities
  - ELO difference (team_a - team_b)
  - MLE attack/defence params + xG
  - Most likely score
  - ELO-implied win probability (logistic, for comparison)

Flags matches where:
  - The lower-ELO team is favoured by >10pp in the model, OR
  - Model win prob diverges from ELO-implied win prob by > threshold

Read-only — does not modify any model code.

Usage:
    python scripts/run_calibration_audit.py
"""

from __future__ import annotations

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

from src.data.api_football_client import ApiFootballClient
from src.data.fixture_provider import get_fixtures, FixtureSource
from src.data.team_snapshot_loader import load_team_snapshots, TeamSnapshot
from src.data.strength_loader import load_strength_params, StrengthParams
from src.app.prediction_runner import RunnerInput, run_full_prediction


_SNAP_DEF = TeamSnapshot(elo=1800.0, ppg=1.5)
_PAR_DEF  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)

# Divergence threshold (percentage points) for flagging
ELO_VS_MODEL_THRESHOLD = 0.10
LOWER_ELO_FAVOURITE_THRESHOLD = 0.10


def elo_implied_win_prob(elo_a: float, elo_b: float) -> float:
    """Standard ELO logistic win-probability formula (no draw modeling).

    P(A beats B) = 1 / (1 + 10^((elo_b - elo_a)/400))

    This is a *pure ELO baseline* — does not include draw probability,
    so it is only used as a directional sanity check, not a target.
    """
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def sep(n=110): print("=" * n)


def main():
    sep()
    print("  World Cup 2026 — Model Sanity / Calibration Audit (Group Stage)")
    sep()

    print("\nLoading model data...")
    snaps  = load_team_snapshots()
    params = load_strength_params()
    print(f"  ELO snapshots: {len(snaps)} teams")
    print(f"  Strength params: {len(params)} teams")

    # Load fixtures (live API with CSV fallback)
    fixtures = []
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if api_key:
        try:
            client = ApiFootballClient(api_key=api_key, cache_dir=str(_ROOT / "data" / "api_cache"))
            prov = get_fixtures(FixtureSource.AUTO, api_client=client, force_refresh=False)
        except Exception as e:
            print(f"  API error: {e} -- falling back to CSV")
            prov = get_fixtures(FixtureSource.CSV)
    else:
        prov = get_fixtures(FixtureSource.CSV)

    print(f"  Fixture source: {prov.source_used.value} ({prov.fixture_count} fixtures)")

    # Filter to group stage only
    group_fixtures = [f for f in prov.fixtures if f.stage == "group"]
    print(f"  Group-stage fixtures: {len(group_fixtures)}")

    rows = []
    flagged = []

    for f in group_fixtures:
        team_a, team_b = f.team_a, f.team_b

        snap_a = snaps.get(team_a, _SNAP_DEF)
        snap_b = snaps.get(team_b, _SNAP_DEF)
        par_a  = params.get(team_a, _PAR_DEF)
        par_b  = params.get(team_b, _PAR_DEF)

        has_data_a = team_a in snaps and team_a in params
        has_data_b = team_b in snaps and team_b in params

        inp = RunnerInput(team_a, team_b, snap_a, snap_b, par_a, par_b)
        r   = run_full_prediction(inp)

        elo_diff = snap_a.elo - snap_b.elo
        elo_implied_a = elo_implied_win_prob(snap_a.elo, snap_b.elo)

        row = {
            "team_a": team_a,
            "team_b": team_b,
            "elo_a": snap_a.elo,
            "elo_b": snap_b.elo,
            "elo_diff": elo_diff,
            "win_a": r.win_a,
            "draw": r.draw,
            "win_b": r.win_b,
            "elo_implied_a": elo_implied_a,
            "xg_a": r.xg_a,
            "xg_b": r.xg_b,
            "alpha_a": par_a.alpha_attack,
            "beta_a": par_a.beta_defense,
            "alpha_b": par_b.alpha_attack,
            "beta_b": par_b.beta_defense,
            "most_likely_score": r.most_likely_score,
            "has_data_a": has_data_a,
            "has_data_b": has_data_b,
            "date": getattr(f, "date", "?"),
        }
        rows.append(row)

        # ── Flag conditions ──────────────────────────────────────────────────
        flags = []

        # 1. Lower-ELO team favoured by >10pp
        if elo_diff < 0 and (r.win_a - r.win_b) > LOWER_ELO_FAVOURITE_THRESHOLD:
            flags.append(
                f"Lower-ELO team A ({team_a}, ELO {snap_a.elo:.0f}) favoured over "
                f"higher-ELO team B ({team_b}, ELO {snap_b.elo:.0f}) by "
                f"{(r.win_a - r.win_b):.1%}"
            )
        elif elo_diff > 0 and (r.win_b - r.win_a) > LOWER_ELO_FAVOURITE_THRESHOLD:
            flags.append(
                f"Lower-ELO team B ({team_b}, ELO {snap_b.elo:.0f}) favoured over "
                f"higher-ELO team A ({team_a}, ELO {snap_a.elo:.0f}) by "
                f"{(r.win_b - r.win_a):.1%}"
            )

        # 2. Model win prob diverges materially from pure-ELO implied prob
        divergence = r.win_a - elo_implied_a
        if abs(divergence) > ELO_VS_MODEL_THRESHOLD:
            flags.append(
                f"Model P(A win)={r.win_a:.1%} vs ELO-implied P(A win)={elo_implied_a:.1%} "
                f"-- divergence {divergence:+.1%}"
            )

        # 3. Missing strength data (defaults used)
        if not has_data_a:
            flags.append(f"{team_a}: no ELO/MLE data -- using defaults (ELO 1800, alpha=1.0, beta=1.0)")
        if not has_data_b:
            flags.append(f"{team_b}: no ELO/MLE data -- using defaults (ELO 1800, alpha=1.0, beta=1.0)")

        if flags:
            flagged.append((row, flags))

    # ── Print full table ──────────────────────────────────────────────────────
    print()
    sep()
    print("  FULL FIXTURE TABLE")
    sep()
    header = (
        f"{'Team A':22s} {'Team B':22s} {'ELO A':>6s} {'ELO B':>6s} {'ΔELO':>7s} "
        f"{'WinA':>6s} {'Draw':>6s} {'WinB':>6s} {'xGA':>5s} {'xGB':>5s} {'MLS':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['team_a']:22s} {row['team_b']:22s} "
            f"{row['elo_a']:6.0f} {row['elo_b']:6.0f} {row['elo_diff']:+7.0f} "
            f"{row['win_a']:6.1%} {row['draw']:6.1%} {row['win_b']:6.1%} "
            f"{row['xg_a']:5.2f} {row['xg_b']:5.2f} {row['most_likely_score']:>5s}"
        )

    # ── Calibration issues report ───────────────────────────────────────────
    print()
    sep()
    print("  POTENTIAL CALIBRATION ISSUES")
    sep()
    print(f"\n  Total group-stage fixtures: {len(rows)}")
    print(f"  Flagged fixtures: {len(flagged)}")
    print()

    for row, flags in flagged:
        print(f"  {row['team_a']} vs {row['team_b']}  ({row['date'][:10]})")
        print(
            f"    ELO: {row['elo_a']:.0f} vs {row['elo_b']:.0f}  (Δ={row['elo_diff']:+.0f})  |  "
            f"Model: {row['win_a']:.1%}/{row['draw']:.1%}/{row['win_b']:.1%}  |  "
            f"ELO-implied A: {row['elo_implied_a']:.1%}  |  "
            f"xG: {row['xg_a']:.2f}-{row['xg_b']:.2f}  |  MLS: {row['most_likely_score']}"
        )
        print(
            f"    MLE: A(α={row['alpha_a']:.2f}, β={row['beta_a']:.2f})  "
            f"B(α={row['alpha_b']:.2f}, β={row['beta_b']:.2f})"
        )
        for fl in flags:
            print(f"    FLAG: {fl}")
        print()

    sep()
    print("  Audit complete. Model NOT modified.")
    sep()


if __name__ == "__main__":
    main()
