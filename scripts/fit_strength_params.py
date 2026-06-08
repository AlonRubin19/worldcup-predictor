"""Fit MLE team strength parameters from match_results.csv.

Reads data/match_results.csv and fits Dixon-Coles Poisson attack/defense
parameters for every team using all matches before WC 2022 (< 2022-11-20).

Run from project root:
    python scripts/fit_strength_params.py

Output:
    data/team_strength_params.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mle_fitter import fit_team_params

_DATA_DIR = Path(__file__).parent.parent / "data"
_CUTOFF_DATE = "2022-11-20"  # WC 2022 start — fit on matches BEFORE this date


def main() -> None:
    matches_path = _DATA_DIR / "match_results.csv"
    if not matches_path.exists():
        print(f"ERROR: {matches_path} not found. Run scripts/build_database.py first.")
        sys.exit(1)

    print(f"Loading {matches_path} ...")
    df = pd.read_csv(matches_path)
    print(f"  {len(df):,} total matches")

    # Filter to training window: all matches BEFORE WC 2022 starts
    train = df[df["date"] < _CUTOFF_DATE].copy()
    print(f"  {len(train):,} matches in training window (before {_CUTOFF_DATE})")

    # Convert to list of dicts for mle_fitter
    records = [
        {
            "date": row["date"],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "team_a_goals": int(row["team_a_goals"]),
            "team_b_goals": int(row["team_b_goals"]),
        }
        for _, row in train.iterrows()
    ]

    print("Fitting MLE parameters (may take 30–60 seconds) ...")
    params = fit_team_params(records, min_matches=5, decay_halflife_days=180)
    print(f"  Fitted parameters for {len(params)} teams")

    # Write output CSV
    rows = [
        {
            "team": p.team,
            "alpha_attack": round(p.alpha_attack, 6),
            "beta_defense": round(p.beta_defense, 6),
            "matches_used": p.matches_used,
            "as_of_date": _CUTOFF_DATE,
        }
        for p in sorted(params.values(), key=lambda x: x.team)
    ]

    out_path = _DATA_DIR / "team_strength_params.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Wrote {len(rows)} teams to {out_path}")

    # Sanity check: WC 2022 teams
    wc_teams = [
        "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
        "Costa Rica", "Croatia", "Denmark", "Ecuador", "England", "France",
        "Germany", "Ghana", "Iran", "Japan", "Mexico", "Morocco", "Netherlands",
        "Poland", "Portugal", "Qatar", "Saudi Arabia", "Senegal", "Serbia",
        "South Korea", "Spain", "Switzerland", "Tunisia", "USA", "Uruguay", "Wales",
    ]
    missing = [t for t in wc_teams if t not in params]
    if missing:
        print(f"\nWARNING: Missing WC 2022 teams: {missing}")
    else:
        print("\nAll 32 WC 2022 teams have fitted parameters.")

    # Print top 5 by attack
    ranked = sorted(params.values(), key=lambda x: -x.alpha_attack)
    print("\nTop 5 by attack strength:")
    for p in ranked[:5]:
        print(f"  {p.team:20s}  α={p.alpha_attack:.4f}  β={p.beta_defense:.4f}  n={p.matches_used}")


if __name__ == "__main__":
    main()
