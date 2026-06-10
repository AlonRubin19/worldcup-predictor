"""Fit MLE team strength parameters from match_results.csv.

Sprint 17 (Fresh Data Refit): Reads data/match_results.csv and fits
Dixon-Coles Poisson attack/defense parameters for every team using ALL
matches up to the latest available date, with exponential time-decay
weighting (decay_halflife_days) so recent national-team form dominates
the fit while older history still contributes.

Previously this script only used matches BEFORE 2022-11-20 (the 2022 World
Cup), which froze alpha/beta at pre-2022 values while ELO/form (loaded
separately from match_results.csv) continued to update. This produced a
~3.5 year mismatch between ELO/form (fresh) and alpha/beta (frozen 2022).
The old output is preserved at data/team_strength_params_2022_archive.csv
for before/after comparison.

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
_DECAY_HALFLIFE_DAYS = 365  # ~1 year half-life: recent form dominates,
                            # but international teams play infrequently
                            # so a full year of history still matters.
_TRAIN_START_DATE = "2014-01-01"  # bound the dataset to ~12 years / 3 WC
                            # cycles so the MLE optimization (numerical
                            # gradient over 2x(#teams) params) finishes in
                            # a reasonable time, while still being far more
                            # current than the previous fixed 2022-11-20 cut.


def main() -> None:
    matches_path = _DATA_DIR / "match_results.csv"
    if not matches_path.exists():
        print(f"ERROR: {matches_path} not found. Run scripts/build_database.py first.")
        sys.exit(1)

    print(f"Loading {matches_path} ...")
    df = pd.read_csv(matches_path)
    print(f"  {len(df):,} total matches")

    cutoff_date = str(df["date"].max())
    train = df[df["date"] >= _TRAIN_START_DATE].copy()
    print(f"  {len(train):,} matches in training window ({_TRAIN_START_DATE} to {cutoff_date}, "
          f"decay half-life {_DECAY_HALFLIFE_DAYS} days)")

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

    print("Fitting MLE parameters (may take 30-60 seconds) ...")
    params = fit_team_params(records, min_matches=5, decay_halflife_days=_DECAY_HALFLIFE_DAYS)
    print(f"  Fitted parameters for {len(params)} teams")

    # Write output CSV
    rows = [
        {
            "team": p.team,
            "alpha_attack": round(p.alpha_attack, 6),
            "beta_defense": round(p.beta_defense, 6),
            "matches_used": p.matches_used,
            "as_of_date": cutoff_date,
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
