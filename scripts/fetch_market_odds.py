"""Fetch real historical bookmaker odds for WC 2022 and write to market_odds_real.csv.

INVESTIGATION RESULT (2026-06-09):
  No free, directly-downloadable CSV source exists for WC 2022 odds.

  Sources investigated:
    football-data.co.uk  → domestic leagues only; no WC 2022 odds
    The Odds API         → HAS WC 2022 odds; requires paid API key

  To activate:
    export ODDS_API_KEY=<your-key>
    python scripts/fetch_market_odds.py

  Without a key this script exits and reports exactly what is missing.
  It does NOT fabricate odds data.

Output:
  data/market_odds_real.csv  (research_valid=true if real data fetched)

Usage:
  python scripts/fetch_market_odds.py [--dry-run] [--output PATH]

  --dry-run     Validate adapters and matching without writing output.
  --output PATH Override default output path.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.market_sources.football_data_uk import FootballDataUKAdapter
from src.data.market_sources.the_odds_api import TheOddsAPIAdapter
from src.data.market_sources.odds_matcher import match_odds_to_match_ids
from src.data.market_sources.base import DataNotAvailableError, MissingCredentialsError

DATA = Path(__file__).parent.parent / "data"
DEFAULT_OUTPUT = DATA / "market_odds_real.csv"

WC_TEAMS = {
    "Qatar", "Ecuador", "Senegal", "Netherlands", "England", "Iran", "USA", "Wales",
    "Argentina", "Saudi Arabia", "Mexico", "Poland", "France", "Australia", "Denmark",
    "Tunisia", "Germany", "Japan", "Spain", "Costa Rica", "Belgium", "Canada",
    "Morocco", "Croatia", "Switzerland", "Cameroon", "Brazil", "Serbia",
    "Uruguay", "South Korea", "Portugal", "Ghana",
}

_D = "=" * 70
_T = "-" * 70


def load_wc2022_match_results() -> pd.DataFrame:
    df = pd.read_csv(DATA / "match_results.csv")
    sp = pd.read_csv(DATA / "team_strength_params.csv")
    sp_teams = set(sp["team"])
    wc = df[
        (df["date"] >= "2022-11-20") &
        (df["date"] <= "2022-12-18") &
        df["team_a"].isin(sp_teams) &
        df["team_b"].isin(sp_teams) &
        df["team_a"].isin(WC_TEAMS) &
        df["team_b"].isin(WC_TEAMS)
    ].copy()
    return wc


def report_source_investigation():
    print(f"\n{_T}")
    print("DATA SOURCE INVESTIGATION REPORT")
    print(_T)

    print("\n  1. football-data.co.uk")
    fdk = FootballDataUKAdapter()
    print(f"     is_available:  {fdk.is_available()}")
    print(f"     source_name:   {fdk.source_name}")
    try:
        fdk.fetch_wc2022()
    except DataNotAvailableError as e:
        print(f"     WC 2022 odds:  NOT AVAILABLE")
        print(f"     Reason: {e}")

    print("\n  2. The Odds API (the-odds-api.com)")
    oda = TheOddsAPIAdapter()
    print(f"     is_available:  {oda.is_available()}")
    print(f"     source_name:   {oda.source_name}")
    if not oda.is_available():
        print(f"     WC 2022 odds:  AVAILABLE but requires paid API key")
        print(f"     Action needed: Set ODDS_API_KEY environment variable")
        print(f"     Cost:          $30+/month (https://the-odds-api.com)")
    else:
        print(f"     WC 2022 odds:  API key found — ready to fetch")


def main():
    parser = argparse.ArgumentParser(description="Fetch real WC 2022 market odds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without writing output")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print(_D)
    print("SPRINT 4.1: REAL MARKET ODDS INGESTION")
    print(_D)

    report_source_investigation()

    # Load WC 2022 match IDs
    wc_df = load_wc2022_match_results()
    print(f"\n{_T}")
    print(f"WC 2022 MATCH IDs AVAILABLE: {len(wc_df)}")
    print(_T)

    # Try The Odds API
    oda = TheOddsAPIAdapter()
    if not oda.is_available():
        print(f"\n{'!' * 70}")
        print("MISSING DATA SOURCE — CANNOT PROCEED")
        print(f"{'!' * 70}")
        print("\n  No source with real WC 2022 odds is currently configured.")
        print("\n  WHAT IS MISSING:")
        print("    - ODDS_API_KEY environment variable (The Odds API)")
        print("      Get a key at: https://the-odds-api.com")
        print("      Plans start at $30/month. Historical data requires paid plan.")
        print("\n  WHAT IS NOT MISSING:")
        print("    - football-data.co.uk: investigated — WC 2022 not available there")
        print("    - Free CSV downloads: investigated — no WC 2022 odds found")
        print("\n  WHAT TO DO NEXT:")
        print("    1. Sign up at https://the-odds-api.com")
        print("    2. Set: export ODDS_API_KEY=<your-key>")
        print("    3. Re-run: python scripts/fetch_market_odds.py")
        print("\n  data/market_odds_real.csv will NOT be written.")
        print("  data/market_odds.csv (placeholder) remains unchanged.")
        print(_D)
        sys.exit(1)

    # Fetch from The Odds API
    print(f"\nFetching WC 2022 odds from The Odds API...")
    try:
        odds_rows = oda.fetch_wc2022()
    except Exception as e:
        print(f"ERROR fetching from The Odds API: {e}")
        sys.exit(1)

    print(f"Fetched {len(odds_rows)} odds rows from API")

    # Match to WC 2022 match IDs
    match_result = match_odds_to_match_ids(odds_rows, wc_df)

    print(f"\n{_T}")
    print("MATCHING REPORT")
    print(_T)
    print(f"  Total odds rows:   {match_result.total_odds_rows}")
    print(f"  Matched:           {match_result.matched_count}")
    print(f"  Unmatched:         {match_result.unmatched_count}")

    if match_result.unmatched:
        print(f"\n  Unmatched games (odds not linked to a match_id):")
        for u in match_result.unmatched:
            print(f"    {u['date']}  {u['team_a']} vs {u['team_b']}")

    # Report overround statistics
    if match_result.matched:
        from src.models.market_implied import decimal_odds_to_implied_probabilities
        overrounds = []
        for r in match_result.matched:
            try:
                imp = decimal_odds_to_implied_probabilities(
                    r.closing_home_odds, r.closing_draw_odds, r.closing_away_odds
                )
                overrounds.append(imp.overround)
            except ValueError:
                pass

        if overrounds:
            avg_or = sum(overrounds) / len(overrounds)
            print(f"\n  Average overround:  {avg_or:.3f} ({avg_or:.1%})")
            print(f"  Min overround:      {min(overrounds):.3f}")
            print(f"  Max overround:      {max(overrounds):.3f}")

    if args.dry_run:
        print(f"\n[dry-run] Would write {match_result.matched_count} rows to {args.output}")
        print("[dry-run] No file written.")
        return

    if not match_result.matched:
        print("\nNo matched rows — nothing to write.")
        sys.exit(1)

    # Write output
    rows = [r.to_dict() for r in match_result.matched]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print("source_type=historical_odds  research_valid=true")

    print(f"\n{_D}")
    print("Next step: run scripts/run_sprint4_report.py with market_odds_real.csv")
    print("to compare model vs real market Brier scores.")
    print(_D)


if __name__ == "__main__":
    main()
