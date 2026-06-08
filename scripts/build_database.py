"""Build match_results.csv and elo_history.csv from martj42 international results.

Run from project root:
    python scripts/build_database.py

Outputs:
    data/match_results.csv    -- completed matches with pre-match ELO and rolling stats
    data/elo_history.csv      -- chronological ELO per (date, team)
"""

import sys
from pathlib import Path
import pandas as pd
import urllib.request

# So we can import from scripts/ and src/ when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.elo_computer import compute_elo_history

_DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)

BASE_XG = 1.35  # fallback for teams with no prior history

# One-way team name normalization to match our project's naming convention
TEAM_NAME_MAP: dict[str, str] = {
    "United States": "USA",
}

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def load_martj42(url: str = _DATA_URL) -> pd.DataFrame:
    """Download and parse the martj42 international results CSV.

    Returns:
        DataFrame with columns: date, home_team, away_team,
        home_score, away_score, tournament, neutral.
        Only completed matches (non-NA scores) are included.
        Sorted by date ascending.
    """
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "worldcup-predictor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        df = pd.read_csv(r)

    print(f"  Loaded {len(df):,} rows")

    # Drop future matches with NA scores
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    print(f"  {len(df):,} completed matches after dropping NA rows")

    # Normalize team names
    df["home_team"] = df["home_team"].map(lambda t: TEAM_NAME_MAP.get(t, t))
    df["away_team"] = df["away_team"].map(lambda t: TEAM_NAME_MAP.get(t, t))

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_rolling_stats(
    prior_matches: list[dict],
    window: int = 10,
) -> dict:
    """Compute last-N rolling stats for one team from their prior match records.

    Args:
        prior_matches: List of dicts with keys:
                       date (str), goals_for (int), goals_against (int), points (int).
                       Sorted chronologically. Must NOT include the current match.
        window: Maximum number of recent matches to use (default 10).

    Returns:
        Dict with keys: goals_for_last_10, goals_against_last_10,
        points_per_game_last_10, matches_available.
        Falls back to baseline values if no prior matches.
    """
    recent = prior_matches[-window:]
    n = len(recent)

    if n == 0:
        return {
            "goals_for_last_10": BASE_XG,
            "goals_against_last_10": BASE_XG,
            "points_per_game_last_10": 1.5,
            "matches_available": 0,
        }

    gf = sum(m["goals_for"] for m in recent) / n
    ga = sum(m["goals_against"] for m in recent) / n
    ppg = sum(m["points"] for m in recent) / n

    return {
        "goals_for_last_10": round(gf, 4),
        "goals_against_last_10": round(ga, 4),
        "points_per_game_last_10": round(ppg, 4),
        "matches_available": n,
    }


def build_team_match_log(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Build a per-team chronological match log from completed match DataFrame.

    Returns:
        {team_name: [{"date": ..., "goals_for": ..., "goals_against": ...,
                      "points": ...}, ...]}
        Each team appears in both home and away matches.
        Sorted oldest-first within each team.
    """
    log: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        date = str(row["date"].date())

        if hg > ag:
            h_pts, a_pts = 3, 0
        elif hg == ag:
            h_pts, a_pts = 1, 1
        else:
            h_pts, a_pts = 0, 3

        for team, gf, ga, pts in [(home, hg, ag, h_pts), (away, ag, hg, a_pts)]:
            if team not in log:
                log[team] = []
            log[team].append({
                "date": date,
                "goals_for": gf,
                "goals_against": ga,
                "points": pts,
            })

    return log


def build_match_results_with_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match ELO and rolling last-10 stats to every match row.

    For each match row, the ELO and rolling stats are computed from all
    information available BEFORE this match — strict no look-ahead.

    Returns:
        DataFrame with the full schema required by pre_match_team_stats_real.csv.
    """
    # Build ELO history first (already guarantees pre-match ELO)
    match_dicts = [
        {
            "date": str(row["date"].date()),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }
        for _, row in df.iterrows()
    ]
    elo_history = compute_elo_history(match_dicts)
    # Index: (date_str, team) -> elo_pre
    elo_lookup: dict[tuple[str, str], float] = {
        (r["date"], r["team"]): r["elo_pre"] for r in elo_history
    }

    # Build per-team running match log (used for rolling stats)
    team_log = build_team_match_log(df)
    # Track index into each team's log as we process chronologically
    team_cursor: dict[str, int] = {team: 0 for team in team_log}

    rows = []
    match_id = 1

    for _, row in df.iterrows():
        date_str = str(row["date"].date())
        home = row["home_team"]
        away = row["away_team"]
        hg = int(row["home_score"])
        ag = int(row["away_score"])

        # Pre-match ELO (leakage-free via compute_elo_history)
        elo_home = elo_lookup.get((date_str, home), 1600.0)
        elo_away = elo_lookup.get((date_str, away), 1600.0)

        # Rolling stats: use only matches strictly before the cursor position
        # team_log is already sorted chronologically; cursor tracks processed count
        prior_home = team_log.get(home, [])[:team_cursor.get(home, 0)]
        prior_away = team_log.get(away, [])[:team_cursor.get(away, 0)]

        stats_home = compute_rolling_stats(prior_home)
        stats_away = compute_rolling_stats(prior_away)

        rows.append({
            "match_id": match_id,
            "date": date_str,
            "team_a": home,
            "team_b": away,
            "team_a_goals": hg,
            "team_b_goals": ag,
            "team_a_elo_pre": round(elo_home, 2),
            "team_b_elo_pre": round(elo_away, 2),
            "team_a_goals_for_last_10": stats_home["goals_for_last_10"],
            "team_a_goals_against_last_10": stats_home["goals_against_last_10"],
            "team_b_goals_for_last_10": stats_away["goals_for_last_10"],
            "team_b_goals_against_last_10": stats_away["goals_against_last_10"],
            "team_a_points_per_game_last_10": stats_home["points_per_game_last_10"],
            "team_b_points_per_game_last_10": stats_away["points_per_game_last_10"],
            "team_a_matches_available": stats_home["matches_available"],
            "team_b_matches_available": stats_away["matches_available"],
        })

        # Advance cursors AFTER processing this match (so this match's stats
        # are not available when computing stats for the NEXT match)
        team_cursor[home] = team_cursor.get(home, 0) + 1
        team_cursor[away] = team_cursor.get(away, 0) + 1
        match_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    _DATA_DIR.mkdir(exist_ok=True)

    # 1. Download and clean source data
    df = load_martj42()

    # 2. Compute ELO history
    print("Computing ELO history...")
    match_dicts = [
        {
            "date": str(row["date"].date()),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }
        for _, row in df.iterrows()
    ]
    elo_records = compute_elo_history(match_dicts)
    elo_df = pd.DataFrame(elo_records)
    out_elo = _DATA_DIR / "elo_history.csv"
    elo_df.to_csv(out_elo, index=False)
    print(f"  Wrote {len(elo_df):,} rows to {out_elo}")

    # 3. Build full match results with pre-match stats
    print("Building match results with rolling stats...")
    results_df = build_match_results_with_stats(df)
    out_results = _DATA_DIR / "match_results.csv"
    results_df.to_csv(out_results, index=False)
    print(f"  Wrote {len(results_df):,} rows to {out_results}")

    # 4. Quick sanity check: WC 2022 rows
    wc = results_df[
        (results_df["date"] >= "2022-11-20") &
        (results_df["date"] <= "2022-12-18")
    ]
    print(f"\nWC 2022 matches in output: {len(wc)}")
    print("Sample (first 3):")
    print(wc.head(3)[["date", "team_a", "team_b", "team_a_elo_pre", "team_b_elo_pre"]].to_string())


if __name__ == "__main__":
    main()
