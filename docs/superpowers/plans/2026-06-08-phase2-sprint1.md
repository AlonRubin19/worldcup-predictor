# Phase 2 Sprint 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest 49,373 real international match results, compute chronological ELO ratings with zero look-ahead bias, fit MLE attack/defense parameters for every WC 2022 team, and produce a strength-adjusted xG calculator that improves Brier score over the V6 placeholder baseline.

**Architecture:** Three-stage pipeline — (1) `scripts/build_database.py` downloads and processes the martj42 dataset into `data/match_results.csv` and `data/elo_history.csv`; (2) `scripts/fit_strength_params.py` reads those files and runs scipy MLE to produce `data/team_strength_params.csv`; (3) `src/models/strength_adjusted_xg.py` and `src/backtesting/strength_runner.py` plug those parameters into the existing prediction pipeline. All existing tests must continue to pass.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy.optimize, pytest (no new dependencies)

---

## Critical Context

### Dataset
- **URL:** `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`
- **Size:** 49,446 rows (header + data), 49,373 completed matches
- **Columns:** `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`
- **NA rows:** Future scheduled matches have `NA` in score columns — must be filtered out before any computation
- **Coverage:** 1872–2026 (including all 64 WC 2022 matches)

### Team Name Mapping
Only one mismatch between martj42 and our project data:
```python
TEAM_NAME_MAP = {
    "United States": "USA",
}
```
Apply this when writing any output CSV so team names are consistent across the project.

### WC 2022 Tournament Scope
- `data/historical_matches.csv` contains 40 matches (group stage through semi-finals, no final)
- The backtest target teams are: Argentina, Australia, Belgium, Brazil, Cameroon, Canada, Costa Rica, Croatia, Denmark, Ecuador, England, France, Germany, Ghana, Iran, Japan, Mexico, Morocco, Netherlands, Poland, Portugal, Qatar, Saudi Arabia, Senegal, Serbia, South Korea, Spain, Switzerland, Tunisia, USA, Uruguay, Wales

### ELO Formula
```
E_a = 1 / (1 + 10^((elo_b - elo_a) / 400))
E_b = 1 - E_a

W_a = 1.0 if home_score > away_score, 0.5 if draw, 0.0 if loss
W_b = 1.0 - W_a

K = 60 (uniform for international football — no home advantage distinction)

elo_a_new = elo_a + K * (W_a - E_a)
elo_b_new = elo_b + K * (W_b - E_b)
```
**Starting ELO:** 1600 for all teams on first appearance.
**Critical:** Store the ELO values BEFORE updating for the current match result.

### MLE Formula
For each match i with time-decay weight w_i:
```
w_i = exp(-lambda * days_ago_i)   where lambda = ln(2) / 180  (half-life 180 days)

lambda_a = alpha_a * beta_b   (no gamma — all WC matches are neutral)
lambda_b = alpha_b * beta_a

log_likelihood += w_i * [
    goals_a * log(lambda_a) - lambda_a +
    goals_b * log(lambda_b) - lambda_b
]
```
Minimize negative log-likelihood using `scipy.optimize.minimize` with method `L-BFGS-B`.
Bounds: all α, β > 0.1.
Training window: all matches with date < 2022-11-20.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/build_database.py` | Create | Download martj42 → compute ELO → compute rolling last-10 stats → write two CSVs |
| `scripts/fit_strength_params.py` | Create | Read match_results.csv → MLE fit per team → write team_strength_params.csv |
| `src/models/mle_fitter.py` | Create | Pure function: given list of matches, return fitted {team: (alpha, beta)} |
| `src/models/strength_adjusted_xg.py` | Create | Calculate xG using α, β, ELO — same interface as pre_match_xg.py |
| `src/backtesting/strength_runner.py` | Create | Run valid backtest using strength-adjusted xG |
| `src/data/strength_loader.py` | Create | Load team_strength_params.csv → dict |
| `data/match_results.csv` | Generate | Real match results + pre-match stats (output of build_database.py) |
| `data/elo_history.csv` | Generate | Pre-match ELO per (date, team) (output of build_database.py) |
| `data/team_strength_params.csv` | Generate | MLE α/β per team (output of fit_strength_params.py) |
| `tests/scripts/test_build_database.py` | Create | ELO leakage checks, rolling stats correctness |
| `tests/models/test_mle_fitter.py` | Create | MLE convergence, sensible output, edge cases |
| `tests/models/test_strength_adjusted_xg.py` | Create | Formula correctness, clamping, monotonicity |
| `tests/backtesting/test_strength_runner.py` | Create | Integration: runs backtest end-to-end |
| `tests/data/test_strength_loader.py` | Create | Load/parse team_strength_params.csv |

---

## Task 1: ELO Computer Module

**Files:**
- Create: `tests/scripts/test_build_database.py` (ELO unit tests only)
- Create: `scripts/__init__.py` (empty, for testability)
- Create: `scripts/elo_computer.py` (pure ELO logic — isolated for testing)

This task isolates the ELO update logic so it can be unit-tested without a network call or 49k rows.

- [ ] **Step 1: Create `scripts/__init__.py`**

```bash
mkdir -p scripts
echo "" > scripts/__init__.py
```

- [ ] **Step 2: Create `tests/scripts/__init__.py`**

```bash
mkdir -p tests/scripts
echo "" > tests/scripts/__init__.py
```

- [ ] **Step 3: Write failing ELO unit tests**

Create `tests/scripts/test_elo_computer.py`:

```python
import pytest
from scripts.elo_computer import (
    compute_expected_score,
    update_elo,
    compute_elo_history,
)


def test_expected_score_equal_ratings():
    assert abs(compute_expected_score(1600, 1600) - 0.5) < 1e-9


def test_expected_score_higher_rated_favoured():
    e = compute_expected_score(1800, 1600)
    assert e > 0.5


def test_expected_scores_sum_to_one():
    ea = compute_expected_score(1750, 1650)
    eb = compute_expected_score(1650, 1750)
    assert abs(ea + eb - 1.0) < 1e-9


def test_elo_increases_on_win():
    new_a, new_b = update_elo(1600, 1600, win_a=True, draw=False)
    assert new_a > 1600
    assert new_b < 1600


def test_elo_decreases_on_loss():
    new_a, new_b = update_elo(1600, 1600, win_a=False, draw=False)
    assert new_a < 1600
    assert new_b > 1600


def test_elo_unchanged_sum_on_draw():
    new_a, new_b = update_elo(1600, 1600, win_a=False, draw=True)
    # Both at 1600 → expect no change on draw
    assert abs(new_a - 1600) < 1e-9
    assert abs(new_b - 1600) < 1e-9


def test_elo_is_zero_sum():
    """Total ELO in the system is conserved."""
    new_a, new_b = update_elo(1800, 1650, win_a=True, draw=False)
    assert abs((new_a + new_b) - (1800 + 1650)) < 1e-6


def test_compute_elo_history_no_leakage():
    """ELO used for match N must NOT include match N's result."""
    matches = [
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "home_score": 3, "away_score": 0},
        {"date": "2020-02-01", "home_team": "A", "away_team": "B",
         "home_score": 0, "away_score": 2},
    ]
    history = compute_elo_history(matches)
    # Row for first match: both teams at starting ELO 1600
    first = [r for r in history if r["date"] == "2020-01-01" and r["team"] == "A"][0]
    assert first["elo_pre"] == 1600.0

    # Row for second match: A's ELO should reflect first match WIN (> 1600)
    second = [r for r in history if r["date"] == "2020-02-01" and r["team"] == "A"][0]
    assert second["elo_pre"] > 1600.0


def test_new_team_starts_at_1600():
    matches = [
        {"date": "2020-01-01", "home_team": "NewTeam", "away_team": "OtherTeam",
         "home_score": 1, "away_score": 0},
    ]
    history = compute_elo_history(matches)
    row = [r for r in history if r["team"] == "NewTeam"][0]
    assert row["elo_pre"] == 1600.0
```

- [ ] **Step 4: Run to confirm they fail**

```bash
python -m pytest tests/scripts/test_elo_computer.py -v
```
Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 5: Create `scripts/elo_computer.py`**

```python
"""Pure ELO computation logic — no I/O, no network calls, easily testable."""

K_FACTOR = 60        # Standard for international football
STARTING_ELO = 1600  # FIFA-style base rating


def compute_expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that team A wins given ELO ratings.

    Uses the standard ELO formula: E_a = 1 / (1 + 10^((elo_b - elo_a) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float,
    elo_b: float,
    win_a: bool,
    draw: bool,
) -> tuple[float, float]:
    """Apply one match result to ELO ratings.

    Args:
        elo_a: Team A's current ELO (PRE-match).
        elo_b: Team B's current ELO (PRE-match).
        win_a: True if team A won.
        draw:  True if the match was a draw (win_a ignored when True).

    Returns:
        (new_elo_a, new_elo_b)
    """
    expected_a = compute_expected_score(elo_a, elo_b)
    expected_b = 1.0 - expected_a

    if draw:
        actual_a, actual_b = 0.5, 0.5
    elif win_a:
        actual_a, actual_b = 1.0, 0.0
    else:
        actual_a, actual_b = 0.0, 1.0

    new_elo_a = elo_a + K_FACTOR * (actual_a - expected_a)
    new_elo_b = elo_b + K_FACTOR * (actual_b - expected_b)
    return new_elo_a, new_elo_b


def compute_elo_history(matches: list[dict]) -> list[dict]:
    """Compute chronological ELO history from a list of match dicts.

    Args:
        matches: List of dicts with keys:
                 date (str ISO), home_team (str), away_team (str),
                 home_score (int), away_score (int).
                 Must be sorted chronologically (oldest first).

    Returns:
        List of dicts: {date, team, elo_pre}
        One row per team per match, elo_pre = ELO BEFORE this match's result.
        This is the anti-leakage guarantee: no future data is embedded.
    """
    ratings: dict[str, float] = {}  # team -> current ELO
    history: list[dict] = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]

        # Initialize new teams at starting ELO
        if home not in ratings:
            ratings[home] = STARTING_ELO
        if away not in ratings:
            ratings[away] = STARTING_ELO

        elo_home_pre = ratings[home]
        elo_away_pre = ratings[away]

        # CRITICAL: Record pre-match ELO BEFORE updating
        history.append({"date": match["date"], "team": home, "elo_pre": elo_home_pre})
        history.append({"date": match["date"], "team": away, "elo_pre": elo_away_pre})

        # Determine outcome
        home_goals = match["home_score"]
        away_goals = match["away_score"]
        win_home = home_goals > away_goals
        draw = home_goals == away_goals

        # Update ratings for NEXT match
        ratings[home], ratings[away] = update_elo(
            elo_home_pre, elo_away_pre,
            win_a=win_home,
            draw=draw,
        )

    return history
```

- [ ] **Step 6: Run ELO tests**

```bash
python -m pytest tests/scripts/test_elo_computer.py -v
```
Expected: 9 tests pass.

- [ ] **Step 7: Run full suite for regressions**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 86 + 9 = 95 tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/__init__.py scripts/elo_computer.py tests/scripts/__init__.py tests/scripts/test_elo_computer.py
git commit -m "feat: ELO computer — chronological ELO with leakage-free pre-match storage"
```

---

## Task 2: Database Build Script

**Files:**
- Create: `scripts/build_database.py`
- Generate: `data/match_results.csv` and `data/elo_history.csv`

This script downloads the martj42 dataset, applies the ELO computer, computes rolling last-10 stats, and writes the two output CSVs.

- [ ] **Step 1: Write rolling stats unit tests**

Append to `tests/scripts/test_elo_computer.py`:

```python
from scripts.build_database import compute_rolling_stats


def test_rolling_stats_uses_only_prior_matches():
    """Stats for match N must not include match N itself."""
    prior = [
        {"date": "2020-01-01", "goals_for": 3, "goals_against": 1, "points": 3},
        {"date": "2020-02-01", "goals_for": 1, "goals_against": 1, "points": 1},
    ]
    stats = compute_rolling_stats(prior, window=10)
    assert stats["goals_for_last_10"] == 2.0       # (3+1)/2
    assert stats["goals_against_last_10"] == 1.0   # (1+1)/2
    assert stats["points_per_game_last_10"] == 2.0  # (3+1)/2
    assert stats["matches_available"] == 2


def test_rolling_stats_empty_prior():
    """Team with no prior matches returns baseline values."""
    stats = compute_rolling_stats([], window=10)
    assert stats["matches_available"] == 0
    assert stats["goals_for_last_10"] == 1.35   # fallback = BASE_XG
    assert stats["goals_against_last_10"] == 1.35
    assert stats["points_per_game_last_10"] == 1.5  # fallback = mid-range


def test_rolling_stats_window_capped_at_10():
    """Only use the 10 most recent prior matches."""
    prior = [
        {"date": f"2020-{i:02d}-01", "goals_for": 2, "goals_against": 0, "points": 3}
        for i in range(1, 15)  # 14 matches
    ]
    stats = compute_rolling_stats(prior, window=10)
    assert stats["matches_available"] == 10
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
python -m pytest tests/scripts/test_elo_computer.py -v -k "rolling"
```
Expected: ImportError.

- [ ] **Step 3: Create `scripts/build_database.py`**

```python
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
```

- [ ] **Step 4: Run the rolling stats tests**

```bash
python -m pytest tests/scripts/test_elo_computer.py -v -k "rolling"
```
Expected: 3 tests pass.

- [ ] **Step 5: Run the full script to generate the data files**

```bash
python scripts/build_database.py
```

Expected output (approx):
```
Downloading https://...
  Loaded 49,446 rows
  49,373 completed matches after dropping NA rows
Computing ELO history...
  Wrote 98,746 rows to data/elo_history.csv
Building match results with rolling stats...
  Wrote 49,373 rows to data/match_results.csv

WC 2022 matches in output: 64
Sample (first 3):
...
```

- [ ] **Step 6: Verify the output files**

```bash
python -c "
import pandas as pd
elo = pd.read_csv('data/elo_history.csv')
res = pd.read_csv('data/match_results.csv')
print('elo_history:', elo.shape, elo.columns.tolist())
print('match_results:', res.shape, res.columns.tolist())

# No NaN values in key columns
assert elo['elo_pre'].isna().sum() == 0, 'ELO has NaN'
assert res['team_a_elo_pre'].isna().sum() == 0, 'team_a_elo_pre has NaN'

# WC 2022 present
wc = res[res['date'] >= '2022-11-20']
print(f'WC 2022+ matches: {len(wc)}')

# Check France pre-WC ELO
france_wc1 = res[(res['team_a'] == 'France') & (res['date'] == '2022-11-23')]
if not france_wc1.empty:
    print(f'France ELO at WC 2022 first match: {france_wc1.iloc[0][\"team_a_elo_pre\"]}')

print('PASS')
"
```

- [ ] **Step 7: Run full suite for regressions**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 95 tests pass (no regressions).

- [ ] **Step 8: Commit**

```bash
git add scripts/build_database.py tests/scripts/test_elo_computer.py data/match_results.csv data/elo_history.csv
git commit -m "feat: build_database.py — martj42 ingestion, ELO history, rolling stats (49,373 matches)"
```

---

## Task 3: MLE Fitter Module

**Files:**
- Create: `tests/models/test_mle_fitter.py`
- Create: `src/models/mle_fitter.py`

Fits Poisson attack (α) and defense (β) parameters per team using maximum likelihood estimation.

- [ ] **Step 1: Write failing MLE tests**

Create `tests/models/test_mle_fitter.py`:

```python
import pytest
import math
from src.models.mle_fitter import fit_team_params, TeamStrengthParams

# ── Minimal fixture ──────────────────────────────────────────────────────────
def _make_match(team_a, team_b, ga, gb, date="2020-01-01", weight=1.0):
    return {
        "date": date,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_goals": ga,
        "team_b_goals": gb,
        "weight": weight,
    }


def _symmetric_matches(n=20):
    """Equal-strength teams: interleave wins and losses."""
    matches = []
    for i in range(n):
        if i % 2 == 0:
            matches.append(_make_match("A", "B", 2, 1, f"2020-{i+1:02d}-01"))
        else:
            matches.append(_make_match("A", "B", 1, 2, f"2020-{i+1:02d}-01"))
    return matches


# ── Tests ────────────────────────────────────────────────────────────────────

def test_returns_dict_of_team_strength_params():
    result = fit_team_params(_symmetric_matches())
    assert isinstance(result, dict)
    assert "A" in result and "B" in result
    assert isinstance(result["A"], TeamStrengthParams)


def test_all_params_positive():
    result = fit_team_params(_symmetric_matches())
    for team, p in result.items():
        assert p.alpha_attack > 0, f"{team} alpha <= 0"
        assert p.beta_defense > 0, f"{team} beta <= 0"


def test_equal_teams_have_similar_params():
    """Symmetric win/loss should produce similar alpha and beta for both teams."""
    result = fit_team_params(_symmetric_matches(n=40))
    assert abs(result["A"].alpha_attack - result["B"].alpha_attack) < 0.3
    assert abs(result["A"].beta_defense - result["B"].beta_defense) < 0.3


def test_strong_attacker_gets_higher_alpha():
    """Team that scores more goals should get a higher attack parameter."""
    matches = (
        [_make_match("Strong", "Weak", 4, 0, f"2020-{i:02d}-01") for i in range(1, 11)] +
        [_make_match("Weak", "Strong", 0, 4, f"2020-{i:02d}-01") for i in range(11, 21)]
    )
    result = fit_team_params(matches)
    assert result["Strong"].alpha_attack > result["Weak"].alpha_attack


def test_solid_defender_gets_higher_beta():
    """Team that concedes fewer goals should get a higher defense parameter (lower concession)."""
    # Solid team concedes 0, Leaky team concedes 3
    matches = (
        [_make_match("Solid", "Leaky", 1, 3, f"2020-{i:02d}-01") for i in range(1, 11)] +
        [_make_match("Leaky", "Solid", 3, 0, f"2020-{i:02d}-01") for i in range(11, 21)]
    )
    result = fit_team_params(matches)
    # beta_defense is a MULTIPLIER on opponent's lambda. Lower beta = harder to score against.
    # Solid concedes 0 goals; Leaky concedes 3 goals. So Solid should have lower beta.
    assert result["Solid"].beta_defense < result["Leaky"].beta_defense


def test_log_likelihood_is_negative():
    """Log-likelihood of Poisson observations must be negative."""
    result = fit_team_params(_symmetric_matches())
    for team, p in result.items():
        assert p.log_likelihood < 0 or math.isnan(p.log_likelihood) is False


def test_minimum_matches_threshold():
    """Teams with fewer than min_matches are excluded from output."""
    matches = [_make_match("A", "B", 1, 0)]  # Only 1 match
    result = fit_team_params(matches, min_matches=5)
    assert "A" not in result
    assert "B" not in result


def test_fit_handles_zero_goals():
    """0-0 draws should not cause log(0) errors."""
    matches = [_make_match("A", "B", 0, 0, f"2020-{i:02d}-01") for i in range(1, 11)]
    result = fit_team_params(matches)  # Should not raise
    assert "A" in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_mle_fitter.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/models/mle_fitter.py`**

```python
"""Maximum Likelihood Estimation of team attack and defense strength parameters.

Implements the Maher (1982) / Dixon-Coles (1997) Poisson model:
    lambda_a = alpha_a * beta_b   (neutral venue — no home advantage)
    lambda_b = alpha_b * beta_a

Parameters are fitted by minimizing negative weighted log-likelihood using
scipy.optimize.minimize with L-BFGS-B and bounds > 0.1.

Usage:
    from src.models.mle_fitter import fit_team_params
    params = fit_team_params(match_records, as_of_date="2022-11-19")
"""

import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class TeamStrengthParams:
    """MLE-fitted attack and defense parameters for one team."""
    team: str
    alpha_attack: float   # Attack strength (1.0 = average)
    beta_defense: float   # Defense multiplier (lower = harder to score against)
    matches_used: int
    log_likelihood: float  # Weighted log-likelihood at the fitted parameters


def _negative_log_likelihood(
    params: np.ndarray,
    team_list: list[str],
    matches: list[dict],
) -> float:
    """Compute negative weighted log-likelihood for all matches.

    params: 1D array of [alpha_A, alpha_B, ..., beta_A, beta_B, ...]
    team_list: ordered list of teams (defines param array layout)
    matches: list of dicts with team_a, team_b, team_a_goals, team_b_goals, weight
    """
    n = len(team_list)
    idx = {team: i for i, team in enumerate(team_list)}
    alpha = params[:n]   # Attack parameters
    beta = params[n:]    # Defense parameters

    total = 0.0
    for m in matches:
        i_a = idx[m["team_a"]]
        i_b = idx[m["team_b"]]
        lam_a = alpha[i_a] * beta[i_b]  # Expected goals for team A
        lam_b = alpha[i_b] * beta[i_a]  # Expected goals for team B
        g_a = m["team_a_goals"]
        g_b = m["team_b_goals"]
        w = m.get("weight", 1.0)

        # Poisson log-likelihood: goals * log(lambda) - lambda  (constant terms dropped)
        # Guard against lam = 0 or negative
        if lam_a <= 0 or lam_b <= 0:
            return 1e10

        ll = w * (
            g_a * math.log(lam_a) - lam_a +
            g_b * math.log(lam_b) - lam_b
        )
        total += ll

    return -total  # Negative because we minimize


def fit_team_params(
    matches: list[dict],
    min_matches: int = 5,
    decay_halflife_days: int = 180,
) -> dict[str, TeamStrengthParams]:
    """Fit Poisson attack/defense parameters via MLE for all teams.

    Args:
        matches: List of dicts with keys:
                 date (str ISO), team_a (str), team_b (str),
                 team_a_goals (int), team_b_goals (int).
                 Weights are computed from date (most recent = highest weight).
                 Optional "weight" key overrides computed weight.
        min_matches: Minimum total appearances required to include a team.
        decay_halflife_days: Half-life for exponential time decay.
                             Matches played decay_halflife_days ago have weight 0.5.

    Returns:
        Dict of {team_name: TeamStrengthParams} for all teams with enough data.
    """
    import datetime

    if not matches:
        return {}

    # Parse dates and compute time-decay weights
    max_date_str = max(m["date"] for m in matches)
    max_date = datetime.date.fromisoformat(max_date_str)
    lambda_decay = math.log(2) / decay_halflife_days

    weighted = []
    for m in matches:
        if "weight" not in m:
            match_date = datetime.date.fromisoformat(m["date"])
            days_ago = (max_date - match_date).days
            w = math.exp(-lambda_decay * days_ago)
        else:
            w = m["weight"]

        weighted.append({**m, "weight": w})

    # Count appearances per team
    appearances: dict[str, int] = {}
    for m in weighted:
        appearances[m["team_a"]] = appearances.get(m["team_a"], 0) + 1
        appearances[m["team_b"]] = appearances.get(m["team_b"], 0) + 1

    # Filter to teams with sufficient data; exclude others from optimization
    eligible = {t for t, cnt in appearances.items() if cnt >= min_matches}
    filtered = [m for m in weighted if m["team_a"] in eligible and m["team_b"] in eligible]

    if not filtered:
        return {}

    team_list = sorted(eligible)
    n = len(team_list)

    # Initial guess: all alphas = 1.0 (average attack), all betas = 1.0 (average defense)
    x0 = np.ones(2 * n)
    # Bounds: all parameters must be > 0.1
    bounds = [(0.1, None)] * (2 * n)

    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(team_list, filtered),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    final_params = result.x
    alpha_vals = final_params[:n]
    beta_vals = final_params[n:]

    # Build per-team output
    output = {}
    for i, team in enumerate(team_list):
        output[team] = TeamStrengthParams(
            team=team,
            alpha_attack=float(alpha_vals[i]),
            beta_defense=float(beta_vals[i]),
            matches_used=appearances[team],
            log_likelihood=-result.fun / len(filtered),  # Per-match average
        )

    return output
```

- [ ] **Step 4: Run MLE tests**

```bash
python -m pytest tests/models/test_mle_fitter.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 103 tests pass (95 + 8).

- [ ] **Step 6: Commit**

```bash
git add src/models/mle_fitter.py tests/models/test_mle_fitter.py
git commit -m "feat: MLE fitter — Poisson attack/defense parameter estimation with time decay"
```

---

## Task 4: Fit Strength Parameters Script

**Files:**
- Create: `scripts/fit_strength_params.py`
- Generate: `data/team_strength_params.csv`

- [ ] **Step 1: Create `scripts/fit_strength_params.py`**

```python
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
```

- [ ] **Step 2: Run the fit script**

```bash
python scripts/fit_strength_params.py
```

Expected output (approx):
```
Loading data/match_results.csv ...
  49,373 total matches
  ~49,200 matches in training window (before 2022-11-20)
Fitting MLE parameters (may take 30-60 seconds) ...
  Fitted parameters for ~200+ teams
All 32 WC 2022 teams have fitted parameters.

Top 5 by attack strength:
  [teams with high alpha values]
```

- [ ] **Step 3: Verify the output**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/team_strength_params.csv')
print(f'Rows: {len(df)}, Columns: {df.columns.tolist()}')
wc = ['France', 'Brazil', 'Argentina', 'England', 'Germany', 'Spain', 'Qatar', 'USA']
print('Sample WC teams:')
print(df[df.team.isin(wc)][['team','alpha_attack','beta_defense','matches_used']].to_string())
assert (df.alpha_attack > 0).all(), 'alpha has non-positive values'
assert (df.beta_defense > 0).all(), 'beta has non-positive values'
print('PASS')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/fit_strength_params.py data/team_strength_params.csv
git commit -m "feat: fit_strength_params.py — MLE α/β for all teams from 49k real matches"
```

---

## Task 5: Strength Loader and Strength-Adjusted xG

**Files:**
- Create: `tests/data/test_strength_loader.py`
- Create: `src/data/strength_loader.py`
- Create: `tests/models/test_strength_adjusted_xg.py`
- Create: `src/models/strength_adjusted_xg.py`

- [ ] **Step 1: Write loader tests**

Create `tests/data/test_strength_loader.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from src.data.strength_loader import load_strength_params, StrengthParams


def _make_csv(tmp_path, rows):
    f = tmp_path / "params.csv"
    df = pd.DataFrame(rows, columns=["team", "alpha_attack", "beta_defense", "matches_used", "as_of_date"])
    df.to_csv(f, index=False)
    return f


def test_returns_dict():
    pass  # inline below


def test_load_returns_dict_of_strength_params(tmp_path):
    csv = _make_csv(tmp_path, [["France", 1.5, 0.7, 24, "2022-11-19"]])
    result = load_strength_params(csv)
    assert isinstance(result, dict)
    assert "France" in result
    assert isinstance(result["France"], StrengthParams)


def test_alpha_beta_correct(tmp_path):
    csv = _make_csv(tmp_path, [["France", 1.52, 0.71, 24, "2022-11-19"]])
    result = load_strength_params(csv)
    assert abs(result["France"].alpha_attack - 1.52) < 1e-9
    assert abs(result["France"].beta_defense - 0.71) < 1e-9


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_strength_params(tmp_path / "missing.csv")
```

- [ ] **Step 2: Write strength-adjusted xG tests**

Create `tests/models/test_strength_adjusted_xg.py`:

```python
import pytest
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg, XG_MIN, XG_MAX
from src.data.strength_loader import StrengthParams


def _params(alpha=1.0, beta=1.0):
    return StrengthParams(alpha_attack=alpha, beta_defense=beta, matches_used=20)


def test_returns_two_floats():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=1900, elo_b=1900,
        params_a=_params(), params_b=_params(),
        ppg_a=1.5, ppg_b=1.5,
    )
    assert isinstance(xg_a, float) and isinstance(xg_b, float)


def test_equal_teams_produce_equal_xg():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=1900, elo_b=1900,
        params_a=_params(1.0, 1.0), params_b=_params(1.0, 1.0),
        ppg_a=1.5, ppg_b=1.5,
    )
    assert abs(xg_a - xg_b) < 1e-9


def test_stronger_attack_increases_xg():
    low, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0), _params(1.0), 1.5, 1.5)
    high, _ = calculate_strength_adjusted_xg(1900, 1900, _params(2.0), _params(1.0), 1.5, 1.5)
    assert high > low


def test_stronger_opponent_defense_decreases_xg():
    base, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0, 1.0), _params(1.0, 1.0), 1.5, 1.5)
    vs_solid, _ = calculate_strength_adjusted_xg(1900, 1900, _params(1.0, 1.0), _params(1.0, 0.5), 1.5, 1.5)
    assert vs_solid < base


def test_higher_elo_increases_xg():
    base, _ = calculate_strength_adjusted_xg(1900, 1900, _params(), _params(), 1.5, 1.5)
    high, _ = calculate_strength_adjusted_xg(2100, 1700, _params(), _params(), 1.5, 1.5)
    assert high > base


def test_output_clamped():
    xg_a, xg_b = calculate_strength_adjusted_xg(
        elo_a=2500, elo_b=1000,
        params_a=_params(10.0, 0.1), params_b=_params(0.1, 10.0),
        ppg_a=3.0, ppg_b=0.0,
    )
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN
```

- [ ] **Step 3: Run to confirm both test files fail**

```bash
python -m pytest tests/data/test_strength_loader.py tests/models/test_strength_adjusted_xg.py -v
```
Expected: ImportError for both.

- [ ] **Step 4: Create `src/data/strength_loader.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"


@dataclass
class StrengthParams:
    alpha_attack: float
    beta_defense: float
    matches_used: int


def load_strength_params(path: Path | None = None) -> dict[str, StrengthParams]:
    """Load MLE team strength parameters from CSV.

    Returns:
        {team_name: StrengthParams}

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT
    if not p.exists():
        raise FileNotFoundError(f"team_strength_params.csv not found: {p}")

    df = pd.read_csv(p)
    return {
        row["team"]: StrengthParams(
            alpha_attack=float(row["alpha_attack"]),
            beta_defense=float(row["beta_defense"]),
            matches_used=int(row["matches_used"]),
        )
        for _, row in df.iterrows()
    }
```

- [ ] **Step 5: Create `src/models/strength_adjusted_xg.py`**

```python
"""Calculate expected goals using MLE-fitted opponent strength parameters.

This replaces the raw goals-average approach in pre_match_xg.py.
Formula: xg_a = BASE_XG * alpha_a * beta_b * form_a * elo_factor_a

The same BASE_XG, XG_MIN, XG_MAX, FORM_BASE, FORM_SCALE constants are used
so both xG calculators are directly comparable.
"""

from src.data.strength_loader import StrengthParams

BASE_XG    = 1.35
XG_MIN     = 0.2
XG_MAX     = 4.5
FORM_BASE  = 0.85
FORM_SCALE = 0.30


def calculate_strength_adjusted_xg(
    elo_a: float,
    elo_b: float,
    params_a: StrengthParams,
    params_b: StrengthParams,
    ppg_a: float,
    ppg_b: float,
) -> tuple[float, float]:
    """Calculate expected goals from MLE strength parameters.

    No manually estimated ratings. No raw goal averages.
    All signal comes from:
    - α: how prolific the team is at scoring (MLE-fitted)
    - β: how easy the opponent is to score against (MLE-fitted)
    - ELO: relative strength for this specific matchup
    - Form: recent PPG as a dynamic form multiplier

    Formula:
        xg_a = BASE_XG * alpha_a * beta_b * form_a * elo_factor_a
        xg_b = BASE_XG * alpha_b * beta_a * form_b * elo_factor_b

        form   = FORM_BASE + (ppg / 3) * FORM_SCALE
        elo_factor_a = 1 + (elo_a - elo_b) / 4000

    Both clamped to [XG_MIN, XG_MAX].
    """
    form_a = FORM_BASE + (ppg_a / 3) * FORM_SCALE
    form_b = FORM_BASE + (ppg_b / 3) * FORM_SCALE

    elo_factor_a = 1 + (elo_a - elo_b) / 4000
    elo_factor_b = 1 + (elo_b - elo_a) / 4000

    xg_a = BASE_XG * params_a.alpha_attack * params_b.beta_defense * form_a * elo_factor_a
    xg_b = BASE_XG * params_b.alpha_attack * params_a.beta_defense * form_b * elo_factor_b

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
```

- [ ] **Step 6: Run all new tests**

```bash
python -m pytest tests/data/test_strength_loader.py tests/models/test_strength_adjusted_xg.py -v
```
Expected: 9 tests pass.

- [ ] **Step 7: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 112 tests pass (103 + 9).

- [ ] **Step 8: Commit**

```bash
git add src/data/strength_loader.py src/models/strength_adjusted_xg.py \
        tests/data/test_strength_loader.py tests/models/test_strength_adjusted_xg.py
git commit -m "feat: strength loader and strength-adjusted xG calculator"
```

---

## Task 6: Strength Runner and Backtest Comparison

**Files:**
- Create: `tests/backtesting/test_strength_runner.py`
- Create: `src/backtesting/strength_runner.py`

This runner uses `match_results.csv` (real data) + `team_strength_params.csv` (MLE params) to run the WC 2022 backtest. It produces `MatchResult` objects compatible with all existing metrics code.

- [ ] **Step 1: Write runner tests**

Create `tests/backtesting/test_strength_runner.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.strength_runner import run_strength_backtest
from src.backtesting.runner import MatchResult


def _make_match_results_csv(tmp_path, rows):
    f = tmp_path / "match_results.csv"
    cols = [
        "match_id","date","team_a","team_b","team_a_goals","team_b_goals",
        "team_a_elo_pre","team_b_elo_pre",
        "team_a_goals_for_last_10","team_a_goals_against_last_10",
        "team_b_goals_for_last_10","team_b_goals_against_last_10",
        "team_a_points_per_game_last_10","team_b_points_per_game_last_10",
        "team_a_matches_available","team_b_matches_available",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(f, index=False)
    return f


def _make_strength_params_csv(tmp_path, rows):
    f = tmp_path / "params.csv"
    pd.DataFrame(rows, columns=["team","alpha_attack","beta_defense","matches_used","as_of_date"]).to_csv(f, index=False)
    return f


def _row(team_a="France", team_b="Brazil", ga=2, gb=1):
    return [1,"2022-11-23",team_a,team_b,ga,gb,2015,2044,1.8,0.7,2.1,0.7,2.5,2.6,10,10]


def _params(teams=("France","Brazil")):
    return [[t, 1.4, 0.8, 20, "2022-11-19"] for t in teams]


def test_returns_list_of_match_result(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_works_with_poisson(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp, model_type="poisson")
    assert results[0].win_a_prob > 0


def test_works_with_dixon_coles(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row()])
    sp = _make_strength_params_csv(tmp_path, _params())
    results = run_strength_backtest(mr, sp, model_type="dixon_coles", rho=-0.20)
    assert results[0].win_a_prob > 0


def test_missing_team_in_params_raises(tmp_path):
    mr = _make_match_results_csv(tmp_path, [_row("France", "Brazil")])
    sp = _make_strength_params_csv(tmp_path, [["France", 1.4, 0.8, 20, "2022-11-19"]])  # Brazil missing
    with pytest.raises(ValueError, match="Brazil"):
        run_strength_backtest(mr, sp)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_strength_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/backtesting/strength_runner.py`**

```python
"""Run backtesting using real match data and MLE-fitted strength parameters.

Unlike run_valid_backtest() (which uses raw goal averages), this runner
uses strength-adjusted xG derived from Dixon-Coles MLE parameters.

Data sources:
    data/match_results.csv        -- real historical pre-match stats
    data/team_strength_params.csv -- MLE α/β for each team
"""

from pathlib import Path
import pandas as pd

from src.data.strength_loader import load_strength_params
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import MatchResult

_DEFAULT_MATCH_RESULTS = Path(__file__).parent.parent.parent / "data" / "match_results.csv"
_DEFAULT_STRENGTH_PARAMS = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"


def run_strength_backtest(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
) -> list[MatchResult]:
    """Run backtest using real data and MLE strength parameters.

    Args:
        match_results_path: Path to match_results.csv. Default: data/match_results.csv.
        strength_params_path: Path to team_strength_params.csv. Default: data/team_strength_params.csv.
        model_type: "poisson" or "dixon_coles".
        rho: Dixon-Coles rho parameter (ignored for Poisson).
        filter_date_from: Optional ISO date string — only include matches from this date.
        filter_date_to: Optional ISO date string — only include matches up to this date.

    Returns:
        list[MatchResult] — same structure as run_backtest() and run_valid_backtest().

    Raises:
        FileNotFoundError: if either CSV is missing.
        ValueError: if a team in match_results is not in strength_params, or invalid model_type.
    """
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")

    mr_path = match_results_path or _DEFAULT_MATCH_RESULTS
    sp_path = strength_params_path or _DEFAULT_STRENGTH_PARAMS

    df = pd.read_csv(mr_path)
    strength = load_strength_params(sp_path)

    if filter_date_from:
        df = df[df["date"] >= filter_date_from]
    if filter_date_to:
        df = df[df["date"] <= filter_date_to]

    results = []
    for _, row in df.iterrows():
        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if team_a not in strength:
            raise ValueError(f"Team '{team_a}' not found in strength parameters")
        if team_b not in strength:
            raise ValueError(f"Team '{team_b}' not found in strength parameters")

        xg_a, xg_b = calculate_strength_adjusted_xg(
            elo_a=float(row["team_a_elo_pre"]),
            elo_b=float(row["team_b_elo_pre"]),
            params_a=strength[team_a],
            params_b=strength[team_b],
            ppg_a=float(row["team_a_points_per_game_last_10"]),
            ppg_b=float(row["team_b_points_per_game_last_10"]),
        )

        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)

        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])

        if goals_a > goals_b:
            actual_outcome = "team_a_win"
        elif goals_a == goals_b:
            actual_outcome = "draw"
        else:
            actual_outcome = "team_b_win"

        probs = {
            "team_a_win": prediction.win_a,
            "draw": prediction.draw,
            "team_b_win": prediction.win_b,
        }
        predicted_outcome = max(probs, key=probs.get)
        top5 = [(g_a, g_b) for g_a, g_b, _ in prediction.top_scorelines]

        results.append(MatchResult(
            date=str(row["date"]),
            team_a=team_a,
            team_b=team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            win_a_prob=prediction.win_a,
            draw_prob=prediction.draw,
            win_b_prob=prediction.win_b,
            predicted_outcome=predicted_outcome,
            top_scorelines=prediction.top_scorelines,
            exact_score_hit=len(top5) > 0 and top5[0] == (goals_a, goals_b),
            in_top_3=(goals_a, goals_b) in top5[:3],
            in_top_5=(goals_a, goals_b) in top5,
            prob_of_actual_result=probs[actual_outcome],
        ))

    return results
```

- [ ] **Step 4: Run runner tests**

```bash
python -m pytest tests/backtesting/test_strength_runner.py -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 116 tests pass (112 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/strength_runner.py tests/backtesting/test_strength_runner.py
git commit -m "feat: strength runner — backtest using real data + MLE strength parameters"
```

---

## Task 7: Backtest Comparison Report

**Files:**
- No new code — runs existing pipeline and prints results

This task produces the Sprint 1 success criteria: **Brier score comparison** between the V6 placeholder baseline and the new real-data MLE-based backtest.

- [ ] **Step 1: Run the WC 2022 strength backtest**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.backtesting.strength_runner import run_strength_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import DEFAULT_RHO_GRID, RhoResult, select_best_rho

# WC 2022 dates
DATE_FROM = '2022-11-20'
DATE_TO   = '2022-12-18'

# === Real data + MLE strength (Sprint 1) ===
po_results = run_strength_backtest(filter_date_from=DATE_FROM, filter_date_to=DATE_TO, model_type='poisson')
po_m = compute_metrics(po_results)

print('=== SPRINT 1 BACKTEST — Real Data + MLE Strength Parameters ===')
print(f'Matches: {po_m.total_matches}')
print(f'1X2 Accuracy: {po_m.accuracy_1x2:.1%}')
print(f'Brier Score:  {po_m.brier_score:.4f}')
print(f'Top 3 Hit:    {po_m.top_3_hit_rate:.1%}')
print()

# Rho grid on real data
print('Dixon-Coles rho grid (real data):')
print('rho     Brier    1X2     Top3')
rho_results = []
for rho in DEFAULT_RHO_GRID:
    dc_m = compute_metrics(run_strength_backtest(
        filter_date_from=DATE_FROM, filter_date_to=DATE_TO,
        model_type='dixon_coles', rho=rho
    ))
    rho_results.append(RhoResult(
        rho=rho, accuracy_1x2=dc_m.accuracy_1x2,
        exact_score_accuracy=dc_m.exact_score_accuracy,
        top_3_hit_rate=dc_m.top_3_hit_rate,
        top_5_hit_rate=dc_m.top_5_hit_rate,
        brier_score=dc_m.brier_score,
        avg_prob_actual_result=dc_m.avg_prob_actual_result,
    ))
    print(f'{rho:+.2f}  {dc_m.brier_score:.4f}  {dc_m.accuracy_1x2:.1%}  {dc_m.top_3_hit_rate:.1%}')

best = select_best_rho(rho_results)
print(f'Best rho: {best.rho:.2f}  (Brier={best.brier_score:.4f})')

# === V6 placeholder baseline ===
from src.backtesting.valid_runner import run_valid_backtest
v6_po = compute_metrics(run_valid_backtest(model_type='poisson'))
print()
print('=== COMPARISON ===')
print(f'V6 placeholder Poisson Brier:          {v6_po.brier_score:.4f}  ({v6_po.total_matches} matches)')
print(f'Sprint 1 real+MLE Poisson Brier:       {po_m.brier_score:.4f}  ({po_m.total_matches} matches)')
print(f'Sprint 1 best DC Brier (rho={best.rho:.2f}): {best.brier_score:.4f}')
improvement = v6_po.brier_score - po_m.brier_score
if improvement > 0:
    print(f'Improvement: {improvement:.4f} ({improvement/v6_po.brier_score*100:.1f}% reduction in Brier score)')
else:
    print(f'No improvement: {improvement:.4f}')
"
```

- [ ] **Step 2: Final full test run**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 116 tests pass.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: Phase 2 Sprint 1 complete — real ELO + match data + MLE strength parameters"
```
