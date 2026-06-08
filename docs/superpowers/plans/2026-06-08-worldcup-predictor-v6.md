# World Cup Predictor v6 — Valid Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structurally valid backtest path that derives xG purely from pre-match statistics, with clear labelling separating it from the illustrative backtest.

**Architecture:** New CSV → new loader → new xG formula → new runner → Streamlit adds two labelled sections. The existing runner and team_ratings.csv are never touched.

**Tech Stack:** Python 3.10+, pandas, streamlit, pytest (no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `data/pre_match_team_stats.csv` | Create — 25 placeholder sample rows |
| `docs/valid_backtest_status.md` | Create — status and sourcing requirements |
| `src/data/pre_match_loader.py` | Create — `PreMatchStats`, `load_pre_match_stats()` |
| `src/models/pre_match_xg.py` | Create — `calculate_pre_match_xg()` |
| `src/backtesting/valid_runner.py` | Create — `run_valid_backtest()` |
| `src/app/app.py` | Modify — Illustrative / Valid labels + valid section |
| `tests/data/test_pre_match_loader.py` | Create |
| `tests/models/test_pre_match_xg.py` | Create |
| `tests/backtesting/test_valid_runner.py` | Create |

---

## Task 1: Pre-Match Stats Dataset + Status Doc

**Files:**
- Create: `data/pre_match_team_stats.csv`
- Create: `docs/valid_backtest_status.md`

- [ ] **Step 1: Create `data/pre_match_team_stats.csv`**

```
# WARNING: PLACEHOLDER DATA — NOT SOURCED FROM REAL PRE-MATCH STATISTICS
# This file is structurally valid but values are illustrative only.
# See docs/valid_backtest_status.md for sourcing requirements.
# goals_for/against columns = avg goals per game over the last 10 matches
# points_per_game columns = avg points per game (win=3, draw=1) over last 10
match_id,date,team_a,team_b,team_a_elo_pre,team_b_elo_pre,team_a_goals_for_last_10,team_a_goals_against_last_10,team_b_goals_for_last_10,team_b_goals_against_last_10,team_a_points_per_game_last_10,team_b_points_per_game_last_10,team_a_matches_available,team_b_matches_available,team_a_goals,team_b_goals
1,2022-11-20,Ecuador,Qatar,1755,1634,1.40,0.90,0.80,1.60,2.10,0.90,10,10,2,0
2,2022-11-21,England,Iran,1978,1714,1.70,0.80,1.10,1.40,2.30,1.40,10,10,6,2
3,2022-11-21,Senegal,Netherlands,1835,1968,1.20,0.90,1.80,0.90,1.90,2.40,10,10,0,2
4,2022-11-22,USA,Wales,1815,1798,1.30,1.00,1.20,1.10,1.80,1.70,10,10,1,1
5,2022-11-22,Argentina,Saudi Arabia,2064,1693,1.80,0.70,1.00,1.30,2.60,1.50,10,10,1,2
6,2022-11-22,Denmark,Tunisia,1904,1675,1.40,0.80,0.90,1.20,2.10,1.30,10,10,0,0
7,2022-11-23,Mexico,Poland,1836,1832,1.30,0.90,1.10,1.00,1.90,1.80,10,10,0,0
8,2022-11-23,France,Australia,2015,1726,2.00,0.70,1.10,1.40,2.50,1.50,10,10,4,1
9,2022-11-24,Morocco,Croatia,1864,1933,1.10,0.90,1.30,0.80,1.80,2.10,10,10,0,0
10,2022-11-24,Germany,Japan,1979,1815,1.90,0.90,1.30,0.90,2.40,1.90,10,10,1,2
11,2022-11-24,Spain,Costa Rica,1969,1706,2.10,0.60,0.90,1.50,2.50,1.40,10,10,7,0
12,2022-11-25,Belgium,Canada,1934,1755,1.60,0.80,1.20,1.10,2.20,1.80,10,10,1,0
13,2022-11-25,Switzerland,Cameroon,1864,1694,1.40,0.90,1.00,1.40,2.00,1.40,10,10,1,0
14,2022-11-26,Uruguay,South Korea,1894,1795,1.20,0.80,1.10,1.10,1.80,1.70,10,10,0,0
15,2022-11-26,Portugal,Ghana,2004,1675,1.80,0.80,1.10,1.50,2.40,1.50,10,10,3,2
16,2022-11-27,Brazil,Serbia,2044,1846,2.10,0.70,1.20,1.10,2.60,1.90,10,10,2,0
17,2022-11-28,France,Denmark,2015,1898,2.00,0.70,1.40,0.80,2.50,2.10,10,10,2,1
18,2022-11-28,Argentina,Mexico,2064,1836,1.80,0.70,1.30,1.00,2.60,1.90,10,10,2,0
19,2022-11-29,Japan,Costa Rica,1815,1706,1.30,0.90,0.90,1.50,1.90,1.40,10,4,0,1
20,2022-11-29,Germany,Spain,1979,1969,1.90,0.90,2.10,0.60,2.40,2.50,10,10,1,1
21,2022-11-30,Belgium,Morocco,1934,1864,1.60,0.80,1.10,0.90,2.20,1.80,10,10,0,2
22,2022-11-30,Croatia,Canada,1933,1755,1.30,0.80,1.20,1.10,2.10,1.80,10,10,4,1
23,2022-12-01,Brazil,Switzerland,2044,1864,2.10,0.70,1.40,0.90,2.60,2.00,10,4,1,0
24,2022-12-01,Portugal,Uruguay,2004,1894,1.80,0.80,1.20,0.80,2.40,1.80,10,10,2,0
25,2022-12-02,South Korea,Ghana,1795,1675,1.10,1.10,1.10,1.50,1.70,1.50,4,10,2,3
```

Note: Rows 19, 23, 25 have one team with matches_available=4 (below the default min of 5) for testing purposes.

- [ ] **Step 2: Verify CSV parses correctly**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/pre_match_team_stats.csv', comment='#')
print(len(df), 'rows')
print(df.columns.tolist())
print('Rows with <5 matches:', len(df[(df.team_a_matches_available < 5) | (df.team_b_matches_available < 5)]))
"
```
Expected: 25 rows, correct columns, 3 rows with <5 matches.

- [ ] **Step 3: Create `docs/valid_backtest_status.md`**

```markdown
# Valid Backtest — Status

## Current Status

**Engineering validity:** ✅ Complete  
**Data validity:** ❌ Placeholder only  

The infrastructure for a valid backtest is fully implemented. The data is not yet real.

## What Is Placeholder

`data/pre_match_team_stats.csv` contains 25 sample rows with plausible but not
historically sourced values. The ELO ratings, goals averages, and form statistics
were manually estimated as illustrative values — they are NOT derived from real
match-by-match records.

## What Would Make This Research-Grade

To replace the placeholder with real data, each row needs:

### ELO ratings
- Source: World Football Elo Ratings (https://www.eloratings.net)
- Method: Download the historical ELO table, join on team name and date.
  Use the ELO value as of the day BEFORE the match date.

### Goals for/against (last 10)
- Source: football-data.org (free tier, requires API key)
- Method: For each team and match date, fetch their last 10 international
  results BEFORE that date. Compute avg goals scored and avg goals conceded.

### Points per game (last 10)
- Source: Same football-data.org API
- Method: Compute 3 for win, 1 for draw, 0 for loss; average over last 10 matches.

### Matches available
- Method: Count of actual historical matches found before the date (may be <10
  for recently-formed national teams or after a long gap).

## Implementation Path

1. Register for football-data.org free API key.
2. Write `scripts/fetch_pre_match_stats.py` that for each row in
   `data/historical_matches.csv`:
   - Queries the team's last 10 international results before the match date
   - Looks up ELO from eloratings.net historical data
   - Writes a verified row to `data/pre_match_team_stats.csv`
3. Remove the `# WARNING` header from the CSV.
4. Update this document to reflect data validity: ✅
5. Re-run the valid backtest. Results will now constitute a genuine
   out-of-sample evaluation.

## Test Coverage

All code paths are tested. The valid runner, pre-match loader, and pre-match xG
calculator have full test suites that do not depend on real data values.
```

- [ ] **Step 4: Commit**

```bash
git add data/pre_match_team_stats.csv docs/valid_backtest_status.md
git commit -m "data: add pre_match_team_stats.csv (placeholder) and valid_backtest_status.md"
```

---

## Task 2: Pre-Match Loader

**Files:**
- Create: `src/data/pre_match_loader.py`
- Create: `tests/data/test_pre_match_loader.py`

- [ ] **Step 1: Create `tests/data/test_pre_match_loader.py`**

```python
import pytest
import pandas as pd
from pathlib import Path
from src.data.pre_match_loader import load_pre_match_stats, PreMatchStats


def _make_csv(tmp_path, rows, header=True):
    """Write a minimal pre-match CSV for testing."""
    f = tmp_path / "stats.csv"
    cols = [
        "match_id","date","team_a","team_b",
        "team_a_elo_pre","team_b_elo_pre",
        "team_a_goals_for_last_10","team_a_goals_against_last_10",
        "team_b_goals_for_last_10","team_b_goals_against_last_10",
        "team_a_points_per_game_last_10","team_b_points_per_game_last_10",
        "team_a_matches_available","team_b_matches_available",
        "team_a_goals","team_b_goals",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f, index=False)
    return f


def _row(match_id=1, ma=10, mb=10):
    return [match_id,"2022-11-20","France","Brazil",2015,2044,
            1.8,0.7,2.1,0.7,2.5,2.6,ma,mb,2,0]


def test_returns_list_of_pre_match_stats(tmp_path):
    csv = _make_csv(tmp_path, [_row()])
    result = load_pre_match_stats(csv)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], PreMatchStats)


def test_comment_lines_are_skipped(tmp_path):
    f = tmp_path / "stats.csv"
    f.write_text(
        "# WARNING: PLACEHOLDER\n"
        "match_id,date,team_a,team_b,team_a_elo_pre,team_b_elo_pre,"
        "team_a_goals_for_last_10,team_a_goals_against_last_10,"
        "team_b_goals_for_last_10,team_b_goals_against_last_10,"
        "team_a_points_per_game_last_10,team_b_points_per_game_last_10,"
        "team_a_matches_available,team_b_matches_available,"
        "team_a_goals,team_b_goals\n"
        "1,2022-11-20,A,B,2000,1900,1.5,0.8,1.2,1.0,2.2,1.8,10,10,2,1\n"
    )
    result = load_pre_match_stats(f)
    assert len(result) == 1


def test_exclude_insufficient_removes_low_match_rows(tmp_path):
    csv = _make_csv(tmp_path, [_row(1, ma=10, mb=10), _row(2, ma=3, mb=10), _row(3, ma=10, mb=4)])
    full = load_pre_match_stats(csv, exclude_insufficient=False)
    filtered = load_pre_match_stats(csv, exclude_insufficient=True, min_matches=5)
    assert len(full) == 3
    assert len(filtered) == 1  # only the row with both >= 5


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pre_match_stats(tmp_path / "missing.csv")


def test_missing_column_raises(tmp_path):
    f = tmp_path / "stats.csv"
    f.write_text("match_id,date\n1,2022-11-20\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_pre_match_stats(f)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/data/test_pre_match_loader.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/data/pre_match_loader.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT_CSV = Path(__file__).parent.parent.parent / "data" / "pre_match_team_stats.csv"

_REQUIRED_COLUMNS = {
    "match_id", "date", "team_a", "team_b",
    "team_a_elo_pre", "team_b_elo_pre",
    "team_a_goals_for_last_10", "team_a_goals_against_last_10",
    "team_b_goals_for_last_10", "team_b_goals_against_last_10",
    "team_a_points_per_game_last_10", "team_b_points_per_game_last_10",
    "team_a_matches_available", "team_b_matches_available",
    "team_a_goals", "team_b_goals",
}


@dataclass
class PreMatchStats:
    """Pre-match statistics for one match, derived only from pre-game records."""
    match_id: int
    date: str
    team_a: str
    team_b: str
    team_a_elo_pre: float
    team_b_elo_pre: float
    team_a_goals_for_last_10: float       # avg goals scored per game, last 10 matches
    team_a_goals_against_last_10: float   # avg goals conceded per game, last 10 matches
    team_b_goals_for_last_10: float
    team_b_goals_against_last_10: float
    team_a_points_per_game_last_10: float
    team_b_points_per_game_last_10: float
    team_a_matches_available: int         # actual matches found (≤10); flag if <min_matches
    team_b_matches_available: int
    team_a_goals: int                     # actual match result (for backtesting)
    team_b_goals: int


def load_pre_match_stats(
    path: Path | None = None,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[PreMatchStats]:
    """Load pre-match statistics from CSV.

    Args:
        path: Path to CSV. Defaults to data/pre_match_team_stats.csv.
        min_matches: Minimum matches_available to consider reliable.
        exclude_insufficient: If True, exclude rows where either team has
                              fewer than min_matches. If False (default),
                              include all rows.

    Raises:
        FileNotFoundError: if CSV is missing.
        ValueError: if required columns are missing.
    """
    p = path if path is not None else _DEFAULT_CSV

    if not p.exists():
        raise FileNotFoundError(f"Pre-match stats file not found: {p}")

    # comment='#' skips the WARNING header lines
    df = pd.read_csv(p, comment='#')

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"pre_match_team_stats.csv missing columns: {sorted(missing)}")

    if exclude_insufficient:
        df = df[
            (df["team_a_matches_available"] >= min_matches) &
            (df["team_b_matches_available"] >= min_matches)
        ]

    stats = []
    for _, row in df.iterrows():
        stats.append(PreMatchStats(
            match_id=int(row["match_id"]),
            date=str(row["date"]),
            team_a=str(row["team_a"]).strip(),
            team_b=str(row["team_b"]).strip(),
            team_a_elo_pre=float(row["team_a_elo_pre"]),
            team_b_elo_pre=float(row["team_b_elo_pre"]),
            team_a_goals_for_last_10=float(row["team_a_goals_for_last_10"]),
            team_a_goals_against_last_10=float(row["team_a_goals_against_last_10"]),
            team_b_goals_for_last_10=float(row["team_b_goals_for_last_10"]),
            team_b_goals_against_last_10=float(row["team_b_goals_against_last_10"]),
            team_a_points_per_game_last_10=float(row["team_a_points_per_game_last_10"]),
            team_b_points_per_game_last_10=float(row["team_b_points_per_game_last_10"]),
            team_a_matches_available=int(row["team_a_matches_available"]),
            team_b_matches_available=int(row["team_b_matches_available"]),
            team_a_goals=int(row["team_a_goals"]),
            team_b_goals=int(row["team_b_goals"]),
        ))

    return stats
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/data/test_pre_match_loader.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 72 tests pass (67 + 5).

- [ ] **Step 6: Commit**

```bash
git add src/data/pre_match_loader.py tests/data/test_pre_match_loader.py
git commit -m "feat: pre-match stats loader — load_pre_match_stats() with comment skip and filtering"
```

---

## Task 3: Pre-Match xG Calculator

**Files:**
- Create: `src/models/pre_match_xg.py`
- Create: `tests/models/test_pre_match_xg.py`

- [ ] **Step 1: Create `tests/models/test_pre_match_xg.py`**

```python
import pytest
from src.models.pre_match_xg import calculate_pre_match_xg, BASE_XG, XG_MIN, XG_MAX
from src.data.pre_match_loader import PreMatchStats


def _match(
    elo_a=1900, elo_b=1900,
    gf_a=1.35, ga_a=1.35,
    gf_b=1.35, ga_b=1.35,
    ppg_a=1.5, ppg_b=1.5,
):
    """Build a test PreMatchStats with sensible defaults (1.0 multipliers)."""
    return PreMatchStats(
        match_id=1, date="2022-11-20", team_a="A", team_b="B",
        team_a_elo_pre=elo_a, team_b_elo_pre=elo_b,
        team_a_goals_for_last_10=gf_a, team_a_goals_against_last_10=ga_a,
        team_b_goals_for_last_10=gf_b, team_b_goals_against_last_10=ga_b,
        team_a_points_per_game_last_10=ppg_a, team_b_points_per_game_last_10=ppg_b,
        team_a_matches_available=10, team_b_matches_available=10,
        team_a_goals=1, team_b_goals=0,
    )


def test_returns_two_floats():
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert isinstance(xg_a, float)
    assert isinstance(xg_b, float)


def test_equal_teams_produce_equal_xg():
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert abs(xg_a - xg_b) < 1e-9


def test_average_team_produces_xg_near_base():
    # All multipliers = 1.0 → xg should be close to BASE_XG
    xg_a, xg_b = calculate_pre_match_xg(_match())
    assert abs(xg_a - BASE_XG) < 0.05


def test_higher_elo_gets_higher_xg():
    xg_a, xg_b = calculate_pre_match_xg(_match(elo_a=2100, elo_b=1700))
    assert xg_a > xg_b


def test_higher_attack_increases_xg():
    base_a, _ = calculate_pre_match_xg(_match(gf_a=1.35))
    high_a, _ = calculate_pre_match_xg(_match(gf_a=2.00))
    assert high_a > base_a


def test_strong_defense_decreases_opponent_xg():
    # Team B with low goals_against → lower xg_a
    normal, _ = calculate_pre_match_xg(_match(ga_b=1.35))
    tight, _ = calculate_pre_match_xg(_match(ga_b=0.70))
    assert tight < normal


def test_higher_form_increases_xg():
    low_form_a, _ = calculate_pre_match_xg(_match(ppg_a=0.5))
    high_form_a, _ = calculate_pre_match_xg(_match(ppg_a=2.8))
    assert high_form_a > low_form_a


def test_output_clamped_to_bounds():
    # Extreme values should clamp
    xg_a, xg_b = calculate_pre_match_xg(_match(
        elo_a=2500, elo_b=1000,
        gf_a=3.0, ga_b=3.0, ppg_a=3.0,
        gf_b=0.3, ga_a=0.3, ppg_b=0.0,
    ))
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_pre_match_xg.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/models/pre_match_xg.py`**

```python
from src.data.pre_match_loader import PreMatchStats

BASE_XG    = 1.35
XG_MIN     = 0.2
XG_MAX     = 4.5
FORM_BASE  = 0.85
FORM_SCALE = 0.30   # range above FORM_BASE (max form = 0.85 + 0.30 = 1.15)


def calculate_pre_match_xg(match: PreMatchStats) -> tuple[float, float]:
    """Calculate expected goals using only pre-match statistics.

    No manually estimated ratings are used. All factors come from:
    - Per-game goal averages over the last 10 matches
    - Per-game points average over the last 10 matches
    - Pre-match ELO ratings

    Formula:
        attack_a  = goals_for_a  / BASE_XG   (1.0 = average)
        defense_b = goals_against_b / BASE_XG (1.0 = average)
        attack_b  = goals_for_b  / BASE_XG
        defense_a = goals_against_a / BASE_XG

        form_a = FORM_BASE + (ppg_a / 3) * FORM_SCALE
        form_b = FORM_BASE + (ppg_b / 3) * FORM_SCALE

        elo_factor_a = 1 + (elo_a - elo_b) / 4000
        elo_factor_b = 1 + (elo_b - elo_a) / 4000

        xg_a = BASE_XG * attack_a * defense_b * form_a * elo_factor_a
        xg_b = BASE_XG * attack_b * defense_a * form_b * elo_factor_b

    Both values are clamped to [XG_MIN, XG_MAX].
    """
    attack_a  = match.team_a_goals_for_last_10     / BASE_XG
    defense_b = match.team_b_goals_against_last_10 / BASE_XG
    attack_b  = match.team_b_goals_for_last_10     / BASE_XG
    defense_a = match.team_a_goals_against_last_10 / BASE_XG

    form_a = FORM_BASE + (match.team_a_points_per_game_last_10 / 3) * FORM_SCALE
    form_b = FORM_BASE + (match.team_b_points_per_game_last_10 / 3) * FORM_SCALE

    elo_factor_a = 1 + (match.team_a_elo_pre - match.team_b_elo_pre) / 4000
    elo_factor_b = 1 + (match.team_b_elo_pre - match.team_a_elo_pre) / 4000

    xg_a = BASE_XG * attack_a * defense_b * form_a * elo_factor_a
    xg_b = BASE_XG * attack_b * defense_a * form_b * elo_factor_b

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
```

- [ ] **Step 4: Run pre-match xG tests**

```bash
python -m pytest tests/models/test_pre_match_xg.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 80 tests pass (72 + 8).

- [ ] **Step 6: Commit**

```bash
git add src/models/pre_match_xg.py tests/models/test_pre_match_xg.py
git commit -m "feat: pre-match xG calculator — no manually estimated ratings"
```

---

## Task 4: Valid Runner

**Files:**
- Create: `src/backtesting/valid_runner.py`
- Create: `tests/backtesting/test_valid_runner.py`

- [ ] **Step 1: Create `tests/backtesting/test_valid_runner.py`**

```python
import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.runner import MatchResult


def _make_stats_csv(tmp_path, rows):
    """Write a minimal pre-match CSV."""
    f = tmp_path / "stats.csv"
    cols = [
        "match_id","date","team_a","team_b",
        "team_a_elo_pre","team_b_elo_pre",
        "team_a_goals_for_last_10","team_a_goals_against_last_10",
        "team_b_goals_for_last_10","team_b_goals_against_last_10",
        "team_a_points_per_game_last_10","team_b_points_per_game_last_10",
        "team_a_matches_available","team_b_matches_available",
        "team_a_goals","team_b_goals",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f, index=False)
    return f


def _row(match_id=1, ma=10, mb=10, ga=2, gb=1):
    return [match_id,"2022-11-20","France","Brazil",2015,2044,
            1.8,0.7,2.1,0.7,2.5,2.6,ma,mb,ga,gb]


def test_returns_list_of_match_results(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_works_with_poisson_model(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv, model_type="poisson")
    assert results[0].win_a_prob > 0


def test_works_with_dixon_coles_model(tmp_path):
    csv = _make_stats_csv(tmp_path, [_row()])
    results = run_valid_backtest(csv, model_type="dixon_coles", rho=-0.20)
    assert results[0].win_a_prob > 0


def test_exclude_insufficient_reduces_results(tmp_path):
    rows = [_row(1, ma=10, mb=10), _row(2, ma=3, mb=10), _row(3, ma=10, mb=4)]
    csv = _make_stats_csv(tmp_path, rows)
    full = run_valid_backtest(csv, exclude_insufficient=False)
    filtered = run_valid_backtest(csv, exclude_insufficient=True, min_matches=5)
    assert len(full) == 3
    assert len(filtered) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_valid_backtest(tmp_path / "missing.csv")


def test_does_not_use_team_ratings_csv(tmp_path, monkeypatch):
    """Verify valid_runner never reads team_ratings.csv."""
    import src.data.loader as loader_module
    call_count = [0]
    original = loader_module.load_team_ratings

    def patched(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "load_team_ratings", patched)
    csv = _make_stats_csv(tmp_path, [_row()])
    run_valid_backtest(csv)
    assert call_count[0] == 0, "valid_runner must not call load_team_ratings()"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_valid_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/backtesting/valid_runner.py`**

```python
from pathlib import Path

from src.data.pre_match_loader import load_pre_match_stats
from src.models.pre_match_xg import calculate_pre_match_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import MatchResult   # reuse same result dataclass


def run_valid_backtest(
    path: Path | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[MatchResult]:
    """Run backtest using ONLY pre-match statistics — no manually estimated ratings.

    Unlike run_backtest() (which reads team_ratings.csv), this function derives
    xG exclusively from per-match pre-game statistics in pre_match_team_stats.csv.

    Args:
        path: Path to pre_match_team_stats.csv. Defaults to the project data file.
        model_type: "poisson" (default) or "dixon_coles".
        rho: Dixon-Coles parameter. Only used when model_type == "dixon_coles".
        min_matches: Minimum historical matches required to be considered reliable.
        exclude_insufficient: If True, skip rows where either team has fewer than
                              min_matches available.

    Returns:
        list[MatchResult] — same structure as run_backtest(), compatible with
        compute_metrics() and all downstream analysis.

    Raises:
        FileNotFoundError: if CSV is missing.
        ValueError: if model_type is invalid.
    """
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")

    match_stats = load_pre_match_stats(
        path=path,
        min_matches=min_matches,
        exclude_insufficient=exclude_insufficient,
    )

    results = []
    for m in match_stats:
        xg_a, xg_b = calculate_pre_match_xg(m)

        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(m.team_a, m.team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(m.team_a, m.team_b, xg_a, xg_b)

        goals_a = m.team_a_goals
        goals_b = m.team_b_goals

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
        exact_score_hit = len(top5) > 0 and top5[0] == (goals_a, goals_b)
        in_top_3 = (goals_a, goals_b) in top5[:3]
        in_top_5 = (goals_a, goals_b) in top5

        results.append(MatchResult(
            date=m.date,
            team_a=m.team_a,
            team_b=m.team_b,
            actual_goals_a=goals_a,
            actual_goals_b=goals_b,
            actual_outcome=actual_outcome,
            win_a_prob=prediction.win_a,
            draw_prob=prediction.draw,
            win_b_prob=prediction.win_b,
            predicted_outcome=predicted_outcome,
            top_scorelines=prediction.top_scorelines,
            exact_score_hit=exact_score_hit,
            in_top_3=in_top_3,
            in_top_5=in_top_5,
            prob_of_actual_result=probs[actual_outcome],
        ))

    return results
```

- [ ] **Step 4: Run valid runner tests**

```bash
python -m pytest tests/backtesting/test_valid_runner.py -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 86 tests pass (80 + 6).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/valid_runner.py tests/backtesting/test_valid_runner.py
git commit -m "feat: valid backtest runner — xG from pre-match stats only"
```

---

## Task 5: Streamlit Labels + Valid Section

**Files:**
- Modify: `src/app/app.py`

- [ ] **Step 1: Add import at top of app.py**

After the rho_tuning import, add:

```python
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.rho_tuning import tune_rho as tune_rho_valid
```

Wait — `tune_rho` in rho_tuning.py uses `run_backtest` (ratings-based). For the valid path we need a separate tuner. Instead, call `run_valid_backtest` in a loop directly in the app. Add only:

```python
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.rho_tuning import DEFAULT_RHO_GRID
```

(DEFAULT_RHO_GRID is already imported via rho_tuning; check if already imported — if so, skip the duplicate)

- [ ] **Step 2: Replace the backtesting tab header**

Change:
```python
    st.markdown("Model accuracy validated against historical international match results.")
```
To:
```python
    st.markdown(
        "This tab shows two separate backtests with clearly different data provenance."
    )
```

- [ ] **Step 3: Add section labels to the existing illustrative backtest**

Before `st.subheader("Model Comparison")` add:

```python
        st.warning(
            "⚠️ **Illustrative Backtest** — uses `team_ratings.csv` (manually estimated by AI). "
            "Ratings were assigned with knowledge of WC 2022 outcomes. "
            "Results are for **engineering validation only**, not accuracy measurement."
        )
```

- [ ] **Step 4: Add the Valid Backtest section at the end of the backtesting tab**

After the rho tuning section (at the very bottom of `with tab_backtest:`), add:

```python
        # ══ Valid Pre-Match Backtest ══════════════════════════════════════════
        st.markdown("---")
        st.subheader("Valid Pre-Match Backtest")
        st.info(
            "📐 **Data provenance:** xG calculated from pre-match statistics only "
            "(goals averages, form, ELO). No manually estimated ratings used.\n\n"
            "⚠️ **PLACEHOLDER DATA:** `pre_match_team_stats.csv` contains sample values, "
            "not real historical records. See `docs/valid_backtest_status.md`."
        )

        valid_results_po = None
        valid_metrics_po = None
        valid_rho_results = None
        valid_best_rho = None
        try:
            valid_results_po = run_valid_backtest(model_type="poisson")
            valid_metrics_po = compute_metrics(valid_results_po)

            # Rho grid search on valid path
            valid_rho_results = []
            for rho_val in DEFAULT_RHO_GRID:
                dc_results = run_valid_backtest(model_type="dixon_coles", rho=rho_val)
                m = compute_metrics(dc_results)
                from src.backtesting.rho_tuning import RhoResult
                valid_rho_results.append(RhoResult(
                    rho=rho_val,
                    accuracy_1x2=m.accuracy_1x2,
                    exact_score_accuracy=m.exact_score_accuracy,
                    top_3_hit_rate=m.top_3_hit_rate,
                    top_5_hit_rate=m.top_5_hit_rate,
                    brier_score=m.brier_score,
                    avg_prob_actual_result=m.avg_prob_actual_result,
                ))
            from src.backtesting.rho_tuning import select_best_rho
            valid_best_rho = select_best_rho(valid_rho_results)
        except Exception as e:
            st.error(f"Valid backtest failed: {e}")

        if valid_metrics_po is not None:
            valid_metrics_data = {
                "Metric": [
                    "Total Matches Tested", "1X2 Accuracy", "Exact Score Accuracy",
                    "Top 3 Hit Rate", "Top 5 Hit Rate",
                    "Brier Score (lower = better)", "Avg P(Actual Result)",
                ],
                "Poisson (pre-match xG)": [
                    str(valid_metrics_po.total_matches),
                    f"{valid_metrics_po.accuracy_1x2:.1%}",
                    f"{valid_metrics_po.exact_score_accuracy:.1%}",
                    f"{valid_metrics_po.top_3_hit_rate:.1%}",
                    f"{valid_metrics_po.top_5_hit_rate:.1%}",
                    f"{valid_metrics_po.brier_score:.4f}",
                    f"{valid_metrics_po.avg_prob_actual_result:.1%}",
                ],
            }
            st.table(pd.DataFrame(valid_metrics_data))

        if valid_rho_results and valid_best_rho:
            st.markdown("**Dixon-Coles rho grid (valid path):**")
            valid_rho_rows = [{
                "rho": f"{r.rho:.2f}", "1X2": f"{r.accuracy_1x2:.1%}",
                "Exact": f"{r.exact_score_accuracy:.1%}", "Top3": f"{r.top_3_hit_rate:.1%}",
                "Brier": f"{r.brier_score:.4f}",
            } for r in valid_rho_results]
            st.table(pd.DataFrame(valid_rho_rows))
            st.caption(f"Best rho (valid path): {valid_best_rho.rho:.2f} "
                       f"(Brier: {valid_best_rho.brier_score:.4f})")
```

- [ ] **Step 5: Verify syntax**

```bash
python -c "
import ast
with open('src/app/app.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('syntax OK')
"
```

- [ ] **Step 6: Run full suite — no regressions**

```bash
python -m pytest -v
```
Expected: 86 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/app/app.py
git commit -m "feat: illustrative/valid backtest labels + valid pre-match backtest section"
```

---

## Task 6: Validation Run

- [ ] **Step 1: Run validation script**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import DEFAULT_RHO_GRID, RhoResult, select_best_rho

# Valid path — Poisson
po_results = run_valid_backtest(model_type='poisson')
po_m = compute_metrics(po_results)

print('=== Valid Pre-Match Backtest — Poisson ===')
print(f'Matches: {po_m.total_matches}')
print(f'1X2 Accuracy: {po_m.accuracy_1x2:.1%}')
print(f'Brier Score:  {po_m.brier_score:.4f}')
print()

# Valid path — Dixon-Coles rho grid
print('rho     Brier    1X2     Top3')
rho_results = []
for rho in DEFAULT_RHO_GRID:
    dc_m = compute_metrics(run_valid_backtest(model_type='dixon_coles', rho=rho))
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
print(f'Best rho: {best.rho:.2f}')
"
```

- [ ] **Step 2: Final test run**

```bash
python -m pytest -v
```
Expected: 86 tests pass.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: v6 validation complete"
```
