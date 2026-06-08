# World Cup Predictor v6 — Valid Backtest Dataset

**Date:** 2026-06-08
**Status:** Approved
**Scope:** Replace manually-estimated ratings with a structurally valid pre-match stats dataset; add a separate valid backtest path.

---

## 1. Goal

Build a second backtest pipeline that derives xG exclusively from per-match pre-game statistics — no manually estimated ratings, no AI-assigned values. The existing "Illustrative Only" backtest is preserved unchanged. The new path is clearly labelled as "structural placeholder" until real historical data is sourced.

---

## 2. What Changes vs v5

| Component | v5 | v6 |
|---|---|---|
| `data/pre_match_team_stats.csv` | does not exist | new — 25 sample rows, placeholder |
| `src/data/pre_match_loader.py` | does not exist | new — `PreMatchStats`, `load_pre_match_stats()` |
| `src/models/pre_match_xg.py` | does not exist | new — `calculate_pre_match_xg()` |
| `src/backtesting/valid_runner.py` | does not exist | new — `run_valid_backtest()` |
| `src/app/app.py` | unlabelled tabs | explicit "Illustrative" / "Valid" labels |
| `src/backtesting/runner.py` | unchanged | unchanged |
| `data/team_ratings.csv` | unchanged | unchanged (illustrative path only) |
| All models | unchanged | unchanged |

---

## 3. Data File

### `data/pre_match_team_stats.csv`

#### Column specification

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Unique match identifier |
| `date` | str | ISO date (YYYY-MM-DD) |
| `team_a` | str | Home/first team name |
| `team_b` | str | Away/second team name |
| `team_a_elo_pre` | float | Team A ELO rating before this match |
| `team_b_elo_pre` | float | Team B ELO rating before this match |
| `team_a_goals_for_last_10` | float | Team A avg goals scored per game, last 10 matches |
| `team_a_goals_against_last_10` | float | Team A avg goals conceded per game, last 10 matches |
| `team_b_goals_for_last_10` | float | Team B avg goals scored per game, last 10 matches |
| `team_b_goals_against_last_10` | float | Team B avg goals conceded per game, last 10 matches |
| `team_a_points_per_game_last_10` | float | Team A avg points per game (3=win, 1=draw), last 10 matches |
| `team_b_points_per_game_last_10` | float | Team B avg points per game, last 10 matches |
| `team_a_matches_available` | int | Actual matches used (≤10; flag if <5) |
| `team_b_matches_available` | int | Actual matches used (≤10; flag if <5) |
| `team_a_goals` | int | Actual goals scored by Team A in this match |
| `team_b_goals` | int | Actual goals scored by Team B in this match |

#### Warning

The 25 sample rows must include a large comment header:

```
# WARNING: PLACEHOLDER DATA — NOT SOURCED FROM REAL PRE-MATCH STATISTICS
# This file is structurally valid but values are illustrative only.
# See docs/valid_backtest_status.md for sourcing requirements.
```

Pandas `read_csv` with `comment='#'` will skip these lines automatically.

#### Sample data

25 rows using WC 2022 matches. Goal stats and ELO values are plausible estimates for pre-tournament state, not derived from actual historical records. All `team_a_matches_available` and `team_b_matches_available` are set to 10 except 3 rows where one team has 4 (to test the <5 flag).

---

## 4. Pre-Match Loader

### `src/data/pre_match_loader.py`

```python
@dataclass
class PreMatchStats:
    match_id: int
    date: str
    team_a: str
    team_b: str
    team_a_elo_pre: float
    team_b_elo_pre: float
    team_a_goals_for_last_10: float       # avg goals/game
    team_a_goals_against_last_10: float   # avg goals conceded/game
    team_b_goals_for_last_10: float
    team_b_goals_against_last_10: float
    team_a_points_per_game_last_10: float
    team_b_points_per_game_last_10: float
    team_a_matches_available: int
    team_b_matches_available: int
    team_a_goals: int                     # actual match result
    team_b_goals: int


def load_pre_match_stats(
    path: Path | None = None,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[PreMatchStats]:
    """Load pre-match statistics from CSV.

    Args:
        path: Path to CSV. Defaults to data/pre_match_team_stats.csv.
        min_matches: Minimum matches_available to consider reliable (default 5).
        exclude_insufficient: If True, exclude rows where either team has
                              fewer than min_matches. If False (default),
                              include all rows but set a warning.

    Returns:
        List of PreMatchStats. If exclude_insufficient=False, all rows included.

    Raises:
        FileNotFoundError: if CSV missing.
        ValueError: if required columns are missing.
    """
```

---

## 5. Pre-Match xG Calculator

### `src/models/pre_match_xg.py`

```python
BASE_XG    = 1.35
XG_MIN     = 0.2
XG_MAX     = 4.5
FORM_BASE  = 0.85
FORM_SCALE = 0.30   # max form adjustment above FORM_BASE


def calculate_pre_match_xg(match: PreMatchStats) -> tuple[float, float]:
    """Calculate expected goals from pre-match statistics only.

    No manually estimated ratings are used. All factors derived from:
    - Goals scored/conceded averages over last 10 matches
    - Points per game over last 10 matches
    - Pre-match ELO ratings

    Formula:
        attack_a  = goals_for_a  / BASE_XG   (normalized to BASE_XG = 1)
        defense_b = goals_against_b / BASE_XG
        attack_b  = goals_for_b  / BASE_XG
        defense_a = goals_against_a / BASE_XG

        form_a = FORM_BASE + (ppg_a / 3) * FORM_SCALE
        form_b = FORM_BASE + (ppg_b / 3) * FORM_SCALE

        elo_factor_a = 1 + (elo_a - elo_b) / 4000
        elo_factor_b = 1 + (elo_b - elo_a) / 4000

        xg_a = BASE_XG * attack_a * defense_b * form_a * elo_factor_a
        xg_b = BASE_XG * attack_b * defense_a * form_b * elo_factor_b

    Both values clamped to [XG_MIN, XG_MAX].

    Raises:
        ValueError: if any stat value that would produce negative xG (e.g. zero attack).
    """
```

---

## 6. Valid Runner

### `src/backtesting/valid_runner.py`

```python
def run_valid_backtest(
    path: Path | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[MatchResult]:
    """Run backtest using only pre-match statistics.

    Unlike run_backtest() (which uses manually-estimated team_ratings.csv),
    this function derives xG exclusively from pre-match stats in pre_match_team_stats.csv.

    Returns the same list[MatchResult] structure so all metrics work identically.
    """
```

Uses `load_pre_match_stats()` → `calculate_pre_match_xg()` → `predict()` or `predict_dixon_coles()`.

---

## 7. Updated Streamlit App

The Backtesting tab gets two clearly labelled sections:

```
⚠️ Illustrative Backtest — uses manually estimated ratings (team_ratings.csv).
   Ratings were assigned by AI with knowledge of WC 2022 outcomes.
   Results are for engineering validation only.

📐 Valid Pre-Match Backtest — uses pre-match stats (pre_match_team_stats.csv).
   xG derived from goals and form statistics only.
   ⚠️ PLACEHOLDER: Data not yet sourced from real historical records.
```

The Valid Backtest section shows:
- Metrics table (same 7 metrics)
- Rho tuning table for the valid path
- Recommendation

---

## 8. Tests

### `tests/data/test_pre_match_loader.py`
- Returns list of PreMatchStats
- CSV with comment lines is parsed correctly
- Exclude_insufficient=True filters rows with <5 matches
- Missing file raises FileNotFoundError
- Missing columns raise ValueError

### `tests/models/test_pre_match_xg.py`
- Returns tuple of two floats
- Average-strength teams produce xG near BASE_XG (1.35)
- Higher ELO team gets higher xG
- Strong attack (high goals_for) increases xG
- Strong defense (low goals_against) decreases opponent xG
- Form: high ppg increases xG
- Output clamped to [XG_MIN, XG_MAX]

### `tests/backtesting/test_valid_runner.py`
- Returns list[MatchResult]
- No team_ratings.csv is accessed (verify by mocking or passing custom path)
- Works with both model_type="poisson" and "dixon_coles"
- exclude_insufficient=True produces fewer results than False when data has <5 rows

---

## 9. Status Document

Create `docs/valid_backtest_status.md` documenting:
- What is implemented and engineering-valid
- What placeholder data means
- Exactly what data sources are needed to make this research-grade
