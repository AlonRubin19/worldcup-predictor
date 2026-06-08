# World Cup Predictor v3 — Design Spec

**Date:** 2026-06-08  
**Status:** Approved  
**Scope:** Backtesting engine to validate xG + Poisson model against historical match results. No external APIs.

---

## 1. Goal

Validate how well the automatic xG + Poisson prediction model performs against real historical match outcomes. Provide accuracy metrics and per-match detail in a new Backtesting tab in the Streamlit app.

---

## 2. What Changes vs v2

| Component | v2 | v3 |
|---|---|---|
| `data/historical_matches.csv` | does not exist | new — 40 sample matches |
| `src/backtesting/runner.py` | empty stub | new — `run_backtest()` |
| `src/backtesting/metrics.py` | does not exist | new — `compute_metrics()` |
| `src/models/poisson.py` | unchanged | unchanged |
| `src/models/xg_calculator.py` | unchanged | unchanged |
| `src/data/loader.py` | unchanged | unchanged |
| `src/app/app.py` | single page | adds Backtesting tab |
| `tests/backtesting/__init__.py` | does not exist | new (empty) |
| `tests/backtesting/test_metrics.py` | does not exist | new |
| `tests/backtesting/test_runner.py` | does not exist | new |

---

## 3. Data Layer

### `data/historical_matches.csv`

Header: `date,team_a,team_b,team_a_goals,team_b_goals`

- `date`: ISO format (YYYY-MM-DD)
- `team_a`, `team_b`: must match team names in `team_ratings.csv`
- `team_a_goals`, `team_b_goals`: integer goals scored

40 sample matches drawn from real-world international fixtures. All teams must exist in `team_ratings.csv`.

---

## 4. Backtesting Runner

### `src/backtesting/runner.py`

```python
@dataclass
class MatchResult:
    date: str
    team_a: str
    team_b: str
    actual_goals_a: int
    actual_goals_b: int
    actual_outcome: str          # "team_a_win", "draw", "team_b_win"
    win_a_prob: float
    draw_prob: float
    win_b_prob: float
    predicted_outcome: str       # outcome with highest predicted probability
    top_scorelines: list[tuple[int, int, float]]
    exact_score_hit: bool        # actual score is the #1 predicted scoreline
    in_top_3: bool               # actual score appears in top 3 predicted scorelines
    in_top_5: bool               # actual score appears in top 5 predicted scorelines
    prob_of_actual_result: float # predicted probability assigned to the actual 1X2 outcome


def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
) -> list[MatchResult]:
    """Run predictions for all historical matches and return per-match results.

    Args:
        matches_path: Path to historical_matches.csv. Defaults to data/historical_matches.csv.
        ratings: Dict from load_team_ratings(). Loaded from file if not provided.

    Raises:
        FileNotFoundError: if matches CSV is missing.
        ValueError: if required columns are missing or a team has no ratings entry.
    """
```

**Logic per match:**
1. Look up ratings for both teams (raise `ValueError` if either team is missing from ratings).
2. Call `calculate_xg(ratings_a, ratings_b)` → `xg_a, xg_b`.
3. Call `predict(team_a, team_b, xg_a, xg_b)` → `PredictionResult`.
4. Determine `actual_outcome` from goals.
5. Determine `predicted_outcome` = outcome with max probability among (win_a, draw, win_b).
6. Check if `(actual_goals_a, actual_goals_b)` appears in `top_scorelines` at position 0 (exact hit), positions 0–2 (top 3), positions 0–4 (top 5).
7. Set `prob_of_actual_result` = the predicted probability for the actual 1X2 outcome.

---

## 5. Metrics

### `src/backtesting/metrics.py`

```python
@dataclass
class BacktestMetrics:
    total_matches: int
    accuracy_1x2: float          # fraction where predicted_outcome == actual_outcome
    exact_score_accuracy: float  # fraction where exact_score_hit is True
    top_3_hit_rate: float        # fraction where in_top_3 is True
    top_5_hit_rate: float        # fraction where in_top_5 is True
    brier_score: float           # multi-class Brier score for 1X2 probabilities
    avg_prob_actual_result: float


def compute_metrics(results: list[MatchResult]) -> BacktestMetrics:
    """Compute aggregate metrics from a list of MatchResult objects.

    Brier score (multi-class):
        BS = (1/N) * sum_i [ (p_win_a_i - o_win_a_i)^2
                           + (p_draw_i  - o_draw_i)^2
                           + (p_win_b_i - o_win_b_i)^2 ]
    where o_* is 1.0 if that outcome occurred, 0.0 otherwise.

    Raises:
        ValueError: if results list is empty.
    """
```

---

## 6. Updated Streamlit App

`src/app/app.py` uses `st.tabs(["⚽ Match Predictor", "📊 Backtesting"])` to add the second tab.

**Backtesting tab:**
1. Load historical matches and run backtest via `run_backtest()`.
2. If it fails, show `st.error()` (do not crash the whole app — predictor tab must still work).
3. Display summary metrics in a two-column table.
4. Display per-match results table: date, match, actual score, predicted outcome, actual outcome, correct (✓/✗), in top 5 (✓/✗), prob of actual result.

---

## 7. Tests

### `tests/backtesting/test_metrics.py`

Required test cases:
- `compute_metrics` on a known result set returns correct `accuracy_1x2`
- Brier score is 0.0 when predictions are perfect (probability 1.0 on actual outcome)
- Brier score is correctly computed for a mixed result set
- Empty results raises `ValueError`
- `exact_score_accuracy`, `top_3_hit_rate`, `top_5_hit_rate` computed correctly

### `tests/backtesting/test_runner.py`

Required test cases:
- `run_backtest` returns a list of `MatchResult` objects
- `actual_outcome` is correctly determined from goals
- `predicted_outcome` is the outcome with the highest probability
- `exact_score_hit` is True when actual score is #1 predicted scoreline
- `in_top_5` is True when actual score is in top 5
- Missing match file raises `FileNotFoundError`
- Missing team in ratings raises `ValueError`

---

## 8. File Structure (additions only)

```
worldcup-predictor/
├── data/
│   └── historical_matches.csv     # new — 40 sample matches
├── src/
│   ├── backtesting/
│   │   ├── __init__.py            # already exists (empty)
│   │   ├── runner.py              # new
│   │   └── metrics.py             # new
│   └── app/
│       └── app.py                 # modified — adds Backtesting tab
└── tests/
    └── backtesting/
        ├── __init__.py            # new (empty)
        ├── test_metrics.py        # new
        └── test_runner.py         # new
```
