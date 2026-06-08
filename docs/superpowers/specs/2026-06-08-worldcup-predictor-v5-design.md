# World Cup Predictor v5 — Design Spec

**Date:** 2026-06-08
**Status:** Approved
**Scope:** Dixon-Coles rho tuning — find the best rho via grid search over historical matches.

---

## 1. Goal

Replace the fixed rho=-0.10 with a data-driven best rho found by evaluating all values in a predefined grid against the historical matches. Present the full grid comparison in the Streamlit app and make a concrete model recommendation.

---

## 2. What Changes vs v4

| Component | v4 | v5 |
|---|---|---|
| `src/backtesting/runner.py` | `model_type` only | adds `rho` parameter (Dixon-Coles only) |
| `src/backtesting/rho_tuning.py` | does not exist | new — `RhoResult`, `tune_rho()`, `select_best_rho()` |
| `src/app/app.py` | comparison table | adds Rho Tuning section |
| `tests/backtesting/test_rho_tuning.py` | does not exist | new |
| `tests/backtesting/test_runner.py` | 11 tests | adds 2 tests for rho parameter |
| All models | unchanged | unchanged |

---

## 3. Runner rho Parameter

**`src/backtesting/runner.py`** signature change:

```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,           # only used when model_type == "dixon_coles"
) -> list[MatchResult]:
```

When `model_type == "dixon_coles"`, call:
```python
prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)
```

The `rho` parameter is silently ignored when `model_type == "poisson"`.

---

## 4. Rho Tuning Module

**`src/backtesting/rho_tuning.py`**

```python
DEFAULT_RHO_GRID = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]

@dataclass
class RhoResult:
    rho: float
    accuracy_1x2: float
    exact_score_accuracy: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    brier_score: float
    avg_prob_actual_result: float


def tune_rho(
    ratings: dict,
    rho_grid: list[float] | None = None,
    matches_path: Path | None = None,
) -> list[RhoResult]:
    """Run Dixon-Coles backtest for each rho in the grid.

    Returns one RhoResult per rho value, in grid order.
    """


def select_best_rho(results: list[RhoResult]) -> RhoResult:
    """Select the best rho by:
    1. Primary:   lowest brier_score
    2. Secondary: highest top_3_hit_rate (tie-break)
    3. Tertiary:  highest exact_score_accuracy (second tie-break)

    Raises ValueError if results is empty.
    """
```

---

## 5. Streamlit Rho Tuning Section

Added inside `with tab_backtest:`, after the existing comparison table.

```
st.subheader("Rho Tuning — Dixon-Coles Parameter Search")
```

Shows:
1. Full rho grid table with columns: rho, 1X2 Acc, Exact, Top3, Top5, Brier, Avg P
2. Best rho highlighted in a caption
3. Recommendation: if best rho Brier < Poisson Brier - 0.001, recommend "Dixon-Coles (rho=X.XX)"; otherwise "Poisson (default)"

---

## 6. Tests

**`tests/backtesting/test_rho_tuning.py`:**
- `tune_rho` returns one result per rho value
- `tune_rho` returns `RhoResult` objects
- `select_best_rho` picks the result with lowest Brier score
- `select_best_rho` uses top_3_hit_rate as tie-breaker when Brier scores are equal
- `select_best_rho` uses exact_score_accuracy as second tie-breaker
- `select_best_rho` raises ValueError on empty list
- rho=0.0 produces Brier score close to Poisson

**`tests/backtesting/test_runner.py` additions:**
- `run_backtest(model_type="dixon_coles", rho=-0.05)` produces different results than `rho=-0.10`
- `run_backtest(model_type="poisson", rho=-0.05)` ignores rho (same result as default)
