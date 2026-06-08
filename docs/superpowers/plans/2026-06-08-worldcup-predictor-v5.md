# World Cup Predictor v5 — Rho Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rho grid search to find the best Dixon-Coles rho parameter empirically, display the full tuning table in the Streamlit app, and make a concrete model recommendation.

**Architecture:** `runner.py` gains a `rho` parameter forwarded to `predict_dixon_coles`; a new `rho_tuning.py` module runs the grid search and selects the best rho; the Streamlit backtesting tab gains a "Rho Tuning" section. All model code is untouched.

**Tech Stack:** Python 3.10+, pandas, streamlit, pytest (no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `src/backtesting/runner.py` | Modify — add `rho` parameter |
| `src/backtesting/rho_tuning.py` | Create — `RhoResult`, `tune_rho()`, `select_best_rho()` |
| `src/app/app.py` | Modify — Rho Tuning section in backtesting tab |
| `tests/backtesting/test_runner.py` | Modify — 2 new tests for rho |
| `tests/backtesting/test_rho_tuning.py` | Create — 7 tests |

---

## Task 1: Runner rho Parameter

**Files:**
- Modify: `src/backtesting/runner.py`
- Modify: `tests/backtesting/test_runner.py`

- [ ] **Step 1: Append 2 failing tests to `tests/backtesting/test_runner.py`**

```python
def test_rho_changes_dixon_coles_results(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    r1 = run_backtest(csv, ratings, model_type="dixon_coles", rho=-0.10)[0]
    r2 = run_backtest(csv, ratings, model_type="dixon_coles", rho=-0.25)[0]
    # Different rho values should produce different win probabilities
    assert abs(r1.win_a_prob - r2.win_a_prob) > 1e-6


def test_rho_ignored_for_poisson_model(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    r1 = run_backtest(csv, ratings, model_type="poisson", rho=-0.10)[0]
    r2 = run_backtest(csv, ratings, model_type="poisson", rho=-0.99)[0]
    # rho is ignored for Poisson — results must be identical
    assert r1.win_a_prob == r2.win_a_prob
    assert r1.draw_prob == r2.draw_prob
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_runner.py -v -k "rho"
```
Expected: TypeError — `run_backtest` doesn't accept `rho` yet.

- [ ] **Step 3: Update `src/backtesting/runner.py`**

Change the function signature from:
```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
    model_type: str = "poisson",
) -> list[MatchResult]:
```
To:
```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
    model_type: str = "poisson",
    rho: float = -0.10,
) -> list[MatchResult]:
```

Update the docstring Args section to add:
```
        rho: Dixon-Coles correction parameter (default -0.10).
             Only used when model_type == "dixon_coles". Ignored for Poisson.
```

Change the prediction dispatch inside the loop from:
```python
        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)
```
To:
```python
        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)
```

- [ ] **Step 4: Run runner tests**

```bash
python -m pytest tests/backtesting/test_runner.py -v
```
Expected: 13 tests pass (11 + 2).

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 60 tests pass (58 + 2).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/runner.py tests/backtesting/test_runner.py
git commit -m "feat: runner accepts rho parameter for Dixon-Coles backtest"
```

---

## Task 2: Rho Tuning Module

**Files:**
- Create: `src/backtesting/rho_tuning.py`
- Create: `tests/backtesting/test_rho_tuning.py`

- [ ] **Step 1: Create `tests/backtesting/test_rho_tuning.py`**

```python
import pytest
from src.backtesting.rho_tuning import tune_rho, select_best_rho, RhoResult, DEFAULT_RHO_GRID
from src.backtesting.metrics import BacktestMetrics
from src.data.loader import load_team_ratings


def _ratings():
    return load_team_ratings()


def test_tune_rho_returns_one_result_per_rho():
    ratings = _ratings()
    grid = [-0.10, -0.05, 0.00]
    results = tune_rho(ratings, rho_grid=grid)
    assert len(results) == 3


def test_tune_rho_returns_rho_result_objects():
    ratings = _ratings()
    results = tune_rho(ratings, rho_grid=[-0.10])
    assert isinstance(results[0], RhoResult)
    assert results[0].rho == -0.10


def test_tune_rho_preserves_grid_order():
    ratings = _ratings()
    grid = [-0.30, -0.10, 0.10]
    results = tune_rho(ratings, rho_grid=grid)
    assert [r.rho for r in results] == grid


def test_select_best_rho_picks_lowest_brier():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.60,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.05, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.2, top_5_hit_rate=0.5, brier_score=0.65,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20


def test_select_best_rho_tiebreak_top3():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.1,
                  top_3_hit_rate=0.35, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20  # higher top_3_hit_rate wins tie


def test_select_best_rho_tiebreak_exact_score():
    results = [
        RhoResult(rho=-0.10, accuracy_1x2=0.5, exact_score_accuracy=0.12,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
        RhoResult(rho=-0.20, accuracy_1x2=0.5, exact_score_accuracy=0.15,
                  top_3_hit_rate=0.30, top_5_hit_rate=0.5, brier_score=0.55,
                  avg_prob_actual_result=0.4),
    ]
    best = select_best_rho(results)
    assert best.rho == -0.20  # higher exact_score wins second tie


def test_select_best_rho_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        select_best_rho([])
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_rho_tuning.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/backtesting/rho_tuning.py`**

```python
from dataclasses import dataclass
from pathlib import Path

from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

DEFAULT_RHO_GRID = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]


@dataclass
class RhoResult:
    """Backtesting metrics for one rho value."""
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

    Args:
        ratings: Dict from load_team_ratings().
        rho_grid: List of rho values to test. Defaults to DEFAULT_RHO_GRID.
        matches_path: Path to historical_matches.csv. Defaults to the project data file.

    Returns:
        One RhoResult per rho value, in the same order as rho_grid.
    """
    grid = rho_grid if rho_grid is not None else DEFAULT_RHO_GRID

    results = []
    for rho in grid:
        match_results = run_backtest(
            matches_path=matches_path,
            ratings=ratings,
            model_type="dixon_coles",
            rho=rho,
        )
        m = compute_metrics(match_results)
        results.append(RhoResult(
            rho=rho,
            accuracy_1x2=m.accuracy_1x2,
            exact_score_accuracy=m.exact_score_accuracy,
            top_3_hit_rate=m.top_3_hit_rate,
            top_5_hit_rate=m.top_5_hit_rate,
            brier_score=m.brier_score,
            avg_prob_actual_result=m.avg_prob_actual_result,
        ))

    return results


def select_best_rho(results: list[RhoResult]) -> RhoResult:
    """Select the best rho from a list of RhoResult objects.

    Selection criteria (in priority order):
    1. Lowest brier_score (primary)
    2. Highest top_3_hit_rate (tie-break)
    3. Highest exact_score_accuracy (second tie-break)

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        raise ValueError("Cannot select best rho from empty results list")

    return min(
        results,
        key=lambda r: (r.brier_score, -r.top_3_hit_rate, -r.exact_score_accuracy),
    )
```

- [ ] **Step 4: Run rho tuning tests**

```bash
python -m pytest tests/backtesting/test_rho_tuning.py -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 67 tests pass (60 + 7).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/rho_tuning.py tests/backtesting/test_rho_tuning.py
git commit -m "feat: rho tuning module — tune_rho() grid search and select_best_rho()"
```

---

## Task 3: Streamlit Rho Tuning Section

**Files:**
- Modify: `src/app/app.py`

- [ ] **Step 1: Add rho_tuning import to `src/app/app.py`**

Add this import at the top of the file (after the existing backtesting imports):

```python
from src.backtesting.rho_tuning import tune_rho, select_best_rho, DEFAULT_RHO_GRID
```

- [ ] **Step 2: Add rho tuning data loading inside `with tab_backtest:`**

After the existing `bt_results_po / bt_results_dc` block (after the `except Exception` clause), add:

```python
    rho_tuning_results = None
    best_rho_result = None
    try:
        rho_tuning_results = tune_rho(all_ratings)
        best_rho_result = select_best_rho(rho_tuning_results)
    except Exception as e:
        st.error(f"Rho tuning failed: {e}")
```

- [ ] **Step 3: Add Rho Tuning section inside the `if bt_metrics_po is not None` block**

After the existing per-match results table (at the very end of the `if bt_metrics_po is not None and bt_metrics_dc is not None:` block), add:

```python
        # ── Rho Tuning ────────────────────────────────────────────────────────
        if rho_tuning_results is not None and best_rho_result is not None:
            st.markdown("---")
            st.subheader("Rho Tuning — Dixon-Coles Parameter Search")

            rho_rows = []
            for r in rho_tuning_results:
                rho_rows.append({
                    "rho": f"{r.rho:.2f}",
                    "1X2 Acc": f"{r.accuracy_1x2:.1%}",
                    "Exact": f"{r.exact_score_accuracy:.1%}",
                    "Top 3": f"{r.top_3_hit_rate:.1%}",
                    "Top 5": f"{r.top_5_hit_rate:.1%}",
                    "Brier": f"{r.brier_score:.4f}",
                    "Avg P": f"{r.avg_prob_actual_result:.1%}",
                })
            st.table(pd.DataFrame(rho_rows))

            st.caption(f"**Best rho:** {best_rho_result.rho:.2f} "
                       f"(Brier: {best_rho_result.brier_score:.4f}, "
                       f"Top 3: {best_rho_result.top_3_hit_rate:.1%})")

            # Recommendation: DC with best rho if it beats Poisson Brier by >0.001.
            poisson_brier = bt_metrics_po.brier_score
            if best_rho_result.brier_score < poisson_brier - 0.001:
                st.success(
                    f"Recommendation: **Dixon-Coles (rho={best_rho_result.rho:.2f})** — "
                    f"Brier {best_rho_result.brier_score:.4f} vs Poisson {poisson_brier:.4f} "
                    f"({poisson_brier - best_rho_result.brier_score:.4f} improvement)"
                )
            else:
                st.info(
                    f"Recommendation: **Poisson (default)** — "
                    f"Dixon-Coles best rho={best_rho_result.rho:.2f} does not improve "
                    f"Brier score by more than 0.001 (DC: {best_rho_result.brier_score:.4f}, "
                    f"Poisson: {poisson_brier:.4f})"
                )
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/app/app.py').read()); print('syntax OK')"
```

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 67 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/app.py
git commit -m "feat: rho tuning section in Streamlit backtesting tab with recommendation"
```

---

## Task 4: Validation Run

- [ ] **Step 1: Run full rho grid and print report**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.data.loader import load_team_ratings
from src.backtesting.rho_tuning import tune_rho, select_best_rho
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

ratings = load_team_ratings()
results = tune_rho(ratings)
best = select_best_rho(results)

poisson_brier = compute_metrics(run_backtest(ratings=ratings, model_type='poisson')).brier_score

print('rho     1X2     Exact   Top3    Top5    Brier   AvgP')
for r in results:
    marker = ' <-- BEST' if r.rho == best.rho else ''
    print(f'{r.rho:+.2f}   {r.accuracy_1x2:.1%}  {r.exact_score_accuracy:.1%}  {r.top_3_hit_rate:.1%}  {r.top_5_hit_rate:.1%}  {r.brier_score:.4f}  {r.avg_prob_actual_result:.1%}{marker}')
print()
print(f'Best rho: {best.rho:.2f} (Brier={best.brier_score:.4f})')
print(f'Poisson Brier: {poisson_brier:.4f}')
improvement = poisson_brier - best.brier_score
if improvement > 0.001:
    print(f'Recommendation: Dixon-Coles rho={best.rho:.2f} (improvement: {improvement:.4f})')
else:
    print(f'Recommendation: Poisson (DC improvement {improvement:.4f} < 0.001 threshold)')
"
```

- [ ] **Step 2: Run all tests final time**

```bash
python -m pytest -v
```
Expected: 67 tests pass.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: v5 validation complete — rho tuning report"
```
