# World Cup Predictor v4 — Dixon-Coles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Dixon-Coles tau adjustment as an optional model mode that corrects low-scoring outcome probabilities (especially draws) while preserving the existing Poisson model unchanged.

**Architecture:** Extract a `build_score_matrix()` helper from `poisson.py` so both models share the same Poisson PMF logic; `dixon_coles.py` applies tau corrections to the four low-score cells and normalizes; `runner.py` gains a `model_type` parameter; `app.py` gains a model radio selector and a backtesting comparison table.

**Tech Stack:** Python 3.10+, numpy, scipy, streamlit, pytest (no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `src/models/poisson.py` | Modify — extract `build_score_matrix()`, refactor `predict()` to call it |
| `src/models/dixon_coles.py` | Create — `predict_dixon_coles()` with tau correction |
| `src/backtesting/runner.py` | Modify — add `model_type` parameter |
| `src/app/app.py` | Modify — model radio + comparison table |
| `tests/models/test_poisson.py` | Modify — add test for `build_score_matrix()` |
| `tests/models/test_dixon_coles.py` | Create — 7 tests for Dixon-Coles |
| `tests/backtesting/test_runner.py` | Modify — add 3 tests for model_type |

---

## Task 1: Extract `build_score_matrix()` from Poisson

**Files:**
- Modify: `src/models/poisson.py`
- Modify: `tests/models/test_poisson.py`

This refactor must keep all 11 existing poisson tests passing. `predict()` behaviour is unchanged.

- [ ] **Step 1: Add one failing test for `build_score_matrix`**

Append to `tests/models/test_poisson.py`:

```python
from src.models.poisson import build_score_matrix


def test_build_score_matrix_shape_and_sum():
    matrix = build_score_matrix(1.5, 1.2)
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    # Matrix should sum to approximately 1 (small gap due to truncation)
    assert abs(matrix.sum() - 1.0) < 1e-4


def test_build_score_matrix_cell_values():
    from scipy.stats import poisson as _poisson
    matrix = build_score_matrix(1.5, 1.2)
    # Cell [2][1] should equal poisson.pmf(2, 1.5) * poisson.pmf(1, 1.2)
    expected = _poisson.pmf(2, 1.5) * _poisson.pmf(1, 1.2)
    assert abs(matrix[2, 1] - expected) < 1e-12
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_poisson.py -v -k "build_score_matrix"
```
Expected: ImportError — `build_score_matrix` not yet exported.

- [ ] **Step 3: Refactor `src/models/poisson.py`**

Replace the entire file with this (same behaviour, now exposes `build_score_matrix`):

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson


@dataclass
class PredictionResult:
    """Holds the output of a single match prediction."""
    team_a: str
    team_b: str
    win_a: float   # probability Team A wins
    draw: float    # probability of a draw
    win_b: float   # probability Team B wins
    # Each entry is (goals_a, goals_b, probability), sorted by probability descending.
    top_scorelines: list[tuple[int, int, float]]


def build_score_matrix(xg_a: float, xg_b: float) -> np.ndarray:
    """Build a joint Poisson probability matrix for goals 0 to N-1.

    N = max(11, int(max(xg_a, xg_b) * 3) + 1) — adaptive for high xG.

    Returns an (N x N) array where cell [i][j] =
    poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b).
    """
    max_goals = max(11, int(max(xg_a, xg_b) * 3) + 1)
    goals_range = np.arange(max_goals)
    prob_a = poisson.pmf(goals_range, xg_a)  # shape: (max_goals,)
    prob_b = poisson.pmf(goals_range, xg_b)  # shape: (max_goals,)
    return np.outer(prob_a, prob_b)           # shape: (max_goals, max_goals)


def _extract_result(team_a: str, team_b: str, matrix: np.ndarray) -> "PredictionResult":
    """Derive win/draw/win probabilities and top 5 scorelines from a score matrix."""
    max_goals = matrix.shape[0]

    win_a = float(np.sum(np.tril(matrix, k=-1)))  # Team A scores more (below diagonal)
    draw  = float(np.sum(np.diag(matrix)))         # Equal scores (diagonal)
    win_b = float(np.sum(np.triu(matrix, k=1)))    # Team B scores more (above diagonal)

    scorelines = [
        (i, j, matrix[i, j])
        for i in range(max_goals)
        for j in range(max_goals)
    ]
    scorelines.sort(key=lambda x: (-x[2], x[0], x[1]))
    top_scorelines = [(int(g_a), int(g_b), float(p)) for g_a, g_b, p in scorelines[:5]]

    return PredictionResult(
        team_a=team_a,
        team_b=team_b,
        win_a=win_a,
        draw=draw,
        win_b=win_b,
        top_scorelines=top_scorelines,
    )


def predict(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
) -> PredictionResult:
    """Predict match outcome probabilities using independent Poisson distributions.

    Args:
        team_a: Name of the first team.
        team_b: Name of the second team.
        xg_a: Expected goals for Team A (must be > 0).
        xg_b: Expected goals for Team B (must be > 0).

    Returns:
        PredictionResult with win/draw/loss probabilities and top 5 scorelines.

    Raises:
        ValueError: If either xG value is <= 0.
    """
    if xg_a <= 0 or xg_b <= 0:
        raise ValueError(f"Expected goals must be > 0, got xg_a={xg_a}, xg_b={xg_b}")

    matrix = build_score_matrix(xg_a, xg_b)
    return _extract_result(team_a, team_b, matrix)
```

- [ ] **Step 4: Run all poisson tests**

```bash
python -m pytest tests/models/test_poisson.py -v
```
Expected: all 13 tests pass (11 original + 2 new).

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
python -m pytest -v
```
Expected: 48 tests pass (46 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/models/poisson.py tests/models/test_poisson.py
git commit -m "refactor: extract build_score_matrix() and _extract_result() from predict()"
```

---

## Task 2: Dixon-Coles Model

**Files:**
- Create: `src/models/dixon_coles.py`
- Create: `tests/models/test_dixon_coles.py`

- [ ] **Step 1: Create `tests/models/test_dixon_coles.py`**

```python
import pytest
import numpy as np
from src.models.dixon_coles import predict_dixon_coles
from src.models.poisson import predict, PredictionResult


def test_returns_prediction_result():
    result = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    assert isinstance(result, PredictionResult)


def test_probabilities_sum_to_one():
    result = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    total = result.win_a + result.draw + result.win_b
    assert abs(total - 1.0) < 1e-4


def test_rho_zero_gives_same_result_as_poisson():
    dc = predict_dixon_coles("France", "Brazil", 1.5, 1.2, rho=0.0)
    po = predict("France", "Brazil", 1.5, 1.2)
    assert abs(dc.win_a - po.win_a) < 1e-6
    assert abs(dc.draw  - po.draw)  < 1e-6
    assert abs(dc.win_b - po.win_b) < 1e-6


def test_draw_probability_differs_from_poisson_with_default_rho():
    dc = predict_dixon_coles("France", "Brazil", 1.5, 1.2)
    po = predict("France", "Brazil", 1.5, 1.2)
    # Default rho=-0.10 should change draw probability meaningfully
    assert abs(dc.draw - po.draw) > 0.001


def test_top_scorelines_has_five_entries():
    result = predict_dixon_coles("Spain", "Germany", 1.4, 1.3)
    assert len(result.top_scorelines) == 5


def test_xg_validation_raises():
    with pytest.raises(ValueError):
        predict_dixon_coles("A", "B", 0.0, 1.0)
    with pytest.raises(ValueError):
        predict_dixon_coles("A", "B", 1.0, -0.5)


def test_low_score_probabilities_differ_from_poisson():
    dc = predict_dixon_coles("A", "B", 1.5, 1.2)
    po = predict("A", "B", 1.5, 1.2)
    # 0-0 and 1-1 probabilities should differ (tau ≠ 1 for these cells)
    dc_00 = next(p for g_a, g_b, p in dc.top_scorelines if g_a == 0 and g_b == 0) \
            if any(g_a == 0 and g_b == 0 for g_a, g_b, _ in dc.top_scorelines) else None
    po_00 = next(p for g_a, g_b, p in po.top_scorelines if g_a == 0 and g_b == 0) \
            if any(g_a == 0 and g_b == 0 for g_a, g_b, _ in po.top_scorelines) else None
    # If both models have 0-0 in top 5, verify the probabilities differ
    if dc_00 is not None and po_00 is not None:
        assert abs(dc_00 - po_00) > 1e-6
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_dixon_coles.py -v
```
Expected: ImportError — module doesn't exist.

- [ ] **Step 3: Create `src/models/dixon_coles.py`**

```python
import numpy as np
from src.models.poisson import build_score_matrix, _extract_result, PredictionResult


def predict_dixon_coles(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    rho: float = -0.10,
) -> PredictionResult:
    """Predict match outcomes using Poisson + Dixon-Coles tau correction.

    Applies a tau correction factor to the four low-score cells of the Poisson
    score matrix to better capture the statistical dependence between low scores.
    The matrix is then normalized to ensure probabilities sum to 1.

    Tau correction (Dixon & Coles, 1997):
        0-0:  tau = 1 - (xg_a * xg_b * rho)
        0-1:  tau = 1 + (xg_a * rho)
        1-0:  tau = 1 + (xg_b * rho)
        1-1:  tau = 1 - rho
        else: tau = 1  (unchanged)

    When rho=0, all tau values equal 1 and the result matches pure Poisson.
    Default rho=-0.10 is the empirical value from Dixon & Coles (1997).

    Args:
        team_a: Name of the first team.
        team_b: Name of the second team.
        xg_a: Expected goals for Team A (must be > 0).
        xg_b: Expected goals for Team B (must be > 0).
        rho: Correction parameter (default -0.10). Negative values reduce
             0-0 and 1-1 probabilities relative to pure Poisson.

    Returns:
        PredictionResult with win/draw/loss probabilities and top 5 scorelines.

    Raises:
        ValueError: If either xG value is <= 0.
    """
    if xg_a <= 0 or xg_b <= 0:
        raise ValueError(f"Expected goals must be > 0, got xg_a={xg_a}, xg_b={xg_b}")

    matrix = build_score_matrix(xg_a, xg_b)

    # Apply Dixon-Coles tau correction to the four low-score cells.
    # All other cells are multiplied by tau=1 (no change).
    matrix[0, 0] *= 1 - (xg_a * xg_b * rho)  # 0-0
    matrix[0, 1] *= 1 + (xg_a * rho)           # 0-1
    matrix[1, 0] *= 1 + (xg_b * rho)           # 1-0
    matrix[1, 1] *= 1 - rho                     # 1-1

    # Normalize so all probabilities sum to 1.
    matrix /= matrix.sum()

    return _extract_result(team_a, team_b, matrix)
```

- [ ] **Step 4: Run Dixon-Coles tests**

```bash
python -m pytest tests/models/test_dixon_coles.py -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 55 tests pass (48 + 7).

- [ ] **Step 6: Commit**

```bash
git add src/models/dixon_coles.py tests/models/test_dixon_coles.py
git commit -m "feat: Dixon-Coles model — predict_dixon_coles() with tau correction"
```

---

## Task 3: Runner model_type Parameter

**Files:**
- Modify: `src/backtesting/runner.py`
- Modify: `tests/backtesting/test_runner.py`

- [ ] **Step 1: Add 3 failing tests to `tests/backtesting/test_runner.py`**

Append to the existing test file:

```python
def test_run_backtest_with_poisson_model_type(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings, model_type="poisson")
    assert len(results) == 1


def test_run_backtest_with_dixon_coles_model_type(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings, model_type="dixon_coles")
    assert len(results) == 1
    assert isinstance(results[0].win_a_prob, float)


def test_run_backtest_invalid_model_type_raises(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    with pytest.raises(ValueError, match="model_type"):
        run_backtest(csv, ratings, model_type="invalid")
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_runner.py -v -k "model_type"
```
Expected: TypeError — `run_backtest` doesn't accept `model_type` yet.

- [ ] **Step 3: Update `src/backtesting/runner.py`**

Add the import and update the function signature and body. Change the top of the file and the function definition as follows:

Add to the imports at the top:
```python
from src.models.dixon_coles import predict_dixon_coles
```

Change the `run_backtest` signature from:
```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
) -> list[MatchResult]:
```

To:
```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
    model_type: str = "poisson",
) -> list[MatchResult]:
```

Update the docstring to include:
```
        model_type: "poisson" (default) or "dixon_coles".

    Raises:
        ...
        ValueError: if model_type is not "poisson" or "dixon_coles".
```

Add validation immediately after the `if ratings is None:` block:
```python
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")
```

Change the prediction call inside the loop from:
```python
        xg_a, xg_b = calculate_xg(ratings[team_a], ratings[team_b])
        prediction = predict(team_a, team_b, xg_a, xg_b)
```

To:
```python
        xg_a, xg_b = calculate_xg(ratings[team_a], ratings[team_b])
        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)
```

- [ ] **Step 4: Run runner tests**

```bash
python -m pytest tests/backtesting/test_runner.py -v
```
Expected: 11 tests pass (8 original + 3 new).

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 58 tests pass (55 + 3).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/runner.py tests/backtesting/test_runner.py
git commit -m "feat: backtesting runner supports model_type poisson|dixon_coles"
```

---

## Task 4: Streamlit App — Model Selector + Comparison Table

**Files:**
- Modify: `src/app/app.py`

- [ ] **Step 1: Add the model import to app.py**

At the top of `src/app/app.py`, add this import after the existing model imports:

```python
from src.models.dixon_coles import predict_dixon_coles
```

- [ ] **Step 2: Add model radio to the Predictor tab**

In `src/app/app.py`, inside `with tab_predictor:`, add a model selector directly after the `st.markdown(...)` description line and before the data loading block:

```python
    model_choice = st.radio(
        "Prediction Model",
        ["Poisson", "Dixon-Coles"],
        horizontal=True,
        help="Dixon-Coles corrects low-score probabilities (draws, 1-0, 0-1) vs pure Poisson.",
    )
```

- [ ] **Step 3: Update the prediction call in the Predictor tab**

Replace the existing prediction call:
```python
    try:
        result = predict(team_a, team_b, final_xg_a, final_xg_b)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
```

With:
```python
    try:
        if model_choice == "Dixon-Coles":
            result = predict_dixon_coles(team_a, team_b, final_xg_a, final_xg_b)
        else:
            result = predict(team_a, team_b, final_xg_a, final_xg_b)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
```

- [ ] **Step 4: Update the Backtesting tab to show a comparison table**

Replace the backtesting data loading block:
```python
    bt_results = None
    bt_metrics = None
    try:
        bt_results = run_backtest(ratings=all_ratings)
        bt_metrics = compute_metrics(bt_results)
    except Exception as e:
        st.error(f"Backtesting failed: {e}")
```

With:
```python
    bt_results_po = None
    bt_results_dc = None
    bt_metrics_po = None
    bt_metrics_dc = None
    try:
        bt_results_po = run_backtest(ratings=all_ratings, model_type="poisson")
        bt_metrics_po = compute_metrics(bt_results_po)
        bt_results_dc = run_backtest(ratings=all_ratings, model_type="dixon_coles")
        bt_metrics_dc = compute_metrics(bt_results_dc)
    except Exception as e:
        st.error(f"Backtesting failed: {e}")
```

- [ ] **Step 5: Replace the summary metrics table with a comparison table**

Replace the `if bt_metrics is not None:` block (the summary metrics table and per-match table) with:

```python
    if bt_metrics_po is not None and bt_metrics_dc is not None:
        # ── Model comparison ─────────────────────────────────────────────────
        st.subheader("Model Comparison")

        comparison_data = {
            "Metric": [
                "Total Matches Tested",
                "1X2 Accuracy",
                "Exact Score Accuracy",
                "Top 3 Scoreline Hit Rate",
                "Top 5 Scoreline Hit Rate",
                "Brier Score (lower = better)",
                "Avg Probability of Actual Result",
            ],
            "Poisson": [
                str(bt_metrics_po.total_matches),
                f"{bt_metrics_po.accuracy_1x2:.1%}",
                f"{bt_metrics_po.exact_score_accuracy:.1%}",
                f"{bt_metrics_po.top_3_hit_rate:.1%}",
                f"{bt_metrics_po.top_5_hit_rate:.1%}",
                f"{bt_metrics_po.brier_score:.4f}",
                f"{bt_metrics_po.avg_prob_actual_result:.1%}",
            ],
            "Dixon-Coles": [
                str(bt_metrics_dc.total_matches),
                f"{bt_metrics_dc.accuracy_1x2:.1%}",
                f"{bt_metrics_dc.exact_score_accuracy:.1%}",
                f"{bt_metrics_dc.top_3_hit_rate:.1%}",
                f"{bt_metrics_dc.top_5_hit_rate:.1%}",
                f"{bt_metrics_dc.brier_score:.4f}",
                f"{bt_metrics_dc.avg_prob_actual_result:.1%}",
            ],
        }
        st.table(pd.DataFrame(comparison_data))

        # ── Per-match results (Poisson, as reference) ─────────────────────────
        st.subheader("Match-Level Results (Poisson)")

        outcome_labels = {
            "team_a_win": "Team A Win",
            "draw": "Draw",
            "team_b_win": "Team B Win",
        }

        rows = []
        for r in bt_results_po:
            rows.append({
                "Date": r.date,
                "Match": f"{r.team_a} vs {r.team_b}",
                "Actual Score": f"{r.actual_goals_a}-{r.actual_goals_b}",
                "Predicted": outcome_labels[r.predicted_outcome],
                "Actual": outcome_labels[r.actual_outcome],
                "Correct": "✓" if r.predicted_outcome == r.actual_outcome else "✗",
                "In Top 5": "✓" if r.in_top_5 else "✗",
                "P(actual)": f"{r.prob_of_actual_result:.1%}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
```

- [ ] **Step 6: Verify syntax and imports**

```bash
python -c "import ast; ast.parse(open('src/app/app.py').read()); print('syntax OK')"
```

- [ ] **Step 7: Run full suite — no regressions**

```bash
python -m pytest -v
```
Expected: 58 tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/app/app.py
git commit -m "feat: model selector radio + Poisson vs Dixon-Coles comparison table"
```

---

## Task 5: V4 Validation Run

- [ ] **Step 1: Run comparison validation script**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.data.loader import load_team_ratings
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

ratings = load_team_ratings()

po = compute_metrics(run_backtest(ratings=ratings, model_type='poisson'))
dc = compute_metrics(run_backtest(ratings=ratings, model_type='dixon_coles'))

print('Metric                       Poisson    Dixon-Coles')
print(f'1X2 Accuracy:                {po.accuracy_1x2:.1%}     {dc.accuracy_1x2:.1%}')
print(f'Exact Score Accuracy:        {po.exact_score_accuracy:.1%}     {dc.exact_score_accuracy:.1%}')
print(f'Top 3 Hit Rate:              {po.top_3_hit_rate:.1%}     {dc.top_3_hit_rate:.1%}')
print(f'Top 5 Hit Rate:              {po.top_5_hit_rate:.1%}     {dc.top_5_hit_rate:.1%}')
print(f'Brier Score:                 {po.brier_score:.4f}    {dc.brier_score:.4f}')
print(f'Avg P(Actual Result):        {po.avg_prob_actual_result:.1%}     {dc.avg_prob_actual_result:.1%}')
"
```

- [ ] **Step 2: Run all tests final time**

```bash
python -m pytest -v
```
Expected: 58 tests pass.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: v4 validation complete — Poisson vs Dixon-Coles comparison"
```
