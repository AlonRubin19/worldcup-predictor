# World Cup Predictor v4 — Design Spec

**Date:** 2026-06-08  
**Status:** Approved  
**Scope:** Dixon-Coles adjustment as an optional model mode alongside the existing Poisson model.

---

## 1. Goal

Improve prediction accuracy for low-scoring outcomes (especially draws) by applying the Dixon-Coles tau correction to the Poisson score matrix. The existing Poisson model is preserved unchanged; Dixon-Coles is an additive option selected by the user.

---

## 2. What Changes vs v3

| Component | v3 | v4 |
|---|---|---|
| `src/models/poisson.py` | `predict()` only | adds `build_score_matrix()` helper |
| `src/models/dixon_coles.py` | does not exist | new — `predict_dixon_coles()` |
| `src/backtesting/runner.py` | single model | adds `model_type` parameter |
| `src/app/app.py` | no model selector | adds model radio + comparison table |
| `tests/models/test_dixon_coles.py` | does not exist | new |
| `PredictionResult` dataclass | unchanged | unchanged |
| `xg_calculator.py` | unchanged | unchanged |

---

## 3. Score Matrix Helper

**`src/models/poisson.py`** gains one public function:

```python
def build_score_matrix(xg_a: float, xg_b: float) -> np.ndarray:
    """Build a Poisson joint probability matrix for goals 0 to max_goals-1.

    Returns an (N x N) numpy array where cell [i][j] =
    poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b).

    N is adaptive: max(11, int(max(xg_a, xg_b) * 3) + 1).
    """
```

The existing `predict()` function is refactored to call `build_score_matrix()` internally. No change to its signature or behaviour.

---

## 4. Dixon-Coles Model

**`src/models/dixon_coles.py`**

```python
def predict_dixon_coles(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
    rho: float = -0.10,
) -> PredictionResult:
```

### Algorithm

1. Validate `xg_a > 0`, `xg_b > 0`. Raise `ValueError` if not.
2. Call `build_score_matrix(xg_a, xg_b)` → `matrix`.
3. Apply tau correction to the four low-score cells:

| Scoreline | tau formula |
|---|---|
| 0-0 | `1 - (xg_a * xg_b * rho)` |
| 0-1 | `1 + (xg_a * rho)` |
| 1-0 | `1 + (xg_b * rho)` |
| 1-1 | `1 - rho` |
| All others | `1` (unchanged) |

4. Multiply each cell by its tau: `matrix[i, j] *= tau(i, j)`.
5. **Normalize** the matrix so all cells sum to 1: `matrix /= matrix.sum()`.
6. Extract `win_a`, `draw`, `win_b` from the normalized matrix (same triangle logic as Poisson).
7. Extract top 5 scorelines (same sort logic as Poisson).
8. Return `PredictionResult(team_a, team_b, win_a, draw, win_b, top_scorelines)`.

### Key invariant

When `rho = 0`, all tau values are 1 and the matrix is identical to the raw Poisson matrix, so `predict_dixon_coles(..., rho=0)` ≈ `predict(...)`.

### Default rho

`rho = -0.10` is a standard empirical value from the Dixon-Coles 1997 paper. Negative rho reduces the probability of 0-0 and 1-1 draws while increasing 1-0 and 0-1 results relative to pure Poisson.

---

## 5. Updated Backtesting Runner

**`src/backtesting/runner.py`**

```python
def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
    model_type: str = "poisson",   # "poisson" | "dixon_coles"
) -> list[MatchResult]:
```

- If `model_type == "poisson"`: calls `predict(team_a, team_b, xg_a, xg_b)`.
- If `model_type == "dixon_coles"`: calls `predict_dixon_coles(team_a, team_b, xg_a, xg_b)`.
- If `model_type` is anything else: raises `ValueError`.
- All other logic unchanged.

---

## 6. Updated Streamlit App

### Predictor Tab

Add a model selector above the xG section:

```python
model_choice = st.radio("Prediction Model", ["Poisson", "Dixon-Coles"], horizontal=True)
```

If "Dixon-Coles" is selected, call `predict_dixon_coles(...)` instead of `predict(...)`.

### Backtesting Tab

Run both models and show a comparison table:

```
| Metric                  | Poisson | Dixon-Coles |
|-------------------------|---------|-------------|
| 1X2 Accuracy            | 57.5%   | XX.X%       |
| Exact Score Accuracy    | 15.0%   | XX.X%       |
| Top 3 Hit Rate          | 25.0%   | XX.X%       |
| Top 5 Hit Rate          | 57.5%   | XX.X%       |
| Brier Score             | 0.5418  | X.XXXX      |
| Avg P(Actual Result)    | 42.7%   | XX.X%       |
```

Followed by the per-match detail table (already exists), showing results for the currently selected model (default: Poisson).

---

## 7. Tests

**`tests/models/test_dixon_coles.py`**

Required test cases:
- `predict_dixon_coles` returns a `PredictionResult`
- Probabilities (win_a + draw + win_b) sum to ≈ 1.0
- `rho=0` gives nearly identical result to `predict()` (within 1e-6)
- Draw probability is different from pure Poisson with default `rho=-0.10`
- Top 5 scorelines list has 5 entries
- `xg_a <= 0` or `xg_b <= 0` raises `ValueError`
- Low-score probabilities (0-0, 1-1) differ from pure Poisson

**`tests/backtesting/test_runner.py`** additions:
- `run_backtest(model_type="poisson")` works
- `run_backtest(model_type="dixon_coles")` works
- `run_backtest(model_type="invalid")` raises `ValueError`

---

## 8. File Structure (additions only)

```
worldcup-predictor/
├── src/
│   └── models/
│       ├── poisson.py              # modified: extract build_score_matrix()
│       └── dixon_coles.py          # new
└── tests/
    └── models/
        └── test_dixon_coles.py     # new
```
