# World Cup Predictor v3 — Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtesting engine that validates the xG + Poisson model against 40 historical international match results and surfaces accuracy metrics in a new Streamlit tab.

**Architecture:** `historical_matches.csv` feeds into `runner.py` which produces a `MatchResult` per match; `metrics.py` aggregates those into a `BacktestMetrics` summary; the Streamlit app gains a second tab that calls both and renders the results. All existing model code is untouched.

**Tech Stack:** Python 3.10+, pandas, streamlit, pytest (no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `data/historical_matches.csv` | Create — 40 sample matches |
| `tests/backtesting/__init__.py` | Create — empty |
| `src/backtesting/runner.py` | Create — `MatchResult` dataclass + `run_backtest()` |
| `src/backtesting/metrics.py` | Create — `BacktestMetrics` dataclass + `compute_metrics()` |
| `tests/backtesting/test_runner.py` | Create — 7 tests for runner |
| `tests/backtesting/test_metrics.py` | Create — 5 tests for metrics |
| `src/app/app.py` | Modify — add `st.tabs` + Backtesting tab |
| `src/models/poisson.py` | No change |
| `src/models/xg_calculator.py` | No change |
| `src/data/loader.py` | No change |

---

## Task 1: Historical Matches Dataset

**Files:**
- Create: `data/historical_matches.csv`

- [ ] **Step 1: Create `data/historical_matches.csv`**

```csv
date,team_a,team_b,team_a_goals,team_b_goals
2022-11-20,Ecuador,Qatar,2,0
2022-11-21,England,Iran,6,2
2022-11-21,Senegal,Netherlands,0,2
2022-11-22,USA,Wales,1,1
2022-11-22,Argentina,Saudi Arabia,1,2
2022-11-22,Denmark,Tunisia,0,0
2022-11-23,Mexico,Poland,0,0
2022-11-23,France,Australia,4,1
2022-11-24,Morocco,Croatia,0,0
2022-11-24,Germany,Japan,1,2
2022-11-24,Spain,Costa Rica,7,0
2022-11-25,Belgium,Canada,1,0
2022-11-25,Switzerland,Cameroon,1,0
2022-11-26,Uruguay,South Korea,0,0
2022-11-26,Portugal,Ghana,3,2
2022-11-27,Brazil,Serbia,2,0
2022-11-28,France,Denmark,2,1
2022-11-28,Argentina,Mexico,2,0
2022-11-29,Japan,Costa Rica,0,1
2022-11-29,Germany,Spain,1,1
2022-11-30,Belgium,Morocco,0,2
2022-11-30,Croatia,Canada,4,1
2022-12-01,Brazil,Switzerland,1,0
2022-12-01,Portugal,Uruguay,2,0
2022-12-02,South Korea,Ghana,2,3
2022-12-02,Cameroon,Serbia,3,3
2022-12-03,Argentina,Australia,2,1
2022-12-03,France,Poland,3,1
2022-12-04,England,Senegal,3,0
2022-12-04,Netherlands,USA,3,1
2022-12-05,Japan,Croatia,1,1
2022-12-05,Brazil,South Korea,4,1
2022-12-06,Morocco,Spain,0,0
2022-12-06,Portugal,Switzerland,6,1
2022-12-09,Argentina,Netherlands,2,2
2022-12-09,France,England,2,1
2022-12-10,Morocco,Portugal,1,0
2022-12-10,Croatia,Brazil,1,1
2022-12-13,Argentina,Croatia,3,0
2022-12-14,France,Morocco,2,0
```

- [ ] **Step 2: Verify the file**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/historical_matches.csv')
print(len(df), 'matches')
print(df.columns.tolist())
print('Sample:', df.iloc[0].tolist())
"
```
Expected: `40 matches`, correct columns, first row shows Ecuador vs Qatar 2-0.

- [ ] **Step 3: Commit**

```bash
git add data/historical_matches.csv
git commit -m "data: add historical_matches.csv — 40 World Cup 2022 group+knockout matches"
```

---

## Task 2: Backtesting Runner

**Files:**
- Create: `tests/backtesting/__init__.py`
- Create: `src/backtesting/runner.py`
- Create: `tests/backtesting/test_runner.py`

- [ ] **Step 1: Create `tests/backtesting/__init__.py`** (empty file)

- [ ] **Step 2: Create `tests/backtesting/test_runner.py`**

```python
import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.runner import run_backtest, MatchResult


def _make_csv(tmp_path, rows):
    """Write a minimal matches CSV for testing."""
    f = tmp_path / "matches.csv"
    df = pd.DataFrame(rows, columns=["date","team_a","team_b","team_a_goals","team_b_goals"])
    df.to_csv(f, index=False)
    return f


def _minimal_ratings():
    """Ratings dict covering the test teams."""
    from src.data.loader import load_team_ratings
    return load_team_ratings()


def test_returns_list_of_match_results(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Brazil",2,1]])
    results = run_backtest(csv, ratings)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_actual_outcome_team_a_win(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",3,0]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "team_a_win"


def test_actual_outcome_draw(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","Germany","Spain",1,1]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "draw"


def test_actual_outcome_team_b_win(tmp_path):
    ratings = _minimal_ratings()
    csv = _make_csv(tmp_path, [["2022-11-20","Qatar","France",0,3]])
    result = run_backtest(csv, ratings)[0]
    assert result.actual_outcome == "team_b_win"


def test_predicted_outcome_is_max_probability(tmp_path):
    ratings = _minimal_ratings()
    # France vs Qatar — France should be heavy favourite
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",2,0]])
    result = run_backtest(csv, ratings)[0]
    # predicted_outcome must be the outcome with the highest probability
    probs = {
        "team_a_win": result.win_a_prob,
        "draw": result.draw_prob,
        "team_b_win": result.win_b_prob,
    }
    assert result.predicted_outcome == max(probs, key=probs.get)


def test_in_top_5_when_score_present(tmp_path):
    ratings = _minimal_ratings()
    # 1-0 results are nearly always in top 5 for typical xG
    csv = _make_csv(tmp_path, [["2022-11-20","France","Qatar",1,0]])
    result = run_backtest(csv, ratings)[0]
    # check the flag is consistent with top_scorelines
    top5_scores = [(g_a, g_b) for g_a, g_b, _ in result.top_scorelines]
    assert result.in_top_5 == ((result.actual_goals_a, result.actual_goals_b) in top5_scores)


def test_missing_file_raises(tmp_path):
    ratings = _minimal_ratings()
    with pytest.raises(FileNotFoundError):
        run_backtest(tmp_path / "nonexistent.csv", ratings)


def test_missing_team_in_ratings_raises(tmp_path):
    csv = _make_csv(tmp_path, [["2022-11-20","France","UnknownTeam",1,0]])
    ratings = {"France": {"elo":2020,"attack_rating":1.18,"defense_rating":0.85,
                          "form_rating":1.08,"squad_rating":1.12}}
    with pytest.raises(ValueError, match="not found in ratings"):
        run_backtest(csv, ratings)
```

- [ ] **Step 3: Run to confirm tests fail**

```bash
python -m pytest tests/backtesting/test_runner.py -v
```
Expected: ImportError — `runner.py` doesn't exist yet.

- [ ] **Step 4: Create `src/backtesting/runner.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data.loader import load_team_ratings
from src.models.xg_calculator import calculate_xg
from src.models.poisson import predict

_MATCHES_CSV = Path(__file__).parent.parent.parent / "data" / "historical_matches.csv"

_REQUIRED_COLUMNS = {"date", "team_a", "team_b", "team_a_goals", "team_b_goals"}


@dataclass
class MatchResult:
    """Predicted vs actual outcome for one historical match."""
    date: str
    team_a: str
    team_b: str
    actual_goals_a: int
    actual_goals_b: int
    actual_outcome: str          # "team_a_win" | "draw" | "team_b_win"
    win_a_prob: float
    draw_prob: float
    win_b_prob: float
    predicted_outcome: str       # outcome with highest predicted probability
    top_scorelines: list[tuple[int, int, float]]
    exact_score_hit: bool        # actual score is #1 predicted scoreline
    in_top_3: bool               # actual score in top 3 predicted scorelines
    in_top_5: bool               # actual score in top 5 predicted scorelines
    prob_of_actual_result: float # predicted probability of the actual 1X2 outcome


def run_backtest(
    matches_path: Path | None = None,
    ratings: dict | None = None,
) -> list[MatchResult]:
    """Run model predictions for all historical matches and return per-match results.

    Args:
        matches_path: Path to historical_matches.csv. Defaults to data/historical_matches.csv.
        ratings: Dict from load_team_ratings(). Loaded from default CSV if not provided.

    Raises:
        FileNotFoundError: if matches CSV is missing.
        ValueError: if a team in the CSV is not found in ratings.
    """
    path = matches_path if matches_path is not None else _MATCHES_CSV

    if not path.exists():
        raise FileNotFoundError(f"Historical matches file not found: {path}")

    if ratings is None:
        ratings = load_team_ratings()

    df = pd.read_csv(path)
    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"historical_matches.csv missing columns: {sorted(missing_cols)}")

    results = []
    for _, row in df.iterrows():
        team_a = str(row["team_a"]).strip()
        team_b = str(row["team_b"]).strip()

        if team_a not in ratings:
            raise ValueError(f"Team '{team_a}' not found in ratings")
        if team_b not in ratings:
            raise ValueError(f"Team '{team_b}' not found in ratings")

        xg_a, xg_b = calculate_xg(ratings[team_a], ratings[team_b])
        prediction = predict(team_a, team_b, xg_a, xg_b)

        goals_a = int(row["team_a_goals"])
        goals_b = int(row["team_b_goals"])

        if goals_a > goals_b:
            actual_outcome = "team_a_win"
        elif goals_a == goals_b:
            actual_outcome = "draw"
        else:
            actual_outcome = "team_b_win"

        # Outcome with the highest predicted probability.
        probs = {
            "team_a_win": prediction.win_a,
            "draw": prediction.draw,
            "team_b_win": prediction.win_b,
        }
        predicted_outcome = max(probs, key=probs.get)

        # Scoreline hit flags.
        top5 = [(g_a, g_b) for g_a, g_b, _ in prediction.top_scorelines]
        exact_score_hit = len(top5) > 0 and top5[0] == (goals_a, goals_b)
        in_top_3 = (goals_a, goals_b) in top5[:3]
        in_top_5 = (goals_a, goals_b) in top5

        prob_of_actual_result = probs[actual_outcome]

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
            exact_score_hit=exact_score_hit,
            in_top_3=in_top_3,
            in_top_5=in_top_5,
            prob_of_actual_result=prob_of_actual_result,
        ))

    return results
```

- [ ] **Step 5: Run runner tests**

```bash
python -m pytest tests/backtesting/test_runner.py -v
```
Expected: 7 tests pass.

- [ ] **Step 6: Run full suite**

```bash
python -m pytest -v
```
Expected: 38 tests pass (31 existing + 7 new).

- [ ] **Step 7: Commit**

```bash
git add tests/backtesting/__init__.py src/backtesting/runner.py tests/backtesting/test_runner.py
git commit -m "feat: backtesting runner — run_backtest() produces MatchResult per match"
```

---

## Task 3: Metrics Engine

**Files:**
- Create: `src/backtesting/metrics.py`
- Create: `tests/backtesting/test_metrics.py`

- [ ] **Step 1: Create `tests/backtesting/test_metrics.py`**

```python
import pytest
from src.backtesting.metrics import compute_metrics, BacktestMetrics
from src.backtesting.runner import MatchResult


def _result(actual, win_a, draw, win_b, exact=False, top3=False, top5=False):
    """Build a minimal MatchResult for metric testing."""
    prob = {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}[actual]
    predicted = max({"team_a_win": win_a, "draw": draw, "team_b_win": win_b},
                    key=lambda k: {"team_a_win": win_a, "draw": draw, "team_b_win": win_b}[k])
    return MatchResult(
        date="2022-11-20", team_a="A", team_b="B",
        actual_goals_a=1, actual_goals_b=0,
        actual_outcome=actual,
        win_a_prob=win_a, draw_prob=draw, win_b_prob=win_b,
        predicted_outcome=predicted,
        top_scorelines=[(1,0,0.10),(2,0,0.08),(0,0,0.07)] if top5 else [(9,9,0.01)]*5,
        exact_score_hit=exact,
        in_top_3=top3,
        in_top_5=top5,
        prob_of_actual_result=prob,
    )


def test_returns_backtest_metrics():
    results = [_result("team_a_win", 0.6, 0.25, 0.15)]
    m = compute_metrics(results)
    assert isinstance(m, BacktestMetrics)


def test_accuracy_1x2_all_correct():
    # predicted_outcome == actual_outcome for all results
    results = [
        _result("team_a_win", 0.6, 0.25, 0.15),  # predicted: team_a_win ✓
        _result("draw",       0.2, 0.50, 0.30),   # predicted: draw ✓
    ]
    m = compute_metrics(results)
    assert m.accuracy_1x2 == pytest.approx(1.0)


def test_accuracy_1x2_none_correct():
    # All results where predicted ≠ actual
    results = [
        _result("team_b_win", 0.6, 0.25, 0.15),  # predicted: team_a_win ✗
        _result("team_a_win", 0.1, 0.30, 0.60),  # predicted: team_b_win ✗
    ]
    m = compute_metrics(results)
    assert m.accuracy_1x2 == pytest.approx(0.0)


def test_brier_score_perfect_prediction():
    # If probability of actual outcome is 1.0, Brier score should be 0.0
    results = [_result("team_a_win", 1.0, 0.0, 0.0)]
    m = compute_metrics(results)
    assert m.brier_score == pytest.approx(0.0, abs=1e-9)


def test_brier_score_worst_prediction():
    # Probability 0.0 assigned to actual outcome → maximum Brier score for 3-class = 2.0
    results = [_result("team_a_win", 0.0, 0.0, 1.0)]
    m = compute_metrics(results)
    # BS = (0-1)^2 + (0-0)^2 + (1-0)^2 = 1 + 0 + 1 = 2.0
    assert m.brier_score == pytest.approx(2.0, abs=1e-9)


def test_hit_rates_computed_correctly():
    results = [
        _result("team_a_win", 0.6, 0.25, 0.15, exact=True,  top3=True,  top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=True,  top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=False, top5=True),
        _result("team_a_win", 0.6, 0.25, 0.15, exact=False, top3=False, top5=False),
    ]
    m = compute_metrics(results)
    assert m.exact_score_accuracy == pytest.approx(0.25)
    assert m.top_3_hit_rate       == pytest.approx(0.5)
    assert m.top_5_hit_rate       == pytest.approx(0.75)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
python -m pytest tests/backtesting/test_metrics.py -v
```
Expected: ImportError — `metrics.py` doesn't exist yet.

- [ ] **Step 3: Create `src/backtesting/metrics.py`**

```python
from dataclasses import dataclass
from src.backtesting.runner import MatchResult


@dataclass
class BacktestMetrics:
    """Aggregate accuracy metrics for a backtesting run."""
    total_matches: int
    accuracy_1x2: float           # fraction where predicted_outcome == actual_outcome
    exact_score_accuracy: float   # fraction where exact_score_hit is True
    top_3_hit_rate: float         # fraction where in_top_3 is True
    top_5_hit_rate: float         # fraction where in_top_5 is True
    brier_score: float            # multi-class Brier score for 1X2 probabilities
    avg_prob_actual_result: float # mean predicted probability of the actual 1X2 outcome


def compute_metrics(results: list[MatchResult]) -> BacktestMetrics:
    """Compute aggregate metrics from a list of MatchResult objects.

    Brier score (multi-class, 3 outcomes):
        BS = (1/N) * sum_i [ (p_win_a - o_win_a)^2
                           + (p_draw  - o_draw )^2
                           + (p_win_b - o_win_b)^2 ]
    where o_* is 1.0 if that outcome occurred, 0.0 otherwise.

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        raise ValueError("Cannot compute metrics on empty results list")

    n = len(results)

    correct_1x2  = sum(1 for r in results if r.predicted_outcome == r.actual_outcome)
    exact_hits   = sum(1 for r in results if r.exact_score_hit)
    top3_hits    = sum(1 for r in results if r.in_top_3)
    top5_hits    = sum(1 for r in results if r.in_top_5)
    total_prob   = sum(r.prob_of_actual_result for r in results)

    brier_total = 0.0
    for r in results:
        # One-hot encode actual outcome.
        o_win_a = 1.0 if r.actual_outcome == "team_a_win" else 0.0
        o_draw  = 1.0 if r.actual_outcome == "draw"       else 0.0
        o_win_b = 1.0 if r.actual_outcome == "team_b_win" else 0.0

        brier_total += (
            (r.win_a_prob - o_win_a) ** 2
            + (r.draw_prob  - o_draw)  ** 2
            + (r.win_b_prob - o_win_b) ** 2
        )

    return BacktestMetrics(
        total_matches=n,
        accuracy_1x2=correct_1x2 / n,
        exact_score_accuracy=exact_hits / n,
        top_3_hit_rate=top3_hits / n,
        top_5_hit_rate=top5_hits / n,
        brier_score=brier_total / n,
        avg_prob_actual_result=total_prob / n,
    )
```

- [ ] **Step 4: Run metrics tests**

```bash
python -m pytest tests/backtesting/test_metrics.py -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```
Expected: 45 tests pass (38 + 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/metrics.py tests/backtesting/test_metrics.py
git commit -m "feat: backtesting metrics — compute_metrics() with Brier score and hit rates"
```

---

## Task 4: Streamlit Backtesting Tab

**Files:**
- Modify: `src/app/app.py`

- [ ] **Step 1: Replace `src/app/app.py` entirely**

```python
import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")

tab_predictor, tab_backtest = st.tabs(["⚽ Match Predictor", "📊 Backtesting"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predictor:
    st.markdown("Select two teams to see auto-calculated expected goals and match outcome predictions.")

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        teams = load_teams()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load teams data: {e}")
        st.stop()

    try:
        all_ratings = load_team_ratings()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load team ratings: {e}")
        st.stop()

    # ── Team selection ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Team A", options=teams, index=0)

    with col2:
        teams_b = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", options=teams_b, index=0)

    # ── Auto-calculate xG ─────────────────────────────────────────────────────
    ratings_a = all_ratings.get(team_a)
    ratings_b = all_ratings.get(team_b)

    if ratings_a is None:
        st.warning(f"No ratings found for {team_a} — using baseline xG ({BASE_XG}).")
    if ratings_b is None:
        st.warning(f"No ratings found for {team_b} — using baseline xG ({BASE_XG}).")

    _AVG = {"elo": 1800, "attack_rating": 1.0, "defense_rating": 1.0,
            "form_rating": 1.0, "squad_rating": 1.0}
    auto_xg_a, auto_xg_b = calculate_xg(
        ratings_a if ratings_a is not None else _AVG,
        ratings_b if ratings_b is not None else _AVG,
    )

    # ── xG inputs ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Expected Goals (xG)")

    override = st.checkbox("Override xG manually", value=False)

    _label = "Auto-calculated xG (overridden below)" if override else "Auto-calculated xG"
    st.info(f"**{_label}** — {team_a}: **{auto_xg_a:.2f}** | {team_b}: **{auto_xg_b:.2f}**")

    col3, col4 = st.columns(2)

    with col3:
        xg_a = st.number_input(
            f"{team_a} xG", min_value=0.1, max_value=5.0,
            value=float(round(auto_xg_a, 1)), step=0.1, format="%.1f",
            disabled=not override, key="xg_a_input",
        )

    with col4:
        xg_b = st.number_input(
            f"{team_b} xG", min_value=0.1, max_value=5.0,
            value=float(round(auto_xg_b, 1)), step=0.1, format="%.1f",
            disabled=not override, key="xg_b_input",
        )

    final_xg_a = xg_a if override else auto_xg_a
    final_xg_b = xg_b if override else auto_xg_b

    st.caption(
        f"**Final xG used** — {team_a}: {final_xg_a:.2f} | {team_b}: {final_xg_b:.2f}"
        + (" *(manual override)*" if override else " *(auto-calculated)*")
    )

    # ── Prediction ────────────────────────────────────────────────────────────
    try:
        result = predict(team_a, team_b, final_xg_a, final_xg_b)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader(f"Prediction: {team_a} vs {team_b}")

    outcome_data = {
        "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
        "Probability": [
            f"{result.win_a:.1%}", f"{result.draw:.1%}", f"{result.win_b:.1%}",
        ],
    }
    st.markdown("**Match Outcome Probabilities**")
    st.table(pd.DataFrame(outcome_data))

    scoreline_data = {
        "Scoreline": [f"{team_a} {g_a} – {g_b} {team_b}" for g_a, g_b, _ in result.top_scorelines],
        "Probability": [f"{p:.1%}" for _, _, p in result.top_scorelines],
    }
    st.markdown("**Top 5 Most Likely Scorelines**")
    st.table(pd.DataFrame(scoreline_data))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown("Model accuracy validated against historical international match results.")

    try:
        bt_ratings = load_team_ratings()
        bt_results = run_backtest(ratings=bt_ratings)
        bt_metrics = compute_metrics(bt_results)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Backtesting failed: {e}")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Model Performance")

    metrics_data = {
        "Metric": [
            "Total Matches Tested",
            "1X2 Accuracy",
            "Exact Score Accuracy",
            "Top 3 Scoreline Hit Rate",
            "Top 5 Scoreline Hit Rate",
            "Brier Score (lower = better)",
            "Avg Probability of Actual Result",
        ],
        "Value": [
            str(bt_metrics.total_matches),
            f"{bt_metrics.accuracy_1x2:.1%}",
            f"{bt_metrics.exact_score_accuracy:.1%}",
            f"{bt_metrics.top_3_hit_rate:.1%}",
            f"{bt_metrics.top_5_hit_rate:.1%}",
            f"{bt_metrics.brier_score:.4f}",
            f"{bt_metrics.avg_prob_actual_result:.1%}",
        ],
    }
    st.table(pd.DataFrame(metrics_data))

    # ── Per-match results ─────────────────────────────────────────────────────
    st.subheader("Match-Level Results")

    outcome_labels = {
        "team_a_win": "Home Win",
        "draw": "Draw",
        "team_b_win": "Away Win",
    }

    rows = []
    for r in bt_results:
        rows.append({
            "Date": r.date,
            "Match": f"{r.team_a} vs {r.team_b}",
            "Actual Score": f"{r.actual_goals_a}–{r.actual_goals_b}",
            "Predicted": outcome_labels[r.predicted_outcome],
            "Actual": outcome_labels[r.actual_outcome],
            "Correct": "✓" if r.predicted_outcome == r.actual_outcome else "✗",
            "In Top 5": "✓" if r.in_top_5 else "✗",
            "P(actual)": f"{r.prob_of_actual_result:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
```

- [ ] **Step 2: Verify syntax and imports**

```bash
python -c "import ast; ast.parse(open('src/app/app.py').read()); print('syntax OK')"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics
print('imports OK')
"
```

- [ ] **Step 3: Run full suite — no regressions**

```bash
python -m pytest -v
```
Expected: 45 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/app/app.py
git commit -m "feat: add Backtesting tab to Streamlit app"
```

---

## Task 5: Validation Run

- [ ] **Step 1: Run backtest and print report**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.data.loader import load_team_ratings
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

ratings = load_team_ratings()
results = run_backtest(ratings=ratings)
m = compute_metrics(results)

print('=== V3 Backtest Results ===')
print(f'Total matches:          {m.total_matches}')
print(f'1X2 accuracy:           {m.accuracy_1x2:.1%}')
print(f'Exact score accuracy:   {m.exact_score_accuracy:.1%}')
print(f'Top 3 hit rate:         {m.top_3_hit_rate:.1%}')
print(f'Top 5 hit rate:         {m.top_5_hit_rate:.1%}')
print(f'Brier score:            {m.brier_score:.4f}')
print(f'Avg P(actual result):   {m.avg_prob_actual_result:.1%}')
print()
correct = [r for r in results if r.predicted_outcome == r.actual_outcome]
print(f'Correct predictions:    {len(correct)} / {m.total_matches}')
"
```

- [ ] **Step 2: Run all tests one final time**

```bash
python -m pytest -v
```
Expected: 45 tests pass.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: v3 validation complete"
```
