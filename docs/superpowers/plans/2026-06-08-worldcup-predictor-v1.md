# World Cup Predictor v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit MVP that predicts World Cup match outcomes using a Poisson distribution model, given two teams and their expected goals.

**Architecture:** A data loader reads team names from a CSV; a pure Poisson model function computes win/draw/loss probabilities and top scorelines; a Streamlit app wires them together with reactive selectbox + number_input controls. The only data seam is `loader.py` — future versions swap that one file to pull from an API or DB.

**Tech Stack:** Python 3.10+, Streamlit, pandas, numpy, scipy, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `data/teams.csv` | Source of truth for team names; one row per team, header `team` |
| `src/__init__.py` | Makes `src` a package |
| `src/data/__init__.py` | Makes `src/data` a package |
| `src/data/loader.py` | `load_teams() -> list[str]` — only file that knows where teams come from |
| `src/models/__init__.py` | Makes `src/models` a package |
| `src/models/poisson.py` | `PredictionResult` dataclass + `predict()` pure function |
| `src/app/__init__.py` | Makes `src/app` a package |
| `src/app/app.py` | Streamlit UI — calls loader and model, renders tables |
| `src/backtesting/__init__.py` | Empty stub — reserved for v2 |
| `tests/__init__.py` | Makes tests a package |
| `tests/data/__init__.py` | Makes tests/data a package |
| `tests/data/test_loader.py` | Tests for `load_teams()` |
| `tests/models/__init__.py` | Makes tests/models a package |
| `tests/models/test_poisson.py` | Tests for `predict()` and `PredictionResult` |
| `requirements.txt` | Runtime dependencies (no pins for v1) |
| `README.md` | Setup and run instructions |

---

## Task 1: Project Scaffold

**Files:**
- Create: `data/teams.csv`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/app/__init__.py`, `src/backtesting/__init__.py`
- Create: `tests/__init__.py`, `tests/data/__init__.py`, `tests/models/__init__.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create `data/teams.csv`**

```csv
team
Algeria
Argentina
Australia
Austria
Belgium
Brazil
Cameroon
Canada
Chile
Colombia
Costa Rica
Croatia
Denmark
Ecuador
Egypt
England
France
Germany
Ghana
Iran
Italy
Japan
Mexico
Morocco
Netherlands
Nigeria
Norway
Panama
Paraguay
Peru
Poland
Portugal
Qatar
Saudi Arabia
Senegal
Serbia
South Korea
Spain
Sweden
Switzerland
Tunisia
Ukraine
Uruguay
USA
```

- [ ] **Step 2: Create all `__init__.py` files**

Create empty files at:
- `src/__init__.py`
- `src/data/__init__.py`
- `src/models/__init__.py`
- `src/app/__init__.py`
- `src/backtesting/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/models/__init__.py`

- [ ] **Step 3: Create `requirements.txt`**

```
streamlit
pandas
numpy
scipy
pytest
```

- [ ] **Step 4: Commit scaffold**

```bash
git init
git add data/teams.csv src/ tests/ requirements.txt
git commit -m "chore: project scaffold — directories, packages, teams data"
```

---

## Task 2: Data Loader

**Files:**
- Create: `src/data/loader.py`
- Test: `tests/data/test_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_loader.py`:

```python
import pytest
from src.data.loader import load_teams


def test_returns_list_of_strings():
    teams = load_teams()
    assert isinstance(teams, list)
    assert all(isinstance(t, str) for t in teams)


def test_returns_expected_teams():
    teams = load_teams()
    assert "Argentina" in teams
    assert "Brazil" in teams
    assert "France" in teams


def test_returns_sorted_list():
    teams = load_teams()
    assert teams == sorted(teams)


def test_no_duplicates():
    teams = load_teams()
    assert len(teams) == len(set(teams))


def test_minimum_team_count():
    teams = load_teams()
    assert len(teams) >= 40
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/data/test_loader.py -v
```
Expected: ImportError or ModuleNotFoundError — `load_teams` doesn't exist yet.

- [ ] **Step 3: Implement `src/data/loader.py`**

```python
import pandas as pd
from pathlib import Path

# Path to the teams CSV relative to this file's location.
# Change this one line (or replace the whole function) to load from a DB or API.
_TEAMS_CSV = Path(__file__).parent.parent.parent / "data" / "teams.csv"


def load_teams() -> list[str]:
    """Load national team names from the local CSV.

    Returns a sorted list of team name strings.
    Raises FileNotFoundError if teams.csv is missing.
    Raises ValueError if the CSV is empty or missing the 'team' column.
    """
    if not _TEAMS_CSV.exists():
        raise FileNotFoundError(f"Teams data file not found: {_TEAMS_CSV}")

    df = pd.read_csv(_TEAMS_CSV)

    if "team" not in df.columns:
        raise ValueError("teams.csv must have a 'team' column header")

    teams = df["team"].dropna().str.strip().tolist()

    if not teams:
        raise ValueError("teams.csv contains no team entries")

    return sorted(teams)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/data/test_loader.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/data/test_loader.py
git commit -m "feat: data loader — load_teams() reads from teams.csv"
```

---

## Task 3: Poisson Prediction Model

**Files:**
- Create: `src/models/poisson.py`
- Test: `tests/models/test_poisson.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_poisson.py`:

```python
import pytest
from src.models.poisson import predict, PredictionResult


def test_returns_prediction_result():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    assert isinstance(result, PredictionResult)


def test_probabilities_sum_to_one():
    result = predict("France", "Germany", 1.3, 1.1)
    total = result.win_a + result.draw + result.win_b
    assert abs(total - 1.0) < 0.01  # allow small floating point gap from truncated matrix


def test_probabilities_are_between_0_and_1():
    result = predict("Spain", "England", 1.4, 1.0)
    assert 0.0 <= result.win_a <= 1.0
    assert 0.0 <= result.draw <= 1.0
    assert 0.0 <= result.win_b <= 1.0


def test_higher_xg_team_has_higher_win_probability():
    result = predict("Brazil", "Qatar", 2.5, 0.5)
    assert result.win_a > result.win_b


def test_equal_xg_gives_symmetric_probabilities():
    result = predict("A", "B", 1.5, 1.5)
    assert abs(result.win_a - result.win_b) < 0.001


def test_top_scorelines_returns_five():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    assert len(result.top_scorelines) == 5


def test_top_scorelines_are_sorted_descending():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    probs = [p for _, _, p in result.top_scorelines]
    assert probs == sorted(probs, reverse=True)


def test_top_scorelines_format():
    result = predict("Brazil", "Argentina", 1.5, 1.2)
    for goals_a, goals_b, prob in result.top_scorelines:
        assert isinstance(goals_a, int)
        assert isinstance(goals_b, int)
        assert 0.0 < prob <= 1.0


def test_xg_validation_raises_on_zero():
    with pytest.raises(ValueError):
        predict("A", "B", 0.0, 1.0)


def test_xg_validation_raises_on_negative():
    with pytest.raises(ValueError):
        predict("A", "B", 1.0, -0.5)


def test_team_names_stored_in_result():
    result = predict("France", "Brazil", 1.2, 1.5)
    assert result.team_a == "France"
    assert result.team_b == "Brazil"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/models/test_poisson.py -v
```
Expected: ImportError — `poisson.py` doesn't exist yet.

- [ ] **Step 3: Implement `src/models/poisson.py`**

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
    top_scorelines: list


def predict(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
) -> PredictionResult:
    """Predict match outcome probabilities using independent Poisson distributions.

    Models each team's goals as a Poisson random variable parameterised by their
    expected goals (xG). The score matrix covers 0–10 goals per team, which
    captures >99.9% of real-match probability mass for typical xG values.

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

    max_goals = 11  # consider scores 0–10 for each team

    # Build probability vectors for each team using the Poisson PMF.
    goals_range = np.arange(max_goals)
    prob_a = poisson.pmf(goals_range, xg_a)  # shape: (11,)
    prob_b = poisson.pmf(goals_range, xg_b)  # shape: (11,)

    # Outer product gives the joint probability matrix.
    # matrix[i][j] = P(Team A scores i) * P(Team B scores j)
    matrix = np.outer(prob_a, prob_b)  # shape: (11, 11)

    # Win/draw probabilities from the score matrix.
    win_a = float(np.sum(np.tril(matrix, k=-1)))  # Team A scores more (below diagonal)
    draw  = float(np.sum(np.diag(matrix)))         # Equal scores (diagonal)
    win_b = float(np.sum(np.triu(matrix, k=1)))    # Team B scores more (above diagonal)

    # Top 5 scorelines: flatten to (goals_a, goals_b, probability) tuples,
    # sort by probability descending, break ties by (goals_a, goals_b) ascending.
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/models/test_poisson.py -v
```
Expected: 11 tests pass.

- [ ] **Step 5: Run all tests to check nothing broken**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/poisson.py tests/models/test_poisson.py
git commit -m "feat: Poisson model — predict() returns win/draw/loss + top scorelines"
```

---

## Task 4: Streamlit App

**Files:**
- Create: `src/app/app.py`

No unit tests for the UI — test manually by running the app and verifying the UI behaves correctly.

- [ ] **Step 1: Implement `src/app/app.py`**

```python
import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams
from src.models.poisson import predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")
st.markdown(
    "Select two teams and enter their expected goals to see predicted match outcomes."
)

# ── Load teams ─────────────────────────────────────────────────────────────────
try:
    teams = load_teams()
except (FileNotFoundError, ValueError) as e:
    st.error(f"Could not load teams data: {e}")
    st.stop()

# ── Team selection ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", options=teams, index=0)

with col2:
    # Filter out Team A so the same team cannot be selected twice.
    teams_b = [t for t in teams if t != team_a]
    team_b = st.selectbox("Team B", options=teams_b, index=0)

# ── Expected goals inputs ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Expected Goals (xG)")

col3, col4 = st.columns(2)

with col3:
    xg_a = st.number_input(
        f"{team_a} xG",
        min_value=0.1,
        max_value=5.0,
        value=1.3,
        step=0.1,
        format="%.1f",
    )

with col4:
    xg_b = st.number_input(
        f"{team_b} xG",
        min_value=0.1,
        max_value=5.0,
        value=1.3,
        step=0.1,
        format="%.1f",
    )

# ── Run prediction ─────────────────────────────────────────────────────────────
result = predict(team_a, team_b, xg_a, xg_b)

# ── Results display ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Prediction: {team_a} vs {team_b}")

# Match outcome probabilities table.
outcome_data = {
    "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
    "Probability": [
        f"{result.win_a:.1%}",
        f"{result.draw:.1%}",
        f"{result.win_b:.1%}",
    ],
}
st.markdown("**Match Outcome Probabilities**")
st.table(pd.DataFrame(outcome_data))

# Top 5 most likely scorelines.
scoreline_data = {
    "Scoreline": [f"{team_a} {g_a} – {g_b} {team_b}" for g_a, g_b, _ in result.top_scorelines],
    "Probability": [f"{p:.1%}" for _, _, p in result.top_scorelines],
}
st.markdown("**Top 5 Most Likely Scorelines**")
st.table(pd.DataFrame(scoreline_data))
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Run the app manually**

```bash
streamlit run src/app/app.py
```

Open the URL shown in the terminal (typically http://localhost:8501).

Verify:
- Page loads with title "World Cup Predictor"
- Team A selectbox shows all teams
- Team B selectbox never shows the same team as Team A
- Changing teams or xG values updates the tables instantly
- Win/Draw/Win probabilities always sum to ~100%
- Top 5 scorelines are shown in descending probability order

- [ ] **Step 4: Commit**

```bash
git add src/app/app.py
git commit -m "feat: Streamlit app — team selector, xG inputs, prediction tables"
```

---

## Task 5: README and Final Polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# World Cup Predictor

A local Python MVP that predicts World Cup match outcomes using a Poisson distribution model.

## How It Works

Given two teams and their expected goals (xG), the model builds a score probability matrix using independent Poisson distributions. From that matrix it derives win/draw/loss probabilities and the five most likely exact scorelines.

## Setup

1. **Clone the repo and enter the directory:**
   ```bash
   cd worldcup-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run src/app/app.py
   ```

4. Open the URL shown in the terminal (default: http://localhost:8501).

## Running Tests

```bash
pytest -v
```

## Project Structure

```
worldcup-predictor/
├── data/
│   └── teams.csv          # National teams list — replace to add/remove teams
├── src/
│   ├── data/
│   │   └── loader.py      # Data seam — swap this file to load from API or DB
│   ├── models/
│   │   └── poisson.py     # Poisson prediction model
│   ├── app/
│   │   └── app.py         # Streamlit UI
│   └── backtesting/       # Reserved for v2
├── requirements.txt
└── README.md
```

## Expanding to Future Versions

- **New teams or ratings:** Replace `src/data/loader.py` — it is the only file that knows where data comes from.
- **Improved model:** Extend `PredictionResult` in `src/models/poisson.py` with ELO adjustments, form factors, or player ratings.
- **Backtesting:** Add historical match replay logic in `src/backtesting/`.
```

- [ ] **Step 2: Run all tests one final time**

```bash
pytest -v
```
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, run instructions, and project structure"
```
