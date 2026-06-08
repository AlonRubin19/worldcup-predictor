# World Cup Predictor v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic xG calculation from a local team ratings CSV, replacing manual xG sliders with auto-filled values and an optional manual override.

**Architecture:** A new `team_ratings.csv` feeds into a new `load_team_ratings()` function in the existing loader; `xg_calculator.py` applies the multi-factor formula; `app.py` wires them together with an override checkbox. The Poisson model is untouched throughout.

**Tech Stack:** Python 3.10+, pandas, streamlit, pytest (existing stack — no new dependencies)

---

## File Map

| File | Change |
|---|---|
| `data/team_ratings.csv` | Create — 45 rows of team ratings |
| `src/data/loader.py` | Modify — add `load_team_ratings()` |
| `src/models/xg_calculator.py` | Create — `calculate_xg()` + constants |
| `src/models/poisson.py` | No change |
| `src/app/app.py` | Modify — auto xG, override checkbox, updated display |
| `tests/data/test_loader.py` | Modify — add 4 tests for `load_team_ratings()` |
| `tests/models/test_xg_calculator.py` | Create — 7 tests for `calculate_xg()` |

---

## Task 1: Team Ratings Dataset

**Files:**
- Create: `data/team_ratings.csv`

No tests for this task — data correctness is verified implicitly by Task 2 loader tests.

- [ ] **Step 1: Create `data/team_ratings.csv`**

Create the file with this exact content (header + 45 data rows):

```csv
team,elo,attack_rating,defense_rating,form_rating,squad_rating
Algeria,1680,0.92,1.05,0.95,0.90
Argentina,2070,1.15,0.88,1.05,1.10
Australia,1730,0.88,1.02,0.90,0.88
Austria,1800,0.95,1.00,0.96,0.92
Belgium,1940,1.08,0.95,0.98,1.05
Brazil,2050,1.12,0.90,1.02,1.08
Cameroon,1700,0.90,1.05,0.92,0.88
Canada,1760,0.88,1.00,0.95,0.88
Chile,1820,0.95,1.00,0.90,0.93
Colombia,1870,1.00,0.98,0.95,0.98
Costa Rica,1710,0.82,1.05,0.88,0.85
Croatia,1940,1.05,0.92,1.00,1.00
Denmark,1910,1.02,0.92,1.00,0.98
Ecuador,1760,0.88,1.02,0.92,0.88
Egypt,1700,0.85,1.05,0.90,0.85
England,1985,1.10,0.92,1.00,1.05
France,2020,1.18,0.85,1.08,1.12
Germany,1985,1.10,0.92,0.98,1.06
Ghana,1680,0.85,1.08,0.88,0.82
Iran,1720,0.80,1.05,0.88,0.82
Italy,1930,1.05,0.90,0.95,1.00
Japan,1820,0.95,0.98,1.00,0.92
Mexico,1840,0.95,1.00,0.92,0.93
Morocco,1870,1.00,0.90,1.02,0.95
Netherlands,1975,1.08,0.92,1.00,1.05
Nigeria,1720,0.90,1.05,0.90,0.88
Norway,1800,0.95,1.00,0.96,0.92
Panama,1650,0.78,1.10,0.88,0.80
Paraguay,1720,0.85,1.05,0.88,0.85
Peru,1780,0.90,1.02,0.90,0.88
Poland,1840,0.95,1.00,0.92,0.93
Portugal,2010,1.15,0.88,1.05,1.10
Qatar,1640,0.72,1.15,0.80,0.75
Saudi Arabia,1700,0.82,1.08,0.88,0.82
Senegal,1840,1.00,0.98,1.00,0.95
Serbia,1850,1.00,0.98,0.95,0.95
South Korea,1800,0.92,1.00,0.95,0.90
Spain,1975,1.08,0.90,0.98,1.04
Sweden,1850,0.98,0.95,0.98,0.95
Switzerland,1870,1.00,0.95,1.00,0.98
Tunisia,1680,0.85,1.08,0.88,0.83
Ukraine,1840,0.98,1.00,0.95,0.95
Uruguay,1900,1.05,0.92,0.98,1.00
USA,1820,0.92,1.00,0.95,0.90
```

- [ ] **Step 2: Verify row count**

Run:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/team_ratings.csv'); print(len(df), 'teams'); print(df.columns.tolist())"
```
Expected: `44 teams` and `['team', 'elo', 'attack_rating', 'defense_rating', 'form_rating', 'squad_rating']`

- [ ] **Step 3: Commit**

```bash
git add data/team_ratings.csv
git commit -m "data: add team_ratings.csv with ELO and multiplier ratings for 44 teams"
```

---

## Task 2: Ratings Loader

**Files:**
- Modify: `src/data/loader.py`
- Test: `tests/data/test_loader.py`

- [ ] **Step 1: Add failing tests to `tests/data/test_loader.py`**

Append these tests to the existing file (do not remove existing tests):

```python
from src.data.loader import load_team_ratings


def test_load_team_ratings_returns_dict():
    ratings = load_team_ratings()
    assert isinstance(ratings, dict)


def test_load_team_ratings_contains_expected_keys():
    ratings = load_team_ratings()
    assert "Argentina" in ratings
    required_keys = {"elo", "attack_rating", "defense_rating", "form_rating", "squad_rating"}
    assert required_keys.issubset(ratings["Argentina"].keys())


def test_load_team_ratings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_team_ratings(tmp_path / "missing.csv")


def test_load_team_ratings_missing_column_raises(tmp_path):
    f = tmp_path / "ratings.csv"
    f.write_text("team,elo\nArgentina,2070\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_team_ratings(f)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/data/test_loader.py -v -k "ratings"
```
Expected: ImportError — `load_team_ratings` not imported yet.

- [ ] **Step 3: Add `load_team_ratings()` to `src/data/loader.py`**

Add after the existing `load_teams()` function:

```python
_RATINGS_CSV = Path(__file__).parent.parent.parent / "data" / "team_ratings.csv"

_REQUIRED_RATING_COLUMNS = {
    "team", "elo", "attack_rating", "defense_rating", "form_rating", "squad_rating"
}


def load_team_ratings(ratings_path: Path | None = None) -> dict[str, dict]:
    """Load team ratings from the local CSV.

    Returns a dict keyed by team name, each value a dict of rating fields:
        {"elo": 2070, "attack_rating": 1.15, "defense_rating": 0.88,
         "form_rating": 1.05, "squad_rating": 1.10}

    Raises FileNotFoundError if the ratings CSV is missing.
    Raises ValueError if required columns are missing.
    """
    path = ratings_path if ratings_path is not None else _RATINGS_CSV

    if not path.exists():
        raise FileNotFoundError(f"Ratings data file not found: {path}")

    df = pd.read_csv(path)

    missing = _REQUIRED_RATING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"team_ratings.csv missing columns: {sorted(missing)}")

    df["team"] = df["team"].str.strip()
    return df.set_index("team")[list(_REQUIRED_RATING_COLUMNS - {"team"})].to_dict("index")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/data/test_loader.py -v
```
Expected: all 12 tests pass (8 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/data/test_loader.py
git commit -m "feat: add load_team_ratings() to data loader"
```

---

## Task 3: xG Calculator

**Files:**
- Create: `src/models/xg_calculator.py`
- Create: `tests/models/test_xg_calculator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_xg_calculator.py`:

```python
import pytest
from src.models.xg_calculator import calculate_xg, BASE_XG, XG_MIN, XG_MAX


def _ratings(elo=1900, attack=1.0, defense=1.0, form=1.0, squad=1.0):
    """Helper: build a ratings dict with sensible defaults."""
    return {
        "elo": elo,
        "attack_rating": attack,
        "defense_rating": defense,
        "form_rating": form,
        "squad_rating": squad,
    }


def test_returns_tuple_of_two_floats():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert isinstance(xg_a, float)
    assert isinstance(xg_b, float)


def test_equal_ratings_give_equal_xg():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert abs(xg_a - xg_b) < 1e-9


def test_higher_elo_team_gets_higher_xg():
    xg_a, xg_b = calculate_xg(_ratings(elo=2000), _ratings(elo=1700))
    assert xg_a > xg_b


def test_lower_elo_team_gets_lower_xg():
    xg_a, xg_b = calculate_xg(_ratings(elo=1700), _ratings(elo=2000))
    assert xg_a < xg_b


def test_strong_attack_increases_xg():
    base_a, _ = calculate_xg(_ratings(attack=1.0), _ratings())
    high_a, _ = calculate_xg(_ratings(attack=1.3), _ratings())
    assert high_a > base_a


def test_strong_defense_decreases_opponent_xg():
    # Strong defense on team B means lower xg_a
    _, xg_b_strong_def = calculate_xg(_ratings(), _ratings(defense=0.7))
    _, xg_b_weak_def   = calculate_xg(_ratings(), _ratings(defense=1.3))
    # When team B has strong defense (0.7), xg_a (which uses defense_b) is lower
    xg_a_strong, _ = calculate_xg(_ratings(), _ratings(defense=0.7))
    xg_a_weak,   _ = calculate_xg(_ratings(), _ratings(defense=1.3))
    assert xg_a_strong < xg_a_weak


def test_extreme_elo_gap_clamps_xg():
    # ELO gap of 4000 → elo_factor = 2.0, could push xG above MAX
    xg_a, xg_b = calculate_xg(_ratings(elo=3000, attack=1.3, form=1.2, squad=1.2),
                               _ratings(elo=1000, attack=0.7, form=0.8, squad=0.8))
    assert xg_a <= XG_MAX
    assert xg_b >= XG_MIN


def test_xg_values_always_within_bounds():
    xg_a, xg_b = calculate_xg(_ratings(), _ratings())
    assert XG_MIN <= xg_a <= XG_MAX
    assert XG_MIN <= xg_b <= XG_MAX
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_xg_calculator.py -v
```
Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement `src/models/xg_calculator.py`**

```python
BASE_XG = 1.35   # average goals per team per match (historical baseline)
XG_MIN  = 0.2    # floor — prevents degenerate near-zero predictions
XG_MAX  = 4.5    # ceiling — prevents unrealistic blowout predictions


def calculate_xg(ratings_a: dict, ratings_b: dict) -> tuple[float, float]:
    """Calculate expected goals for both teams from their ratings.

    Uses a multiplicative formula where each factor adjusts the base xG:
      - attack_rating: how well the team creates chances
      - defense_rating (opponent's): how well the opponent suppresses chances
      - form_rating: recent match form
      - squad_rating: overall squad quality
      - elo_factor: relative strength adjustment from ELO difference

    The elo_factor for team A and B are mirrors: if A gets a boost, B gets
    an equal reduction, ensuring elo_factor + (2 - elo_factor) = 2 (constant sum).

    Args:
        ratings_a: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating
        ratings_b: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating

    Returns:
        (xg_a, xg_b) — both clamped to [XG_MIN, XG_MAX]
    """
    elo_factor = 1 + ((ratings_a["elo"] - ratings_b["elo"]) / 4000)

    xg_a = (
        BASE_XG
        * ratings_a["attack_rating"]
        * ratings_b["defense_rating"]   # opponent defense affects how much A can score
        * ratings_a["form_rating"]
        * ratings_a["squad_rating"]
        * elo_factor
    )

    xg_b = (
        BASE_XG
        * ratings_b["attack_rating"]
        * ratings_a["defense_rating"]   # opponent defense affects how much B can score
        * ratings_b["form_rating"]
        * ratings_b["squad_rating"]
        * (2 - elo_factor)              # mirror: ensures symmetry at equal ELO
    )

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/models/test_xg_calculator.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest -v
```
Expected: all tests pass (12 data + 11 poisson + 8 xg_calculator = 31 total).

- [ ] **Step 6: Commit**

```bash
git add src/models/xg_calculator.py tests/models/test_xg_calculator.py
git commit -m "feat: xG calculator — calculate_xg() from team ratings with ELO factor"
```

---

## Task 4: Update Streamlit App

**Files:**
- Modify: `src/app/app.py`

No unit tests — verify by running the app manually or via syntax/import checks.

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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")
st.markdown(
    "Select two teams to see auto-calculated expected goals and match outcome predictions."
)

# ── Load data ──────────────────────────────────────────────────────────────────
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

# ── Team selection ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", options=teams, index=0)

with col2:
    # Filter out Team A so the same team cannot be selected twice.
    teams_b = [t for t in teams if t != team_a]
    team_b = st.selectbox("Team B", options=teams_b, index=0)

# ── Auto-calculate xG from ratings ────────────────────────────────────────────
ratings_a = all_ratings.get(team_a)
ratings_b = all_ratings.get(team_b)

# Warn and fall back to BASE_XG for teams missing from the ratings file.
if ratings_a is None:
    st.warning(f"No ratings found for {team_a} — using baseline xG ({BASE_XG}).")
if ratings_b is None:
    st.warning(f"No ratings found for {team_b} — using baseline xG ({BASE_XG}).")

if ratings_a is not None and ratings_b is not None:
    auto_xg_a, auto_xg_b = calculate_xg(ratings_a, ratings_b)
else:
    auto_xg_a = BASE_XG
    auto_xg_b = BASE_XG

# ── xG section ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Expected Goals (xG)")

st.info(
    f"**Auto-calculated xG** — {team_a}: **{auto_xg_a:.2f}** | {team_b}: **{auto_xg_b:.2f}**"
)

override = st.checkbox("Override xG manually", value=False)

col3, col4 = st.columns(2)

with col3:
    xg_a = st.number_input(
        f"{team_a} xG",
        min_value=0.1,
        max_value=5.0,
        value=float(round(auto_xg_a, 1)),
        step=0.1,
        format="%.1f",
        disabled=not override,
        key="xg_a_input",
    )

with col4:
    xg_b = st.number_input(
        f"{team_b} xG",
        min_value=0.1,
        max_value=5.0,
        value=float(round(auto_xg_b, 1)),
        step=0.1,
        format="%.1f",
        disabled=not override,
        key="xg_b_input",
    )

# Use auto values unless override is active.
final_xg_a = xg_a if override else auto_xg_a
final_xg_b = xg_b if override else auto_xg_b

st.caption(
    f"**Final xG used** — {team_a}: {final_xg_a:.2f} | {team_b}: {final_xg_b:.2f}"
    + (" *(manual override)*" if override else " *(auto-calculated)*")
)

# ── Run prediction ─────────────────────────────────────────────────────────────
try:
    result = predict(team_a, team_b, final_xg_a, final_xg_b)
except ValueError as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

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

- [ ] **Step 2: Verify syntax and imports**

```bash
python -c "import ast; ast.parse(open('src/app/app.py').read()); print('syntax OK')"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict
print('imports OK')
"
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
python -m pytest -v
```
Expected: all 31 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/app/app.py
git commit -m "feat: update Streamlit app with auto xG, override checkbox, final xG display"
```

---

## Task 5: Validation Scenarios

Run the 3 required validation scenarios and confirm all checks pass.

- [ ] **Step 1: Run validation script**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
from src.data.loader import load_team_ratings
from src.models.xg_calculator import calculate_xg
from src.models.poisson import predict

scenarios = [
    ('Argentina', 'Brazil'),
    ('France', 'Qatar'),
    ('Denmark', 'Ukraine'),
]

ratings = load_team_ratings()

for team_a, team_b in scenarios:
    xg_a, xg_b = calculate_xg(ratings[team_a], ratings[team_b])
    r = predict(team_a, team_b, xg_a, xg_b)
    total = r.win_a + r.draw + r.win_b
    print(f'--- {team_a} vs {team_b} (xG {xg_a:.2f} vs {xg_b:.2f}) ---')
    print(f'  {team_a} Win : {r.win_a:.4%}')
    print(f'  Draw         : {r.draw:.4%}')
    print(f'  {team_b} Win : {r.win_b:.4%}')
    print(f'  Total        : {total:.6f}')
    print(f'  Top 5:')
    for g_a, g_b, p in r.top_scorelines:
        print(f'    {team_a} {g_a} - {g_b} {team_b}  ({p:.4%})')
    print()
"
```

- [ ] **Step 2: Verify all checks manually**
  - [ ] Probabilities sum to ~100% for each scenario
  - [ ] Argentina ≈ Brazil (similar ratings → similar win probabilities)
  - [ ] France win probability >> Qatar win probability
  - [ ] Denmark win probability > Ukraine win probability

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: v2 validation complete — all scenarios verified"
```
