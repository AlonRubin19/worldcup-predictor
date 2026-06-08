# World Cup Predictor v1 — Design Spec

**Date:** 2026-06-08  
**Status:** Approved  
**Scope:** Local Python MVP, no external APIs, Streamlit UI

---

## 1. Goal

Build a local World Cup match prediction tool that uses a Poisson distribution model to calculate win/draw/loss probabilities and top exact scorelines given two teams and their expected goals.

---

## 2. Project Structure

```
worldcup-predictor/
├── data/
│   └── teams.csv              # ~46 national teams, one name per row
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # load_teams() → list[str]
│   ├── models/
│   │   ├── __init__.py
│   │   └── poisson.py         # predict() → PredictionResult
│   ├── app/
│   │   ├── __init__.py
│   │   └── app.py             # Streamlit entry point
│   └── backtesting/
│       └── __init__.py        # reserved for v2
├── requirements.txt
└── README.md
```

`app.py` adds the project root to `sys.path` at startup so `src.*` imports resolve without installing the package.

---

## 3. Data Layer

**File:** `data/teams.csv`  
Single column with header `team`, one name per row, ~46 national teams.

**File:** `src/data/loader.py`  
Exports one function: `load_teams() -> list[str]`  
Reads `teams.csv` via pandas and returns a sorted list of team names.  
If the file is missing or empty, raises a `FileNotFoundError` or `ValueError` — Streamlit will surface this as a red error banner, which is acceptable for v1.  
This is the **only file** that knows where team data comes from — future versions replace this function to pull from a DB, API, or fixture feed without touching the app or model.

---

## 4. Poisson Model

**File:** `src/models/poisson.py`

### Input
```python
predict(team_a: str, team_b: str, xg_a: float, xg_b: float) -> PredictionResult
```

### Input validation
`xg_a` and `xg_b` must be in range [0.1, 5.0]. The Streamlit UI enforces this via `number_input` min/max. The model also asserts both values are > 0 and raises `ValueError` if not.

### Algorithm
1. Build a 11×11 score probability matrix (goals 0–10 for each team).
2. Each cell `[i][j]` = `poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b)`.
3. Sum cells above diagonal → P(Team A wins).
4. Sum cells on diagonal → P(Draw).
5. Sum cells below diagonal → P(Team B wins).
6. Flatten matrix, sort by probability descending; tiebreak by `(goals_a, goals_b)` ascending for determinism. Take top 5.

### Output
```python
@dataclass
class PredictionResult:
    team_a: str
    team_b: str
    win_a: float       # probability 0–1
    draw: float
    win_b: float
    top_scorelines: list[tuple[int, int, float]]  # (goals_a, goals_b, probability)
```

---

## 5. Streamlit App

**File:** `src/app/app.py`

### UI Components
- Page title and subtitle.
- Two `st.selectbox` controls: Team A and Team B.
  - Team B options filter out the currently selected Team A to prevent duplicate selection.
- Two `st.number_input` controls: xG for Team A and Team B (range 0.1–5.0, step 0.1, default 1.3).
- Predictions recalculate reactively on any input change (no submit button).

### Output Tables
1. **Match Outcome Probabilities** — three-row table: Team A Win / Draw / Team B Win with percentage values.
2. **Top 5 Most Likely Scorelines** — table of score (e.g. "2 - 1") and probability percentage.

---

## 6. Expandability Notes

- `loader.py` is the data seam. Future versions swap only this module to pull from football-data.org, a local SQLite DB, or a live fixture feed.
- `PredictionResult` dataclass can be extended with ELO adjustment, player ratings modifier, or form factor fields in later versions.
- `src/backtesting/` is scaffolded but empty — will hold historical match replay logic in v2.

---

## 7. Dependencies

```
streamlit
pandas
numpy
scipy
```

No version pins in v1 (install latest). No external football data APIs in v1.

---

## 8. Run Instructions

```bash
pip install -r requirements.txt
streamlit run src/app/app.py
```
