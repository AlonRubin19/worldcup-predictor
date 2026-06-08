# World Cup Predictor v2 — Design Spec

**Date:** 2026-06-08  
**Status:** Approved  
**Scope:** Automatic xG calculation from local team ratings dataset; no external APIs

---

## 1. Goal

Remove the need for manual xG input. Instead, auto-calculate expected goals from a local team ratings dataset using a multi-factor formula. Keep manual override available for power users. The Poisson model is unchanged.

---

## 2. What Changes vs v1

| Component | v1 | v2 |
|---|---|---|
| `data/teams.csv` | team names | unchanged |
| `data/team_ratings.csv` | does not exist | new — full ratings per team |
| `src/data/loader.py` | `load_teams()` | adds `load_team_ratings()` |
| `src/models/xg_calculator.py` | does not exist | new — formula + clamping |
| `src/models/poisson.py` | unchanged | unchanged |
| `src/app/app.py` | manual xG sliders | auto xG + optional override |
| `tests/models/test_xg_calculator.py` | does not exist | new — formula and edge case tests |

---

## 3. Data Layer

### `data/team_ratings.csv`

Header: `team,elo,attack_rating,defense_rating,form_rating,squad_rating`

- **elo**: Integer ELO rating (e.g. 2070 for Argentina). Typical range 1600–2100.
- **attack_rating**: Float multiplier, 1.0 = average. Typical range 0.7–1.3.
- **defense_rating**: Float multiplier, 1.0 = average attacker facing average defense. Lower = stronger defense (harder to score against). Typical range 0.7–1.3.
- **form_rating**: Float multiplier based on recent results. 1.0 = average. Range 0.8–1.2.
- **squad_rating**: Float multiplier for overall squad quality. 1.0 = average. Range 0.8–1.2.

45 teams — all teams from `data/teams.csv` must have a corresponding row.

### `src/data/loader.py`

Adds one function alongside the existing `load_teams()`:

```python
def load_team_ratings(ratings_path: Path | None = None) -> dict[str, dict]:
    """Load team ratings from CSV.

    Returns a dict keyed by team name:
        {
            "Argentina": {
                "elo": 2070,
                "attack_rating": 1.15,
                "defense_rating": 0.88,
                "form_rating": 1.05,
                "squad_rating": 1.10,
            },
            ...
        }

    Raises FileNotFoundError if the ratings CSV is missing.
    Raises ValueError if required columns are missing or any team has NaN values.
    """
```

The path defaults to `data/team_ratings.csv` relative to `__file__`, same pattern as `load_teams()`.

---

## 4. xG Calculator

### `src/models/xg_calculator.py`

```python
BASE_XG = 1.35
XG_MIN  = 0.2
XG_MAX  = 4.5

def calculate_xg(ratings_a: dict, ratings_b: dict) -> tuple[float, float]:
    """Calculate expected goals for both teams using team ratings.

    Formula:
        elo_factor = 1 + ((elo_a - elo_b) / 4000)

        xg_a = BASE_XG
               * attack_a
               * defense_b          # opponent defense weakens or strengthens xG
               * form_a
               * squad_a
               * elo_factor

        xg_b = BASE_XG
               * attack_b
               * defense_a
               * form_b
               * squad_b
               * (2 - elo_factor)   # mirror of elo_factor for team B

    Both values are clamped to [XG_MIN, XG_MAX].

    Args:
        ratings_a: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating
        ratings_b: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating

    Returns:
        (xg_a, xg_b) — clamped floats
    """
```

**Key invariant:** When both teams have identical ratings, `elo_factor = 1.0` and `2 - elo_factor = 1.0`, so `xg_a == xg_b`.

---

## 5. Updated Streamlit App

### New UI flow

1. User selects Team A and Team B (unchanged).
2. App loads ratings and auto-calculates xG for both teams.
3. App displays auto-calculated xG values.
4. **"Override xG manually"** checkbox (default: unchecked).
   - If unchecked: xG inputs are shown but disabled (display only).
   - If checked: xG inputs become editable (same range 0.1–5.0 as v1).
5. "Final xG used" line shows the effective values going into `predict()`.
6. Prediction table and top 5 scorelines (unchanged from v1).

### Error handling

If `load_team_ratings()` fails (missing file, missing column), show `st.error()` + `st.stop()`. The user must not see a Python traceback.

If a selected team has no ratings entry, show `st.warning()` with the team name and fall back to `xg = BASE_XG (1.35)` for that team.

---

## 6. Tests

### `tests/models/test_xg_calculator.py`

Required test cases:
- Equal ratings → equal xG for both teams
- Higher ELO team gets higher xG
- Lower ELO team gets lower xG
- Strong attack rating increases xG
- Strong defense rating (low multiplier) decreases opponent xG
- Extreme ELO gap → xG is clamped to [0.2, 4.5]
- Output is always a tuple of two floats

### `tests/data/test_loader.py` additions

- `load_team_ratings()` returns a dict
- Dict contains expected keys for a known team
- Missing file raises `FileNotFoundError`
- Missing column raises `ValueError`

---

## 7. File Structure (additions only)

```
worldcup-predictor/
├── data/
│   ├── teams.csv              # unchanged
│   └── team_ratings.csv       # new
├── src/
│   ├── data/
│   │   └── loader.py          # adds load_team_ratings()
│   ├── models/
│   │   ├── poisson.py         # unchanged
│   │   └── xg_calculator.py   # new
│   └── app/
│       └── app.py             # updated UI
└── tests/
    ├── data/
    │   └── test_loader.py     # adds rating loader tests
    └── models/
        └── test_xg_calculator.py  # new
```

---

## 8. Ratings Data — Representative Values

All attack/defense/form/squad ratings use 1.0 as the global average. ELO values are representative of real-world rankings (not sourced from a live API).

| Team | ELO | Attack | Defense | Form | Squad |
|---|---|---|---|---|---|
| Argentina | 2070 | 1.15 | 0.88 | 1.05 | 1.10 |
| Brazil | 2050 | 1.12 | 0.90 | 1.02 | 1.08 |
| France | 2020 | 1.18 | 0.85 | 1.08 | 1.12 |
| England | 1985 | 1.10 | 0.92 | 1.00 | 1.05 |
| Spain | 1975 | 1.08 | 0.90 | 0.98 | 1.04 |
| *(full table in team_ratings.csv)* | | | | | |

Average teams (e.g. Panama, Qatar) will have ratings close to 1.0 across the board with ELO ~1600–1750.
