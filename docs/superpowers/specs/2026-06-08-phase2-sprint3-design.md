# Phase 2 Sprint 3 Design — Player Impact Engine

**Date:** 2026-06-08  
**Status:** Approved  
**Scope:** Add a player/squad availability xG modifier to the real-data MLE pipeline. All data is placeholder until real sources are wired in Sprint 4+.

---

## 1. Goal

Build a complete, testable player impact pipeline that modifies xG based on squad availability and starting XI strength. When key players are unavailable, the team's expected goals are reduced (or increased for opponents). The pipeline is structurally valid but labelled "Engineering validation only" since player data is not yet research-grade.

**Sprint 3 success criteria:**
- Player impact pipeline works end-to-end
- xG changes measurably when key players are out
- xG unchanged when all expected starters are fit (squad_factor = 1.0)
- Existing models unaffected (no regressions)
- 40-match backtest compares MLE+DC vs MLE+DC+Player Impact
- Report shows specific matches where player impact changed probabilities

---

## 2. Data Files

### `data/player_profiles.csv`

One row per player. Covers all 32 WC 2022 teams with ~11–15 players each (~400 rows total).

| Column | Type | Description |
|---|---|---|
| `player_id` | str | Unique ID (e.g. `ARG_001`) |
| `player_name` | str | Full name |
| `team` | str | National team name (matches our convention) |
| `position` | str | GK / DEF / MID / FWD |
| `club` | str | Club team name |
| `minutes_last_90_days` | float | Club + international minutes, rolling 90 days |
| `national_team_minutes_last_12_months` | float | National team minutes, rolling 12 months |
| `goals_per_90` | float | Goals per 90 minutes (national + club) |
| `assists_per_90` | float | Assists per 90 |
| `xg_per_90` | float | Expected goals per 90 |
| `xa_per_90` | float | Expected assists per 90 |
| `defensive_actions_per_90` | float | Pressures + tackles + interceptions per 90 |
| `international_caps` | int | Total senior international appearances |
| `base_impact_score` | float | Composite score 0.0–1.0 (1.0 = world-class) |

**`base_impact_score` formula** (for generating placeholder data):
```
base = 0.3 * (goals_per_90 / 1.0) + 0.2 * (assists_per_90 / 0.5)
      + 0.2 * (xg_per_90 / 1.0) + 0.1 * (xa_per_90 / 0.5)
      + 0.1 * (defensive_actions_per_90 / 10.0)
      + 0.1 * min(international_caps / 100, 1.0)
Clamped to [0.1, 1.0].
```

**Placeholder data guidelines:**
- Top players (Messi, Mbappé, Ronaldo, Neymar): base_impact_score ~0.85–0.95
- Regular starters: 0.50–0.75
- Fringe players: 0.30–0.50
- Include Benzema for France with `base_impact_score = 0.85` (he withdrew injured — for demonstration)
- Include 3–5 notable injury/absence cases from WC 2022

### `data/match_player_availability.csv`

One row per (match, team, player). Covers all 40 historical_matches.csv matches for all players.

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Matches historical_matches.csv row order (1-indexed) |
| `date` | str | ISO date |
| `team` | str | Team name |
| `player_id` | str | Foreign key to player_profiles |
| `expected_starter` | bool | True = expected in starting XI |
| `availability_status` | str | `fit` / `doubtful` / `out` / `suspended` |
| `availability_factor` | float | 1.0 (fit), 0.6 (doubtful), 0.0 (out/suspended) |
| `form_factor` | float | 0.7–1.3, represents recent form relative to baseline |

**Most rows:** all expected starters fit, form_factor = 1.0, availability_factor = 1.0.

**Interesting rows (for demonstration):**
- France, match 1 (vs Australia): Benzema `out`, availability_factor = 0.0
- France, all matches: Benzema excluded (he pulled out before the tournament)
- Germany, match 3: Neuer `doubtful`, availability_factor = 0.6
- Portugal, match 1: One key midfielder `doubtful`
- Brazil, QF: Neymar `doubtful` (was actually injured in group stage)

---

## 3. Player Loader (`src/data/player_loader.py`)

```python
@dataclass
class PlayerProfile:
    player_id: str
    player_name: str
    team: str
    position: str   # GK / DEF / MID / FWD
    club: str
    minutes_last_90_days: float
    national_team_minutes_last_12_months: float
    goals_per_90: float
    assists_per_90: float
    xg_per_90: float
    xa_per_90: float
    defensive_actions_per_90: float
    international_caps: int
    base_impact_score: float  # composite 0.0–1.0


@dataclass
class PlayerAvailability:
    match_id: int
    date: str
    team: str
    player_id: str
    expected_starter: bool
    availability_status: str   # fit / doubtful / out / suspended
    availability_factor: float  # 1.0, 0.6, 0.0
    form_factor: float           # 0.7–1.3


def load_player_profiles(path: Path | None = None) -> dict[str, PlayerProfile]:
    """Returns {player_id: PlayerProfile}."""


def load_match_availability(path: Path | None = None) -> list[PlayerAvailability]:
    """Returns list of all availability rows."""


def get_team_profiles(profiles: dict[str, PlayerProfile], team: str) -> list[PlayerProfile]:
    """Returns all PlayerProfiles for a team, sorted by base_impact_score desc."""


def get_match_availability(
    availability: list[PlayerAvailability],
    match_id: int,
    team: str,
) -> list[PlayerAvailability]:
    """Returns availability rows for one team in one match."""
```

---

## 4. Player Impact Model (`src/models/player_impact.py`)

### Formula (exact)

```python
# Per-player match impact
player_match_impact = base_impact_score * availability_factor * form_factor

# Starting XI strength: average impact of expected starters
starting_xi_strength = mean(player_match_impact for expected_starter=True)

# Baseline: average of top 11 players by base_impact_score (full squad fit)
baseline_xi_strength = mean(sorted(base_impact_scores, desc)[:11])

# Squad factor
squad_factor = starting_xi_strength / baseline_xi_strength

# Clamp
squad_factor = max(0.85, min(1.15, squad_factor))

# Apply to xG
team_xg_adjusted = team_xg * squad_factor
```

### Edge cases

- If team has fewer than 11 players in profiles: use all available.
- If no expected_starter rows in availability: squad_factor = 1.0 (no adjustment).
- If baseline_xi_strength == 0: squad_factor = 1.0 (guard against division by zero).

### API

```python
def compute_squad_factor(
    profiles: list[PlayerProfile],         # all profiles for this team
    availability: list[PlayerAvailability], # availability for this match/team
) -> float:
    """Return squad_factor in [0.85, 1.15]. Returns 1.0 if insufficient data."""


def apply_player_impact(
    xg_a: float,
    xg_b: float,
    profiles_a: list[PlayerProfile],
    profiles_b: list[PlayerProfile],
    availability_a: list[PlayerAvailability],
    availability_b: list[PlayerAvailability],
) -> tuple[float, float]:
    """Apply squad_factor to both teams' xG. Clamps to [0.2, 4.5]."""
```

---

## 5. Player Impact Runner (`src/backtesting/player_impact_runner.py`)

Extends the MLE strength pipeline with the player impact adjustment.

```python
def run_player_impact_backtest(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    player_profiles_path: Path | None = None,
    availability_path: Path | None = None,
    model_type: str = "dixon_coles",
    rho: float = -0.30,
) -> list[MatchResult]:
    """MLE strength + player impact adjustment → Poisson/DC prediction.

    Falls back to squad_factor = 1.0 for any match without player data.
    """
```

Pipeline per match:
1. Look up pre-match stats via match_resolver (ELO + rolling stats)
2. Call `calculate_strength_adjusted_xg()` → base xG_a, xG_b
3. Call `apply_player_impact()` → adjusted xG_a, xG_b  
4. Call `predict()` or `predict_dixon_coles()`
5. Return `MatchResult`

---

## 6. Tests

### `tests/data/test_player_loader.py` (6 tests)
- `load_player_profiles` returns `dict[str, PlayerProfile]`
- `load_match_availability` returns `list[PlayerAvailability]`
- `get_team_profiles` returns only that team, sorted desc by base_impact_score
- `get_match_availability` filters by match_id and team
- Missing file raises `FileNotFoundError`
- Missing columns raise `ValueError`

### `tests/models/test_player_impact.py` (8 tests)
- `compute_squad_factor` returns 1.0 when all fit and form=1.0
- Returns < 1.0 when key player (high base_impact_score) is out
- Returns > 1.0 when key players have form_factor > 1.0
- Clamped to [0.85, 1.15]
- Returns 1.0 with no availability data (safe default)
- Returns 1.0 with no expected starters flagged
- `apply_player_impact` applies factor to both teams independently
- Output clamped to [0.2, 4.5]

### `tests/backtesting/test_player_impact_runner.py` (5 tests)
- Returns `list[MatchResult]`
- Match count equals historical_matches.csv row count
- Works with `model_type="poisson"` and `model_type="dixon_coles"`
- When all players are fit with form_factor=1.0, output equals MLE run without player impact
- When Benzema is out for France, France xG is reduced

---

## 7. Placeholder Data Labelling

Both CSVs must include a comment header:

```
# WARNING: PLACEHOLDER DATA — not sourced from real player records.
# Engineering validation only. See docs/player_data_status.md for sourcing requirements.
```

The player_impact_runner output is labelled in the report as:
```
"⚠️ Player Impact (PLACEHOLDER DATA — engineering validation only)"
```

---

## 8. Files Not Changed

`run_backtest()`, `run_valid_backtest()`, `run_strength_backtest()`, `compute_metrics()`, `poisson.py`, `dixon_coles.py`, `mle_fitter.py`, `strength_adjusted_xg.py` — all unchanged.
