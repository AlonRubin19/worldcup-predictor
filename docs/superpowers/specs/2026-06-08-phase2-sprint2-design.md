# Phase 2 Sprint 2 Design — Research-Valid Backtest

**Date:** 2026-06-08  
**Status:** Ready for review  
**Scope:** Three focused fixes that turn Sprint 1 infrastructure into a research-valid 40-match apples-to-apples backtest

---

## 1. Goal

Produce a valid, comparable 40-match WC 2022 backtest across four models, with the MLE parameter scale fixed and all match lookups working bidirectionally.

**Sprint 2 success criteria:**
- 40/40 WC 2022 matches matched (up from 15/40)
- MLE parameters normalized: no team at bounds, mean(log α) ≈ 0
- Full 40-match report with four-model comparison
- Clear recommendation: does MLE improve over rolling-stats baseline?

---

## 2. Three Priorities

### Priority 1 — Bidirectional Match Resolver

**Problem:** `historical_matches.csv` stores Qatar as `team_b` (Ecuador vs Qatar), while `martj42` records it as `home_team = Qatar`. When joining on `(date, team_a, team_b)`, 25 of 40 matches fail.

**Solution:** A resolver that tries both orderings and always aligns output to `historical_matches.csv` convention.

**Module:** `src/data/match_resolver.py`

```python
@dataclass
class ResolvedMatchStats:
    date: str
    team_a: str                          # Always from historical_matches convention
    team_b: str
    team_a_elo_pre: float
    team_b_elo_pre: float
    team_a_goals_for_last_10: float
    team_a_goals_against_last_10: float
    team_b_goals_for_last_10: float
    team_b_goals_against_last_10: float
    team_a_points_per_game_last_10: float
    team_b_points_per_game_last_10: float
    team_a_matches_available: int
    team_b_matches_available: int
    was_reversed: bool                   # True if the DB row had teams in opposite order


def resolve_match(
    date: str,
    team_a: str,
    team_b: str,
    match_results_df: pd.DataFrame,
) -> ResolvedMatchStats | None:
    """
    Look up pre-match stats for (team_a, team_b) on given date.
    Tries (team_a, team_b) first; if missing, tries (team_b, team_a) and swaps
    all columns back to team_a/team_b convention.
    Returns None if no match found in either order.
    """


def resolve_all_matches(
    historical_df: pd.DataFrame,
    match_results_df: pd.DataFrame,
) -> tuple[list[ResolvedMatchStats], list[dict]]:
    """
    Resolve every row in historical_df against match_results_df.
    Returns (resolved_list, unresolved_list).
    unresolved_list contains dicts with date/team_a/team_b for warning logging.
    """
```

**Column swapping logic when reversed:**
```
team_a_elo_pre        ← DB row's team_b_elo_pre
team_b_elo_pre        ← DB row's team_a_elo_pre
team_a_goals_for_last_10       ← DB row's team_b_goals_for_last_10
team_a_goals_against_last_10   ← DB row's team_b_goals_against_last_10
team_b_goals_for_last_10       ← DB row's team_a_goals_for_last_10
team_b_goals_against_last_10   ← DB row's team_a_goals_against_last_10
team_a_points_per_game_last_10 ← DB row's team_b_points_per_game_last_10
team_b_points_per_game_last_10 ← DB row's team_a_points_per_game_last_10
team_a_matches_available       ← DB row's team_b_matches_available
team_b_matches_available       ← DB row's team_a_matches_available
```

**Tests (in `tests/data/test_match_resolver.py`):**
- `test_exact_order_match` — finds row with correct team ordering, no swap
- `test_reversed_order_match` — finds row with reversed teams, swaps stats back
- `test_elo_correctly_swapped` — after reversal, team_a_elo_pre is the DB's team_b_elo_pre
- `test_ppg_correctly_swapped` — same for ppg
- `test_no_match_returns_none` — unrecognized date/team pair returns None
- `test_resolve_all_returns_unresolved_list` — returns correct unresolved count

---

### Priority 2 — Normalized MLE Parameters

**Problem:** The MLE has an identifiability issue: multiplying all α by constant C and dividing all β by C yields the same likelihood. Without a constraint, the optimizer produces arbitrarily-scaled parameters. Argentina hits `β = 0.10` (lower bound), indicating the true solution wants β < 0.10.

**Solution:** Post-optimization normalization using the geometric mean.

**Algorithm:**
```python
# After scipy.optimize.minimize converges:
log_alphas = [log(p.alpha_attack) for p in fitted.values()]
scale = exp(mean(log_alphas))           # geometric mean of alphas
for team in fitted:
    fitted[team].alpha_attack /= scale  # renormalize alpha
    fitted[team].beta_defense *= scale  # compensate so α*β unchanged
# Result: mean(log α) = 0, all products α_a * β_b unchanged
```

This is mathematically equivalent to the original fit — predicted goals `λ = α_a * β_b` are identical before and after normalization, because the scale cancels out.

**Why this fixes bounds:** Before normalization, Brazil's `α = 6.83` anchors the whole scale. Argentina's true `β` would be `0.03` without bounds. After normalizing to `mean(log α) = 0`, the scale shifts so all parameters are meaningful.

**After normalization expected behavior:**
- No WC 2022 team hits α or β = 0.10 lower bound
- `mean(log α)` across all fitted teams ≈ 0 (within ±0.01)
- Brazil α ≈ 1.5–2.5 (still highest, but not 6.83)
- Argentina β > 0.15 (no longer at bound)

**Tests (in `tests/models/test_mle_fitter.py` — additional tests):**
- `test_mean_log_alpha_near_zero` — after fitting and normalization, mean(log α) ≈ 0
- `test_no_team_at_lower_bound` — no team's α or β == 0.10 in a well-specified dataset
- `test_normalization_preserves_expected_goals` — α_a * β_b unchanged before and after normalization
- `test_normalized_params_unchanged_prediction` — xG predictions identical before/after

---

### Priority 3 — Apples-to-Apples 40-Match Report

**Four models on the same 40 WC 2022 matches:**

| # | Model | xG Source | Runner |
|---|---|---|---|
| 1 | Illustrative (existing) | `team_ratings.csv` (AI-estimated) | `run_backtest()` |
| 2 | Real rolling stats | `match_results.csv` (Sprint 1, real history) | new `run_rolling_stats_backtest()` wrapper |
| 3 | MLE opponent strength | `team_strength_params.csv` (normalized) | `run_strength_backtest()` |
| 4 | MLE + Dixon-Coles (best ρ) | Same as Model 3 with tuned ρ | `run_strength_backtest(model_type='dixon_coles', rho=best)` |

**Model 2 detail:** Uses the bidirectional resolver to extract real pre-match rolling stats for all 40 WC 2022 matches, then applies `calculate_pre_match_xg()` (the existing raw-stats formula, not MLE).

**Comparison table columns:** model, matches, 1X2 accuracy, Brier score, exact score acc, top 3 hit rate, top 5 hit rate, avg P(actual result).

**Script:** `scripts/run_sprint2_report.py` — standalone script that produces a printable comparison table.

---

## 3. File Map

| File | Action |
|---|---|
| `src/data/match_resolver.py` | Create — bidirectional resolver |
| `tests/data/test_match_resolver.py` | Create — 6 tests |
| `src/models/mle_fitter.py` | Modify — add normalization step, 4 new tests |
| `scripts/fit_strength_params.py` | Modify — re-run with normalization (regenerates CSV) |
| `data/team_strength_params.csv` | Regenerate — normalized parameters |
| `scripts/run_sprint2_report.py` | Create — 40-match comparison report |
| All existing files | Unchanged |

---

## 4. Data Flow

```
data/historical_matches.csv (40 rows, team order = truth)
    +
data/match_results.csv (49,373 rows, martj42 team order)
    ↓
src/data/match_resolver.py
    → 40 ResolvedMatchStats (ELO + rolling stats in correct order)
    ↓
    ├── calculate_pre_match_xg() → Model 2 xG
    └── calculate_strength_adjusted_xg() → Model 3/4 xG
    ↓
predict() / predict_dixon_coles()
    ↓
MatchResult × 40
    ↓
compute_metrics()
    ↓
Comparison table
```

---

## 5. Tests

### New tests in `tests/data/test_match_resolver.py` (6 tests)
- exact order match
- reversed order match with correct stat swapping
- ELO correctly swapped on reversal
- PPG correctly swapped on reversal
- returns None for unknown match
- resolve_all returns correct unresolved count

### New tests in `tests/models/test_mle_fitter.py` (4 additional)
- mean(log α) ≈ 0 after fitting
- no parameter hits the 0.10 lower bound in a well-specified dataset
- normalization preserves α_a * β_b products
- predictions unchanged by normalization

---

## 6. No New Dependencies

All three priorities use only: pandas, numpy, scipy, pytest — all already installed.
