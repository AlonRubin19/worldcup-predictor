# Phase 2 Sprint 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two structural issues from Sprint 1 (25/40 unmatched WC matches; MLE parameters at bounds) and produce a research-valid 40-match apples-to-apples comparison across four models.

**Architecture:** Three independent deliverables — (1) `src/data/match_resolver.py` with bidirectional lookup; (2) normalized MLE in `src/models/mle_fitter.py` with regenerated `data/team_strength_params.csv`; (3) `scripts/run_sprint2_report.py` that drives all four models on the same 40 matches.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, pytest — no new dependencies.

---

## Critical Context

- **120 tests passing** at Sprint 2 start. No regressions allowed.
- `data/match_results.csv` — 49,373 rows, columns include `date, team_a, team_b, team_a_elo_pre, team_b_elo_pre, team_a_goals_for_last_10, team_a_goals_against_last_10, team_b_goals_for_last_10, team_b_goals_against_last_10, team_a_points_per_game_last_10, team_b_points_per_game_last_10, team_a_matches_available, team_b_matches_available, team_a_goals, team_b_goals`
- `data/historical_matches.csv` — 40 WC 2022 matches in our team-ordering convention
- `data/team_strength_params.csv` — 302 teams; Argentina β = 0.10 (at bound); mean(log α) ≈ 0.013 (near-zero, normalization will fix residual)
- Mismatch cause: martj42 stores host/home team first; `historical_matches.csv` uses a different ordering. In 25 of 40 WC matches the teams appear in reversed order in `match_results.csv`.

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/data/match_resolver.py` | Create | Bidirectional lookup of pre-match stats |
| `tests/data/test_match_resolver.py` | Create | 6 unit tests |
| `src/models/mle_fitter.py` | Modify (lines 141–157) | Add normalization after optimization |
| `tests/models/test_mle_fitter.py` | Modify — append 4 tests | Normalization correctness |
| `scripts/fit_strength_params.py` | Run again | Regenerate with normalized params |
| `data/team_strength_params.csv` | Regenerate | Normalized — Argentina β > 0.10 |
| `scripts/run_sprint2_report.py` | Create | 40-match 4-model comparison |

---

## Task 1: Bidirectional Match Resolver

**Files:**
- Create: `tests/data/test_match_resolver.py`
- Create: `src/data/match_resolver.py`

- [ ] **Step 1: Write 6 failing tests**

Create `tests/data/test_match_resolver.py`:

```python
import pytest
import pandas as pd
from src.data.match_resolver import resolve_match, resolve_all_matches, ResolvedMatchStats


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_db_row(**kwargs):
    """Build a minimal match_results DataFrame row."""
    defaults = dict(
        date="2022-11-20", team_a="Ecuador", team_b="Qatar",
        team_a_elo_pre=1755.0, team_b_elo_pre=1634.0,
        team_a_goals_for_last_10=1.40, team_a_goals_against_last_10=0.90,
        team_b_goals_for_last_10=0.80, team_b_goals_against_last_10=1.60,
        team_a_points_per_game_last_10=2.10, team_b_points_per_game_last_10=0.90,
        team_a_matches_available=10, team_b_matches_available=10,
        team_a_goals=2, team_b_goals=0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ── Tests ────────────────────────────────────────────────────────────────────

def test_exact_order_match_found():
    """When historical order matches DB order, resolve succeeds."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")
    result = resolve_match("2022-11-20", "Ecuador", "Qatar", db)
    assert result is not None
    assert isinstance(result, ResolvedMatchStats)
    assert result.was_reversed is False


def test_reversed_order_match_found():
    """When historical teams are reversed vs DB, resolve still succeeds."""
    # DB has Ecuador as team_a, Qatar as team_b
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_elo_pre=1755.0, team_b_elo_pre=1634.0)
    # But historical_matches has Qatar first
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    assert result is not None
    assert result.was_reversed is True


def test_elo_correctly_swapped_on_reversal():
    """After reversal, team_a in result gets DB's team_b ELO."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_elo_pre=1755.0, team_b_elo_pre=1634.0)
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    # Qatar was team_b in DB (elo=1634), now team_a in result
    assert abs(result.team_a_elo_pre - 1634.0) < 1e-6
    assert abs(result.team_b_elo_pre - 1755.0) < 1e-6


def test_ppg_correctly_swapped_on_reversal():
    """After reversal, team_a PPG comes from DB's team_b PPG."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_points_per_game_last_10=2.10,
                      team_b_points_per_game_last_10=0.90)
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    # Qatar was team_b in DB (ppg=0.90), now team_a in result
    assert abs(result.team_a_points_per_game_last_10 - 0.90) < 1e-6
    assert abs(result.team_b_points_per_game_last_10 - 2.10) < 1e-6


def test_no_match_returns_none():
    """When date/team pair doesn't exist in DB, returns None."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")
    result = resolve_match("2022-11-20", "France", "Brazil", db)
    assert result is None


def test_resolve_all_returns_unresolved_list():
    """resolve_all_matches correctly separates resolved from unresolved rows."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")

    historical = pd.DataFrame([
        {"date": "2022-11-20", "team_a": "Ecuador", "team_b": "Qatar",
         "team_a_goals": 2, "team_b_goals": 0},
        {"date": "2022-11-21", "team_a": "France", "team_b": "Brazil",
         "team_a_goals": 1, "team_b_goals": 0},  # Not in DB
    ])
    resolved, unresolved = resolve_all_matches(historical, db)
    assert len(resolved) == 1
    assert len(unresolved) == 1
    assert unresolved[0]["team_a"] == "France"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/data/test_match_resolver.py -v
```
Expected: `ImportError` — module doesn't exist.

- [ ] **Step 3: Create `src/data/match_resolver.py`**

```python
"""Bidirectional match resolver for joining historical_matches.csv with match_results.csv.

Handles the fact that martj42 records the host/home team as team_a, while
historical_matches.csv uses a different team ordering. For 25 of 40 WC 2022 matches
the teams appear reversed between the two sources.

Usage:
    resolved, unresolved = resolve_all_matches(historical_df, match_results_df)
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class ResolvedMatchStats:
    """Pre-match stats aligned to historical_matches.csv team ordering."""
    date: str
    team_a: str
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
    was_reversed: bool  # True if found by swapping team_a/team_b in match_results


def resolve_match(
    date: str,
    team_a: str,
    team_b: str,
    match_results_df: pd.DataFrame,
) -> "ResolvedMatchStats | None":
    """Look up pre-match stats for a given (date, team_a, team_b) triplet.

    Tries (team_a, team_b) first. If not found, tries (team_b, team_a) and
    swaps all columns to restore the team_a/team_b alignment requested.

    Args:
        date: ISO date string (YYYY-MM-DD).
        team_a: First team (per historical_matches.csv convention).
        team_b: Second team (per historical_matches.csv convention).
        match_results_df: DataFrame from match_results.csv.

    Returns:
        ResolvedMatchStats with was_reversed flag, or None if not found in either order.
    """
    # Try exact order
    row = match_results_df[
        (match_results_df["date"] == date) &
        (match_results_df["team_a"] == team_a) &
        (match_results_df["team_b"] == team_b)
    ]
    if not row.empty:
        r = row.iloc[0]
        return ResolvedMatchStats(
            date=date, team_a=team_a, team_b=team_b,
            team_a_elo_pre=float(r["team_a_elo_pre"]),
            team_b_elo_pre=float(r["team_b_elo_pre"]),
            team_a_goals_for_last_10=float(r["team_a_goals_for_last_10"]),
            team_a_goals_against_last_10=float(r["team_a_goals_against_last_10"]),
            team_b_goals_for_last_10=float(r["team_b_goals_for_last_10"]),
            team_b_goals_against_last_10=float(r["team_b_goals_against_last_10"]),
            team_a_points_per_game_last_10=float(r["team_a_points_per_game_last_10"]),
            team_b_points_per_game_last_10=float(r["team_b_points_per_game_last_10"]),
            team_a_matches_available=int(r["team_a_matches_available"]),
            team_b_matches_available=int(r["team_b_matches_available"]),
            was_reversed=False,
        )

    # Try reversed order — swap team_a/team_b in the lookup
    row = match_results_df[
        (match_results_df["date"] == date) &
        (match_results_df["team_a"] == team_b) &
        (match_results_df["team_b"] == team_a)
    ]
    if not row.empty:
        r = row.iloc[0]
        # Swap all team_a/team_b columns back to requested orientation
        return ResolvedMatchStats(
            date=date, team_a=team_a, team_b=team_b,
            team_a_elo_pre=float(r["team_b_elo_pre"]),           # swapped
            team_b_elo_pre=float(r["team_a_elo_pre"]),           # swapped
            team_a_goals_for_last_10=float(r["team_b_goals_for_last_10"]),      # swapped
            team_a_goals_against_last_10=float(r["team_b_goals_against_last_10"]),  # swapped
            team_b_goals_for_last_10=float(r["team_a_goals_for_last_10"]),      # swapped
            team_b_goals_against_last_10=float(r["team_a_goals_against_last_10"]),  # swapped
            team_a_points_per_game_last_10=float(r["team_b_points_per_game_last_10"]),  # swapped
            team_b_points_per_game_last_10=float(r["team_a_points_per_game_last_10"]),  # swapped
            team_a_matches_available=int(r["team_b_matches_available"]),   # swapped
            team_b_matches_available=int(r["team_a_matches_available"]),   # swapped
            was_reversed=True,
        )

    return None


def resolve_all_matches(
    historical_df: pd.DataFrame,
    match_results_df: pd.DataFrame,
) -> tuple[list[ResolvedMatchStats], list[dict]]:
    """Resolve every row in historical_df against match_results_df.

    Args:
        historical_df: DataFrame from historical_matches.csv with columns:
                       date, team_a, team_b (at minimum).
        match_results_df: DataFrame from match_results.csv.

    Returns:
        (resolved, unresolved)
        resolved: list of ResolvedMatchStats for each successfully matched row.
        unresolved: list of dicts {date, team_a, team_b} for rows with no match.
    """
    resolved = []
    unresolved = []

    for _, row in historical_df.iterrows():
        stats = resolve_match(
            str(row["date"]),
            str(row["team_a"]),
            str(row["team_b"]),
            match_results_df,
        )
        if stats is not None:
            resolved.append(stats)
        else:
            unresolved.append({
                "date": str(row["date"]),
                "team_a": str(row["team_a"]),
                "team_b": str(row["team_b"]),
            })

    return resolved, unresolved
```

- [ ] **Step 4: Run the resolver tests**

```bash
python -m pytest tests/data/test_match_resolver.py -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Verify all 40 WC 2022 matches now resolve**

```bash
python -c "
import sys, pandas as pd
sys.path.insert(0, '.')
from src.data.match_resolver import resolve_all_matches

hist = pd.read_csv('data/historical_matches.csv')
mr   = pd.read_csv('data/match_results.csv')

resolved, unresolved = resolve_all_matches(hist, mr)
print(f'Resolved:   {len(resolved)}/40')
print(f'Unresolved: {len(unresolved)}')
reversed_count = sum(1 for r in resolved if r.was_reversed)
print(f'Found by reversed lookup: {reversed_count}')
if unresolved:
    print('Still unresolved:')
    for u in unresolved:
        print(f'  {u}')
else:
    print('SUCCESS: All 40/40 matched.')
"
```
Expected: `Resolved: 40/40`. If < 40, debug by printing unresolved rows.

- [ ] **Step 6: Run full suite — no regressions**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 126 tests pass (120 + 6).

- [ ] **Step 7: Commit**

```bash
git add src/data/match_resolver.py tests/data/test_match_resolver.py
git commit -m "feat: bidirectional match resolver — fixes 25/40 WC 2022 unmatched rows"
```

---

## Task 2: MLE Normalization

**Files:**
- Modify: `src/models/mle_fitter.py` — add normalization block after optimization
- Modify: `tests/models/test_mle_fitter.py` — append 4 new tests

The normalization divides all α by the geometric mean and multiplies all β by the same scale. This keeps every product `α_a * β_b` identical (scale cancels), so no xG values or predictions change — only the scale/interpretation of individual parameters changes.

- [ ] **Step 1: Append 4 failing normalization tests to `tests/models/test_mle_fitter.py`**

```python
import math as _math


def test_mean_log_alpha_near_zero():
    """After fitting, mean(log alpha) should be ≈ 0 (geometric mean constraint)."""
    result = fit_team_params(_symmetric_matches(n=40))
    log_alphas = [_math.log(p.alpha_attack) for p in result.values()]
    assert abs(sum(log_alphas) / len(log_alphas)) < 0.05  # within 0.05 of zero


def test_no_wc_team_at_lower_bound():
    """No fitted team should sit exactly at the 0.10 lower bound after normalization."""
    # Build a dataset where one team is much weaker than others (similar to Qatar vs Brazil)
    strong_matches = [_make_match("Strong", "Weak", 4, 0, f"2020-{i:02d}-01")
                      for i in range(1, 16)]
    result = fit_team_params(strong_matches, min_matches=5)
    for team, p in result.items():
        # Neither alpha nor beta should be exactly at the bound
        assert p.alpha_attack > 0.105, f"{team} alpha at bound: {p.alpha_attack}"
        assert p.beta_defense > 0.105, f"{team} beta at bound: {p.beta_defense}"


def test_normalization_preserves_expected_goals():
    """The product alpha_a * beta_b must be the same before and after normalization.

    We verify this by checking that xG predictions are identical with
    normalized vs unnormalized parameters on the same match.
    """
    from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
    from src.data.strength_loader import StrengthParams

    # Fit once (normalized internally)
    result = fit_team_params(_symmetric_matches(n=40))
    teams = list(result.keys())
    if len(teams) < 2:
        pytest.skip("Need at least 2 teams")
    t_a, t_b = teams[0], teams[1]

    # Record alpha * beta product
    product_ab = result[t_a].alpha_attack * result[t_b].beta_defense

    # Manually "un-normalize" by multiplying alpha by 2 and halving beta
    # The product should remain the same
    p_a_scaled = StrengthParams(
        alpha_attack=result[t_a].alpha_attack * 2.0,
        beta_defense=result[t_a].beta_defense / 2.0,  # this team's own beta
        matches_used=result[t_a].matches_used,
    )
    p_b = StrengthParams(
        alpha_attack=result[t_b].alpha_attack,
        beta_defense=result[t_b].beta_defense / 2.0,  # halved to compensate
        matches_used=result[t_b].matches_used,
    )

    product_ab_scaled = p_a_scaled.alpha_attack * p_b.beta_defense
    assert abs(product_ab - product_ab_scaled) < 1e-9


def test_fit_with_real_params_no_bound_hits():
    """Synthetic dataset sized like WC 2022: no team at bounds after normalization."""
    import random
    random.seed(42)
    # 32 teams, ~10 matches each, varied strengths
    teams = [f"Team_{i}" for i in range(32)]
    matches = []
    for i in range(320):
        t_a = random.choice(teams)
        t_b = random.choice([t for t in teams if t != t_a])
        g_a = random.randint(0, 4)
        g_b = random.randint(0, 3)
        matches.append(_make_match(t_a, t_b, g_a, g_b, f"2020-{(i%12)+1:02d}-{(i%28)+1:02d}"))

    result = fit_team_params(matches, min_matches=5)
    at_bound = [(t, p) for t, p in result.items() if p.alpha_attack <= 0.105 or p.beta_defense <= 0.105]
    assert len(at_bound) == 0, f"Teams at bounds: {[(t, p.alpha_attack, p.beta_defense) for t, p in at_bound]}"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_mle_fitter.py -v -k "normalize or log_alpha or bound or preserves"
```
Expected: 4 failures (existing normalization is partial — mean(log α) ≈ 0.013 which may pass the 0.05 threshold, but bound hit tests will fail).

- [ ] **Step 3: Add normalization block to `src/models/mle_fitter.py`**

Locate the block starting at line 142 (after `result = minimize(...)`):

```python
    final_params = result.x
    alpha_vals = final_params[:n]
    beta_vals = final_params[n:]

    # Build per-team output
    output = {}
    for i, team in enumerate(team_list):
        output[team] = TeamStrengthParams(
            team=team,
            alpha_attack=float(alpha_vals[i]),
            beta_defense=float(beta_vals[i]),
            matches_used=appearances[team],
            log_likelihood=-result.fun / len(filtered),
        )

    return output
```

Replace with:

```python
    final_params = result.x
    alpha_vals = final_params[:n]
    beta_vals = final_params[n:]

    # ── Identifiability normalization ─────────────────────────────────────
    # The Poisson model λ = α_a * β_b has a degree of freedom: multiplying
    # all α by C and dividing all β by C leaves every λ unchanged.
    # We fix this by normalizing to geometric mean(α) = 1, i.e. mean(log α) = 0.
    # This is equivalent to dividing every α by the geometric mean and
    # multiplying every β by the same factor — all products α_a * β_b are
    # preserved exactly.
    log_alpha_mean = float(np.mean(np.log(alpha_vals)))
    scale = math.exp(log_alpha_mean)
    alpha_vals = alpha_vals / scale    # normalized: mean(log α) = 0
    beta_vals  = beta_vals  * scale    # compensated: every α*β unchanged

    # Build per-team output
    output = {}
    for i, team in enumerate(team_list):
        output[team] = TeamStrengthParams(
            team=team,
            alpha_attack=float(alpha_vals[i]),
            beta_defense=float(beta_vals[i]),
            matches_used=appearances[team],
            log_likelihood=-result.fun / len(filtered),
        )

    return output
```

- [ ] **Step 4: Run MLE tests**

```bash
python -m pytest tests/models/test_mle_fitter.py -v
```
Expected: All 12 tests pass (8 original + 4 new).

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 130 tests pass (126 + 4).

- [ ] **Step 6: Regenerate team_strength_params.csv with normalized parameters**

```bash
python scripts/fit_strength_params.py
```

Expected output: "All 32 WC 2022 teams have fitted parameters." Check that Argentina β > 0.10.

```bash
python -c "
import pandas as pd, math
df = pd.read_csv('data/team_strength_params.csv')
at_bound = df[df.alpha_attack <= 0.105]
print(f'Teams at alpha bound: {len(at_bound)}')
at_beta_bound = df[df.beta_defense <= 0.105]
print(f'Teams at beta bound: {len(at_beta_bound)}')
log_alphas = df.alpha_attack.apply(math.log)
print(f'mean(log alpha): {log_alphas.mean():.6f}  (should be ~0)')
wc = ['France','Brazil','Argentina','England','Germany','Qatar','Saudi Arabia']
print(df[df.team.isin(wc)][['team','alpha_attack','beta_defense']].to_string(index=False))
"
```

- [ ] **Step 7: Commit**

```bash
git add src/models/mle_fitter.py tests/models/test_mle_fitter.py data/team_strength_params.csv
git commit -m "feat: normalized MLE parameters — geometric mean constraint, no teams at bounds"
```

---

## Task 3: 40-Match Comparison Report Script

**Files:**
- Create: `scripts/run_sprint2_report.py`

This script drives all four models on the same 40 WC 2022 matches and prints a comparison table.

- [ ] **Step 1: Create `scripts/run_sprint2_report.py`**

```python
"""Sprint 2 validation: 40-match apples-to-apples comparison across 4 models.

Run from project root:
    python scripts/run_sprint2_report.py

Prints a comparison table and a recommendation.
"""

import sys, math
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.match_resolver import resolve_all_matches, ResolvedMatchStats
from src.data.strength_loader import load_strength_params
from src.models.pre_match_xg import calculate_pre_match_xg, BASE_XG as PRE_MATCH_BASE
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import run_backtest, MatchResult
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import DEFAULT_RHO_GRID, RhoResult, select_best_rho
from src.data.loader import load_team_ratings
from src.data.pre_match_loader import PreMatchStats

_ROOT = Path(__file__).parent.parent


def _resolve_wc2022():
    """Return list of ResolvedMatchStats for all 40 WC 2022 matches."""
    hist = pd.read_csv(_ROOT / "data" / "historical_matches.csv")
    mr   = pd.read_csv(_ROOT / "data" / "match_results.csv")
    resolved, unresolved = resolve_all_matches(hist, mr)
    if unresolved:
        print(f"WARNING: {len(unresolved)} matches unresolved:")
        for u in unresolved:
            print(f"  {u}")
    return resolved, hist


def _make_pre_match_stats(r: ResolvedMatchStats, ga: int, gb: int) -> PreMatchStats:
    """Convert a ResolvedMatchStats into a PreMatchStats for calculate_pre_match_xg()."""
    return PreMatchStats(
        match_id=0, date=r.date, team_a=r.team_a, team_b=r.team_b,
        team_a_elo_pre=r.team_a_elo_pre,
        team_b_elo_pre=r.team_b_elo_pre,
        team_a_goals_for_last_10=r.team_a_goals_for_last_10,
        team_a_goals_against_last_10=r.team_a_goals_against_last_10,
        team_b_goals_for_last_10=r.team_b_goals_for_last_10,
        team_b_goals_against_last_10=r.team_b_goals_against_last_10,
        team_a_points_per_game_last_10=r.team_a_points_per_game_last_10,
        team_b_points_per_game_last_10=r.team_b_points_per_game_last_10,
        team_a_matches_available=r.team_a_matches_available,
        team_b_matches_available=r.team_b_matches_available,
        team_a_goals=ga, team_b_goals=gb,
    )


def _make_match_result(r: ResolvedMatchStats, pred, ga: int, gb: int) -> MatchResult:
    """Build a MatchResult from prediction and actuals."""
    if ga > gb: actual = "team_a_win"
    elif ga == gb: actual = "draw"
    else: actual = "team_b_win"
    probs = {"team_a_win": pred.win_a, "draw": pred.draw, "team_b_win": pred.win_b}
    top5 = [(g_a, g_b) for g_a, g_b, _ in pred.top_scorelines]
    return MatchResult(
        date=r.date, team_a=r.team_a, team_b=r.team_b,
        actual_goals_a=ga, actual_goals_b=gb, actual_outcome=actual,
        win_a_prob=pred.win_a, draw_prob=pred.draw, win_b_prob=pred.win_b,
        predicted_outcome=max(probs, key=probs.get),
        top_scorelines=pred.top_scorelines,
        exact_score_hit=len(top5) > 0 and top5[0] == (ga, gb),
        in_top_3=(ga, gb) in top5[:3],
        in_top_5=(ga, gb) in top5,
        prob_of_actual_result=probs[actual],
    )


def run_model2_rolling_stats(resolved, hist):
    """Model 2: Real rolling stats → calculate_pre_match_xg() → Poisson."""
    results = []
    hist_idx = hist.set_index(["date", "team_a", "team_b"])
    for r in resolved:
        key = (r.date, r.team_a, r.team_b)
        row = hist_idx.loc[key]
        ga, gb = int(row["team_a_goals"]), int(row["team_b_goals"])
        pms = _make_pre_match_stats(r, ga, gb)
        xg_a, xg_b = calculate_pre_match_xg(pms)
        pred = predict(r.team_a, r.team_b, xg_a, xg_b)
        results.append(_make_match_result(r, pred, ga, gb))
    return results


def run_model3_mle(resolved, hist, strength_params, model_type="poisson", rho=-0.10):
    """Model 3/4: Real data + MLE strength params → Poisson or Dixon-Coles."""
    results = []
    hist_idx = hist.set_index(["date", "team_a", "team_b"])
    for r in resolved:
        if r.team_a not in strength_params or r.team_b not in strength_params:
            continue
        key = (r.date, r.team_a, r.team_b)
        row = hist_idx.loc[key]
        ga, gb = int(row["team_a_goals"]), int(row["team_b_goals"])
        xg_a, xg_b = calculate_strength_adjusted_xg(
            r.team_a_elo_pre, r.team_b_elo_pre,
            strength_params[r.team_a], strength_params[r.team_b],
            r.team_a_points_per_game_last_10, r.team_b_points_per_game_last_10,
        )
        if model_type == "dixon_coles":
            pred = predict_dixon_coles(r.team_a, r.team_b, xg_a, xg_b, rho=rho)
        else:
            pred = predict(r.team_a, r.team_b, xg_a, xg_b)
        results.append(_make_match_result(r, pred, ga, gb))
    return results


def print_row(label, m, n):
    print(f"  {label:<35s}  {n:>4}  {m.accuracy_1x2:>6.1%}  {m.brier_score:>7.4f}  "
          f"{m.exact_score_accuracy:>6.1%}  {m.top_3_hit_rate:>6.1%}  "
          f"{m.top_5_hit_rate:>6.1%}  {m.avg_prob_actual_result:>6.1%}")


def main():
    print("Loading data...")
    resolved, hist = _resolve_wc2022()
    strength_params = load_strength_params()
    all_ratings = load_team_ratings()

    print(f"  {len(resolved)} / 40 WC 2022 matches resolved")
    print()

    # ── Model 1: Illustrative (team_ratings.csv) ─────────────────────────
    m1_results = run_backtest(ratings=all_ratings)
    m1 = compute_metrics(m1_results)

    # ── Model 2: Real rolling stats → pre_match_xg ───────────────────────
    m2_results = run_model2_rolling_stats(resolved, hist)
    m2 = compute_metrics(m2_results)

    # ── Model 3: Real data + MLE strength → Poisson ───────────────────────
    m3_results = run_model3_mle(resolved, hist, strength_params)
    m3 = compute_metrics(m3_results)

    # ── Model 4: MLE + Dixon-Coles (rho grid) ────────────────────────────
    print("Running rho grid for Model 4...")
    rho_results = []
    for rho in DEFAULT_RHO_GRID:
        dc_results = run_model3_mle(resolved, hist, strength_params, "dixon_coles", rho)
        dc_m = compute_metrics(dc_results)
        rho_results.append(RhoResult(
            rho=rho, accuracy_1x2=dc_m.accuracy_1x2,
            exact_score_accuracy=dc_m.exact_score_accuracy,
            top_3_hit_rate=dc_m.top_3_hit_rate,
            top_5_hit_rate=dc_m.top_5_hit_rate,
            brier_score=dc_m.brier_score,
            avg_prob_actual_result=dc_m.avg_prob_actual_result,
        ))
    best_rho = select_best_rho(rho_results)
    m4_results = run_model3_mle(resolved, hist, strength_params, "dixon_coles", best_rho.rho)
    m4 = compute_metrics(m4_results)

    # ── Print comparison table ────────────────────────────────────────────
    print()
    print("=" * 95)
    print("  SPRINT 2 BACKTEST — WC 2022 (apples-to-apples)")
    print("=" * 95)
    print(f"  {'Model':<35s}  {'N':>4}  {'1X2':>6}  {'Brier':>7}  {'Exact':>6}  "
          f"{'Top3':>6}  {'Top5':>6}  {'AvgP':>6}")
    print("-" * 95)
    print_row("1. Illustrative (AI ratings)", m1, m1.total_matches)
    print_row("2. Real rolling stats (Poisson)", m2, m2.total_matches)
    print_row("3. Real data + MLE (Poisson)", m3, m3.total_matches)
    print_row(f"4. MLE + Dixon-Coles (rho={best_rho.rho:.2f})", m4, m4.total_matches)
    print("=" * 95)
    print()

    # ── Full rho grid ─────────────────────────────────────────────────────
    print("Dixon-Coles rho grid (Model 4 sweep):")
    print(f"  {'rho':>6}  {'Brier':>7}  {'1X2':>6}  {'Top3':>6}  {'Exact':>6}")
    for r in rho_results:
        marker = " <-- best" if r.rho == best_rho.rho else ""
        print(f"  {r.rho:>+6.2f}  {r.brier_score:>7.4f}  {r.accuracy_1x2:>6.1%}  "
              f"{r.top_3_hit_rate:>6.1%}  {r.exact_score_accuracy:>6.1%}{marker}")
    print()

    # ── Recommendation ────────────────────────────────────────────────────
    best_brier = min(m1.brier_score, m2.brier_score, m3.brier_score, m4.brier_score)
    best_label = {
        m1.brier_score: "Model 1 (Illustrative)",
        m2.brier_score: "Model 2 (Real rolling stats)",
        m3.brier_score: "Model 3 (MLE Poisson)",
        m4.brier_score: f"Model 4 (MLE + DC rho={best_rho.rho:.2f})",
    }[best_brier]

    mle_vs_rolling = m2.brier_score - m3.brier_score
    print("Recommendation:")
    print(f"  Best Brier overall: {best_label} ({best_brier:.4f})")
    print(f"  MLE vs rolling stats: {mle_vs_rolling:+.4f} "
          f"({'MLE improves' if mle_vs_rolling > 0 else 'Rolling stats better or equal'})")
    if mle_vs_rolling > 0.005:
        print("  → RECOMMEND: Use MLE opponent strength for future development.")
    elif mle_vs_rolling > 0:
        print("  → MARGINAL: MLE helps slightly. Continue MLE but investigate calibration.")
    else:
        print("  → INCONCLUSIVE: MLE does not improve over rolling stats on this sample.")
        print("    Possible causes: small sample (40 matches), scale of normalization,")
        print("    or xG formula design. Investigate before concluding.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report**

```bash
python scripts/run_sprint2_report.py
```

Expected output: header "SPRINT 2 BACKTEST", table with 4 rows, rho grid, recommendation.

- [ ] **Step 3: Run full suite — no regressions**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 130 tests pass.

- [ ] **Step 4: Final commit**

```bash
git add scripts/run_sprint2_report.py
git commit -m "feat: Sprint 2 complete — 40-match apples-to-apples comparison script"
```

---

## Task 4: Final Verification

- [ ] **Step 1: Confirm 40/40 resolved, params normalized**

```bash
python -c "
import sys, pandas as pd, math
sys.path.insert(0, '.')
from src.data.match_resolver import resolve_all_matches
hist = pd.read_csv('data/historical_matches.csv')
mr   = pd.read_csv('data/match_results.csv')
resolved, unresolved = resolve_all_matches(hist, mr)
print(f'Sprint 2 success checks:')
print(f'  Matches resolved:    {len(resolved)}/40  ({'PASS' if len(resolved)==40 else 'FAIL'})')

df = pd.read_csv('data/team_strength_params.csv')
at_bound = (df.alpha_attack <= 0.105).sum() + (df.beta_defense <= 0.105).sum()
mean_log_a = df.alpha_attack.apply(math.log).mean()
print(f'  Params at bounds:    {at_bound}          ({'PASS' if at_bound==0 else 'FAIL'})')
print(f'  mean(log alpha):     {mean_log_a:.6f}   ({'PASS' if abs(mean_log_a)<0.01 else 'FAIL'})')
"
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest -v
```
Expected: 130 tests pass.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: Phase 2 Sprint 2 complete — research-valid 40-match backtest"
```
