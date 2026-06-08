# Phase 2 Sprint 1 Design — ELO Ingestion, Match Ingestion, Opponent Strength MLE

**Date:** 2026-06-08  
**Status:** Ready for review  
**Scope:** First three deliverables of Phase 2: real historical ELO, real match results, and MLE-fitted opponent strength parameters

---

## 1. Goal

Replace placeholder data with real historical sources and fit opponent strength parameters using maximum likelihood estimation. This establishes the "Real Data Layer" (Module 1) and "Opponent Strength Adjustment" (Module 2) from the Phase 2 architecture.

Success criterion: **Brier score improvement vs. current V6 baseline** when using strength-adjusted xG instead of raw goal averages.

---

## 2. Data Sources

### Source: martj42/international-results (GitHub)

- **URL:** https://github.com/martj42/international_football_results_1872-2017
- **Data:** ~47,000 international matches from 1872 to 2017
- **Format:** CSV with columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`
- **Why this source:** Complete historical record, covers WC 2022 period, publicly available, no API key required
- **Access:** Download as CSV directly from repo or via pandas `read_csv(URL)` from raw GitHub link

### Extension: WC 2022 matches

- Existing `data/historical_matches.csv` contains the 40 group + knockout matches
- Will be cross-referenced to ensure continuity (no duplicates, consistent team names)

---

## 3. Deliverables

### Deliverable 1: `data/elo_history.csv`

**Purpose:** Chronological ELO rating history for all teams.

**Schema:**
```
date, team, elo_rating
2022-11-20, France, 2043
2022-11-20, Brazil, 2055
...
```

**Generation process:**
1. Load martj42 CSV
2. Sort by date ascending (chronological order)
3. For each team: initialize ELO to 1600 (FIFA standard base)
4. Iterate through matches chronologically
5. For each match: store pre-match ELO for both teams, then update ELO using K-factor
6. Write to CSV: one row per (team, date) with pre-match ELO

**ELO Update Formula (Elo International Standard):**
```
expected_home = 1 / (1 + 10^((elo_away - elo_home) / 400))
expected_away = 1 - expected_home

if home_score > away_score:
    actual_home = 1.0
    actual_away = 0.0
elif home_score == away_score:
    actual_home = 0.5
    actual_away = 0.5
else:
    actual_home = 0.0
    actual_away = 1.0

K = 60 (standard for international football)
elo_home_new = elo_home + K * (actual_home - expected_home)
elo_away_new = elo_away + K * (actual_away - expected_away)
```

**Key requirements:**
- No look-ahead bias: ELO for match N uses only matches 1 through N-1
- Handle team name variations (e.g., "South Korea" vs "Korea", "England" vs "Great Britain")
- Start both teams at 1600 on first appearance
- Neutral venue doesn't change K-factor for international football (different from club)

---

### Deliverable 2: `data/match_results.csv`

**Purpose:** Complete match history with pre-match and actual results.

**Schema:**
```
date, team_a, team_b, team_a_goals, team_b_goals, 
team_a_elo_pre, team_b_elo_pre, 
team_a_goals_for_last_10, team_a_goals_against_last_10,
team_b_goals_for_last_10, team_b_goals_against_last_10,
team_a_points_per_game_last_10, team_b_points_per_game_last_10,
team_a_matches_available, team_b_matches_available
```

**Generation process:**
1. For each match in chronological order:
   - Look up pre-match ELO from `elo_history.csv` (largest date ≤ match date)
   - Compute last-10 stats: goals for/against per game, points per game
   - Count actual matches available in the window
2. Store result

**Key requirements:**
- Last-10 window: the 10 most recent matches BEFORE the current match date
- If team has fewer than 10 matches before this date, compute average over available (e.g., if team has only 3 prior matches, use those 3)
- No future data: all statistics use only matches before the current match's date
- Points: win=3, draw=1, loss=0

---

### Deliverable 3: `data/team_strength_params.csv`

**Purpose:** MLE-fitted attack/defense strength parameters for each team.

**Schema:**
```
team, alpha_attack, beta_defense, as_of_date, matches_used, log_likelihood
France, 1.518, 0.694, 2022-11-20, 24, -68.432
Brazil, 1.487, 0.742, 2022-11-20, 26, -71.105
...
```

**Generation process (Maximum Likelihood Estimation):**

1. **Training window:** All matches up to and including 2022-11-19 (the day before WC 2022 starts)
2. **Time decay:** Weight each match with exponential decay `w = 0.99^(days_ago)` to prioritize recent form
3. **For each team:** Estimate `alpha` (attack strength) and `beta` (defense strength) by maximizing:

```
Log-Likelihood = Σ { 
    w_i * [
        observed_home_goals_i * log(λ_home_i) - λ_home_i - log(observed_home_goals_i)! +
        observed_away_goals_i * log(λ_away_i) - λ_away_i - log(observed_away_goals_i)!
    ]
}

where:
λ_home_i = α_home * β_away_i * γ   (γ = 1.0 for international, no home advantage effect)
λ_away_i = α_away * β_home_i
```

4. **Optimization:** Use scipy.optimize.minimize to fit α and β for each team
5. **Constraints:** α, β > 0.1 (prevent unrealistic extreme values)
6. **Output:** One row per team with fitted parameters and training statistics

**Key requirements:**
- Each team gets one (α, β) pair as of 2022-11-19
- Compute using all matches before WC 2022 start date
- Time decay ensures recent form is weighted more
- Log-likelihood is Poisson likelihood (matches treated as independent goals counts)

---

## 4. Data Flow Diagram

```
martj42/international-results CSV
    ↓
    ├─→ Compute ELO chronologically → data/elo_history.csv
    │
    └─→ Extract match metadata + results → data/match_results.csv
         ↓
         └─→ Compute last-10 rolling stats
         
                         ↓
                   
       All matches up to 2022-11-19 (pre-WC)
         ↓
         └─→ MLE fit for each team → data/team_strength_params.csv
         
                         ↓

            All three CSVs loaded into memory
         ↓
         └─→ For each WC 2022 match:
             - Look up pre-match ELO
             - Look up last-10 stats
             - Look up strength params (α, β)
             - Generate pre_match_team_stats_real.csv
                         ↓
                    Valid backtest ready
```

---

## 5. Architecture

### New modules

**`scripts/ingest_elo_and_matches.py`**
- Downloads/loads martj42 CSV
- Computes ELO history
- Generates match_results.csv
- Handles team name normalization
- ~150–200 lines

**`scripts/fit_strength_params.py`**
- Reads match_results.csv
- Performs MLE fit for each team
- Generates team_strength_params.csv
- ~100–150 lines

**`src/data/strength_adjusted_xg.py`** (used in later stages)
- Replace `calculate_pre_match_xg()` with version that reads α, β instead of raw goal averages
- Formula: `xg_a = BASE_XG * α_a * β_b * form_a * elo_factor_a`
- Identical interface to current `calculate_pre_match_xg()`

### Existing modules (unchanged)

- `src/backtesting/valid_runner.py` — updated to use `strength_adjusted_xg()` once available
- `src/backtesting/metrics.py`, `poisson.py`, `dixon_coles.py` — unchanged
- `src/app/app.py` — unchanged in Sprint 1

---

## 6. Testing Strategy

### `tests/data/test_elo_ingestion.py`
- ELO for a new team starts at 1600
- After 1 win, ELO increases
- After 1 loss, ELO decreases
- Draw → no net change (K/2 to each)
- Last match before date N is not used in ELO for match N

### `tests/data/test_match_ingestion.py`
- Match results CSV has all required columns
- All matches sorted by date
- Last-10 stats are computed over correct window (10 most recent, not last calendar 10 days)
- If team has <10 prior matches, average uses available matches
- ELO values match elo_history.csv

### `tests/models/test_mle_fitter.py`
- MLE fit converges for known test data
- α > 0 and β > 0 for all teams
- Teams with more wins have higher α
- Teams with more clean sheets have higher β
- Log-likelihood is negative (correct for likelihood)

### `tests/backtesting/test_strength_adjusted_xg.py`
- Strong attacker (high α) produces higher xg_a
- Strong defender (high β) reduces opponent xg
- Formula: `xg = BASE_XG * α * β_opp * form * elo_factor` is applied correctly
- Output clamped to [0.2, 4.5]

---

## 7. Success Criteria

✅ **Data quality:**
- Zero rows with missing pre-match ELO
- Zero rows with missing last-10 stats (falling back to available matches)
- All WC 2022 teams present in elo_history.csv with ELO as of 2022-11-19

✅ **No leakage:**
- ELO for match N is pre-match (before goals in match N)
- Last-10 stats for match N use only matches before N
- MLE parameters fitted only on matches before WC 2022

✅ **Backtesting improvement:**
- Real data backtest produces valid Brier score
- Strength-adjusted xG Brier score < Poisson Brier score on WC 2022 historical

✅ **Tests:**
- All new tests pass
- No regressions in existing tests

---

## 8. Timeline

- **Hours 0–2:** Ingest ELO + compute history
- **Hours 2–4:** Ingest matches + compute last-10 stats
- **Hours 4–8:** MLE fit for strength parameters
- **Hours 8–10:** Integration + testing
- **Total:** ~2 days work (accounting for testing and debugging)

---

## 9. What Comes Next (Sprint 2)

Once Sprint 1 is complete:
- `strength_adjusted_xg.py` will read α, β from team_strength_params.csv
- Valid runner will use strength-adjusted xG instead of raw goal averages
- Backtest will show whether MLE parameters improve Brier score
- If yes: deploy to app; if no: investigate and iterate
