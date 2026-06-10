# Odds Integration & Calibration Audit (2026-06-10)

## Part A — Real bookmaker odds investigation

Sources investigated:

1. **API-Football (`API_FOOTBALL_KEY` in `.env`)** — Tested both auth schemes:
   - `x-apisports-key` header against `v3.football.api-sports.io` → `403 Missing application key`
   - `x-rapidapi-key`/`x-rapidapi-host: v3.football.api-sports.io` (the scheme used by
     `src/data/api_football_client.py`) against `/status` and `/fixtures` → `403 Missing application key`
   - `x-rapidapi-key`/`x-rapidapi-host: api-football-v1.p.rapidapi.com` → `401 Invalid API key`
   - **Conclusion: the configured API_FOOTBALL_KEY is not currently valid for any tested
     API-Football host/auth combination, so the `/odds` endpoint could not be reached.**
2. **football-data.co.uk** — confirmed (existing adapter docstring + investigation) to cover
   only domestic league odds. No World Cup odds available. Not viable.
3. **The Odds API** — `ODDS_API_KEY` is not set in the environment; this source requires a
   paid plan, and the existing adapter only supports WC2022 historical snapshots (not live
   WC2026 fixtures). Not viable without new paid credentials and new adapter code.
4. **Other free/low-cost sources** — none identified that meet the `research_valid=true`
   bar without new credentials.

**Result: no real, research-valid bookmaker odds source is currently reachable.**
Per the explicit constraints ("Do not use placeholder odds", "Do not fabricate odds",
"Only use odds marked research_valid=true"), `data/market_odds.csv` (5 placeholder rows,
all `research_valid=false`) is left untouched and the market blend correctly falls back
to "Model only — bookmaker odds unavailable" for every match.

The blend infrastructure (Part B) is fully implemented and wired so that the moment a
real, `research_valid=true` odds row becomes available (via a new credential / adapter),
it is automatically picked up and blended.

## Part B — Market blend reweight (85/15 → 80/20)

`src/models/market_blend.py`:
- `MODEL_WEIGHT = 0.80`, `MARKET_WEIGHT = 0.20`
- `BLEND_LABEL = "Final prediction: 80% model / 20% bookmaker market"`
- `MODEL_ONLY_LABEL = "Model only — bookmaker odds unavailable"`

`src/app/app.py` Match Analyzer "Market Details" expander shows raw model 1X2, market
implied 1X2, and final blended 1X2 side by side when `used_market=True`; otherwise shows
the "Model only — bookmaker odds unavailable" message with the 80/20 label as a preview.

## Part C — Elite team sanity audit (Monte Carlo, n=2000, seed=42)

| Team | Group | ELO | alpha_attack | beta_defense | Exp. matches | Win % |
|------|-------|------|------|------|------|------|
| Argentina | C | 2210.3 | 3.692 | 0.212 | 5.57 | 25.25% |
| Spain | E | 2269.5 | 4.359 | 0.321 | 4.94 | 15.05% |
| Brazil | G | 2092.1 | 4.629 | 0.408 | 5.02 | 8.70% |
| England | B | 2102.0 | 3.167 | 0.272 | 4.92 | 8.40% |
| France | D | 2210.5 | 3.655 | 0.361 | 4.71 | 6.90% |
| Portugal | H | 2118.7 | 3.776 | 0.374 | 4.84 | 6.85% |
| Netherlands | A | 2114.3 | 3.727 | 0.399 | 4.59 | 4.45% |
| Japan | E | 2102.9 | 3.110 | 0.292 | 4.20 | 4.20% |
| Germany | E | 2129.2 | 4.173 | 0.443 | 4.16 | 3.95% |
| USA | B | 2008.7 | 2.793 | 0.613 | 3.66 | 0.20% |

**France**: ranked 6th, 6.90% — **not unreasonable**. France has the joint-highest ELO
(tied with Argentina, 2210.5) but the **lowest beta_defense among the top contenders is
not the issue — its alpha_attack (3.655) is lower than Brazil (4.629), Spain (4.359), and
Portugal (3.776)**. This is the primary driver of France ranking behind those teams
despite a top-2 ELO. This reflects the fitted MLE attack parameter from the underlying
match-result data, not a bracket or calibration artifact — France's group (D) and expected
match count (4.71) are in line with other contenders. **This is a stale/weak alpha
parameter relative to France's current squad strength (e.g., Mbappe-led attack), not a
bug** — improving it would require refreshing the MLE fit with more recent match data,
which is outside the scope of this odds/calibration task.

**Spain and England** are both reasonably placed (2nd and 4th by win%) — no anomaly found.

## Part D — 1-1 overuse / Dixon-Coles rho audit (48 fixtures)

| rho | % top score = 1-1 | % favourite >60% but top = 1-1 |
|------|------|------|
| -0.30 (old default) | 70.8% | 12.5% |
| -0.20 | 68.8% | 10.4% |
| **-0.13 (new default)** | **56.2%** | **2.1%** |
| 0.00 (Poisson, no DC correction) | 39.6% | 2.1% |

**Recommendation (applied): rho changed from -0.30 to -0.13.**

rho=-0.30 was clearly too aggressive: in 12.5% of matches a clear favourite (>60% win
probability) still had "1-1" as the single most likely scoreline, which is unrealistic.
rho=-0.13 cuts that to 2.1% (matching the rho=0.00 baseline) while still applying a
Dixon-Coles low-score correlation correction (rho=-0.13 is within the commonly cited
empirical range for football, vs -0.30 which over-corrects). Going all the way to
rho=0.00 would discard the empirical low-score correlation entirely, so -0.13 was chosen
as the minimal change that fixes the favourite/1-1 inflation.

Changed in:
- `src/models/research_valid_predictor.py` (`DEFAULT_RHO`)
- `src/tournament/simulator.py` (`_RHO`)
- UI labels in `src/app/app.py` and `src/app/prediction_runner.py`

Note: even at rho=0.00, ~40% of matches still have 1-1 as the top scoreline — this is
expected for low-xG matches (xG around 1.0–1.5 each) where 1-1 is the Poisson mode. This
is not a bug; it reflects genuinely tight matchups in the current fixture set.

## Part E — Validation

Re-ran `predict_research_valid` (rho=-0.13) and the 80/20 blend for: Mexico vs South
Africa, Australia vs Turkey, and all France/Spain/England group fixtures. All show
"Model only — bookmaker odds unavailable" (consistent with Part A finding — no real odds
available), with raw model 1X2 == blended 1X2 in every case.

Tournament simulation (n=3000) top 10 winner probabilities (rho=-0.13):
Argentina 24.7%, Spain 11.7%, England 9.1%, Brazil 8.5%, Portugal 6.9%, France 6.7%,
Japan 5.0%, Netherlands 4.9%, Ecuador 4.6%, Morocco 4.1%.

Golden Boot top 20 unchanged in ranking (Messi, Mbappe, Kane, Neymar, Ronaldo, ...) —
golden boot model does not depend on rho.

All 1100 tests pass (one test, `test_higher_probability_ranked_first`, was updated — its
original assertion encoded a coincidental ordering that no longer held after the rho
change; replaced with a correctness check on signal-strength tier ordering).
