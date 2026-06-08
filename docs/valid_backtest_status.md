# Valid Backtest — Status

## Current Status

**Engineering validity:** ✅ Complete  
**Data validity:** ❌ Placeholder only  

The infrastructure for a valid backtest is fully implemented. The data is not yet real.

## What Is Placeholder

`data/pre_match_team_stats.csv` contains 25 sample rows with plausible but not
historically sourced values. The ELO ratings, goals averages, and form statistics
were manually estimated as illustrative values — they are NOT derived from real
match-by-match records.

## What Would Make This Research-Grade

To replace the placeholder with real data, each row needs:

### ELO ratings
- Source: World Football Elo Ratings (https://www.eloratings.net)
- Method: Download the historical ELO table, join on team name and date.
  Use the ELO value as of the day BEFORE the match date.

### Goals for/against (last 10)
- Source: football-data.org (free tier, requires API key)
- Method: For each team and match date, fetch their last 10 international
  results BEFORE that date. Compute avg goals scored and avg goals conceded.

### Points per game (last 10)
- Source: Same football-data.org API
- Method: Compute 3 for win, 1 for draw, 0 for loss; average over last 10 matches.

### Matches available
- Method: Count of actual historical matches found before the date (may be <10
  for recently-formed national teams or after a long gap).

## Implementation Path

1. Register for football-data.org free API key.
2. Write `scripts/fetch_pre_match_stats.py` that for each row in
   `data/historical_matches.csv`:
   - Queries the team's last 10 international results before the match date
   - Looks up ELO from eloratings.net historical data
   - Writes a verified row to `data/pre_match_team_stats.csv`
3. Remove the `# WARNING` header from the CSV.
4. Update this document to reflect data validity: ✅
5. Re-run the valid backtest. Results will now constitute a genuine
   out-of-sample evaluation.

## Test Coverage

All code paths are tested. The valid runner, pre-match loader, and pre-match xG
calculator have full test suites that do not depend on real data values.
