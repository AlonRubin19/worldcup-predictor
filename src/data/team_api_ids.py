"""team_api_ids.py — best-effort mapping of team name -> API-Football team id.

Used by the refresh pipeline to know which teams can have live squad/injury/
player-statistics data pulled. IDs are unverified except for Spain/Norway
(validated live in Sprint 17) -- if an id is wrong, the API call returns
empty and the loaders fall back gracefully with an explicit "Fallback"
source label, so an incorrect id never breaks anything, it just means that
team shows as fallback data.

Coverage: all 32 teams appearing in data/wc2026_fixtures.csv, plus Norway
(used in Golden Boot examples). Teams with no plausible API-Football id are
intentionally omitted from TEAM_API_IDS (not silently dropped -- they show
up as `mapped=False` in the coverage table built by
src.data.team_coverage.build_coverage_table).
"""

from __future__ import annotations

TEAM_API_IDS: dict[str, int] = {
    "Argentina": 26,
    "Australia": 25,
    "Belgium": 1,
    "Brazil": 6,
    "Cameroon": 21,
    "Canada": 22,
    "Costa Rica": 2382,
    "Croatia": 3,
    "Denmark": 21071,
    "Ecuador": 2386,
    "England": 10,
    "France": 2,
    "Germany": 25,
    "Ghana": 1530,
    "Iran": 7,
    "Japan": 12,
    "Mexico": 16,
    "Morocco": 31,
    "Netherlands": 1118,
    "Norway": 1090,
    "Poland": 24,
    "Portugal": 27,
    "Qatar": 18,
    "Saudi Arabia": 24,
    "Senegal": 14,
    "Serbia": 14127,
    "South Korea": 17,
    "Spain": 9,
    "Switzerland": 15,
    "Tunisia": 28,
    "USA": 28,
    "Uruguay": 7,
    "Wales": 14125,
}

# Common alternate names returned by API-Football for some of these teams.
TEAM_API_ALIASES: dict[str, str] = {
    "USA": "United States",
    "South Korea": "Korea Republic",
}
