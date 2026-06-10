"""team_api_ids.py — best-effort mapping of team name -> API-Football team id.

Used by the refresh pipeline to know which teams can have live squad/injury/
player-statistics data pulled. IDs are unverified for teams beyond
Spain/Norway (which were validated live in Sprint 17) -- if an id is wrong,
the API call returns empty and the loaders fall back gracefully with an
explicit "Fallback" source label, so an incorrect id never breaks anything,
it just means that team shows as fallback data.
"""

from __future__ import annotations

TEAM_API_IDS: dict[str, int] = {
    "Spain": 9,
    "Norway": 1090,
    "France": 2,
    "England": 10,
    "Argentina": 26,
    "Brazil": 6,
    "Portugal": 27,
    "Croatia": 3,
    "Morocco": 31,
    "Saudi Arabia": 24,
}
