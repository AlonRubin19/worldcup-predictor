"""Tournament fixture filtering, sorting, and grouping helpers.

All functions are pure (no Streamlit dependency) so they are fully testable.
"""

from __future__ import annotations

from src.tournament.fixtures import Fixture

# ── Status display labels ─────────────────────────────────────────────────────

STATUS_LABELS: dict[str, str] = {
    "NS":  "Upcoming",
    "TBD": "TBD",
    "1H":  "1st Half",
    "HT":  "Half Time",
    "2H":  "2nd Half",
    "ET":  "Extra Time",
    "BT":  "Break (Pens)",
    "P":   "Penalties",
    "SUSP":"Suspended",
    "INT": "Interrupted",
    "FT":  "Finished",
    "AET": "Finished (AET)",
    "PEN": "Finished (Pens)",
    "PST": "Postponed",
    "CANC":"Cancelled",
    "ABD": "Abandoned",
    "AWD": "Awarded",
    "WO":  "Walkover",
    "LIVE":"Live",
}

_FINISHED_STATUSES = {"FT", "AET", "PEN", "ABD", "AWD", "WO"}
_LIVE_STATUSES     = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE"}
_UPCOMING_STATUSES = {"NS", "TBD", "PST"}


def get_status_label(status: str) -> str:
    """Return a human-readable label for an API status short code."""
    return STATUS_LABELS.get(status, status)


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_fixtures(
    fixtures: list[Fixture],
    *,
    stage: str | None = None,
    group: str | None = None,
    team: str | None = None,
    date: str | None = None,
    statuses: list[str] | None = None,
) -> list[Fixture]:
    """Filter a fixture list by any combination of criteria.

    All filters are AND-combined.  Passing None for a criterion means
    "don't filter by this field".

    Args:
        fixtures:  Source fixture list.
        stage:     Internal stage value, e.g. "group", "round_of_16".
        group:     Group letter, e.g. "A".
        team:      Team name; matches if present in either team_a or team_b
                   (case-insensitive substring match).
        date:      ISO date string YYYY-MM-DD for exact date match.
        statuses:  List of API status codes; fixture included if its status
                   is in this list.

    Returns:
        Filtered list (original objects, not copies).
    """
    result = fixtures
    if stage is not None:
        result = [f for f in result if f.stage == stage]
    if group is not None:
        result = [f for f in result if f.group == group]
    if team is not None:
        _t = team.lower()
        result = [
            f for f in result
            if _t in f.team_a.lower() or _t in f.team_b.lower()
        ]
    if date is not None:
        result = [f for f in result if f.date == date]
    if statuses is not None:
        _s = set(statuses)
        result = [f for f in result if f.status in _s]
    return result


# ── Next / today helpers ──────────────────────────────────────────────────────

def get_next_fixtures(
    fixtures: list[Fixture],
    n: int = 5,
    today: str | None = None,
) -> list[Fixture]:
    """Return the next N upcoming (not finished) fixtures, sorted by date.

    Args:
        fixtures: Full fixture list.
        n:        Maximum number of fixtures to return.
        today:    ISO date string YYYY-MM-DD.  If provided, fixtures before
                  this date are excluded (regardless of status).

    Returns:
        Up to n fixtures, sorted ascending by date, excluding finished matches.
    """
    result = [f for f in fixtures if f.status not in _FINISHED_STATUSES]
    if today is not None:
        result = [f for f in result if f.date >= today]
    result = sorted(result, key=lambda f: f.date)
    return result[:n]


def get_today_fixtures(
    fixtures: list[Fixture],
    today: str | None = None,
) -> list[Fixture]:
    """Return all fixtures scheduled for today.

    Args:
        fixtures: Full fixture list.
        today:    ISO date string YYYY-MM-DD.

    Returns:
        Fixtures where date == today.
    """
    if today is None:
        import datetime
        today = datetime.date.today().isoformat()
    return [f for f in fixtures if f.date == today]


# ── Unique value extractors ───────────────────────────────────────────────────

def get_unique_stages(fixtures: list[Fixture]) -> list[str]:
    """Return sorted unique stage values from a fixture list."""
    return sorted({f.stage for f in fixtures})


def get_unique_groups(fixtures: list[Fixture]) -> list[str]:
    """Return sorted unique group letters (non-empty) from a fixture list."""
    return sorted({f.group for f in fixtures if f.group})


def get_unique_teams(fixtures: list[Fixture]) -> list[str]:
    """Return sorted unique team names from a fixture list."""
    teams: set[str] = set()
    for f in fixtures:
        teams.add(f.team_a)
        teams.add(f.team_b)
    return sorted(teams)
