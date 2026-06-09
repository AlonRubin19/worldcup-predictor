"""Fixture provider abstraction for the Daily Match Board.

Supports three modes:
  - CSV:  Load fixtures from the static CSV file (always available).
  - API:  Fetch live upcoming fixtures from API-Football (requires key).
  - AUTO: Use API if a key is available and reachable; fall back to CSV.

No live HTTP calls are made here — the ApiFootballClient handles that,
and tests inject a mock _fetcher to keep this module fully testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.api_football_client import ApiFootballClient

from src.data.live_fixture_loader import (
    LiveFixture,
    fetch_upcoming_fixtures,
    map_team_name,
    TEAM_NAME_MAP,
)
from src.tournament.fixtures import Fixture, load_fixtures as _csv_load_fixtures

# ── Default CSV path ──────────────────────────────────────────────────────────

_DEFAULT_CSV = Path("data/world_cup_fixture_sample.csv")

# ── Known teams set (loaded lazily from teams.csv) ────────────────────────────

def _load_known_teams() -> set[str]:
    teams_path = Path("data/teams.csv")
    if not teams_path.exists():
        return set()
    import csv
    teams: set[str] = set()
    with open(teams_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("team_name", row.get("team", "")).strip()
            if name:
                teams.add(name)
    return teams


# ── Enums & dataclasses ───────────────────────────────────────────────────────

class FixtureSource(str, Enum):
    CSV  = "csv"
    API  = "api"
    AUTO = "auto"


@dataclass
class ProviderResult:
    """Result returned by get_fixtures()."""
    fixtures: list[Fixture]
    source_used: FixtureSource
    api_connected: bool
    mapping_warnings: list[str] = field(default_factory=list)

    @property
    def fixture_count(self) -> int:
        return len(self.fixtures)


# ── Stage parser ──────────────────────────────────────────────────────────────

_STAGE_MAP: list[tuple[str, str]] = [
    # pattern (case-insensitive)    stage value
    (r"group\s*stage",              "group"),
    (r"round\s*of\s*16",            "round_of_16"),
    (r"quarter.?final",             "quarter_final"),
    (r"semi.?final",                "semi_final"),
    (r"3rd\s*place|third\s*place",  "third_place"),
    (r"\bfinal\b",                  "final"),
]


def parse_stage(round_str: str) -> str:
    """Map an API-Football round string to an internal stage value.

    Returns "group" for any group-stage round, the appropriate knockout
    stage for well-known rounds, and "unknown" for anything else.
    """
    s = round_str.strip().lower()
    for pattern, stage in _STAGE_MAP:
        if re.search(pattern, s):
            return stage
    return "unknown"


# ── LiveFixture → Fixture converter ──────────────────────────────────────────

def convert_live_fixture_to_fixture(lf: LiveFixture) -> Fixture:
    """Convert an API-Football LiveFixture to the internal Fixture format.

    - match_id : str(fixture_id)
    - date     : YYYY-MM-DD (strips time/tz component if present)
    - stage    : parsed from lf.round
    - group    : "" (API /fixtures does not return group letter)
    - team_a   : lf.home_team  (already mapped by live_fixture_loader)
    - team_b   : lf.away_team
    """
    # Date: take only the date portion
    date_part = lf.date[:10] if len(lf.date) >= 10 else lf.date

    return Fixture(
        match_id=str(lf.fixture_id),
        stage=parse_stage(lf.round),
        group="",
        date=date_part,
        team_a=lf.home_team,
        team_b=lf.away_team,
        status=lf.status_short or "NS",
    )


# ── Warning generator ─────────────────────────────────────────────────────────

def _collect_mapping_warnings(
    live_fixtures: list[LiveFixture],
    known_teams: set[str],
) -> list[str]:
    """Return warning strings for teams that could not be mapped."""
    warnings: list[str] = []
    seen: set[str] = set()
    for lf in live_fixtures:
        for api_name in (lf.home_team_api, lf.away_team_api):
            mapped = map_team_name(api_name)
            if mapped not in known_teams and api_name not in seen:
                warnings.append(
                    f"Unknown team mapping: '{api_name}' — not found in teams.csv"
                )
                seen.add(api_name)
    return warnings


# ── CSV provider ──────────────────────────────────────────────────────────────

def _get_csv_fixtures(csv_path: Path | str | None) -> ProviderResult:
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
    fixtures = _csv_load_fixtures(str(path))
    return ProviderResult(
        fixtures=fixtures,
        source_used=FixtureSource.CSV,
        api_connected=False,
        mapping_warnings=[],
    )


# ── API provider ──────────────────────────────────────────────────────────────

def _get_api_fixtures(
    api_client: "ApiFootballClient | None",
    league_id: int,
    season: int,
    force_refresh: bool,
) -> ProviderResult:
    """Fetch fixtures from API-Football.

    Returns a failed result (api_connected=False, empty fixtures) on any error.
    """
    from src.data.api_football_client import ApiKeyMissingError, CACHE_TTL_FIXTURES

    if api_client is None:
        return ProviderResult(
            fixtures=[],
            source_used=FixtureSource.API,
            api_connected=False,
        )

    ttl = 0 if force_refresh else CACHE_TTL_FIXTURES

    try:
        live_fixtures = fetch_upcoming_fixtures(
            client=api_client,
            league_id=league_id,
            season=season,
            ttl_seconds=ttl,
        )
    except ApiKeyMissingError:
        return ProviderResult(
            fixtures=[],
            source_used=FixtureSource.API,
            api_connected=False,
        )
    except Exception:
        return ProviderResult(
            fixtures=[],
            source_used=FixtureSource.API,
            api_connected=False,
        )

    fixtures = [convert_live_fixture_to_fixture(lf) for lf in live_fixtures]
    known_teams = _load_known_teams()
    warnings = _collect_mapping_warnings(live_fixtures, known_teams)

    return ProviderResult(
        fixtures=fixtures,
        source_used=FixtureSource.API,
        api_connected=True,
        mapping_warnings=warnings,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def get_fixtures(
    mode: FixtureSource | str = FixtureSource.AUTO,
    api_client: "ApiFootballClient | None" = None,
    csv_path: "Path | str | None" = None,
    league_id: int = 1,
    season: int = 2026,
    force_refresh: bool = False,
) -> ProviderResult:
    """Retrieve fixtures for the Daily Match Board.

    Args:
        mode:          FixtureSource.CSV / API / AUTO (or their string values).
        api_client:    Configured ApiFootballClient (required for API/AUTO modes).
        csv_path:      Override the default CSV path for CSV mode.
        league_id:     API-Football league ID (1 = FIFA World Cup).
        season:        Season year.
        force_refresh: Bypass the file cache and re-fetch from the API.

    Returns:
        ProviderResult with fixtures, source metadata, and mapping warnings.
    """
    if isinstance(mode, str):
        mode = FixtureSource(mode)

    if mode is FixtureSource.CSV:
        return _get_csv_fixtures(csv_path)

    if mode is FixtureSource.API:
        return _get_api_fixtures(api_client, league_id, season, force_refresh)

    # AUTO: try API first; fall back to CSV on any failure
    api_result = _get_api_fixtures(api_client, league_id, season, force_refresh)
    if api_result.api_connected:
        return api_result
    return _get_csv_fixtures(csv_path)
