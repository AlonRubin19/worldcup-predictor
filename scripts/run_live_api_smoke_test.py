#!/usr/bin/env python
"""Live API-Football Pro smoke test.

Loads API_FOOTBALL_KEY from .env (via python-dotenv), fetches upcoming
fixtures for a configurable league/season, prints a mapping audit, then
attempts to retrieve lineups for the first three fixtures.

Usage:
    python scripts/run_live_api_smoke_test.py
    python scripts/run_live_api_smoke_test.py --league 1 --season 2026
    python scripts/run_live_api_smoke_test.py --max-lineup-attempts 5

Never crashes on missing key or unavailable lineups.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles (handles accented team names).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Load .env before any src imports ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError, CACHE_TTL_FIXTURES, CACHE_TTL_LINEUPS
from src.data.live_fixture_loader import fetch_upcoming_fixtures
from src.data.live_lineup_loader import fetch_lineups
from src.data.team_mapping_audit import audit_team_mappings
from src.data.loader import load_teams


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _sep("=")
    print(f"  {title}")
    _sep("=")


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}")


def _info(msg: str) -> None:
    print(f"  [--]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Core smoke functions (importable for testing)
# ─────────────────────────────────────────────────────────────────────────────

def check_api_connection(api_key: str) -> tuple[bool, str]:
    """Return (connected: bool, message: str)."""
    if not api_key:
        return False, (
            "API_FOOTBALL_KEY is not set.\n"
            "  Set it in your .env file or as an environment variable.\n"
            "  Example: API_FOOTBALL_KEY=your_key_here"
        )
    return True, f"API key found ({len(api_key)} chars)."


def fetch_fixtures_report(
    client: ApiFootballClient,
    league_id: int,
    season: int,
) -> tuple[list, str]:
    """Fetch fixtures and return (fixtures, status_message).

    Returns ([], error_message) on failure — never raises.
    """
    try:
        fixtures = fetch_upcoming_fixtures(
            client, league_id=league_id, season=season,
            ttl_seconds=CACHE_TTL_FIXTURES,
        )
        return fixtures, f"Fetched {len(fixtures)} fixture(s) from API."
    except ApiKeyMissingError as exc:
        return [], f"API key missing: {exc}"
    except Exception as exc:
        return [], f"Fixture fetch failed: {exc}"


def fetch_lineup_report(
    client: ApiFootballClient,
    fixture_id: int,
    fixture_label: str,
) -> tuple[list, str]:
    """Attempt to fetch lineups for one fixture.

    Returns (entries, status_message) — never raises.
    """
    try:
        entries = fetch_lineups(client, fixture_id=fixture_id,
                                ttl_seconds=CACHE_TTL_LINEUPS)
        if entries:
            teams = sorted({e.team for e in entries})
            starters = sum(1 for e in entries if e.expected_starter)
            return entries, (
                f"Lineup available — {starters} starters across teams: "
                + ", ".join(teams)
            )
        return [], f"Lineup not yet published for fixture {fixture_id}."
    except ApiKeyMissingError as exc:
        return [], f"API key missing: {exc}"
    except Exception as exc:
        return [], f"Lineup fetch failed for {fixture_label}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(
    league_id: int = 1,
    season: int = 2026,
    max_lineup_attempts: int = 3,
) -> None:
    api_key = os.environ.get("API_FOOTBALL_KEY", "")

    _section("API-Football Pro Smoke Test")
    print(f"  League: {league_id}  |  Season: {season}")

    # ── 1. Connection check ───────────────────────────────────────────────────
    _section("1. API Connection")
    connected, msg = check_api_connection(api_key)
    ((_ok if connected else _warn)(msg))

    if not connected:
        _warn("Cannot proceed without an API key. Exiting.")
        return

    client = ApiFootballClient(
        api_key=api_key,
        cache_dir=_ROOT / "data" / "api_cache",
    )

    # ── 2. Fixture endpoint ───────────────────────────────────────────────────
    _section("2. Fixture Endpoint")
    fixtures, fix_msg = fetch_fixtures_report(client, league_id, season)
    ((_ok if fixtures else _warn)(fix_msg))

    if not fixtures:
        _warn("No fixtures returned — cannot run mapping audit or lineup tests.")
        return

    # ── 3. Mapping audit ──────────────────────────────────────────────────────
    _section("3. Team Name Mapping Audit")
    known_teams = set(load_teams())
    audit = audit_team_mappings(fixtures, known_teams=known_teams)

    _ok(f"Exact matches  : {audit.exact_count}")
    _ok(f"Mapped aliases : {audit.mapped_count}")

    if audit.unknown_count > 0:
        _warn(f"Unknown teams  : {audit.unknown_count} — need mapping entries:")
        for name in audit.unknown_teams:
            _warn(f"    '{name}'  ->  ???  (add to TEAM_NAME_MAP)")
    else:
        _ok("Unknown teams  : 0 — all teams mapped!")

    if audit.mapped_teams:
        _info("Active aliases:")
        for api_name, internal in audit.mapped_teams:
            _info(f"    '{api_name}'  ->  '{internal}'")

    # ── 4. Fixture table ──────────────────────────────────────────────────────
    _section("4. Fixture Table")
    print(f"  {'ID':>8}  {'Date':<26}  {'Home':<22}  {'Away':<22}  {'Map'}")
    _sep()
    for f in fixtures:
        from src.data.team_mapping_audit import classify_team
        h_cls = classify_team(f.home_team_api, known_teams).value
        a_cls = classify_team(f.away_team_api, known_teams).value
        home_disp = f.home_team if f.home_team != f.home_team_api else f.home_team
        away_disp = f.away_team if f.away_team != f.away_team_api else f.away_team
        print(
            f"  {f.fixture_id:>8}  {f.date:<26}  "
            f"{home_disp:<22}  {away_disp:<22}  "
            f"{h_cls}/{a_cls}"
        )

    # ── 5. Lineup endpoint ────────────────────────────────────────────────────
    _section("5. Lineup Endpoint (first {max_lineup_attempts} fixtures)")
    for f in fixtures[:max_lineup_attempts]:
        label = f"{f.home_team} vs {f.away_team}"
        entries, lineup_msg = fetch_lineup_report(client, f.fixture_id, label)
        ((_ok if entries else _info)(f"[{f.fixture_id}] {label}: {lineup_msg}"))

    # ── 6. Summary ────────────────────────────────────────────────────────────
    _section("6. Summary")
    _ok("API: connected")
    _ok("Fixture endpoint: working")
    if audit.unknown_count == 0:
        _ok("Team mappings: complete")
    else:
        _warn(f"Team mappings: {audit.unknown_count} unknown team(s) — update TEAM_NAME_MAP")
        _warn("Suggested additions for src/data/live_fixture_loader.py TEAM_NAME_MAP:")
        for name in audit.unknown_teams:
            _warn(f'    "{name}": "<internal_name>",')

    print()
    _sep()
    print("  Smoke test complete.")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="API-Football Pro smoke test and mapping audit."
    )
    parser.add_argument("--league", type=int, default=1,
                        help="API-Football league ID (default: 1 = FIFA World Cup)")
    parser.add_argument("--season", type=int, default=2026,
                        help="Season year (default: 2026)")
    parser.add_argument("--max-lineup-attempts", type=int, default=3,
                        help="Number of fixtures to attempt lineup fetch for (default: 3)")
    args = parser.parse_args()

    run_smoke_test(
        league_id=args.league,
        season=args.season,
        max_lineup_attempts=args.max_lineup_attempts,
    )
