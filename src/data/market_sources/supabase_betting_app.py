"""supabase_betting_app.py — real bookmaker odds via the sibling betting-app's
Supabase project.

The companion World Cup betting app (C:/projects/world_cup) already runs a
working odds sync (`supabase/functions/sync-odds`) that pulls 1X2 odds from
The Odds API (averaged across bookmakers) and stores them in its `match_odds`
table, joined to its `matches` table.

We reuse that already-synced data by reading it directly from the betting
app's Supabase REST API using its public (anon/publishable) API key — no
new credentials required, and no placeholder/fabricated data.

Rows are cached to disk (TTL-based) and matched to predictor team names by
(date, normalized home team, normalized away team).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

SUPABASE_URL = "https://lnnpcppwivsjtatedqcg.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_obIvrcZ0l_cJrTIcXAhMbg_cryVnMKc"

_CACHE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "betting_app_odds.json"
_CACHE_TTL_SECONDS = 3600  # 1 hour


_NAME_MAP = {
    "türkiye": "turkey",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "usa": "usa",
    "united states": "usa",
}


def _normalize(name: str) -> str:
    n = name.strip().lower()
    return _NAME_MAP.get(n, n)


@dataclass
class BettingAppOddsRow:
    team_a: str
    team_b: str
    date: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str
    updated_at: str


def _fetch_json(path: str) -> list[dict]:
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/{path}", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_live() -> list[BettingAppOddsRow]:
    matches = _fetch_json(
        "matches?select=id,kickoff_time,home_team:home_team_id(name_en),"
        "away_team:away_team_id(name_en)&status=in.(scheduled,live)&limit=500"
    )
    odds = _fetch_json("match_odds?select=*")

    match_by_id = {m["id"]: m for m in matches}

    rows: list[BettingAppOddsRow] = []
    for o in odds:
        m = match_by_id.get(o.get("match_id"))
        if m is None:
            continue
        home = (m.get("home_team") or {}).get("name_en")
        away = (m.get("away_team") or {}).get("name_en")
        kickoff = m.get("kickoff_time")
        if not (home and away and kickoff):
            continue
        rows.append(BettingAppOddsRow(
            team_a=home,
            team_b=away,
            date=str(kickoff)[:10],
            home_odds=float(o["home_odds"]),
            draw_odds=float(o["draw_odds"]),
            away_odds=float(o["away_odds"]),
            bookmaker=str(o.get("bookmaker", "")),
            updated_at=str(o.get("updated_at", "")),
        ))
    return rows


def get_betting_app_odds(force_refresh: bool = False) -> list[BettingAppOddsRow]:
    """Return cached (or freshly-fetched) real odds rows from the betting app."""
    if not force_refresh and _CACHE_PATH.exists():
        age = time.time() - _CACHE_PATH.stat().st_mtime
        if age < _CACHE_TTL_SECONDS:
            data = json.loads(_CACHE_PATH.read_text())
            return [BettingAppOddsRow(**row) for row in data]

    try:
        rows = _fetch_live()
    except (requests.RequestException, KeyError, ValueError):
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            return [BettingAppOddsRow(**row) for row in data]
        return []

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps([row.__dict__ for row in rows]))
    return rows


def find_odds_for_match(team_a: str, team_b: str) -> BettingAppOddsRow | None:
    """Find a real odds row for (team_a, team_b), in either order, by name."""
    rows = get_betting_app_odds()
    na, nb = _normalize(team_a), _normalize(team_b)

    for row in rows:
        ra, rb = _normalize(row.team_a), _normalize(row.team_b)
        if ra == na and rb == nb:
            return row
        if ra == nb and rb == na:
            return BettingAppOddsRow(
                team_a=row.team_b, team_b=row.team_a, date=row.date,
                home_odds=row.away_odds, draw_odds=row.draw_odds, away_odds=row.home_odds,
                bookmaker=row.bookmaker, updated_at=row.updated_at,
            )
    return None
