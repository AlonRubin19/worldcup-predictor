"""market_odds_loader.py — look up bookmaker odds for a match by team names.

Reads data/market_odds.csv and converts decimal odds to normalized implied
1X2 probabilities. Only rows with research_valid=true are usable for the
market blend -- placeholder odds are returned with research_valid=False so
callers never blend on placeholder data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "market_odds.csv"


@dataclass
class MarketOddsResult:
    win_a: float | None
    draw: float | None
    win_b: float | None
    research_valid: bool
    bookmaker: str | None = None


def _implied_probs(odds_a: float, odds_d: float, odds_b: float) -> tuple[float, float, float]:
    inv = (1.0 / odds_a, 1.0 / odds_d, 1.0 / odds_b)
    total = sum(inv)
    return tuple(x / total for x in inv)  # type: ignore[return-value]


def get_market_odds_for_match(
    team_a: str, team_b: str, path: Path | None = None,
) -> MarketOddsResult:
    """Return implied 1X2 probabilities for (team_a, team_b), if any.

    Matches on (team_a, team_b) in either order (swapping home/away odds if
    reversed). Returns research_valid=False if no row found or the row is a
    placeholder.
    """
    p = path if path is not None else _DEFAULT
    if not Path(p).exists():
        return MarketOddsResult(None, None, None, research_valid=False)

    df = pd.read_csv(p)

    direct = df[(df["team_a"] == team_a) & (df["team_b"] == team_b)]
    reversed_match = df[(df["team_a"] == team_b) & (df["team_b"] == team_a)]

    if not direct.empty:
        row = direct.iloc[0]
        oa, od, ob = row["closing_home_odds"], row["closing_draw_odds"], row["closing_away_odds"]
    elif not reversed_match.empty:
        row = reversed_match.iloc[0]
        ob, od, oa = row["closing_home_odds"], row["closing_draw_odds"], row["closing_away_odds"]
    else:
        return MarketOddsResult(None, None, None, research_valid=False)

    research_valid = bool(row["research_valid"]) and str(row["research_valid"]).lower() not in (
        "false", "0", ""
    )

    win_a, draw, win_b = _implied_probs(float(oa), float(od), float(ob))
    return MarketOddsResult(
        win_a=win_a, draw=draw, win_b=win_b,
        research_valid=research_valid,
        bookmaker=str(row.get("bookmaker", "")),
    )
