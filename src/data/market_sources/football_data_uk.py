"""Adapter for football-data.co.uk historical odds CSVs.

INVESTIGATION RESULT (2026-06-09):
  football-data.co.uk covers domestic league odds for 31+ seasons
  across ~47 leagues. The site has recently added a WorldCup.xlsx file,
  but this is for WC 2026 (not WC 2022).

  World Cup 2022 (Qatar) odds are NOT available from this source.

  For WC 2022 odds use The Odds API (the_odds_api.py) — requires API key.
  See: https://the-odds-api.com/sports/fifa-world-cup-odds.html

  This adapter is still useful for domestic league calibration work
  and for future sprints using club-level matches.

CSV format (football-data.co.uk domestic leagues):
  Columns include: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
                   B365H, B365D, B365A, PSH, PSD, PSA, ...
  Date format: DD/MM/YY (two-digit year)
  Odds: decimal format, B365H/D/A = Bet365 home/draw/away
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data.market_sources.base import (
    OddsAdapter,
    MarketOddsRow,
    DataNotAvailableError,
)

_WC2022_UNAVAILABLE_MSG = (
    "World Cup 2022 odds are not available from football-data.co.uk. "
    "The site covers domestic leagues only. The WorldCup.xlsx file on "
    "the site is for WC 2026, not WC 2022. "
    "To obtain WC 2022 historical odds, use The Odds API "
    "(https://the-odds-api.com) with a paid API key."
)

_DATE_FORMATS = ["%d/%m/%y", "%d/%m/%Y"]


def _parse_date(raw: str) -> str:
    """Parse DD/MM/YY or DD/MM/YYYY → YYYY-MM-DD."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw.strip()


class FootballDataUKAdapter(OddsAdapter):
    """Adapter for football-data.co.uk domestic league CSV files."""

    @property
    def source_name(self) -> str:
        return "football_data_co_uk"

    def is_available(self) -> bool:
        return True  # No credentials required — public CSV downloads

    def fetch_wc2022(self) -> list[MarketOddsRow]:
        raise DataNotAvailableError(_WC2022_UNAVAILABLE_MSG)

    def parse_league_csv(
        self,
        path: Path,
        season: str,
        league: str,
        bookmaker_home_col: str = "B365H",
        bookmaker_draw_col: str = "B365D",
        bookmaker_away_col: str = "B365A",
        bookmaker_name: str = "Bet365",
    ) -> list[MarketOddsRow]:
        """Parse a domestic league CSV from football-data.co.uk.

        Args:
            path: Path to the CSV file.
            season: Season label (e.g. "2022-23"), stored for reference.
            league: League code (e.g. "E0"), stored for reference.
            bookmaker_*_col: Column names for the chosen bookmaker's odds.
            bookmaker_name: Human-readable bookmaker name.

        Returns:
            list[MarketOddsRow] with research_valid=True.

        Raises:
            FileNotFoundError: if CSV does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"League CSV not found: {path}")

        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            # Skip rows with missing odds
            if pd.isna(r.get(bookmaker_home_col)) or \
               pd.isna(r.get(bookmaker_draw_col)) or \
               pd.isna(r.get(bookmaker_away_col)):
                continue

            date_str = _parse_date(str(r["Date"]))
            home_odds = float(r[bookmaker_home_col])
            draw_odds = float(r[bookmaker_draw_col])
            away_odds = float(r[bookmaker_away_col])

            rows.append(MarketOddsRow(
                match_id="",  # domestic matches — no match_id assigned here
                date=date_str,
                team_a=str(r["HomeTeam"]),
                team_b=str(r["AwayTeam"]),
                bookmaker=bookmaker_name,
                opening_home_odds=home_odds,
                opening_draw_odds=draw_odds,
                opening_away_odds=away_odds,
                closing_home_odds=home_odds,  # single snapshot: opening == closing
                closing_draw_odds=draw_odds,
                closing_away_odds=away_odds,
                source_type=self.source_name,
                research_valid=True,
            ))

        return rows
