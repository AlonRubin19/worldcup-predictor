"""Load bookmaker market odds from CSV."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "market_odds.csv"


@dataclass
class MarketOdds:
    match_id: str
    date: str
    team_a: str
    team_b: str
    bookmaker: str
    opening_home_odds: float
    opening_draw_odds: float
    opening_away_odds: float
    closing_home_odds: float
    closing_draw_odds: float
    closing_away_odds: float
    source_type: str
    research_valid: bool


def load_market_odds(path: Path | None = None) -> list[MarketOdds]:
    """Load market odds from CSV.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT
    if not Path(p).exists():
        raise FileNotFoundError(f"market_odds.csv not found: {p}")

    df = pd.read_csv(p)
    records = []
    for _, row in df.iterrows():
        rv = str(row["research_valid"]).strip().lower()
        records.append(MarketOdds(
            match_id=str(row["match_id"]),
            date=str(row["date"]),
            team_a=str(row["team_a"]),
            team_b=str(row["team_b"]),
            bookmaker=str(row["bookmaker"]),
            opening_home_odds=float(row["opening_home_odds"]),
            opening_draw_odds=float(row["opening_draw_odds"]),
            opening_away_odds=float(row["opening_away_odds"]),
            closing_home_odds=float(row["closing_home_odds"]),
            closing_draw_odds=float(row["closing_draw_odds"]),
            closing_away_odds=float(row["closing_away_odds"]),
            source_type=str(row["source_type"]).strip(),
            research_valid=rv in ("true", "1", "yes"),
        ))
    return records
