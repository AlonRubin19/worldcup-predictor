"""Base adapter interface and shared types for bookmaker odds sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


class DataNotAvailableError(Exception):
    """Raised when an adapter cannot provide the requested data."""


class MissingCredentialsError(Exception):
    """Raised when required credentials (API key, token) are absent."""


@dataclass
class MarketOddsRow:
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["research_valid"] = str(d["research_valid"]).lower()
        return d


class OddsAdapter(ABC):
    """Abstract base for all bookmaker odds source adapters."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier string for this source (used in source_type column)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this adapter can currently fetch data."""

    @abstractmethod
    def fetch_wc2022(self) -> list[MarketOddsRow]:
        """Fetch historical odds for all 2022 FIFA World Cup matches.

        Raises:
            DataNotAvailableError: if source does not carry WC 2022 data.
            MissingCredentialsError: if credentials are required but absent.
        """
