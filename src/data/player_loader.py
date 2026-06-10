"""Load player profiles and match availability from CSV files."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT_PROFILES = Path(__file__).parent.parent.parent / "data" / "player_profiles.csv"
_DEFAULT_AVAILABILITY = Path(__file__).parent.parent.parent / "data" / "match_player_availability.csv"


@dataclass
class PlayerProfile:
    player_id: str
    player_name: str
    team: str
    position: str
    club: str
    minutes_last_90_days: float
    national_team_minutes_last_12_months: float
    goals_per_90: float
    assists_per_90: float
    xg_per_90: float
    xa_per_90: float
    defensive_actions_per_90: float
    international_caps: int
    base_impact_score: float
    source_type: str = "placeholder"
    research_valid: bool = False
    penalty_taker: bool = False


VALID_SOURCE_TYPES = frozenset(
    {"placeholder", "historical_lineup", "pre_match_report", "manual_assumption"}
)


@dataclass
class PlayerAvailability:
    match_id: str
    date: str
    team: str
    player_id: str
    expected_starter: bool
    availability_status: str
    availability_factor: float
    form_factor: float
    source_type: str = "placeholder"
    research_valid: bool = False


def load_player_profiles(path: Path | None = None) -> dict[str, PlayerProfile]:
    """Load player profiles from CSV, keyed by player_id.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT_PROFILES
    if not Path(p).exists():
        raise FileNotFoundError(f"player_profiles.csv not found: {p}")

    df = pd.read_csv(p)
    return {
        str(row["player_id"]): PlayerProfile(
            player_id=str(row["player_id"]),
            player_name=str(row["player_name"]),
            team=str(row["team"]),
            position=str(row["position"]),
            club=str(row["club"]),
            minutes_last_90_days=float(row["minutes_last_90_days"]),
            national_team_minutes_last_12_months=float(row["national_team_minutes_last_12_months"]),
            goals_per_90=float(row["goals_per_90"]),
            assists_per_90=float(row["assists_per_90"]),
            xg_per_90=float(row["xg_per_90"]),
            xa_per_90=float(row["xa_per_90"]),
            defensive_actions_per_90=float(row["defensive_actions_per_90"]),
            international_caps=int(row["international_caps"]),
            base_impact_score=float(row["base_impact_score"]),
            source_type=str(row["source_type"]) if "source_type" in df.columns else "placeholder",
            research_valid=bool(row["research_valid"]) if "research_valid" in df.columns else False,
            penalty_taker=bool(row["penalty_taker"]) if "penalty_taker" in df.columns else False,
        )
        for _, row in df.iterrows()
    }


def load_match_availability(path: Path | None = None) -> list[PlayerAvailability]:
    """Load match player availability records from CSV.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT_AVAILABILITY
    if not Path(p).exists():
        raise FileNotFoundError(f"match_player_availability.csv not found: {p}")

    df = pd.read_csv(p)
    has_source_type = "source_type" in df.columns
    has_research_valid = "research_valid" in df.columns

    records = []
    for _, row in df.iterrows():
        starter_val = str(row["expected_starter"]).strip().lower()

        source_type = "placeholder"
        if has_source_type:
            source_type = str(row["source_type"]).strip()

        research_valid = False
        if has_research_valid:
            rv = str(row["research_valid"]).strip().lower()
            research_valid = rv in ("true", "1", "yes")

        records.append(PlayerAvailability(
            match_id=str(row["match_id"]),
            date=str(row["date"]),
            team=str(row["team"]),
            player_id=str(row["player_id"]),
            expected_starter=starter_val in ("true", "1", "yes"),
            availability_status=str(row["availability_status"]),
            availability_factor=float(row["availability_factor"]),
            form_factor=float(row["form_factor"]),
            source_type=source_type,
            research_valid=research_valid,
        ))
    return records
