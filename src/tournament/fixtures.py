"""Load and represent tournament fixture data."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_VALID_STAGES = {"group", "round_of_16", "quarter_final", "semi_final", "final"}

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"


@dataclass
class Fixture:
    match_id: str
    stage: str
    group: str        # empty string for knockout matches
    date: str
    team_a: str
    team_b: str
    status: str = "NS"   # API status short code; "NS"=not started, "FT"=finished, etc.


def load_fixtures(path: Path | None = None) -> list[Fixture]:
    """Load tournament fixtures from CSV.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = Path(path) if path is not None else _DEFAULT
    if not p.exists():
        raise FileNotFoundError(f"Fixture file not found: {p}")

    df = pd.read_csv(p)
    df = df.fillna({"group": ""})

    return [
        Fixture(
            match_id=str(row["match_id"]),
            stage=str(row["stage"]),
            group=str(row["group"]),
            date=str(row["date"]),
            team_a=str(row["team_a"]),
            team_b=str(row["team_b"]),
        )
        for _, row in df.iterrows()
    ]
