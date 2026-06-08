import pandas as pd
from pathlib import Path

# Path to the teams CSV relative to this file's location.
# Change this one line (or replace the whole function) to load from a DB or API.
_TEAMS_CSV = Path(__file__).parent.parent.parent / "data" / "teams.csv"


def load_teams(csv_path: Path | None = None) -> list[str]:
    """Load national team names from the local CSV.

    Returns a sorted list of team name strings.
    Raises FileNotFoundError if teams.csv is missing.
    Raises ValueError if the CSV is empty or missing the 'team' column.

    Args:
        csv_path: Optional path override. Defaults to data/teams.csv relative to this file.
                  Pass a custom path in tests to avoid touching the real file.
    """
    path = csv_path if csv_path is not None else _TEAMS_CSV

    if not path.exists():
        raise FileNotFoundError(f"Teams data file not found: {path}")

    df = pd.read_csv(path)

    if "team" not in df.columns:
        raise ValueError("teams.csv must have a 'team' column header")

    teams = df["team"].dropna().str.strip().tolist()

    if not teams:
        raise ValueError("teams.csv contains no team entries")

    return sorted(teams)
