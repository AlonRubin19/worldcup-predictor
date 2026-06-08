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


_RATINGS_CSV = Path(__file__).parent.parent.parent / "data" / "team_ratings.csv"

_REQUIRED_RATING_COLUMNS = {
    "team", "elo", "attack_rating", "defense_rating", "form_rating", "squad_rating"
}


def load_team_ratings(ratings_path: Path | None = None) -> dict[str, dict]:
    """Load team ratings from the local CSV.

    Returns a dict keyed by team name, each value a dict of rating fields:
        {"elo": 2070, "attack_rating": 1.15, "defense_rating": 0.88,
         "form_rating": 1.05, "squad_rating": 1.10}

    Raises FileNotFoundError if the ratings CSV is missing.
    Raises ValueError if required columns are missing.
    """
    path = ratings_path if ratings_path is not None else _RATINGS_CSV

    if not path.exists():
        raise FileNotFoundError(f"Ratings data file not found: {path}")

    df = pd.read_csv(path)

    missing = _REQUIRED_RATING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"team_ratings.csv missing columns: {sorted(missing)}")

    df["team"] = df["team"].str.strip()
    return df.set_index("team")[list(_REQUIRED_RATING_COLUMNS - {"team"})].to_dict("index")
