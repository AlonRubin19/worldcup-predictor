from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT_CSV = Path(__file__).parent.parent.parent / "data" / "pre_match_team_stats.csv"

_REQUIRED_COLUMNS = {
    "match_id", "date", "team_a", "team_b",
    "team_a_elo_pre", "team_b_elo_pre",
    "team_a_goals_for_last_10", "team_a_goals_against_last_10",
    "team_b_goals_for_last_10", "team_b_goals_against_last_10",
    "team_a_points_per_game_last_10", "team_b_points_per_game_last_10",
    "team_a_matches_available", "team_b_matches_available",
    "team_a_goals", "team_b_goals",
}


@dataclass
class PreMatchStats:
    """Pre-match statistics for one match, derived only from pre-game records."""
    match_id: int
    date: str
    team_a: str
    team_b: str
    team_a_elo_pre: float
    team_b_elo_pre: float
    team_a_goals_for_last_10: float       # avg goals scored per game, last 10 matches
    team_a_goals_against_last_10: float   # avg goals conceded per game, last 10 matches
    team_b_goals_for_last_10: float
    team_b_goals_against_last_10: float
    team_a_points_per_game_last_10: float
    team_b_points_per_game_last_10: float
    team_a_matches_available: int         # actual matches found (<=10); flag if <min_matches
    team_b_matches_available: int
    team_a_goals: int                     # actual match result (for backtesting)
    team_b_goals: int


def load_pre_match_stats(
    path: Path | None = None,
    min_matches: int = 5,
    exclude_insufficient: bool = False,
) -> list[PreMatchStats]:
    """Load pre-match statistics from CSV.

    Args:
        path: Path to CSV. Defaults to data/pre_match_team_stats.csv.
        min_matches: Minimum matches_available to consider reliable.
        exclude_insufficient: If True, exclude rows where either team has
                              fewer than min_matches. If False (default),
                              include all rows.

    Raises:
        FileNotFoundError: if CSV is missing.
        ValueError: if required columns are missing.
    """
    p = path if path is not None else _DEFAULT_CSV

    if not p.exists():
        raise FileNotFoundError(f"Pre-match stats file not found: {p}")

    # comment='#' skips the WARNING header lines
    df = pd.read_csv(p, comment='#')

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"pre_match_team_stats.csv missing columns: {sorted(missing)}")

    if exclude_insufficient:
        df = df[
            (df["team_a_matches_available"] >= min_matches) &
            (df["team_b_matches_available"] >= min_matches)
        ]

    stats = []
    for _, row in df.iterrows():
        stats.append(PreMatchStats(
            match_id=int(row["match_id"]),
            date=str(row["date"]),
            team_a=str(row["team_a"]).strip(),
            team_b=str(row["team_b"]).strip(),
            team_a_elo_pre=float(row["team_a_elo_pre"]),
            team_b_elo_pre=float(row["team_b_elo_pre"]),
            team_a_goals_for_last_10=float(row["team_a_goals_for_last_10"]),
            team_a_goals_against_last_10=float(row["team_a_goals_against_last_10"]),
            team_b_goals_for_last_10=float(row["team_b_goals_for_last_10"]),
            team_b_goals_against_last_10=float(row["team_b_goals_against_last_10"]),
            team_a_points_per_game_last_10=float(row["team_a_points_per_game_last_10"]),
            team_b_points_per_game_last_10=float(row["team_b_points_per_game_last_10"]),
            team_a_matches_available=int(row["team_a_matches_available"]),
            team_b_matches_available=int(row["team_b_matches_available"]),
            team_a_goals=int(row["team_a_goals"]),
            team_b_goals=int(row["team_b_goals"]),
        ))

    return stats
