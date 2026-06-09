"""Load latest ELO and PPG snapshot for each team from match_results.csv."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "match_results.csv"


@dataclass
class TeamSnapshot:
    elo: float
    ppg: float


def load_team_snapshots(path: Path | None = None) -> dict[str, TeamSnapshot]:
    """Return the most recent ELO and PPG for every team in match_results.csv.

    For each team, finds the latest row where the team appears (as team_a or
    team_b), and reads the pre-match ELO and points-per-game from that row.

    Returns:
        {team_name: TeamSnapshot(elo, ppg)}

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT
    if not Path(p).exists():
        raise FileNotFoundError(f"match_results.csv not found: {p}")

    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date")

    snapshots: dict[str, TeamSnapshot] = {}

    # Build long-form: each row contributes one entry per team
    rows_a = df[["date", "team_a", "team_a_elo_pre", "team_a_points_per_game_last_10"]].rename(
        columns={
            "team_a": "team",
            "team_a_elo_pre": "elo",
            "team_a_points_per_game_last_10": "ppg",
        }
    )
    rows_b = df[["date", "team_b", "team_b_elo_pre", "team_b_points_per_game_last_10"]].rename(
        columns={
            "team_b": "team",
            "team_b_elo_pre": "elo",
            "team_b_points_per_game_last_10": "ppg",
        }
    )

    combined = pd.concat([rows_a, rows_b], ignore_index=True)
    combined = combined.sort_values("date")

    # Keep only the last row per team (latest date)
    latest = combined.groupby("team").last().reset_index()

    for _, row in latest.iterrows():
        snapshots[row["team"]] = TeamSnapshot(
            elo=float(row["elo"]),
            ppg=float(row["ppg"]),
        )

    return snapshots
