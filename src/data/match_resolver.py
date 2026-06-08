"""Bidirectional match resolver for joining historical_matches.csv with match_results.csv.

Handles the fact that martj42 records the host/home team as team_a, while
historical_matches.csv uses a different team ordering. For 25 of 40 WC 2022 matches
the teams appear reversed between the two sources.

Usage:
    resolved, unresolved = resolve_all_matches(historical_df, match_results_df)
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class ResolvedMatchStats:
    """Pre-match stats aligned to historical_matches.csv team ordering."""
    date: str
    team_a: str
    team_b: str
    team_a_elo_pre: float
    team_b_elo_pre: float
    team_a_goals_for_last_10: float
    team_a_goals_against_last_10: float
    team_b_goals_for_last_10: float
    team_b_goals_against_last_10: float
    team_a_points_per_game_last_10: float
    team_b_points_per_game_last_10: float
    team_a_matches_available: int
    team_b_matches_available: int
    was_reversed: bool  # True if found by swapping team_a/team_b in match_results


def resolve_match(
    date: str,
    team_a: str,
    team_b: str,
    match_results_df: pd.DataFrame,
) -> "ResolvedMatchStats | None":
    """Look up pre-match stats for a given (date, team_a, team_b) triplet.

    Tries (team_a, team_b) first. If not found, tries (team_b, team_a) and
    swaps all columns to restore the team_a/team_b alignment requested.

    Args:
        date: ISO date string (YYYY-MM-DD).
        team_a: First team (per historical_matches.csv convention).
        team_b: Second team (per historical_matches.csv convention).
        match_results_df: DataFrame from match_results.csv.

    Returns:
        ResolvedMatchStats with was_reversed flag, or None if not found in either order.
    """
    # Try exact order
    row = match_results_df[
        (match_results_df["date"] == date) &
        (match_results_df["team_a"] == team_a) &
        (match_results_df["team_b"] == team_b)
    ]
    if not row.empty:
        r = row.iloc[0]
        return ResolvedMatchStats(
            date=date, team_a=team_a, team_b=team_b,
            team_a_elo_pre=float(r["team_a_elo_pre"]),
            team_b_elo_pre=float(r["team_b_elo_pre"]),
            team_a_goals_for_last_10=float(r["team_a_goals_for_last_10"]),
            team_a_goals_against_last_10=float(r["team_a_goals_against_last_10"]),
            team_b_goals_for_last_10=float(r["team_b_goals_for_last_10"]),
            team_b_goals_against_last_10=float(r["team_b_goals_against_last_10"]),
            team_a_points_per_game_last_10=float(r["team_a_points_per_game_last_10"]),
            team_b_points_per_game_last_10=float(r["team_b_points_per_game_last_10"]),
            team_a_matches_available=int(r["team_a_matches_available"]),
            team_b_matches_available=int(r["team_b_matches_available"]),
            was_reversed=False,
        )

    # Try reversed order — swap team_a/team_b in the lookup
    row = match_results_df[
        (match_results_df["date"] == date) &
        (match_results_df["team_a"] == team_b) &
        (match_results_df["team_b"] == team_a)
    ]
    if not row.empty:
        r = row.iloc[0]
        # Swap all team_a/team_b columns back to requested orientation
        return ResolvedMatchStats(
            date=date, team_a=team_a, team_b=team_b,
            team_a_elo_pre=float(r["team_b_elo_pre"]),           # swapped
            team_b_elo_pre=float(r["team_a_elo_pre"]),           # swapped
            team_a_goals_for_last_10=float(r["team_b_goals_for_last_10"]),      # swapped
            team_a_goals_against_last_10=float(r["team_b_goals_against_last_10"]),  # swapped
            team_b_goals_for_last_10=float(r["team_a_goals_for_last_10"]),      # swapped
            team_b_goals_against_last_10=float(r["team_a_goals_against_last_10"]),  # swapped
            team_a_points_per_game_last_10=float(r["team_b_points_per_game_last_10"]),  # swapped
            team_b_points_per_game_last_10=float(r["team_a_points_per_game_last_10"]),  # swapped
            team_a_matches_available=int(r["team_b_matches_available"]),   # swapped
            team_b_matches_available=int(r["team_a_matches_available"]),   # swapped
            was_reversed=True,
        )

    return None


def resolve_all_matches(
    historical_df: pd.DataFrame,
    match_results_df: pd.DataFrame,
) -> tuple[list[ResolvedMatchStats], list[dict]]:
    """Resolve every row in historical_df against match_results_df.

    Args:
        historical_df: DataFrame from historical_matches.csv with columns:
                       date, team_a, team_b (at minimum).
        match_results_df: DataFrame from match_results.csv.

    Returns:
        (resolved, unresolved)
        resolved: list of ResolvedMatchStats for each successfully matched row.
        unresolved: list of dicts {date, team_a, team_b} for rows with no match.
    """
    resolved = []
    unresolved = []

    for _, row in historical_df.iterrows():
        stats = resolve_match(
            str(row["date"]),
            str(row["team_a"]),
            str(row["team_b"]),
            match_results_df,
        )
        if stats is not None:
            resolved.append(stats)
        else:
            unresolved.append({
                "date": str(row["date"]),
                "team_a": str(row["team_a"]),
                "team_b": str(row["team_b"]),
            })

    return resolved, unresolved
