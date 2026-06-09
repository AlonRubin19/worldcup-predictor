"""Match odds rows to match_results.csv match IDs using bidirectional team lookup.

Handles the case where the odds source lists teams in reverse order
relative to match_results.csv (home/away conventions differ).
When teams are reversed, home/away odds are swapped accordingly.
"""

from dataclasses import dataclass
import pandas as pd

from src.data.market_sources.base import MarketOddsRow


@dataclass
class OddsMatchResult:
    matched: list[MarketOddsRow]
    unmatched: list[dict]
    total_odds_rows: int
    matched_count: int
    unmatched_count: int


def match_odds_to_match_ids(
    odds_rows: list[MarketOddsRow],
    match_results_df: pd.DataFrame,
) -> OddsMatchResult:
    """Resolve each odds row to a match_id from match_results_df.

    Tries (date, team_a, team_b) first. On failure tries reversed teams
    and swaps home/away odds to maintain team_a orientation.

    Args:
        odds_rows: Raw odds rows from any adapter (match_id may be empty).
        match_results_df: DataFrame from match_results.csv with at minimum
                          columns: match_id, date, team_a, team_b.

    Returns:
        OddsMatchResult with matched rows (match_id filled in) and
        unmatched rows listed by {date, team_a, team_b}.
    """
    matched: list[MarketOddsRow] = []
    unmatched: list[dict] = []

    for row in odds_rows:
        resolved = _resolve(row, match_results_df)
        if resolved is not None:
            matched.append(resolved)
        else:
            unmatched.append({
                "date": row.date,
                "team_a": row.team_a,
                "team_b": row.team_b,
            })

    return OddsMatchResult(
        matched=matched,
        unmatched=unmatched,
        total_odds_rows=len(odds_rows),
        matched_count=len(matched),
        unmatched_count=len(unmatched),
    )


def _resolve(row: MarketOddsRow, df: pd.DataFrame) -> "MarketOddsRow | None":
    # Try exact order
    hit = df[
        (df["date"] == row.date) &
        (df["team_a"] == row.team_a) &
        (df["team_b"] == row.team_b)
    ]
    if not hit.empty:
        from dataclasses import replace
        return replace(row, match_id=str(int(hit.iloc[0]["match_id"])))

    # Try reversed order — swap home/away odds to restore team_a orientation
    hit = df[
        (df["date"] == row.date) &
        (df["team_a"] == row.team_b) &
        (df["team_b"] == row.team_a)
    ]
    if not hit.empty:
        from dataclasses import replace
        return replace(
            row,
            match_id=str(int(hit.iloc[0]["match_id"])),
            team_a=row.team_b,
            team_b=row.team_a,
            opening_home_odds=row.opening_away_odds,
            opening_draw_odds=row.opening_draw_odds,
            opening_away_odds=row.opening_home_odds,
            closing_home_odds=row.closing_away_odds,
            closing_draw_odds=row.closing_draw_odds,
            closing_away_odds=row.closing_home_odds,
        )

    return None
