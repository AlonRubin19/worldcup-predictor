"""Daily Match Board — pure formatter functions for the Today's Matches tab.

No Streamlit imports — fully testable.

Data flow:
  Fixture + MatchPrediction → MatchBoardRow → format_board_row_as_dict → Streamlit df
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tournament.fixtures import Fixture


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchPrediction:
    """All pre-computed prediction outputs needed for one board row."""
    win_a: float
    draw: float
    win_b: float
    most_likely_score: str       # e.g. "1-0"
    over_25: float               # P(total goals > 2.5)
    btts_yes: float              # P(both teams score)
    top_signal: str              # e.g. "1X: 87.0%"
    top_signal_strength: str     # "Strong" | "Moderate" | "Weak"
    confidence_label: str        # "High" | "Medium" | "Low"
    is_research_valid: bool = True


@dataclass
class MatchBoardRow:
    """One fully-built row for the daily match board."""
    match_id: str
    date: str
    stage: str
    group: str
    team_a: str
    team_b: str
    win_a: float
    draw: float
    win_b: float
    most_likely_score: str
    over_25: float
    btts_yes: float
    top_signal: str
    top_signal_strength: str
    confidence_label: str
    is_research_valid: bool


# ─────────────────────────────────────────────────────────────────────────────
# Filter / sort
# ─────────────────────────────────────────────────────────────────────────────

def filter_fixtures_by_date(fixtures: list[Fixture], date_str: str) -> list[Fixture]:
    """Return all fixtures whose date field equals date_str (YYYY-MM-DD)."""
    return [f for f in fixtures if f.date == date_str]


def sort_matches_by_datetime(fixtures: list[Fixture]) -> list[Fixture]:
    """Return fixtures sorted by date ascending (stable sort)."""
    return sorted(fixtures, key=lambda f: f.date)


# ─────────────────────────────────────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────────────────────────────────────

def build_match_board_row(fixture: Fixture, prediction: MatchPrediction) -> MatchBoardRow:
    """Combine one Fixture with its MatchPrediction into a MatchBoardRow."""
    return MatchBoardRow(
        match_id=fixture.match_id,
        date=fixture.date,
        stage=fixture.stage,
        group=fixture.group,
        team_a=fixture.team_a,
        team_b=fixture.team_b,
        win_a=float(prediction.win_a),
        draw=float(prediction.draw),
        win_b=float(prediction.win_b),
        most_likely_score=prediction.most_likely_score,
        over_25=float(prediction.over_25),
        btts_yes=float(prediction.btts_yes),
        top_signal=prediction.top_signal,
        top_signal_strength=prediction.top_signal_strength,
        confidence_label=prediction.confidence_label,
        is_research_valid=prediction.is_research_valid,
    )


def build_daily_match_rows(
    fixtures: list[Fixture],
    predictions: dict[str, MatchPrediction],
) -> list[MatchBoardRow]:
    """Build all board rows for a set of fixtures.

    Fixtures without a matching prediction entry are silently skipped.

    Args:
        fixtures: List of Fixture objects to display.
        predictions: Dict keyed by match_id → MatchPrediction.

    Returns:
        List of MatchBoardRow, one per fixture that has a prediction.
    """
    rows = []
    for f in fixtures:
        pred = predictions.get(f.match_id)
        if pred is None:
            continue
        rows.append(build_match_board_row(f, pred))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Display formatter
# ─────────────────────────────────────────────────────────────────────────────

_CONF_EMOJI = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
_SIG_EMOJI  = {"Strong": "🟢", "Moderate": "🟡", "Weak": "🔴"}


def format_board_row_as_dict(row: MatchBoardRow) -> dict:
    """Format a MatchBoardRow into a flat dict suitable for a Streamlit dataframe."""
    conf_badge = f"{_CONF_EMOJI.get(row.confidence_label, '')} {row.confidence_label}"
    sig_badge  = f"{_SIG_EMOJI.get(row.top_signal_strength, '')} {row.top_signal_strength}"
    validity   = "Research-valid" if row.is_research_valid else "⚠ Engineering-valid"

    group_label = f"Group {row.group}" if row.group else row.stage.replace("_", " ").title()

    return {
        "Date":          row.date,
        "Stage":         group_label,
        "Match":         f"{row.team_a} vs {row.team_b}",
        f"{row.team_a} Win": f"{row.win_a:.1%}",
        "Draw":          f"{row.draw:.1%}",
        f"{row.team_b} Win": f"{row.win_b:.1%}",
        "Score":         row.most_likely_score,
        "O/U 2.5":       f"{row.over_25:.1%}",
        "BTTS":          f"{row.btts_yes:.1%}",
        "Top Signal":    row.top_signal,
        "Signal":        sig_badge,
        "Confidence":    conf_badge,
        "Data":          validity,
    }
