"""SelectedFixture — session-state payload for the Today's Matches → Match Analyzer flow.

When a user clicks "Analyze in Match Analyzer" on the Today's Matches board,
a SelectedFixture is stored in st.session_state["selected_fixture"].

The Match Analyzer reads it to:
  - pre-select Team A and Team B
  - display a fixture banner (teams, date, source)
  - use the API fixture_id for live lineup fetching (when source_type="api")

All fields are str or None so Streamlit session_state can serialise them
without pickle issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.fixture_provider import FixtureSource
    from src.tournament.fixtures import Fixture


@dataclass
class SelectedFixture:
    """Serialisable snapshot of a fixture chosen from the Daily Match Board.

    Attributes:
        fixture_id:  API-Football fixture ID as a string (e.g. "855744"),
                     or the CSV match_id string.  None is not expected but
                     accepted defensively.
        source_type: "api" | "csv"
        team_a:      Home team internal name.
        team_b:      Away team internal name.
        date:        ISO date string, YYYY-MM-DD.
        stage:       Internal stage value (e.g. "group", "round_of_16").
        group:       Group letter if known (e.g. "A"), else "".
    """
    fixture_id:  str | None
    source_type: str
    team_a:      str
    team_b:      str
    date:        str
    stage:       str
    group:       str


# ── Factory ───────────────────────────────────────────────────────────────────

def create_selected_fixture(
    fixture: "Fixture",
    source_used: "FixtureSource",
) -> SelectedFixture:
    """Build a SelectedFixture from a Fixture and the provider source that produced it.

    Args:
        fixture:     Internal Fixture object (from fixture_provider or CSV loader).
        source_used: FixtureSource.API or FixtureSource.CSV (AUTO is never
                     passed here — callers resolve AUTO to the actual source first).

    Returns:
        SelectedFixture ready to store in st.session_state.
    """
    from src.data.fixture_provider import FixtureSource

    source_type = "api" if source_used is FixtureSource.API else "csv"

    return SelectedFixture(
        fixture_id=fixture.match_id,
        source_type=source_type,
        team_a=fixture.team_a,
        team_b=fixture.team_b,
        date=fixture.date,
        stage=fixture.stage,
        group=fixture.group,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_fixture_id(sf: SelectedFixture | None) -> int | None:
    """Return the API-Football fixture ID as an integer, or None.

    Returns None when:
      - sf is None
      - source_type != "api"
      - fixture_id is None or non-numeric (e.g. CSV match IDs like "m_001")
    """
    if sf is None:
        return None
    if sf.source_type != "api":
        return None
    if sf.fixture_id is None:
        return None
    try:
        return int(sf.fixture_id)
    except (ValueError, TypeError):
        return None


def is_valid_selected_fixture(sf: "SelectedFixture | None") -> bool:
    """Return True if sf is a usable fixture payload.

    A valid fixture has both team names non-empty and different.
    """
    if sf is None:
        return False
    if not isinstance(sf, SelectedFixture):
        return False
    if not sf.team_a or not sf.team_b:
        return False
    if sf.team_a == sf.team_b:
        return False
    return True
