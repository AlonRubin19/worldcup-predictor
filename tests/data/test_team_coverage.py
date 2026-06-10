from __future__ import annotations

from src.data.team_coverage import build_coverage_table, CoverageRow


def test_build_coverage_table_marks_mapped_and_unmapped_teams():
    teams = ["Spain", "France", "Atlantis"]
    api_ids = {"Spain": 9, "France": 2}

    table = build_coverage_table(teams, api_ids)

    assert len(table) == 3
    by_team = {r.team: r for r in table}
    assert by_team["Spain"].mapped is True
    assert by_team["Spain"].api_team_id == 9
    assert by_team["France"].mapped is True
    assert by_team["Atlantis"].mapped is False
    assert by_team["Atlantis"].api_team_id is None


def test_coverage_row_summary_counts_no_silent_drops():
    teams = ["Spain", "Atlantis"]
    api_ids = {"Spain": 9}

    table = build_coverage_table(teams, api_ids)

    assert sum(1 for r in table if r.mapped) == 1
    assert sum(1 for r in table if not r.mapped) == 1
    # every input team appears -- nothing silently missing
    assert {r.team for r in table} == set(teams)


def test_apply_refresh_results_fills_availability_flags():
    from src.data.refresh_pipeline import RefreshSummary, TeamRefreshResult
    from src.data.team_coverage import apply_refresh_results

    table = build_coverage_table(["Spain", "France"], {"Spain": 9, "France": 2})
    summary = RefreshSummary(
        timestamp="now",
        teams=[
            TeamRefreshResult(
                team="Spain", api_team_id=9,
                squad_count=5, squad_source="API-Football live squad (/players/squads)",
                injury_count=1, injury_source="API-Football injuries (/injuries)",
                stats_count=5, stats_source="API-Football player stats (/players)",
                used_live_data=True,
            ),
            TeamRefreshResult(
                team="France", api_team_id=2,
                squad_count=0, squad_source="Fallback: API-Football returned no squad data",
                injury_count=0, injury_source="Fallback: API-Football returned no squad data",
                stats_count=0, stats_source="Fallback: API-Football returned no squad data",
                used_live_data=False,
            ),
        ],
    )

    updated = apply_refresh_results(table, summary)
    by_team = {r.team: r for r in updated}
    assert by_team["Spain"].squad_available is True
    assert by_team["Spain"].injuries_available is True
    assert by_team["Spain"].player_stats_available is True
    assert by_team["France"].squad_available is False
    assert by_team["France"].injuries_available is False
    assert by_team["France"].player_stats_available is False
