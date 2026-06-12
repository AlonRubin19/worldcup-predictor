from src.tournament.group_simulation import simulate_group_stage
from src.tournament.bracket_2026 import GROUPS_2026


def test_simulate_group_stage_returns_tables_for_all_groups():
    result = simulate_group_stage()

    assert len(result.fixture_results) == 72
    assert set(result.group_tables.keys()) == set(GROUPS_2026)

    for g, table in result.group_tables.items():
        assert len(table) == 4
        teams_in_table = {s.team for s in table}
        assert len(teams_in_table) == 4

    # Table should be sorted by points (descending) as a basic sanity check.
    for table in result.group_tables.values():
        points = [s.points for s in table]
        assert points == sorted(points, reverse=True)


def test_simulate_group_stage_qualification():
    result = simulate_group_stage()
    assert len(result.qualified) == 32
    assert len(set(result.qualified)) == 32


def test_fixture_results_have_predictions():
    result = simulate_group_stage()
    sample = result.fixture_results[0]
    assert sample.prediction.team1 == sample.team_a
    assert sample.goals_a >= 0 and sample.goals_b >= 0
