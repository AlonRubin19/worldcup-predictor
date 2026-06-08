from src.data.pre_match_loader import PreMatchStats

BASE_XG    = 1.35
XG_MIN     = 0.2
XG_MAX     = 4.5
FORM_BASE  = 0.85
FORM_SCALE = 0.30   # range above FORM_BASE (max form = 0.85 + 0.30 = 1.15)


def calculate_pre_match_xg(match: PreMatchStats) -> tuple[float, float]:
    """Calculate expected goals using only pre-match statistics.

    No manually estimated ratings are used. All factors come from:
    - Per-game goal averages over the last 10 matches
    - Per-game points average over the last 10 matches
    - Pre-match ELO ratings

    Formula:
        attack_a  = goals_for_a  / BASE_XG   (1.0 = average)
        defense_b = goals_against_b / BASE_XG (1.0 = average)
        attack_b  = goals_for_b  / BASE_XG
        defense_a = goals_against_a / BASE_XG

        form_a = FORM_BASE + (ppg_a / 3) * FORM_SCALE
        form_b = FORM_BASE + (ppg_b / 3) * FORM_SCALE

        elo_factor_a = 1 + (elo_a - elo_b) / 4000
        elo_factor_b = 1 + (elo_b - elo_a) / 4000

        xg_a = BASE_XG * attack_a * defense_b * form_a * elo_factor_a
        xg_b = BASE_XG * attack_b * defense_a * form_b * elo_factor_b

    Both values are clamped to [XG_MIN, XG_MAX].
    """
    attack_a  = match.team_a_goals_for_last_10     / BASE_XG
    defense_b = match.team_b_goals_against_last_10 / BASE_XG
    attack_b  = match.team_b_goals_for_last_10     / BASE_XG
    defense_a = match.team_a_goals_against_last_10 / BASE_XG

    form_a = FORM_BASE + (match.team_a_points_per_game_last_10 / 3) * FORM_SCALE
    form_b = FORM_BASE + (match.team_b_points_per_game_last_10 / 3) * FORM_SCALE

    elo_factor_a = 1 + (match.team_a_elo_pre - match.team_b_elo_pre) / 4000
    elo_factor_b = 1 + (match.team_b_elo_pre - match.team_a_elo_pre) / 4000

    xg_a = BASE_XG * attack_a * defense_b * form_a * elo_factor_a
    xg_b = BASE_XG * attack_b * defense_a * form_b * elo_factor_b

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
