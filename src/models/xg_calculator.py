BASE_XG = 1.35   # average goals per team per match (historical baseline)
XG_MIN  = 0.2    # floor — prevents degenerate near-zero predictions
XG_MAX  = 4.5    # ceiling — prevents unrealistic blowout predictions


def calculate_xg(ratings_a: dict, ratings_b: dict) -> tuple[float, float]:
    """Calculate expected goals for both teams from their ratings.

    Uses a multiplicative formula where each factor adjusts the base xG:
      - attack_rating: how well the team creates chances
      - defense_rating (opponent's): how well the opponent suppresses chances
      - form_rating: recent match form
      - squad_rating: overall squad quality
      - elo_factor: relative strength adjustment from ELO difference

    The elo_factor for team A and B are mirrors: if A gets a boost, B gets
    an equal reduction, ensuring elo_factor + (2 - elo_factor) = 2 (constant sum).

    Args:
        ratings_a: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating
        ratings_b: dict with keys elo, attack_rating, defense_rating, form_rating, squad_rating

    Returns:
        (xg_a, xg_b) — both clamped to [XG_MIN, XG_MAX]
    """
    elo_factor = 1 + ((ratings_a["elo"] - ratings_b["elo"]) / 4000)

    xg_a = (
        BASE_XG
        * ratings_a["attack_rating"]
        * ratings_b["defense_rating"]   # opponent defense affects how much A can score
        * ratings_a["form_rating"]
        * ratings_a["squad_rating"]
        * elo_factor
    )

    xg_b = (
        BASE_XG
        * ratings_b["attack_rating"]
        * ratings_a["defense_rating"]   # opponent defense affects how much B can score
        * ratings_b["form_rating"]
        * ratings_b["squad_rating"]
        * (2 - elo_factor)              # mirror: ensures symmetry at equal ELO
    )

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b))),
    )
