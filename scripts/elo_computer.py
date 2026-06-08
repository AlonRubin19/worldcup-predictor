"""Pure ELO computation logic — no I/O, no network calls, easily testable."""

K_FACTOR = 60        # Standard for international football
STARTING_ELO = 1600  # FIFA-style base rating


def compute_expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that team A wins given ELO ratings.

    Uses the standard ELO formula: E_a = 1 / (1 + 10^((elo_b - elo_a) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float,
    elo_b: float,
    win_a: bool,
    draw: bool,
) -> tuple[float, float]:
    """Apply one match result to ELO ratings.

    Args:
        elo_a: Team A's current ELO (PRE-match).
        elo_b: Team B's current ELO (PRE-match).
        win_a: True if team A won.
        draw:  True if the match was a draw (win_a ignored when True).

    Returns:
        (new_elo_a, new_elo_b)
    """
    expected_a = compute_expected_score(elo_a, elo_b)
    expected_b = 1.0 - expected_a

    if draw:
        actual_a, actual_b = 0.5, 0.5
    elif win_a:
        actual_a, actual_b = 1.0, 0.0
    else:
        actual_a, actual_b = 0.0, 1.0

    new_elo_a = elo_a + K_FACTOR * (actual_a - expected_a)
    new_elo_b = elo_b + K_FACTOR * (actual_b - expected_b)
    return new_elo_a, new_elo_b


def compute_elo_history(matches: list[dict]) -> list[dict]:
    """Compute chronological ELO history from a list of match dicts.

    Args:
        matches: List of dicts with keys:
                 date (str ISO), home_team (str), away_team (str),
                 home_score (int), away_score (int).
                 Must be sorted chronologically (oldest first).

    Returns:
        List of dicts: {date, team, elo_pre}
        One row per team per match, elo_pre = ELO BEFORE this match's result.
        This is the anti-leakage guarantee: no future data is embedded.
    """
    ratings: dict[str, float] = {}  # team -> current ELO
    history: list[dict] = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]

        # Initialize new teams at starting ELO
        if home not in ratings:
            ratings[home] = STARTING_ELO
        if away not in ratings:
            ratings[away] = STARTING_ELO

        elo_home_pre = ratings[home]
        elo_away_pre = ratings[away]

        # CRITICAL: Record pre-match ELO BEFORE updating
        history.append({"date": match["date"], "team": home, "elo_pre": elo_home_pre})
        history.append({"date": match["date"], "team": away, "elo_pre": elo_away_pre})

        # Determine outcome
        home_goals = match["home_score"]
        away_goals = match["away_score"]
        win_home = home_goals > away_goals
        draw = home_goals == away_goals

        # Update ratings for NEXT match
        ratings[home], ratings[away] = update_elo(
            elo_home_pre, elo_away_pre,
            win_a=win_home,
            draw=draw,
        )

    return history
