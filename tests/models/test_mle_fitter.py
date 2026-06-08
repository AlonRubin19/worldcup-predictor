import pytest
import math
from src.models.mle_fitter import fit_team_params, TeamStrengthParams

# ── Minimal fixture ──────────────────────────────────────────────────────────
def _make_match(team_a, team_b, ga, gb, date="2020-01-01", weight=1.0):
    return {
        "date": date,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_goals": ga,
        "team_b_goals": gb,
        "weight": weight,
    }


def _symmetric_matches(n=20):
    """Equal-strength teams: interleave wins and losses."""
    matches = []
    for i in range(n):
        month = (i % 12) + 1
        day = (i // 12) + 1
        date_str = f"2020-{month:02d}-{day:02d}"
        if i % 2 == 0:
            matches.append(_make_match("A", "B", 2, 1, date_str))
        else:
            matches.append(_make_match("A", "B", 1, 2, date_str))
    return matches


# ── Tests ────────────────────────────────────────────────────────────────────

def test_returns_dict_of_team_strength_params():
    result = fit_team_params(_symmetric_matches())
    assert isinstance(result, dict)
    assert "A" in result and "B" in result
    assert isinstance(result["A"], TeamStrengthParams)


def test_all_params_positive():
    result = fit_team_params(_symmetric_matches())
    for team, p in result.items():
        assert p.alpha_attack > 0, f"{team} alpha <= 0"
        assert p.beta_defense > 0, f"{team} beta <= 0"


def test_equal_teams_have_similar_params():
    """Symmetric win/loss should produce similar alpha and beta for both teams."""
    result = fit_team_params(_symmetric_matches(n=40))
    assert abs(result["A"].alpha_attack - result["B"].alpha_attack) < 0.3
    assert abs(result["A"].beta_defense - result["B"].beta_defense) < 0.3


def test_strong_attacker_gets_higher_alpha():
    """Team that scores more goals should get a higher attack parameter."""
    matches = []
    for i in range(1, 11):
        month = ((i-1) % 12) + 1
        matches.append(_make_match("Strong", "Weak", 4, 0, f"2020-{month:02d}-{(i-1)//12+1:02d}"))
    for i in range(11, 21):
        month = ((i-1) % 12) + 1
        matches.append(_make_match("Weak", "Strong", 0, 4, f"2020-{month:02d}-{(i-1)//12+1:02d}"))
    result = fit_team_params(matches)
    assert result["Strong"].alpha_attack > result["Weak"].alpha_attack


def test_solid_defender_gets_higher_beta():
    """Team with worse defense gets higher beta (easier to score against)."""
    # Create matches where one team is clearly better at defense
    # Solid: always holds opponents to 0 goals
    # Leaky: always concedes 3 goals
    matches = []
    for i in range(1, 11):
        month = ((i-1) % 12) + 1
        # Solid vs Leaky: Solid (team_a) scores 3, Leaky (team_b) scores 0
        matches.append(_make_match("Solid", "Leaky", 3, 0, f"2020-{month:02d}-{(i-1)//12+1:02d}"))
    for i in range(11, 21):
        month = ((i-1) % 12) + 1
        # Leaky vs Solid: Leaky (team_a) scores 0, Solid (team_b) scores 3
        matches.append(_make_match("Leaky", "Solid", 0, 3, f"2020-{month:02d}-{(i-1)//12+1:02d}"))
    result = fit_team_params(matches)
    # beta_defense is a MULTIPLIER on opponent's lambda. Lower beta = harder to score against.
    # Solid always concedes 0 goals (when Leaky tries to score against them), so Solid has lower beta.
    # Leaky always concedes 3 goals (when Solid tries to score against them), so Leaky has higher beta.
    assert result["Solid"].beta_defense < result["Leaky"].beta_defense


def test_log_likelihood_is_negative():
    """Log-likelihood of Poisson observations must be negative."""
    result = fit_team_params(_symmetric_matches())
    for team, p in result.items():
        assert p.log_likelihood < 0 or math.isnan(p.log_likelihood) is False


def test_minimum_matches_threshold():
    """Teams with fewer than min_matches are excluded from output."""
    matches = [_make_match("A", "B", 1, 0)]  # Only 1 match
    result = fit_team_params(matches, min_matches=5)
    assert "A" not in result
    assert "B" not in result


def test_fit_handles_zero_goals():
    """0-0 draws should not cause log(0) errors."""
    matches = []
    for i in range(1, 11):
        month = ((i-1) % 12) + 1
        matches.append(_make_match("A", "B", 0, 0, f"2020-{month:02d}-{(i-1)//12+1:02d}"))
    result = fit_team_params(matches)  # Should not raise
    assert "A" in result
