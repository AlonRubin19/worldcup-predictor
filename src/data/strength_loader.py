from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"


@dataclass
class StrengthParams:
    alpha_attack: float
    beta_defense: float
    matches_used: int


def load_strength_params(path: Path | None = None) -> dict[str, StrengthParams]:
    """Load MLE team strength parameters from CSV.

    Returns:
        {team_name: StrengthParams}

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT
    if not p.exists():
        raise FileNotFoundError(f"team_strength_params.csv not found: {p}")

    df = pd.read_csv(p)
    return {
        row["team"]: StrengthParams(
            alpha_attack=float(row["alpha_attack"]),
            beta_defense=float(row["beta_defense"]),
            matches_used=int(row["matches_used"]),
        )
        for _, row in df.iterrows()
    }
