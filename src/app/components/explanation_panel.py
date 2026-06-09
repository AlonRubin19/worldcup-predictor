"""Explanation panel — pure-data logic layer + Streamlit rendering."""

from dataclasses import dataclass
from src.explainability.driver import DriverContribution, PredictionExplanation


@dataclass
class ValidityBadge:
    label: str
    color: str    # "green" | "orange"
    tooltip: str


def format_driver_table(drivers: list[DriverContribution]) -> list[dict]:
    """Convert driver list to table rows for display.

    Dixon-Coles adjustment (magnitude=0, direction=neutral) is shown as
    context-only with no numeric magnitude.
    """
    rows = []
    for d in drivers:
        is_context = d.name == "Dixon-Coles adjustment" and d.magnitude == 0.0

        direction_symbol = {
            "positive": "+ positive",
            "negative": "- negative",
            "neutral":  "  neutral",
        }.get(d.direction, d.direction)

        rows.append({
            "Driver":    d.name,
            "Team":      d.team,
            "Direction": direction_symbol,
            "Magnitude": "Context only" if is_context else f"{d.magnitude:.4f}",
            "Detail":    d.description,
        })
    return rows


def format_validity_badge(is_research_valid: bool, warnings: list[str]) -> ValidityBadge:
    """Return a ValidityBadge describing data validity status."""
    if is_research_valid and not warnings:
        return ValidityBadge(
            label="Research-valid",
            color="green",
            tooltip=(
                "This prediction uses historical ELO and MLE-fitted attack/defense "
                "parameters from the validated backtest pipeline."
            ),
        )

    warning_text = " | ".join(warnings) if warnings else "Some data is not research-valid."
    return ValidityBadge(
        label="Engineering-valid only",
        color="orange",
        tooltip=warning_text,
    )


# ── Streamlit rendering ───────────────────────────────────────────────────────

def render_explanation_panel(expl: PredictionExplanation, is_research_valid: bool) -> None:
    """Render the explainability panel in the Streamlit app."""
    import streamlit as st
    import pandas as pd
    from src.explainability.report import generate_report

    badge = format_validity_badge(is_research_valid, expl.warnings)
    _icon = "✅" if badge.color == "green" else "⚠️"
    st.caption(f"{_icon} Data validity: **{badge.label}** — {badge.tooltip}")

    st.markdown("**Key Reasons**")
    st.info(generate_report(expl))

    if expl.drivers:
        with st.expander("Driver details", expanded=False):
            rows = format_driver_table(expl.drivers)
            st.table(pd.DataFrame(rows)[["Driver", "Team", "Direction", "Magnitude"]])

    for w in expl.warnings:
        st.warning(w)
