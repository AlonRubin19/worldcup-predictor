"""Regression tests for app.py runtime crashes (NameError, etc.).

Uses Streamlit's AppTest to actually run the app script and click the
Tournament Simulation / Golden Boot buttons, catching any NameError or
other exception raised during the run (e.g. the ``datetime`` /
``_dt`` alias mismatch that previously crashed both pages).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).parent.parent.parent / "src" / "app" / "app.py"


def _assert_no_exceptions(at: AppTest) -> None:
    for block in at.exception:
        raise AssertionError(f"App raised an exception: {block.value!r}")


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(_APP_PATH)).run(timeout=60)
    _assert_no_exceptions(at)


def test_tournament_simulation_run_does_not_crash():
    at = AppTest.from_file(str(_APP_PATH)).run(timeout=60)
    _assert_no_exceptions(at)

    run_btn = next(b for b in at.button if b.label == "Run Tournament Simulation")
    run_btn.click().run(timeout=120)

    _assert_no_exceptions(at)
    assert "mc_run_at" in at.session_state
    assert "mc_result" in at.session_state


def test_golden_boot_projection_does_not_crash():
    at = AppTest.from_file(str(_APP_PATH)).run(timeout=60)
    _assert_no_exceptions(at)

    run_btn = next(b for b in at.button if b.label == "Run Golden Boot Projection")
    run_btn.click().run(timeout=120)

    _assert_no_exceptions(at)
