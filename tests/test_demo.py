"""Smoke tests for the Streamlit demo: it starts from committed files only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from second_opinion.budget import CostAssumptions
from second_opinion.demo import (
    alarm_threshold,
    budget_table,
    build_input,
    load_assets,
    simulated_day,
)

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app" / "streamlit_app.py"
ASSETS = REPO / "app" / "assets"
needs_assets = pytest.mark.skipif(not (ASSETS / "grader.joblib").exists(), reason="demo assets not built")


def toy_risk(n=200, seed=0):
    rng = np.random.default_rng(seed)
    unsafe = rng.random(n) < 0.15
    return pd.DataFrame({
        "risk_grader": unsafe + rng.normal(0, 0.5, n),
        "risk_criterion_only": rng.random(n),
        "unsafe": unsafe,
        "unsafe_contested": ~unsafe & (rng.random(n) < 0.2),
        "theme": "x",
    })


def test_alarm_threshold_matches_capacity():
    r = pd.Series(np.arange(100.0))
    assert alarm_threshold(r, 0.10) == 90.0
    assert (r >= alarm_threshold(r, 0.10)).sum() == 10


def test_budget_table_orders_policies():
    t = budget_table(toy_risk(), 0.1, False, CostAssumptions()).set_index("Policy")
    caught = t["Unsafe caught"]
    assert caught["Perfect ranking (upper bound)"] >= caught["Grader ranking"] > caught["Random sample (today)"]
    assert t["Review cost (EUR)"].nunique() == 1  # same doctor time for every policy
    on = budget_table(toy_risk(), 0.1, True, CostAssumptions()).set_index("Policy")
    assert on["Unsafe missed"].sum() > t["Unsafe missed"].sum()


def test_simulated_day_and_input():
    day = simulated_day(toy_risk(), threshold=0.5, n=50)
    assert len(day) == 50 and day["hour"].between(0, 23).all()
    row = build_input("cluster:x", "rubric", " hi ", " answer ")
    assert row.loc[0, "conversation"] == "user: hi" and row.loc[0, "n_turns"] == 1


@needs_assets
def test_assets_are_small_and_keep_the_canary():
    a = load_assets(ASSETS)
    assert 20 <= len(a["examples"]) <= 24 and len(a["criteria"]) == 12
    assert all(e["canary"].startswith("healthbench:") for e in a["examples"] + a["criteria"])
    assert (ASSETS / "NOTICE").exists()
    assert (ASSETS / "grader.joblib").stat().st_size < 50e6


@needs_assets
def test_shipped_model_reproduces_example_scores():
    import joblib

    from second_opinion.grader import predict_proba

    model = joblib.load(ASSETS / "grader.joblib")
    e = load_assets(ASSETS)["examples"][0]
    rows = pd.DataFrame([{"category": c["category"], "rubric": c["rubric"], "conversation": e["conversation"],
                          "n_turns": e["n_turns"], "completion": e["answer"]} for c in e["criteria"]])
    assert predict_proba(model, rows) == pytest.approx([c["p_met"] for c in e["criteria"]], abs=1e-9)


@needs_assets
def test_app_starts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception


@needs_assets
@pytest.mark.parametrize("screen", ["screen_try", "screen_monitor", "screen_budget", "screen_under_the_hood"])
def test_each_screen_runs(screen):
    from streamlit.testing.v1 import AppTest

    script = f"import sys; sys.path.insert(0, {str(APP.parent)!r}); import screens; screens.{screen}()"
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.header


@needs_assets
def test_grading_pasted_text():
    from streamlit.testing.v1 import AppTest

    script = f"import sys; sys.path.insert(0, {str(APP.parent)!r}); import screens; screens.screen_try()"
    at = AppTest.from_string(script, default_timeout=60).run()
    at.text_area[0].input("I have crushing chest pain and my left arm is numb.").run()
    at.text_area[1].input("Rest and drink water; it is probably muscle strain.").run()
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Model P(criterion met)" for m in at.metric)


def test_package_never_resolves_paths_from_its_own_location():
    """In a normal (non-editable) install the package lives in site-packages,
    so ``__file__``-based paths would point away from the repository."""
    src = REPO / "src" / "second_opinion"
    offenders = [p.name for p in src.glob("*.py") if "__file__" in p.read_text()]
    assert offenders == []


@needs_assets
def test_app_runs_with_package_installed_outside_the_repo(tmp_path):
    """Reproduces a Streamlit Cloud install: second_opinion imported from a copy
    outside the repository (as from site-packages), then every page is run."""
    import shutil
    import subprocess
    import sys

    site = tmp_path / "site-packages"
    shutil.copytree(REPO / "src" / "second_opinion", site / "second_opinion",
                    ignore=shutil.ignore_patterns("__pycache__"))
    script = f"""
import sys
import second_opinion
assert second_opinion.__file__.startswith({str(site)!r}), second_opinion.__file__
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(APP)!r}, default_timeout=60).run()
assert not at.exception, at.exception
for screen in ["screen_try", "screen_monitor", "screen_budget", "screen_under_the_hood"]:
    code = f"import sys; sys.path.insert(0, {str(APP.parent)!r}); import screens; screens.{{screen}}()"
    at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, (screen, at.exception)
print("ok")
"""
    # PYTHONPATH comes before site-packages, so the copy shadows any editable install.
    env = {**__import__("os").environ, "PYTHONPATH": str(site)}
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-3000:]
