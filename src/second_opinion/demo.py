"""Logic behind the Streamlit demo, kept out of the UI so it can be tested.

Everything here works on precomputed files in app/assets/ (built by
scripts/build_demo_assets.py); nothing is trained at startup. Callers pass the
folders: the package never resolves paths from its own location, because a
normal pip install puts it in site-packages, away from the repository files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from second_opinion.budget import CostAssumptions, expected_caught

def load_assets(assets: Path) -> dict:
    """Small files only; the model is loaded separately (it is larger)."""
    examples = [json.loads(line) for line in (assets / "examples.jsonl").read_text().splitlines() if line]
    return {
        "risk": pd.read_csv(assets / "risk_scores.csv", index_col=0),
        "examples": examples,
        "criteria": json.loads((assets / "criteria.json").read_text()),
    }


def load_results(results: Path) -> dict:
    return {name: json.loads((results / f"{name}.json").read_text())
            for name in ("grader_classical", "safety_alarm", "label_summary", "data_summary")}


def short_name(category: str) -> str:
    """'cluster:emergency_referrals_emergent_emergency_behavior' -> readable text."""
    return category.removeprefix("cluster:").replace("_", " ").replace("-", " ")


def alarm_threshold(risk: pd.Series, capacity_share: float) -> float:
    """Risk above which an answer is sent to a doctor, so that about
    ``capacity_share`` of answers are reviewed."""
    risk = risk.dropna().to_numpy()
    k = max(1, int(round(capacity_share * len(risk))))
    return float(np.sort(risk)[::-1][min(k, len(risk)) - 1])


def build_input(category: str, rubric: str, user_message: str, answer: str) -> pd.DataFrame:
    """One meta-eval-shaped row for a pasted answer (single user turn)."""
    return pd.DataFrame([{
        "category": category,
        "rubric": rubric,
        "conversation": f"user: {user_message.strip()}",
        "n_turns": 1,
        "completion": answer.strip(),
    }])


def budget_table(
    risk: pd.DataFrame, capacity_share: float, contested_is_unsafe: bool,
    costs: CostAssumptions,
) -> pd.DataFrame:
    """Caught / missed / false alarms / cost per policy at one capacity, per 1,000 answers."""
    y = risk["unsafe"].to_numpy(bool)
    if contested_is_unsafe:
        y = y | risk["unsafe_contested"].to_numpy(bool)
    n = len(y)
    k = int(round(capacity_share * n))
    scale = 1000 / n
    policies = {
        "Grader ranking": risk["risk_grader"].to_numpy(),
        "Criterion-only ranking": risk["risk_criterion_only"].to_numpy(),
        "Random sample (today)": np.zeros(n),
        "Perfect ranking (upper bound)": y.astype(float),
    }
    rows = []
    reviews = k * scale
    cost = reviews * costs.cost_per_review_eur
    for name, s in policies.items():
        caught = expected_caught(s, y, k)
        rows.append({
            "Policy": name,
            "Unsafe caught": caught * scale,
            "Unsafe missed": (y.sum() - caught) * scale,
            "False alarms": (k - caught) * scale,
            "Share of unsafe caught": caught / y.sum() if y.sum() else np.nan,
            "Doctor hours": reviews * costs.minutes_per_review / 60,
            "Review cost (EUR)": cost,
            "EUR per unsafe caught": cost / (caught * scale) if caught else np.nan,
        })
    return pd.DataFrame(rows)


def recall_curve(risk: pd.DataFrame, contested_is_unsafe: bool, shares=None) -> pd.DataFrame:
    y = risk["unsafe"].to_numpy(bool)
    if contested_is_unsafe:
        y = y | risk["unsafe_contested"].to_numpy(bool)
    shares = np.arange(0.01, 0.51, 0.01) if shares is None else shares
    cols = {"Grader": risk["risk_grader"].to_numpy(),
            "Criterion-only": risk["risk_criterion_only"].to_numpy(),
            "Random": np.zeros(len(y)), "Perfect": y.astype(float)}
    out = [{"Capacity": s, "Policy": name,
            "Share of unsafe caught": expected_caught(v, y, int(round(s * len(y)))) / y.sum()}
           for s in shares for name, v in cols.items()]
    return pd.DataFrame(out)


def simulated_day(risk: pd.DataFrame, threshold: float, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A SIMULATED day of traffic: benchmark answers given random arrival hours.

    Uses held-out (out-of-fold) grader risk, so no answer is scored by a model
    that saw it. Not real traffic.
    """
    rng = np.random.default_rng(seed)
    day = risk.iloc[rng.choice(len(risk), size=min(n, len(risk)), replace=False)].copy()
    day["hour"] = rng.integers(0, 24, len(day))
    day["flagged"] = day["risk_grader"] >= threshold
    day["agrees"] = day["flagged"] == day["unsafe"]
    return day.sort_values("hour")
