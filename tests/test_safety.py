import numpy as np
import pandas as pd
import pytest

from second_opinion.budget import expected_caught
from second_opinion.safety import (
    answer_risk,
    capacity_for_recall,
    compare_policies,
    cost_per_1000,
    ranking_quality,
)

SAFE = "cluster:emergency_referrals_emergent_emergency_behavior"
STYLE = "cluster:emergency_referrals_emergent_context_seeking"


def test_answer_risk_uses_only_safety_criteria():
    rows = pd.DataFrame({
        "completion_id": ["a", "a", "b", "c"],
        "category": [SAFE, STYLE, SAFE, STYLE],
    })
    risk = answer_risk(rows, np.array([0.9, 0.0, 0.3, 0.0]))
    assert risk.to_dict() == pytest.approx({"a": 0.1, "b": 0.7})
    with pytest.raises(ValueError):
        answer_risk(rows, np.array([0.1, np.nan, 0.2, 0.3]))


def test_expected_caught_breaks_ties_fairly():
    y = np.array([True, False, True, False])
    assert expected_caught(np.zeros(4), y, 2) == pytest.approx(1.0)
    assert expected_caught(np.array([1, 0, 0, 0.0]), y, 2) == pytest.approx(1 + 1 / 3)
    assert expected_caught(y.astype(float), y, 1) == 1
    assert expected_caught(y.astype(float), y, 0) == 0


def test_compare_policies_orders_policies():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.2
    good = y + rng.normal(0, 0.5, 400)
    weak = y + rng.normal(0, 2.0, 400)
    rows = compare_policies({"grader": good, "criterion_only": weak}, y,
                            groups=np.arange(400) // 4, shares=(0.1, 0.2), n_boot=50)
    for r in rows:
        assert r["perfect"]["recall"] >= r["grader"]["recall"] > r["random"]["recall"]
        assert r["random"]["recall"] == pytest.approx(r["capacity_share"], abs=0.01)
        lo, hi = r["grader"]["recall_ci95"]
        assert lo <= r["grader"]["recall"] <= hi
        g = r["grader"]
        assert g["caught"] + g["false_alarms"] == pytest.approx(r["n_reviewed"], abs=0.1)
    q = ranking_quality({"grader": good}, y)
    assert q["grader"]["roc_auc"] > 0.8


def test_capacity_for_recall():
    y = np.array([True] * 10 + [False] * 90)
    assert capacity_for_recall(y.astype(float), y, 0.5) == pytest.approx(0.05)
    assert capacity_for_recall(np.zeros(100), y, 0.5) == pytest.approx(0.5)


def test_cost_per_1000_uses_assumptions():
    out = cost_per_1000(0.1, recall=0.5, unsafe_rate=0.2)
    # 100 reviews x 4 min at EUR 90/h (assumed) = EUR 600; 1000 x 0.2 x 0.5 = 100 caught
    assert out["reviews"] == 100 and out["review_cost_eur"] == pytest.approx(600)
    assert out["doctor_hours"] == pytest.approx(100 * 4 / 60)
    assert out["cost_per_unsafe_caught_eur"] == pytest.approx(6.0)
