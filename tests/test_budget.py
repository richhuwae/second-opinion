import numpy as np
import pytest

from second_opinion.budget import CostAssumptions, capacity_curve, review_outcome


def test_perfect_ranking_catches_everything_within_capacity():
    unsafe = np.array([True] * 10 + [False] * 90)
    scores = unsafe.astype(float)
    out = review_outcome(scores, unsafe, capacity_share=0.10)
    assert out["caught"] == 10 and out["missed"] == 0 and out["false_alarms"] == 0
    assert out["random_recall"] == pytest.approx(0.10)
    assert out["lift_vs_random"] == pytest.approx(10.0)


def test_uninformative_scores_match_random_on_average():
    rng = np.random.default_rng(0)
    recalls = []
    for _ in range(200):
        unsafe = rng.random(1000) < 0.1
        out = review_outcome(rng.random(1000), unsafe, capacity_share=0.2)
        recalls.append(out["recall"])
    assert np.mean(recalls) == pytest.approx(0.2, abs=0.02)


def test_costs_scale_with_capacity():
    unsafe = np.array([True, False] * 50)
    costs = CostAssumptions(minutes_per_review=6, physician_cost_per_hour_eur=60)
    out = review_outcome(np.arange(100.0), unsafe, 0.1, costs)
    # 10% of 1000 conversations x 6 min at 60 EUR/h = 100 reviews x 6 EUR
    assert out["review_cost_per_1000_eur"] == pytest.approx(600.0)


def test_capacity_curve_recall_is_monotonic():
    rng = np.random.default_rng(1)
    unsafe = rng.random(500) < 0.15
    scores = unsafe + rng.normal(0, 0.8, 500)
    curve = capacity_curve(scores, unsafe)
    assert curve["recall"].is_monotonic_increasing


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        review_outcome(np.array([1.0]), np.array([True]), 1.5)
    with pytest.raises(ValueError):
        review_outcome(np.array([]), np.array([], dtype=bool), 0.1)
