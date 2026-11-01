import pandas as pd

from second_opinion.labels import (
    SAFETY_CRITERIA,
    answer_labels,
    high_stakes_conversations,
    summarize_labels,
    theme,
)

SAFE = "cluster:emergency_referrals_emergent_emergency_behavior"
STYLE = "cluster:emergency_referrals_emergent_context_seeking"


def rows():
    # a1: safety criterion failed by majority -> unsafe, not good
    # a2: safety tie -> contested, not good
    # a3: all met -> good
    # a4: only a non-safety criterion, failed -> not covered, not unsafe, not good
    return pd.DataFrame(
        [
            ("p1", "a1", SAFE, [False, False]),
            ("p1", "a1", STYLE, [True, True]),
            ("p1", "a2", SAFE, [True, False]),
            ("p2", "a3", SAFE, [True, True, False]),
            ("p2", "a3", STYLE, [True, True]),
            ("p2", "a4", "cluster:hedging_no-uncertainty_hedges", [False, False]),
        ],
        columns=["prompt_id", "completion_id", "category", "labels"],
    )


def test_answer_labels():
    lab = answer_labels(rows())
    assert SAFE in SAFETY_CRITERIA and STYLE not in SAFETY_CRITERIA
    assert lab.loc["a1", "unsafe"] and not lab.loc["a1", "good"]
    assert lab.loc["a2", "unsafe_contested"] and not lab.loc["a2", "unsafe"]
    assert lab.loc["a3", "good"] and lab.loc["a3", "unsafe_any_physician"] and not lab.loc["a3", "unsafe"]
    assert not lab.loc["a4", "has_safety_criterion"] and not lab.loc["a4", "unsafe"]
    assert not lab.loc["a4", "good"]
    assert lab.loc["a1", "n_criteria"] == 2


def test_summarize_labels():
    lab = answer_labels(rows())
    out = summarize_labels(lab, high_stakes={"p1"})
    assert out["n_answers"] == 4 and out["n_conversations"] == 2
    assert out["answers_unsafe"] == 1 and out["answers_unsafe_contested"] == 1
    assert out["answers_good"] == 1
    assert out["conversations_with_unsafe_answer"] == 1
    assert out["conversations_all_answers_good"] == 0
    assert out["high_stakes"]["share_unsafe_high_stakes"] == 0.5
    assert out["high_stakes"]["share_unsafe_other"] == 0.0


def test_theme_and_high_stakes():
    assert theme("cluster:health_data_tasks_not-enough-info-to-complete-task_safety") == "health_data_tasks"
    ev = [
        {"prompt_id": "x", "rubrics": [{"points": -9, "tags": ["level:example"]}]},
        {"prompt_id": "y", "rubrics": [{"points": -5, "tags": ["level:example"]},
                                       {"points": -9, "tags": ["level:cluster"]}]},
    ]
    assert high_stakes_conversations(ev) == {"x"}


def test_labels_doc_lists_every_safety_criterion():
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "labels.md").read_text()
    for name in SAFETY_CRITERIA:
        assert f"`{name.removeprefix('cluster:')}`" in doc
    assert len(SAFETY_CRITERIA) == 12
