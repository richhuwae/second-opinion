import numpy as np
import pandas as pd
import pytest

from second_opinion import grader
from second_opinion.experiment import physician_ceiling, physician_pooled, run_experiment

TOPICS = ["hydration", "ibuprofen", "emergency", "allergy", "sleep", "vaccine", "insulin", "fever"]


def synthetic_meta_eval(n_convos: int = 120, seed: int = 0) -> pd.DataFrame:
    """Fake rows in the loader's output format (not real data).

    A criterion 'Mentions <topic>' is met exactly when the answer mentions the
    topic, so a grader that reads the answer can learn it and one that reads
    only the criterion cannot. Physicians agree 90% of the time.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_convos):
        topic = TOPICS[c % len(TOPICS)]
        for k in range(4):
            mentioned = rng.choice(TOPICS, size=2, replace=False).tolist()
            criterion = TOPICS[(c + k) % len(TOPICS)]
            truth = criterion in mentioned
            n_doc = int(rng.integers(1, 4))
            labels = [bool(truth if rng.random() < 0.9 else not truth) for _ in range(n_doc)]
            rows.append(
                {
                    "prompt_id": f"p{c}",
                    "category": f"cluster:{topic}",
                    "conversation": f"user: question about {topic} number {c}",
                    "n_turns": 1,
                    "completion": f"Here is advice on {mentioned[0]} and also {mentioned[1]}.",
                    "rubric": f"Mentions {criterion} in the answer.",
                    "labels": labels,
                    "physician_ids": [f"doc{(c + i) % 7}" for i in range(n_doc)],
                    "share_met": sum(labels) / n_doc,
                }
            )
    return pd.DataFrame(rows)


def test_soft_labels_keep_total_weight_per_row():
    df = synthetic_meta_eval(10)
    rows, y, w = grader.soft_label_training_set(df)
    assert len(rows) == len(y) == len(w)
    assert set(rows) <= set(range(len(df)))
    assert w.sum() == pytest.approx(len(df))
    assert (w > 0).all()
    # 'met' weight equals the physician share for each original row
    assert w[y].sum() == pytest.approx(df["share_met"].sum())


def test_grouped_split_keeps_conversations_together():
    df = synthetic_meta_eval(50)
    train, test = grader.grouped_split(df, 0.2, seed=0)
    assert set(df.iloc[train]["prompt_id"]).isdisjoint(df.iloc[test]["prompt_id"])
    assert len(train) + len(test) == len(df)


def test_group_key_falls_back_to_conversation_text():
    df = pd.DataFrame({"prompt_id": [None, None, "x"], "conversation": ["a", "a", "b"]})
    keys = grader.group_key(df)
    assert keys[0] == keys[1] != keys[2]


def test_handcrafted_overlap_and_negation():
    df = pd.DataFrame(
        {
            "rubric": ["Mentions ibuprofen dosing.", "Fails to advise rest."],
            "completion": ["Take ibuprofen with food; dosing is 400mg.", "Drink water?"],
            "conversation": ["user: hi", "user: hi"],
            "n_turns": [1, 1],
        }
    )
    f = grader.handcrafted_features(df)
    assert f.loc[0, "overlap_share"] == pytest.approx(1.0)
    assert f.loc[1, "overlap_share"] == 0.0
    assert f.loc[1, "criterion_negative"] == 1.0 and f.loc[0, "criterion_negative"] == 0.0
    assert f.loc[1, "answer_questions"] == 1.0


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
def test_grader_learns_answer_dependent_criterion(kind):
    df = synthetic_meta_eval(160)
    train, test = grader.grouped_split(df, 0.25, seed=1)
    model = grader.fit(df.iloc[train], kind)
    proba = grader.predict_proba(model, df.iloc[test])
    assert ((proba >= 0) & (proba <= 1)).all()
    preds = proba >= 0.5
    truth = df.iloc[test]["share_met"].to_numpy() > 0.5
    assert (preds == truth).mean() > 0.8


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        grader.build_model("transformer")


def test_choose_threshold_finds_separating_cut():
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = [[False], [False], [False], [True], [True], [True]]
    t = grader.choose_threshold(proba, labels)
    assert 0.3 < t <= 0.7


def test_expand_per_physician():
    p, y = grader.expand_per_physician(np.array([0.2, 0.9]), [[True, False], [True]])
    assert p.tolist() == [0.2, 0.2, 0.9] and y.tolist() == [True, False, True]


def test_physician_ceiling_reflects_agreement():
    df = synthetic_meta_eval(120)
    ceiling = physician_ceiling(df, min_rows=5)
    assert ceiling["n_physicians"] > 0
    assert 0.7 < ceiling["mean"] <= 1.0  # physicians agree ~90% by construction


def test_run_experiment_end_to_end():
    df = synthetic_meta_eval(160)
    out = run_experiment(df, kinds=("logreg",), n_folds=3, n_boot=50, min_category_rows=10)
    g, b = out["grader"], out["baselines"]
    assert out["model_selection"]["chosen"] == "logreg"
    assert g["f1_balanced"] > 0.75
    lo, hi = g["f1_balanced_ci95"]
    assert lo <= g["f1_balanced"] <= hi
    # The criterion alone carries no signal here, so reading the answer must win.
    assert g["f1_balanced"] > b["rubric_only"] + 0.15
    assert out["data"]["n_train_rows"] + out["data"]["n_test_rows"] == len(df)
    assert out["by_category"] and all(r["n_rows"] >= 10 for r in out["by_category"])
    m = out["multi_physician_rows"]
    assert 0 < m["n_rows"] < out["data"]["n_test_rows"]
    assert m["grader_f1"] > m["rubric_only_f1"]
    lo, hi = m["grader_f1_ci95"]
    assert lo <= m["grader_f1"] <= hi
    # Synthetic physicians agree 90% of the time.
    assert 0.75 < m["physician_pooled_f1"] <= 1


def test_physician_pooled():
    df = pd.DataFrame({"labels": [[True, True], [False, False], [True], [True, False]]})
    # Pairs: (T,T) (T,T) (F,F) (F,F) (T,F) (F,T); single-label row ignored.
    # pos: precision 2/3, recall 2/3; neg: same -> balanced F1 = 2/3.
    assert abs(physician_pooled(df) - 2 / 3) < 1e-9
    assert physician_pooled(pd.DataFrame({"labels": [[True]]})) is None
