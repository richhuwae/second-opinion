from second_opinion.explore import looks_french, summarize
from tests.test_grader import synthetic_meta_eval


def test_summarize_counts():
    df = synthetic_meta_eval(20).assign(row_id=range(80), completion_id=None)
    df["n_physicians"] = df["labels"].map(len)
    df["unanimous"] = df["labels"].map(lambda x: len(set(x)) == 1)
    df["majority_met"] = df["labels"].map(lambda x: None if 2 * sum(x) == len(x) else 2 * sum(x) > len(x))
    out = summarize(df)
    assert out["n_rows"] == 80 and out["n_conversations"] == 20
    assert out["rows_per_conversation_median"] == 4
    assert sum(r["rows"] for r in out["by_category"]) == 80
    assert out["share_french_conversations"] == 0


def test_looks_french():
    assert looks_french("Bonjour, je pense que les douleurs sont dans le dos")
    assert not looks_french("I think the pain is in my back")
    assert not looks_french("")
