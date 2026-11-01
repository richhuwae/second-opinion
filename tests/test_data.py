import hashlib
import json

import pandas as pd
import pytest

from second_opinion.data import load_meta_eval

# Hand-written fixture in the HealthBench meta-eval format (not real data).
ROWS = [
    {
        "prompt": [{"role": "user", "content": "I have chest pain and my left arm is numb."}],
        "completion": "Please call emergency services now.",
        "rubric": "Advises the user to seek emergency care immediately.",
        "binary_labels": [True, True, False],
        "anonymized_physician_ids": ["a", "b", "c"],
        "category": "cluster:emergency",
        "prompt_id": "p1",
        "completion_id": "c1",
    },
    {
        "prompt": [{"role": "user", "content": "Can I take ibuprofen?"}],
        "completion": "Yes.",
        "rubric": "Asks about relevant medical history before advising.",
        "binary_labels": [False, True],
        "anonymized_physician_ids": ["a", "d"],
        "category": "cluster:context_seeking",
    },
]


def _write(tmp_path, rows):
    path = tmp_path / "meta.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_loads_and_derives_columns(tmp_path):
    df = load_meta_eval(_write(tmp_path, ROWS))
    assert len(df) == 2
    first, second = df.iloc[0], df.iloc[1]
    assert first["conversation"].startswith("user: I have chest pain")
    assert first["n_physicians"] == 3 and first["share_met"] == pytest.approx(2 / 3)
    assert first["majority_met"] == True and first["unanimous"] == False  # noqa: E712 (numpy bools)
    assert second["majority_met"] is None  # 1-1 tie
    assert pd.isna(second["prompt_id"])  # optional field absent


def test_rejects_missing_fields(tmp_path):
    bad = [{k: v for k, v in ROWS[0].items() if k != "binary_labels"}]
    with pytest.raises(ValueError, match="binary_labels"):
        load_meta_eval(_write(tmp_path, bad))


def test_rejects_mismatched_physician_ids(tmp_path):
    bad = [dict(ROWS[0], anonymized_physician_ids=["a"])]
    with pytest.raises(ValueError, match="physician ids"):
        load_meta_eval(_write(tmp_path, bad))


def _fake_mirrors(tmp_path, monkeypatch, content: bytes):
    from second_opinion import data

    good, empty = tmp_path / "good", tmp_path / "empty"
    good.mkdir()
    empty.mkdir()
    (good / "f.jsonl").write_bytes(content)
    monkeypatch.setattr(data, "FILES", {"x": "f.jsonl"})
    monkeypatch.setattr(data, "SHA256", {"x": hashlib.sha256(b"expected").hexdigest()})
    monkeypatch.setattr(data, "MIRRORS", (empty.as_uri(), good.as_uri()))
    return data


def test_download_falls_back_to_next_mirror_and_verifies(tmp_path, monkeypatch):
    data = _fake_mirrors(tmp_path, monkeypatch, b"expected")
    path = data.download("x", dest_dir=tmp_path / "out")
    assert path.read_bytes() == b"expected"
    assert data.download("x", dest_dir=tmp_path / "out") == path  # cached, re-verified


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    data = _fake_mirrors(tmp_path, monkeypatch, b"tampered")
    with pytest.raises(OSError, match="checksum mismatch"):
        data.download("x", dest_dir=tmp_path / "out")
    assert not list((tmp_path / "out").iterdir())  # nothing half-written left behind
