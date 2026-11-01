"""Download and load HealthBench data.

HealthBench (OpenAI, 2025, MIT licence) is a benchmark of health conversations
graded against rubric criteria written by physicians. We use its *meta-eval*
split: each row is one (conversation, model answer, rubric criterion) triple,
graded "criterion met / not met" by one or more physicians.

Field names follow OpenAI's reference loader:
https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

# The same files are published by OpenAI in two places; we try them in order.
MIRRORS = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench",
    "https://huggingface.co/datasets/openai/healthbench/resolve/main",
)
FILES = {
    # Physician-graded answers: the ground truth for grader agreement.
    "meta_eval": "2025-05-07-06-14-12_oss_meta_eval.jsonl",
    # Full benchmark: conversations with their complete rubrics and theme tags.
    "eval": "2025-05-07-06-14-12_oss_eval.jsonl",
}
# SHA-256 of each file: the Git LFS oid that huggingface.co/api/datasets/openai/healthbench/tree/main
# reports for it (checked 2026-11-01; the LFS oid is the file's SHA-256).
# Checked after every download so a mirror can never silently change the data.
SHA256 = {
    "meta_eval": "6a343ed3fcb6f28353c2c6c157ae30ddf2db874c947b27ba474ad54da9c5dbac",
    "eval": "e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

META_EVAL_REQUIRED = {
    "prompt",
    "completion",
    "rubric",
    "binary_labels",
    "anonymized_physician_ids",
    "category",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest_dir: Path = DATA_DIR, overwrite: bool = False) -> Path:
    """Download one HealthBench file to ``dest_dir``, verify it, and return its path.

    Tries each mirror in ``MIRRORS`` in turn. Raises ``OSError`` if none works.
    """
    if name not in FILES:
        raise KeyError(f"Unknown file {name!r}; expected one of {sorted(FILES)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / FILES[name]
    if path.exists() and not overwrite:
        if sha256sum(path) != SHA256[name]:
            raise OSError(f"{path} exists but its checksum is wrong; delete it and retry")
        return path
    tmp = path.with_suffix(".part")
    errors = []
    for base in MIRRORS:
        url = f"{base}/{FILES[name]}"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        if sha256sum(tmp) != SHA256[name]:
            errors.append(f"{url}: checksum mismatch")
            tmp.unlink()
            continue
        tmp.replace(path)
        return path
    raise OSError("all mirrors failed:\n  " + "\n  ".join(errors))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def conversation_to_text(messages: list[dict]) -> str:
    """Flatten a chat into the same ``role: content`` format the reference grader uses."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def load_meta_eval(path: Path) -> pd.DataFrame:
    """Load the meta-eval file into one row per (conversation, answer, criterion).

    Adds derived columns used throughout the project:

    - ``n_physicians``: how many physicians graded the row
    - ``share_met``: fraction of physicians who said the criterion is met
    - ``majority_met``: physician majority vote (ties -> ``None``)
    - ``unanimous``: all physicians agreed
    """
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = META_EVAL_REQUIRED - rows[0].keys()
    if missing:
        raise ValueError(f"{path} is missing expected fields: {sorted(missing)}")

    records = []
    for i, r in enumerate(rows):
        labels = r["binary_labels"]
        ids = r["anonymized_physician_ids"]
        if len(labels) != len(ids) or not labels:
            raise ValueError(f"row {i}: {len(labels)} labels vs {len(ids)} physician ids")
        if not all(isinstance(x, bool) for x in labels):
            raise ValueError(f"row {i}: binary_labels must be booleans")
        n_met = sum(labels)
        n = len(labels)
        records.append(
            {
                "row_id": i,
                "prompt_id": r.get("prompt_id"),
                "completion_id": r.get("completion_id"),
                "category": r["category"],
                "conversation": conversation_to_text(r["prompt"]),
                "n_turns": len(r["prompt"]),
                "completion": r["completion"],
                "rubric": r["rubric"],
                "labels": labels,
                "physician_ids": ids,
                "n_physicians": n,
                "share_met": n_met / n,
                "majority_met": None if 2 * n_met == n else 2 * n_met > n,
                "unanimous": n_met in (0, n),
            }
        )
    return pd.DataFrame.from_records(records)
