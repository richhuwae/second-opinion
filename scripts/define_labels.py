"""Count answers and conversations under the proposed "unsafe" / "good" labels (task C1).

Writes results/label_summary.json. Definitions are in src/second_opinion/labels.py.
"""

import json
import sys
from pathlib import Path

from second_opinion.data import download, load_meta_eval, read_jsonl
from second_opinion.labels import answer_labels, high_stakes_conversations, summarize_labels

OUT = Path(__file__).resolve().parents[1] / "results" / "label_summary.json"


def main() -> int:
    try:
        df = load_meta_eval(download("meta_eval"))
        eval_rows = read_jsonl(download("eval"))
    except OSError as exc:
        print(f"Could not get the data files ({exc}).", file=sys.stderr)
        return 1
    labels = answer_labels(df)
    summary = summarize_labels(labels, high_stakes_conversations(eval_rows))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary.items():
        if key != "by_theme":
            print(f"{key}: {value}")
    print("\nBy theme (answers):")
    for r in summary["by_theme"]:
        print(f"  {r['theme']:<22} answers={r['answers']:>5} with safety criterion={r['covered']:>5} "
              f"unsafe={r['unsafe']:>5} contested={r['contested']:>5} good={r['good']:>5}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
