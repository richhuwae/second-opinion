"""Print and save descriptive facts about the meta-eval file (task C1).

Writes results/data_summary.json.
"""

import json
import sys
from pathlib import Path

from second_opinion.data import download, load_meta_eval
from second_opinion.explore import summarize

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "results" / "data_summary.json"


def main() -> int:
    try:
        df = load_meta_eval(download("meta_eval", RAW))
    except OSError as exc:
        print(f"Could not get the meta-eval file ({exc}).", file=sys.stderr)
        return 1
    summary = summarize(df)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    for key, value in summary.items():
        if key != "by_category":
            print(f"{key}: {value}")
    print("\nBy theme:")
    for row in summary["by_category"]:
        print(f"  {row['category']:<45} rows={row['rows']:>6}  met={row['share_met']:.2f}  unanimous={row['unanimous']:.2f}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
