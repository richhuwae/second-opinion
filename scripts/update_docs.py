"""Fill the numbers and tables in the README, write-up and model card from results/*.json.

    python scripts/update_docs.py          # rewrite the docs in place
    python scripts/update_docs.py --check  # exit 1 if any doc is out of date

Only text between <!--n:key--> ... <!--/n--> and <!--table:key--> ... <!--/table-->
markers is touched. Keys are defined in src/second_opinion/report.py.
"""

import argparse
import sys
from pathlib import Path

from second_opinion.report import fill, load_results

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "WRITEUP.md", ROOT / "docs" / "MODEL_CARD.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale docs, change nothing")
    args = parser.parse_args()
    results = load_results(ROOT / "results")
    stale = []
    for doc in DOCS:
        old = doc.read_text()
        new = fill(old, results)
        if new != old:
            stale.append(doc.relative_to(ROOT))
            if not args.check:
                doc.write_text(new)
    verb = "Out of date" if args.check else "Updated"
    for s in stale:
        print(f"{verb}: {s}")
    if not stale:
        print("All docs match the results.")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
