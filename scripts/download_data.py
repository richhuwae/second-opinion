"""Download the HealthBench files used by this project into data/raw/."""

import sys

from second_opinion.data import FILES, download


def main() -> int:
    for name in FILES:
        try:
            path = download(name)
        except OSError as exc:
            print(f"Could not download {name}: {exc}", file=sys.stderr)
            return 1
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
