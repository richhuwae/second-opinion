# Working on Second Opinion

Read `docs/PLAN.md` first: it holds the goal, agreed scope (core vs optional),
the demo spec, the decisions log and current status. Update its Status and
Decisions log when something changes.

- Setup: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`, then `pytest`.
- Data: `python scripts/download_data.py` (writes to `data/raw/`, which is gitignored).
- Never put a number in the README, write-up or demo unless it was computed by code in this repo.
  Report disappointing results too.
- Cost and time figures are assumptions: label them as such.
- This repository is public: no personal details, no course material, no real patient data.
- Keep writing concise and plain; the owner prefers direct, honest wording.
