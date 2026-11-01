"""The README, write-up and model card must match results/*.json exactly."""

import csv
from pathlib import Path

import pytest

from second_opinion.report import (
    NUMBER,
    capacity_curve,
    fill,
    load_results,
    numbers,
    unmarked_numbers,
)

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "README.md", REPO / "docs" / "WRITEUP.md", REPO / "docs" / "MODEL_CARD.md"]
# Units and names, not results.
ALLOWED = ("1,000", "95%", "10%", "80/20")


@pytest.fixture(scope="module")
def results():
    return load_results(REPO / "results")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_matches_results(doc, results):
    text = doc.read_text()
    assert fill(text, results) == text, f"{doc.name} is stale: run python scripts/update_docs.py"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_doc_has_no_unmarked_numbers(doc):
    stray = unmarked_numbers(doc.read_text(), ALLOWED)
    assert not stray, f"{doc.name} has numbers not filled from results: {stray}"


def test_fill_rewrites_a_wrong_number_and_rejects_unknown_keys(results):
    vals = numbers(results)
    assert fill("<!--n:grader_f1-->0.999<!--/n-->", results) == f"<!--n:grader_f1-->{vals['grader_f1']}<!--/n-->"
    with pytest.raises(KeyError):
        fill("<!--n:not_a_key-->1<!--/n-->", results)
    with pytest.raises(KeyError):
        fill("<!--table:not_a_table-->\nx\n<!--/table-->", results)


def test_unmarked_numbers_catches_strays():
    assert unmarked_numbers("caught 38% of 1,473 answers, F1 0.69") == ["38%", "1,473", "0.69"]
    assert unmarked_numbers("caught <!--n:x-->38%<!--/n--> `0.5` [a](b/0.5.png)") == []


def test_readme_headline_quotes_the_four_policies():
    head = (REPO / "README.md").read_text().split("![")[0]
    keys = {k for k, _ in NUMBER.findall(head)}
    for p in ("grader", "random", "criterion_only", "perfect"):
        assert f"share_for_50_{p}" in keys and f"recall_10_{p}" in keys


def test_headline_chart_points_match_results(results):
    saved = list(csv.DictReader((REPO / "docs" / "figures" / "caught_vs_capacity.csv").open()))
    expected = capacity_curve(results)
    assert len(saved) == len(expected)
    for s, e in zip(saved, expected):
        assert s["policy"] == e["policy"] and s["source"] == e["source"]
        assert float(s["capacity_share"]) == pytest.approx(e["capacity_share"], abs=1e-12)
        assert float(s["recall"]) == pytest.approx(e["recall"], abs=1e-12)
    assert (REPO / "docs" / "figures" / "caught_vs_capacity.png").stat().st_size > 10_000


def test_capacity_curve_is_monotone(results):
    pts = capacity_curve(results)
    for p in ("grader", "criterion_only", "random", "perfect"):
        ys = [d["recall"] for d in pts if d["policy"] == p]
        assert ys == sorted(ys) and ys[0] == 0.0
