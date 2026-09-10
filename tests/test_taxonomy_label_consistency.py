"""
(c) Inria

HORIZONTAL verification of `planktonzilla_taxonomy.csv`, run as a test.

`planktonzilla/planktonzilla_dataset/utils/verify_label_consistency.py` groups the table by
`Raw_Labels` — the key `build_taxonomy_lookup` joins on, so two rows sharing one are describing
the same imagefolder class directory in two different source datasets — and reports the groups
whose rows publish different taxa.

This is the axis `verify_taxonomy_ids.py` cannot see. That module checks each row VERTICALLY:
it takes the taxon a row claims and asks WoRMS / NCBI / Wikidata whether the identifiers agree.
Its findings group by `proposed_label`, which files two disagreeing rows under two different
taxa and so never puts them side by side. `Raw_Labels=Harpacticoida` published as the order
`harpacticoida` by five datasets and as the genus `euterpina` by three passes every one of its
23 checks, because aphia 115348 genuinely *is* Euterpina (KI-31, CODE_REVIEW finding 1.6).

Two kinds of test live here, and the distinction matters:

  * ENGINE tests build tiny synthetic tables and assert the checker fires — and does not fire —
    where it should. They are what makes a green data-state test mean something.
  * DATA-STATE tests run the checker over the real shipped table. The gate is "no UNWAIVED
    finding and no stale waiver": the published table is frozen on HuggingFace Hub under the
    zero-behavioural-drift rule, so a defect found here is adjudicated in
    `utils/LABEL_CONSISTENCY_WAIVERS.json` with its reason rather than silently corrected.
    Adding a waiver is a reviewed act; a NEW disagreement — a CSV edit, or a new source dataset
    reusing an existing label for a different taxon — arrives unwaived and turns this red.

Network-free BY CONSTRUCTION, and `test_the_check_imports_nothing_that_speaks_http` proves it
from the import graph rather than asserting it.

Run the check directly with:
    python -m planktonzilla.planktonzilla_dataset.utils.verify_label_consistency --all
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import json
import subprocess
import sys

import pytest

from planktonzilla.planktonzilla_dataset.utils import verify_label_consistency as vlc

RANKS = vlc.RANKS

# The vocabulary LABEL_CONSISTENCY_WAIVERS.json documents in its own `_comment`.
CATEGORIES = {
    "mislabel_defect",
    "rank_inflation_defect",
    "nomenclature_drift",
    "rank_treatment",
    "needs_taxonomic_adjudication",
    "bucket_naming",
}


# ── Fixtures over the real shipped data ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rows():
    return vlc.read_rows()


@pytest.fixture(scope="module")
def findings(rows):
    return vlc.check_label_consistency(rows)


@pytest.fixture(scope="module")
def waivers():
    return vlc.load_waivers()


@pytest.fixture(scope="module")
def by_label(findings):
    return {finding.raw_label: finding for finding in findings}


# ── Engine tests: synthetic tables, one classification at a time ────────────────────────────
def _row(dataset, raw_label, label, **ranks):
    """One CSV row: every rank column blank unless named."""
    return {"Dataset": dataset, "Raw_Labels": raw_label, "proposed_label": label} | {
        rank: ranks.get(rank.lower(), "") for rank in RANKS
    }


def _check(*rows):
    return vlc.check_label_consistency(list(rows))


def test_rows_that_agree_produce_no_finding():
    """The common case: two datasets, one label, the same taxon. Silence."""
    assert (
        _check(
            _row("a", "copepoda", "copepoda", phylum="arthropoda", **{"class": "copepoda"}),
            _row("b", "copepoda", "copepoda", phylum="arthropoda", **{"class": "copepoda"}),
        )
        == []
    )


def test_a_deeper_lineage_under_the_same_label_is_rank_inflation():
    """The `harpacticoida` shape: one side extends the other with nothing contradicting."""
    (finding,) = _check(
        _row("a", "harpacticoida", "harpacticoida", phylum="arthropoda", order="harpacticoida"),
        _row("b", "harpacticoida", "euterpina", phylum="arthropoda", order="harpacticoida", genus="euterpina"),
    )
    assert (finding.check, finding.severity) == (vlc.RANK_INFLATION, "ERROR")
    assert finding.labels == ("euterpina", "harpacticoida")
    assert finding.conflicts == ()
    # Shallowest first, so the report reads as the inflation it is.
    assert [label for label, _, _ in finding.variants] == ["harpacticoida", "euterpina"]
    assert [rank for _, rank, _ in finding.variants] == ["Order", "Genus"]


def test_a_taxon_given_to_a_bucket_with_no_ranks_is_also_rank_inflation():
    """The limiting case: `unknown` against a real class. Inflation from nothing at all."""
    (finding,) = _check(
        _row("a", "unknown", "unknown"),
        _row("b", "unknown", "thecofilosea", kingdom="chromista", **{"class": "thecofilosea"}),
    )
    assert finding.check == vlc.RANK_INFLATION
    assert [rank for _, rank, _ in finding.variants] == ["", "Class"]


def test_two_lineages_diverging_at_a_shared_rank_is_a_contradiction():
    """Both populate Genus and disagree, so the label cannot be both."""
    (finding,) = _check(
        _row("a", "neoceratium", "neoceratium", **{"class": "dinophyceae"}, genus="neoceratium"),
        _row("b", "neoceratium", "tripos", **{"class": "dinophyceae"}, genus="tripos"),
    )
    assert (finding.check, finding.severity) == (vlc.CONTRADICTION, "ERROR")
    assert finding.conflicts == (("Genus", "neoceratium", "tripos"),)


def test_a_contradiction_outranks_an_inflation_in_the_same_group():
    """A group can be both; the stronger claim is the one reported."""
    (finding,) = _check(
        _row("a", "creseidae", "creseidae", family="creseidae"),
        _row("b", "creseidae", "clio pyramidata", family="cliidae", genus="clio", species="pyramidata"),
    )
    assert finding.check == vlc.CONTRADICTION
    assert ("Family", "cliidae", "creseidae") in finding.conflicts


def test_two_bucket_names_with_no_lineage_are_a_warning_not_an_error():
    """Nothing is given a taxon it does not have, so this is vocabulary drift."""
    (finding,) = _check(_row("a", "darkrods", "other"), _row("b", "darkrods", "shape"))
    assert (finding.check, finding.severity) == (vlc.LABEL_DISAGREEMENT, "WARN")
    assert finding.labels == ("other", "shape")


def test_one_lineage_under_two_names_is_still_reported():
    """Identical ranks, different `proposed_label` — caught, not swallowed by the lineage key."""
    (finding,) = _check(
        _row("a", "x", "copepoda", **{"class": "copepoda"}),
        _row("b", "x", "copepods", **{"class": "copepoda"}),
    )
    assert finding.check == vlc.LABEL_DISAGREEMENT
    assert finding.labels == ("copepoda", "copepods")


def test_labels_group_across_source_casing_and_whitespace():
    """`Harpacticoida` and `harpacticoida` are one folder name in two datasets' conventions.

    Grouping them apart would hide exactly the disagreement this module exists to find —
    and the shipped table really does carry both casings (KI-9 leaves `Raw_Labels` alone).
    """
    (finding,) = _check(
        _row("a", "Harpacticoida", "harpacticoida", order="harpacticoida"),
        _row("b", " harpacticoida ", "euterpina", order="harpacticoida", genus="euterpina"),
    )
    assert finding.raw_label == "harpacticoida"
    assert finding.n_rows == 2


def test_distinct_labels_do_not_cross_talk():
    """Disagreement is only ever within one `Raw_Labels` group."""
    assert (
        _check(
            _row("a", "copepoda", "copepoda", **{"class": "copepoda"}),
            _row("b", "annelida", "annelida", phylum="annelida"),
        )
        == []
    )


def test_the_finding_id_survives_a_new_dataset_but_not_a_new_taxon():
    """What the waiver mechanism rests on: reviewed once, re-reviewed when the claim changes."""
    base = [
        _row("a", "acantharia", "acantharia", **{"class": "acantharia"}),
        _row("b", "acantharia", "amphibelone", **{"class": "acantharia"}, genus="amphibelone"),
    ]
    (original,) = _check(*base)

    # A third dataset adopting a taxon already published under the label: same finding.
    (unchanged,) = _check(*base, _row("c", "acantharia", "amphibelone", **{"class": "acantharia"}, genus="amphibelone"))
    assert unchanged.finding_id == original.finding_id
    assert unchanged.n_rows == 3

    # A third dataset publishing a NEW taxon: a new finding, so the old waiver stops covering it.
    (changed,) = _check(*base, _row("c", "acantharia", "astrolonche", **{"class": "acantharia"}, genus="astrolonche"))
    assert changed.finding_id != original.finding_id


def test_waivers_split_findings_and_flag_stale_entries():
    (finding,) = _check(
        _row("a", "annelida", "annelida", phylum="annelida"),
        _row("b", "annelida", "poeobius", phylum="annelida", genus="poeobius"),
    )
    unwaived, waived, stale = vlc.apply_waivers([finding], {finding.finding_id: {"reason": "reviewed"}})
    assert (unwaived, waived) == ([], [finding])
    assert stale == []

    unwaived, waived, stale = vlc.apply_waivers([finding], {"deadbeefcafe": {"reason": "gone"}})
    assert (unwaived, waived, stale) == ([finding], [], ["deadbeefcafe"])


def test_a_missing_waiver_file_means_no_waivers(tmp_path):
    assert vlc.load_waivers(tmp_path / "nope.json") == {}


def test_the_cli_exits_nonzero_on_an_unwaived_finding(tmp_path, capsys):
    """The CLI is the gate a human runs; it must fail loudly, not just print."""
    csv_path = tmp_path / "taxo.csv"
    header = ",".join(("Dataset", "Raw_Labels", *RANKS, "proposed_label"))
    csv_path.write_text(
        f"{header}\na,annelida,,annelida,,,,,,annelida\nb,annelida,,annelida,,,,poeobius,,poeobius\n",
        encoding="utf-8",
    )
    assert vlc.main(["--csv", str(csv_path), "--waivers", str(tmp_path / "none.json")]) == 1
    assert "UNWAIVED" in capsys.readouterr().out

    waivers = tmp_path / "waivers.json"
    (finding,) = vlc.check_label_consistency(vlc.read_rows(csv_path))
    waivers.write_text(json.dumps({"waivers": [{"finding_id": finding.finding_id, "reason": "reviewed"}]}), encoding="utf-8")
    assert vlc.main(["--csv", str(csv_path), "--waivers", str(waivers)]) == 0


def test_the_json_output_is_parseable_and_carries_the_finding_ids(tmp_path, capsys):
    """`--json` is what a reviewer diffs between two revisions of the table."""
    csv_path = tmp_path / "taxo.csv"
    header = ",".join(("Dataset", "Raw_Labels", *RANKS, "proposed_label"))
    csv_path.write_text(
        f"{header}\na,annelida,,annelida,,,,,,annelida\nb,annelida,,annelida,,,,poeobius,,poeobius\n",
        encoding="utf-8",
    )
    assert vlc.main(["--csv", str(csv_path), "--waivers", str(tmp_path / "none.json"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["by_check"] == {"rank_inflation": 1}
    assert [entry["raw_label"] for entry in payload["unwaived"]] == ["annelida"]
    assert len(payload["unwaived"][0]["finding_id"]) == 12


def test_the_check_imports_nothing_that_speaks_http():
    """Network-free by construction, proven from the import graph rather than asserted.

    The vertical checker needs a committed snapshot to stay offline; this one needs nothing
    but the CSV, and that is the whole argument for it being cheap to run on every push.
    """
    probe = (
        "import sys;"
        "import planktonzilla.planktonzilla_dataset.utils.verify_label_consistency;"
        "print([m for m in ('requests', 'urllib.request', 'http.client', 'aiohttp') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=root)
    assert result.stdout.strip() == "[]", f"HTTP machinery reached the import graph: {result.stdout}"


# ── Data-state tests: the real shipped table ───────────────────────────────────────────────
def test_no_unwaived_disagreement_and_no_stale_waiver(findings, waivers):
    """THE GATE. Every disagreement standing today is adjudicated; a new one turns this red."""
    unwaived, _, stale = vlc.apply_waivers(findings, waivers)
    assert not unwaived, "unwaived label disagreement(s):\n" + "\n".join(f.describe() for f in unwaived)
    assert not stale, f"waiver(s) matching no current finding — delete them: {stale}"


def test_the_shipped_inventory_is_exactly_what_was_adjudicated(findings, rows):
    """Pins the measured state, so a CSV edit that changes the count cannot pass unnoticed."""
    assert len(rows) == 2358
    assert len(vlc.group_by_raw_label(rows)) == 1622
    assert vlc.summarize(findings) == {
        "total": 20,
        "by_severity": {"ERROR": 18, "WARN": 2},
        "by_check": {"label_disagreement": 2, "lineage_contradiction": 5, "rank_inflation": 13},
        "rows_touched": 80,
    }


def test_the_motivating_case_is_caught(by_label):
    """CODE_REVIEW finding 1.6, first case — the one every vertical authority check passes."""
    finding = by_label["harpacticoida"]
    assert finding.check == vlc.RANK_INFLATION
    assert finding.labels == ("euterpina", "harpacticoida")
    datasets = {dataset for _, _, sources in finding.variants for dataset in sources}
    assert {"tara_pacific_hsn", "tara_pacific_manta", "zooscan"} <= datasets


def test_the_second_case_of_finding_1_6_is_caught(by_label):
    """`Creseidae` published under family Cliidae — a contradiction, not merely a finer rank."""
    finding = by_label["creseidae"]
    assert finding.check == vlc.CONTRADICTION
    assert ("Family", "cliidae", "creseidae") in finding.conflicts


def test_the_non_taxonomic_buckets_given_real_taxa_are_caught(by_label):
    """`unknown`, `other_living` and `filament` are the starkest cases: a residual bucket
    cannot be a taxon, so a rank cell under one is a claim the folder name contradicts."""
    for label, taxon in (("unknown", "thecofilosea"), ("other_living", "monstrilloida"), ("filament", "cyanophyceae")):
        finding = by_label[label]
        assert finding.check == vlc.RANK_INFLATION, label
        assert taxon in finding.labels, label
        assert "" in [rank for _, rank, _ in finding.variants], f"{label} should have a no-rank side"


def test_every_waiver_is_justified_and_categorized(waivers):
    """A waiver without a reason is an undocumented defect; that is what this file exists against."""
    assert len(waivers) == 20, "expected the 20 adjudicated findings"
    for finding_id, entry in waivers.items():
        assert len(finding_id) == 12, f"{finding_id} is not a 12-char finding id"
        assert entry.get("category") in CATEGORIES, f"waiver {finding_id} has category {entry.get('category')!r}"
        assert len(entry.get("reason", "")) >= 60, f"waiver {finding_id} has a reason too short to be a justification"
        assert entry.get("raw_label"), f"waiver {finding_id} names no label"


def test_the_defects_are_recorded_as_defects_not_dismissed(waivers):
    """Waiving is documenting, not excusing: most of these do mislabel images.

    Only `nomenclature_drift` and `bucket_naming` are cases where no image carries a taxon it
    should not; if that balance ever inverts, someone has been closing findings too easily.
    """
    categories = [entry["category"] for entry in waivers.values()]
    benign = sum(category in {"nomenclature_drift", "bucket_naming"} for category in categories)
    assert benign == 3, categories
    assert sum(category.endswith("_defect") for category in categories) == 15


def test_the_known_issue_is_documented(waivers):
    """KI-31 carries the write-up; the waiver file carries the per-finding adjudication."""
    text = (root / "planktonzilla" / "planktonzilla_dataset" / "utils" / "KNOWN_ISSUES.md").read_text(encoding="utf-8")
    assert "KI-31" in text
    for label in ("harpacticoida", "creseidae"):
        assert label in text.lower()
