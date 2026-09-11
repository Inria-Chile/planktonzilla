"""
(c) Inria

Gate for step 4 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` — concurrent add-source PRs.

Every design the panel evaluated conflicted here, and the report says so plainly: two branches each
appending a new concept, a new identifier and a new source stanza conflicted on ``concepts.csv``,
``identifiers.csv`` and the descriptor in design A; on ``source.tsv`` in design B; on
``taxonomy.yaml`` in design C; on three files in design D even with disjoint id ranges. Only the
per-source mapping file is conflict-free by construction.

So this module does not assert that the fix works — it performs the merge. Two branches are built
in a scratch repository, each adding one source exactly as a real PR would, and ``git merge`` runs
for real with the repository's own ``.gitattributes``. Measured before the file existed: three
conflicts. After: none, with both branches' rows present in both ledgers.

Two things union merge does NOT fix, both handled rather than hoped away:

* it can leave a ledger unsorted — cosmetic, and restored by ``fmt`` in step 6;
* it will happily keep two rows claiming the same id — a failing check, asserted below.

And one claim this module deliberately does not make: that nothing can ever conflict. Union merge
is applied to exactly three files, each an append-only ledger of independent rows where "keep both"
is always what a curator would have typed. On any file where two edits can contradict each other it
would be the wrong strategy, which is why nothing else has it.
"""

import shutil
import subprocess
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, validate
from planktonzilla.planktonzilla_dataset.taxonomy.model import read_tsv, write_tsv

PACKAGE = loader.PACKAGE_DIR
GITATTRIBUTES = Path(root) / ".gitattributes"
UNION_MERGED = ("taxon.tsv", "identifier.tsv", "merged.tsv")


def _git(repo: Path, *args, check=True):
    return subprocess.run(
        ["git", "-c", "user.email=t@example.org", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )


def _add_source(repo: Path, name: str, taxon_id: str) -> None:
    """Exactly what adding a source touches: one new file, and a row on each shared ledger."""
    data = repo / "data"
    template = read_tsv(data / "mappings" / "zoolake.tsv")[0]
    write_tsv(
        data / "mappings" / f"{name}.tsv",
        tuple(k for k in template if k != "__file__"),
        [{**template, "datasetID": name, "verbatimIdentification": f"{name}_class", "taxonID": taxon_id, "concept": name}],
    )
    with (data / "taxon.tsv").open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{taxon_id}\t\tunranked\t{name}\taccepted\t\t\tbucket\tnew source\n")
    with (data / "identifier.tsv").open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{taxon_id}\tskos:exactMatch\tworms:{taxon_id[-6:]}\tsemapv:UnspecifiedMatching\t\t\t\t\n")


@pytest.fixture
def scratch_repo(tmp_path):
    """A git repository holding the package, with the repository's real .gitattributes."""
    repo = tmp_path / "repo"
    (repo / "data").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKAGE, repo / "data")
    # The committed rules, rebased onto this layout so the SAME lines are exercised.
    rules = GITATTRIBUTES.read_text(encoding="utf-8").replace("planktonzilla/planktonzilla_dataset/taxonomy/data/", "data/")
    (repo / ".gitattributes").write_text(rules, encoding="utf-8")
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _two_concurrent_add_source_branches(repo: Path):
    base = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(repo, "checkout", "-qb", "alpha")
    _add_source(repo, "alphanet", "pzt:900001")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add alphanet")
    _git(repo, "checkout", "-q", base)
    _git(repo, "checkout", "-qb", "beta")
    _add_source(repo, "betanet", "pzt:900002")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add betanet")


def test_two_concurrent_add_source_branches_merge_without_conflict(scratch_repo):
    """THE GATE, performed rather than asserted. Three conflicts before this file existed."""
    _two_concurrent_add_source_branches(scratch_repo)

    merge = _git(scratch_repo, "merge", "alpha", check=False)
    conflicted = _git(scratch_repo, "diff", "--name-only", "--diff-filter=U").stdout.split()

    assert conflicted == [], f"conflicts: {conflicted}\n{merge.stdout}{merge.stderr}"
    assert merge.returncode == 0


def test_both_branches_rows_survive_the_merge(scratch_repo):
    """Union merge must keep BOTH sides, not pick one. A silent loss here is a lost source."""
    _two_concurrent_add_source_branches(scratch_repo)
    _git(scratch_repo, "merge", "alpha", check=False)

    taxa = {row["taxonID"]: row["scientificName"] for row in read_tsv(scratch_repo / "data" / "taxon.tsv")}
    subjects = {row["subject_id"] for row in read_tsv(scratch_repo / "data" / "identifier.tsv")}
    mappings = {path.stem for path in (scratch_repo / "data" / "mappings").glob("*.tsv")}

    assert taxa.get("pzt:900001") == "alphanet" and taxa.get("pzt:900002") == "betanet"
    assert {"pzt:900001", "pzt:900002"} <= subjects
    assert {"alphanet", "betanet"} <= mappings


def test_the_conflict_is_real_without_the_merge_rules(tmp_path):
    """The control. Without .gitattributes the same two branches conflict on both ledgers.

    Worth performing rather than trusting: if git ever merged these cleanly on its own, the rules
    below would be cargo, and this test would say so.
    """
    repo = tmp_path / "bare"
    shutil.copytree(PACKAGE, repo / "data")
    _git(repo, "init", "-q", ".")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    _two_concurrent_add_source_branches(repo)
    _git(repo, "merge", "alpha", check=False)
    conflicted = set(_git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split())

    assert {"data/taxon.tsv", "data/identifier.tsv"} <= conflicted


def test_a_duplicate_id_that_union_merge_lets_through_is_a_failing_check(tmp_path):
    """Union merge keeps both rows even when both claim the same id. That has to be caught.

    This is the cost of the strategy, and the reason it is paired with the validator rather than
    adopted on its own.
    """
    work = tmp_path / "data"
    shutil.copytree(PACKAGE, work)
    rows = read_tsv(work / "taxon.tsv")
    columns = tuple(k for k in rows[0] if k != "__file__")
    clone = {**rows[10], "scientificName": "a-name-the-other-branch-chose"}
    write_tsv(work / "taxon.tsv", columns, [*rows, clone])

    report = validate.validate(work, registered=sorted(constants.DATASET_IMPORT_CONFIGS))

    assert "primary_key" in {finding.check for finding in report.errors}


def test_union_merge_is_scoped_to_the_three_append_only_ledgers():
    """Only files where "keep both" is always right. Anywhere else it would silently merge nonsense."""
    rules = [
        line.split()
        for line in GITATTRIBUTES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    union = {Path(pattern).name for pattern, *attrs in rules if "merge=union" in attrs}

    assert union == set(UNION_MERGED)
    assert "mappings" not in " ".join(str(r) for r in union), "a per-source file never needs union"


def test_the_package_and_the_frozen_csv_are_pinned_to_lf():
    """A CRLF checkout would break the byte gate before anyone edited anything."""
    text = GITATTRIBUTES.read_text(encoding="utf-8")

    assert "planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv  text eol=lf" in text
    assert "taxonomy/data/**/*.tsv  text eol=lf" in text
    assert "taxonomy/data/*.json    text eol=lf" in text


# The second registry, removed
def test_the_descriptor_is_not_a_second_source_registry():
    """Adding a source touches one registry, not two that can drift and both conflict."""
    descriptor = validate.load_descriptor(PACKAGE)
    mapping = next(r for r in descriptor["resources"] if r["name"] == "mapping")

    assert "path" not in mapping, "the descriptor enumerates sources again"
    assert mapping["pathGlob"] == "mappings/*.tsv"


def test_registry_drift_is_measured_against_constants_in_both_directions(tmp_path):
    """constants.DATASET_IMPORT_CONFIGS is the registry — the one adding a source already touches."""
    work = tmp_path / "data"
    shutil.copytree(PACKAGE, work)
    shutil.copy(work / "mappings" / "zoolake.tsv", work / "mappings" / "unregistered.tsv")

    report = validate.validate(work, registered=sorted(constants.DATASET_IMPORT_CONFIGS))
    drift = [f for f in report.findings if f.check == "registry_drift"]

    assert [f.locator for f in drift] == ["unregistered"]
    assert "not a registered source" in drift[0].detail

    # And the other way: registered with no file on disk.
    (work / "mappings" / "unregistered.tsv").unlink()
    (work / "mappings" / "zoolake.tsv").unlink()
    report = validate.validate(work, registered=sorted(constants.DATASET_IMPORT_CONFIGS))
    drift = [f for f in report.findings if f.check == "registry_drift"]

    assert [f.locator for f in drift] == ["zoolake"]
    assert "no mapping file on disk" in drift[0].detail


def test_adding_a_source_needs_no_descriptor_edit(scratch_repo):
    """The whole point: one new file, plus the constants.py entry it already needed."""
    before = (scratch_repo / "data" / "datapackage.json").read_bytes()
    _add_source(scratch_repo, "gammanet", "pzt:900003")

    assert (scratch_repo / "data" / "datapackage.json").read_bytes() == before
    assert "gammanet" in {p.stem for p in (scratch_repo / "data" / "mappings").glob("*.tsv")}
