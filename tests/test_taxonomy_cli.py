"""
(c) Inria

``pz_taxonomy`` — the curator's front end, tested through ``main(argv)`` rather than through the
functions it calls, because the thing being asserted is what a curator gets when they type it.

One property runs through all of it: **nothing writes without ``--apply``.** The tools this
replaces wrote on every run, and one of them destroyed 644 rows when re-run with no arguments at
all. So the dry run is not a convenience mode here, it is the default, and a test that only checked
the write path would miss the half that matters.
"""

import shutil
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest

from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR, cli, loader
from planktonzilla.planktonzilla_dataset.taxonomy.model import read_tsv


@pytest.fixture
def package(tmp_path):
    work = tmp_path / "data"
    shutil.copytree(PACKAGE_DIR, work)
    loader.cache_clear()
    yield work
    loader.cache_clear()


def run(package, *argv):
    return cli.main(["--package", str(package), *argv])


def fingerprint(package):
    return {path: path.read_bytes() for path in sorted(package.rglob("*")) if path.is_file()}


# The default
@pytest.mark.parametrize(
    "argv",
    [
        ("fmt",),
        ("rename", "pzt:000071", "bosmina renamed"),
        ("retire", "pzt:000071", "pzt:000072", "--reason", "duplicate"),
        ("clear-id", "pzt:000071", "aphia_ID", "--reason", "withdrawn"),
    ],
)
def test_no_command_writes_without_apply(package, argv):
    """The property that makes this CLI safe to hand to a curator, on every write command."""
    before = fingerprint(package)

    assert run(package, *argv) == 0
    assert fingerprint(package) == before


def test_apply_is_what_writes(package, capsys):
    """The control: without ``--apply`` these are the same command and only one of them writes."""
    assert run(package, "rename", "pzt:000071", "bosmina renamed", "--apply") == 0

    assert "applied 2 change(s)" in capsys.readouterr().out
    assert loader.load_taxonomy(package).taxon("pzt:000071").scientific_name == "bosmina renamed"
    assert [row for row in read_tsv(package / "taxon.tsv") if row["scientificName"] == "bosmina renamed"]


def test_a_dry_run_says_what_it_would_do_and_how_to_do_it(package, capsys):
    assert run(package, "rename", "pzt:000071", "bosmina renamed") == 0

    out = capsys.readouterr().out
    assert "would apply 2 change(s)" in out
    assert "'bosmina fatalis' -> 'bosmina renamed'" in out
    assert "--apply" in out


# check
def test_check_reports_the_committed_package_as_error_free(package, capsys):
    """168 warnings and no errors is the committed state; a non-zero exit would be CI red forever."""
    assert run(package, "check") == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_check_exits_non_zero_on_an_error(package, capsys):
    """A validator whose exit code does not move is a validator CI cannot use."""
    path = package / "mappings" / "zoolake.tsv"
    rows = path.read_text(encoding="utf-8").split("\n")
    rows[1] = rows[1].replace("\tliving\t", "\tnot_a_root_class\t", 1)
    path.write_text("\n".join(rows), encoding="utf-8")

    assert run(package, "check") == 1
    assert "ERROR" in capsys.readouterr().out


# show, render, diff
def test_show_names_the_concept_behind_every_class_directory(package, capsys):
    assert run(package, "show", "zoolake") == 0

    out = capsys.readouterr().out
    assert "35 class directories in zoolake." in out
    assert "pzt:000661" in out and "aphanizomenon" in out


def test_render_writes_the_committed_bytes(package, tmp_path, capsys):
    from planktonzilla.planktonzilla_dataset import constants

    out_path = tmp_path / "rendered.csv"
    assert run(package, "render", "--out", str(out_path)) == 0

    assert out_path.read_bytes() == constants.DEFAULT_TAXONOMY_CSV_FILENAME.read_bytes()
    assert "2358 rows" in capsys.readouterr().out


def test_diff_against_the_committed_csv_is_empty_on_a_clean_package(package, capsys):
    assert run(package, "diff") == 0
    assert capsys.readouterr().out.strip() == "no changes"


def test_diff_names_the_published_cells_a_curation_moved(package, capsys):
    run(package, "rename", "pzt:000071", "bosmina renamed", "--apply")
    capsys.readouterr()

    assert run(package, "diff") == 0
    out = capsys.readouterr().out
    assert "would apply 2 change(s)" in out
    assert "proposed_label" in out and "Species" in out


# Refusals
def test_a_refusal_is_one_line_and_exits_two(package, capsys):
    """A curator reading a traceback to find the sentence that says what to do is the tool failing."""
    assert run(package, "retire", "pzt:000070", "pzt:000069", "--reason", "redundant") == 2

    captured = capsys.readouterr()
    assert captured.err.startswith("refused: ")
    assert "re-parent them before retiring it" in captured.err
    assert len(captured.err.strip().splitlines()) == 1


@pytest.mark.parametrize("argv", [("retire", "pzt:000071", "pzt:000072"), ("clear-id", "pzt:000071", "aphia_ID")])
def test_a_reason_is_required_by_the_parser_not_defaulted(package, argv):
    """There is no default reason. An id lost with no reason recorded is how ids get lost."""
    with pytest.raises(SystemExit) as exit_code:
        run(package, *argv)

    assert exit_code.value.code == 2


# Step 9 — release, the CI summary, the generated runbook
def test_a_release_freezes_todays_order_and_pins_it(package, capsys):
    """A release answers "which rows did we publish, in which order?" — needed because the order
    WITHIN a label is not derivable: 56 runs covering 545 rows of the frozen prefix obey no key."""
    from planktonzilla.planktonzilla_dataset.taxonomy.model import read_tsv

    assert run(package, "release", "v1.1", "--apply") == 0

    cut = package / "release" / "v1.1"
    order = read_tsv(cut / "legacy_row_order.tsv")
    assert len(order) == 2358
    assert [row["sequence"] for row in order] == [str(i) for i in range(2358)]
    assert (cut / "sha256").read_text(encoding="utf-8").strip()
    # Per-release, and empty: a correction pinned against v1.0's bytes says nothing about v1.1.
    assert read_tsv(cut / "legacy_overrides.tsv") == []


def test_cutting_a_release_never_edits_an_existing_one(package, capsys):
    """v1.0's manifest and pin are what let the loader prove the frozen block has not drifted."""
    before = {path: path.read_bytes() for path in (package / "release" / "v1.0").iterdir()}

    assert run(package, "release", "v1.0", "--apply") == 2
    assert "already exists" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in (package / "release" / "v1.0").iterdir()} == before


def test_release_writes_nothing_without_apply(package):
    assert run(package, "release", "v1.1") == 0
    assert not (package / "release" / "v1.1").exists()


def test_the_ci_summary_is_one_sentence_a_reviewer_can_act_on(package, capsys):
    """A 2358-line CSV diff does not say whether a curation moved published data. This does."""
    run(package, "rename", "pzt:000071", "bosmina renamed", "--apply")
    capsys.readouterr()

    assert run(package, "diff", "--summary") == 0
    out = capsys.readouterr().out
    assert "**2 published cell(s) changed on 1 row(s)**" in out
    assert "- `proposed_label`: 1" in out and "- `Species`: 1" in out


def test_the_ci_summary_says_so_when_nothing_published_moved(package, capsys):
    """Provenance, vocabularies and predicates all move without moving a published cell."""
    assert run(package, "diff", "--summary") == 0
    assert capsys.readouterr().out.strip() == "No published cell changed."


def test_the_committed_runbook_is_what_the_generator_produces():
    """A hand-written runbook rots silently — wrong counts, renamed commands, nothing red.

    This is the test that makes "generated" mean something: the committed file has to equal what
    the command produces from today's package, or the suite says so.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR

    committed = (Path(root) / "docs" / "TAXONOMY_RUNBOOK.md").read_text(encoding="utf-8")

    assert committed == cli.runbook(PACKAGE_DIR), "docs/TAXONOMY_RUNBOOK.md is stale: run `pz_taxonomy runbook --out`"


def test_every_command_documents_itself_in_the_runbook():
    """The table is generated from the handlers' docstrings, so a command added without a sentence
    explaining itself shows up as **undocumented** rather than as nothing at all."""
    from planktonzilla.planktonzilla_dataset.taxonomy import PACKAGE_DIR

    assert "**undocumented**" not in cli.runbook(PACKAGE_DIR)


def test_the_retirement_switch_is_documented_and_not_thrown():
    """Step 9 documents it; throwing it waits on a golden diff against the published Hub artefact.

    Pinned because the switch is one line and the temptation is to flip it once `diff` and `check`
    are both clean — which they are. What they cannot show is that the 17.4 M-row join on the Hub
    matches, and the CSV is the only checkable artefact until something does.
    """
    from planktonzilla.planktonzilla_dataset import constants
    from planktonzilla.planktonzilla_dataset.taxonomy import loader

    assert loader.RETIREMENT_SWITCH_THROWN is False
    assert loader.default_source() == constants.DEFAULT_TAXONOMY_CSV_FILENAME
