"""
(c) Inria

Pin the ``daplankton`` block of ``planktonzilla_taxonomy.csv`` against the frozen
class-dir contract.

This exists for one reason: the taxonomy join is a SILENT LEFT JOIN. An image whose
``(dataset, Raw_Labels)`` pair has no CSV row does not fail the build — it gets null
taxonomy and null external IDs for every column, and the run carries on. The whole-source
coverage check in ``make_planktonzilla.check_taxonomy_csv`` is per-SOURCE and
non-blocking, so a source with 44 class dirs and 3 rows passes it; only
``log_lookup_coverage`` says anything, and only as a warning on the ``pz_planktonzilla``
path. One mistyped class name therefore costs a few thousand untaxonomised images with
nothing red anywhere.

So the Raw_Labels set is pinned byte-exactly against the class dirs recorded in
``tests/fixtures/daplankton/daplankton_class_dirs.tsv``, which came from enumerating the
real archive. Same guard, same shape, as ``test_frepj_taxonomy_coverage.py``.

The reuse tests below pin the other half of the contract. DAPlankton_SEA follows the
label scheme of SYKE-plankton_IFCB_2022 — its 31 classes are a strict subset of that
dataset's 50 — so those rows reuse the syke taxonomy and IDs VERBATIM rather than being
looked up a second time. That is not tidiness: ``test_forward_id_mapping_is_clean`` fails
if one ``proposed_label`` ends up carrying two different values in any ID column, so an
independent re-lookup that disagreed by one digit would break the suite.

Reads only committed files. No network.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
from pathlib import Path

from planktonzilla.dataset_import import daplankton_layout as dl
from planktonzilla.planktonzilla_dataset import constants

DATASET = "daplankton"
CLASS_DIRS_TSV = Path(__file__).parent / "fixtures" / "daplankton" / "daplankton_class_dirs.tsv"

# The four DAPlankton class dirs whose name differs from the syke Raw_Labels they reuse:
# DAPlankton writes an underscore where SYKE writes a hyphen. Spelled out rather than
# derived, so a silent re-spelling upstream shows up here as a failure and not as a rule
# quietly matching something new.
_SEA_TO_SYKE = {
    "Cryptophyceae_Teleaulax": "Cryptophyceae-Teleaulax",
    "Dolichospermum_Anabaenopsis": "Dolichospermum-Anabaenopsis",
    "Dolichospermum_Anabaenopsis_coiled": "Dolichospermum-Anabaenopsis-coiled",
    "Snowella_Woronichinia": "Snowella-Woronichinia",
}

# LAB classes that reuse a row already in the file rather than a fresh lookup.
# Levanderina_fissa is the interesting one: WoRMS and NCBI give Levanderina fissa no
# family, so the file's contiguous-prefix rule stops it at class — which makes its
# proposed_label `dinophyceae`, a label the file already carries. It therefore has to
# reuse THAT row's IDs, or one label would end up with two id sets.
_LAB_REUSE = {
    "Diatoma_tenuis": ("zoocamnet", "Diatoma tenuis"),
    "Melosira_arctica": ("syke_ifcb_2022", "Melosira_arctica"),
    "Peridiniella_catenata": ("syke_ifcb_2022", "Peridiniella_catenata_chain"),
    "Levanderina_fissa": ("syke_ifcb_2022", "Dinophyceae"),
}

# The nine taxa with no prior row anywhere in the file, resolved against WoRMS, NCBI,
# Wikidata (exact P225) and BOLD on 2026-08-27. A blank ID means the authority could not
# confirm one, which is the correct outcome rather than a gap to fill later: blanks are
# skipped by the forward-ID test, a wrong value is not.
_RESOLVED = {
    "Apocalathium_malmogiense",
    "Chrysotila_roscoffensis",
    "Gymnodinium_corollarium",
    "Kryptoperidium_foliaceum",
    "Nephroselmis_pyriformis",
    "Rhinomonas_nottbecki",
    "Rhodomonas_salina",
    "Teleaulax_acuta",
    "Tetraselmis_sp",
}

_ID_COLS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "BOLD_ID", "ecotaxa_ID")
_TAXONOMY_COLS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "proposed_label")


def _read_csv_rows():
    with constants.DEFAULT_TAXONOMY_CSV_FILENAME.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_by_key():
    return {(row["Dataset"], row["Raw_Labels"]): row for row in _read_csv_rows()}


def _daplankton_rows():
    return [row for row in _read_csv_rows() if row["Dataset"] == DATASET]


def _frozen_class_dirs():
    return [line.split("\t")[0] for line in CLASS_DIRS_TSV.read_text().splitlines()[1:]]


def test_coverage_matches_the_frozen_class_dirs():
    """The Raw_Labels set IS the class-dir set: byte-exact, 44, no duplicates.

    This is the guard against the silent left join. Set equality in BOTH directions
    matters — a missing label means untaxonomised images, an extra one means a row that
    will never match anything and quietly rots.
    """
    class_dirs = _frozen_class_dirs()
    assert len(class_dirs) == dl.N_CLASS_DIRS == 44

    raw_labels = [row["Raw_Labels"] for row in _daplankton_rows()]

    assert len(raw_labels) == len(set(raw_labels)), "duplicate daplankton Raw_Labels"
    assert len(raw_labels) == 44
    assert set(raw_labels) == set(class_dirs)


def test_every_row_is_accounted_for_as_reused_or_resolved():
    """Each of the 44 rows is one or the other — no third, undocumented category."""
    labels = {row["Raw_Labels"] for row in _daplankton_rows()}
    sea_or_lab_reuse = (labels - _RESOLVED) - set(_LAB_REUSE)

    assert _RESOLVED <= labels
    assert set(_LAB_REUSE) <= labels
    # Whatever is left reuses a syke_ifcb_2022 row under its own name or the hyphen variant.
    assert len(sea_or_lab_reuse) == 31
    assert set(_SEA_TO_SYKE) <= sea_or_lab_reuse


def test_reused_rows_are_verbatim_copies_of_their_source():
    """Taxonomy and IDs are copied, never re-derived — a one-digit drift would fail the suite."""
    by_key = _rows_by_key()

    for row in _daplankton_rows():
        label = row["Raw_Labels"]
        if label in _RESOLVED:
            continue
        source_key = _LAB_REUSE.get(label) or ("syke_ifcb_2022", _SEA_TO_SYKE.get(label, label))
        source = by_key.get(source_key)
        assert source is not None, f"«{label}» claims to reuse {source_key}, which is not in the CSV"

        for column in (*_TAXONOMY_COLS, *_ID_COLS, "plankton", "living", "root_class", "qualifier"):
            assert row[column] == source[column], (
                f"«{label}» diverges from {source_key} on {column}: {row[column]!r} != {source[column]!r}"
            )


def test_the_sea_classes_are_a_subset_of_syke_ifcb_2022():
    """The documented relationship, asserted rather than trusted.

    DAPlankton_SEA follows SYKE-plankton_IFCB_2022's label scheme, which is WHY its rows
    can be reused. If a re-release added a sea class SYKE does not have, that reuse would
    stop being valid and this fails before a wrong row is copied.
    """
    by_key = _rows_by_key()
    sea_classes = {
        row["class_dir"] for row in csv.DictReader(CLASS_DIRS_TSV.open(newline=""), delimiter="\t") if row["in_sea"] == "1"
    }

    assert len(sea_classes) == dl.N_SEA_CLASSES == 31
    for label in sorted(sea_classes):
        syke_label = _SEA_TO_SYKE.get(label, label)
        assert ("syke_ifcb_2022", syke_label) in by_key, f"sea class «{label}» has no syke_ifcb_2022 counterpart"


def test_every_row_has_a_proposed_label():
    """A null proposed_label would give those images no usable label at all."""
    empty = [row["Raw_Labels"] for row in _daplankton_rows() if not (row["proposed_label"] or "").strip()]
    assert empty == []


def test_resolved_rows_leave_unconfirmed_ids_blank_rather_than_guessed():
    """Blank is the correct answer when an authority could not confirm an ID.

    Pinned because the tempting "fix" is to fill these in from a plausible-looking search
    hit. Two such values were caught during resolution: EcoTaxa ids that came from a
    different id space than the one this column uses. A wrong ID breaks
    test_forward_id_mapping_is_clean the moment the same taxon appears under another
    source; a blank never does.
    """
    rows = {row["Raw_Labels"]: row for row in _daplankton_rows()}

    # BOLD was unreachable for every resolved taxon (Cloudflare 403 on every endpoint).
    assert all(rows[label]["BOLD_ID"] == "" for label in _RESOLVED)
    # WoRMS holds no record for Rhinomonas nottbeckii under any spelling.
    assert rows["Rhinomonas_nottbecki"]["aphia_ID"] == ""
    # Only Apocalathium malmogiense got a species-level EcoTaxa node that round-trips to
    # its own AphiaID; the rest are blank rather than coarsened to a clade id.
    assert rows["Apocalathium_malmogiense"]["ecotaxa_ID"] == "58496"
    assert all(rows[label]["ecotaxa_ID"] == "" for label in _RESOLVED - {"Apocalathium_malmogiense"})


def test_source_spelling_is_kept_in_raw_labels_even_where_it_is_wrong():
    """Raw_Labels is a join key against the image folders, not a place to correct names.

    DAPlankton ships ``Kryptoperidium_foliaceum`` — the accepted genus is
    *Kryptoperidinium*, so the archive's own dir name drops the ``-in-``. The label keeps
    the source's spelling (it has to match the folder) while the taxonomy columns carry
    the accepted one. Verified against the real archive listing on 2026-08-27.
    """
    row = {r["Raw_Labels"]: r for r in _daplankton_rows()}["Kryptoperidium_foliaceum"]

    assert row["Genus"] == "kryptoperidinium"
    assert row["proposed_label"] == "kryptoperidinium foliaceum"
    assert "Kryptoperidium_foliaceum" in _frozen_class_dirs()
