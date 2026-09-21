"""
(c) Inria

Characterization pin for the TRAINING LABEL VOCABULARY — the projection of the taxonomy
table that decides the integer class ids of the released models.

``build_only_plankton`` derives its ``ClassLabel`` names by sorting the distinct
``" ".join(non-empty ranks)`` strings of the plankton rows
(``gen_planktonzilla_only_plankton.py:95``). Nothing pinned the result. That made this the
frozen contract with the least protection AND the most invisible failure mode: an edit that
adds, removes or re-spells one taxonomy string silently renumbers every class id after it,
and a model checkpoint's ``id2label`` then points at the wrong taxon with no test going red,
no column changing, and nothing visible in the CSV diff.

The exposure is not hypothetical. The six sources already curated but not yet published move
the class id of **591 of the 599** v1.0 names when they land (pinned below). The published
vocabulary therefore has to be a frozen artifact keyed to a release, not a value recomputed
from whatever the table happens to hold — which is what
``docs/TAXONOMY_IMPLEMENTATION_PLAN.md`` schedules as its own step. This module is the gate
that has to exist first, so that step (and every migration step after it) has something to
be measured against.

Two vocabularies are pinned, as committed TSV files rather than as bare counts, so a diff
shows WHICH names moved and to which class id. **Step 8 moved them into the package**, where
``build_only_plankton`` now reads them, so they stopped being test data that happened to match
the code and became the artifact the build encodes against:

    taxonomy/data/vocab/labels/v1.0_taxpath.tsv   599 names, the 15 published sources
    taxonomy/data/vocab/labels/v1.2_taxpath.tsv   850 names, all 21 registered sources

What this module asserts is unchanged and is now worth more: that the SHIPPED file equals what a
pure projection of the committed table produces today. It is the bridge between the frozen artifact
and the live data, and it goes red when they part company — which is the moment a new tag is due.

Predicate, stated because it is easy to get wrong: these are projections of the COMMITTED
TABLE, one row per ``(Dataset, Raw_Labels)`` class directory. The published dataset's
vocabulary is the same set minus any class directory that contributed no image, which cannot
be checked offline — ``samples.json`` counts images per source, not per class directory. What
this module pins is exactly what M7 asks for: that a pure projection of the table reproduces
today's names, in today's order.
"""

import json
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import pytest
from datasets import Dataset

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.gen_planktonzilla_only_plankton import (
    TAXONOMY_COLS,
    build_only_plankton,
)
from planktonzilla.planktonzilla_dataset.generate_planktonzilla import build_taxonomy_lookup
from planktonzilla.planktonzilla_dataset.taxonomy import render as taxonomy_render

# Step 8 moved these out of tests/fixtures/ and into the package: a released vocabulary is a
# SHIPPED artifact that the build reads, not test data that happens to match. This module keeps
# pointing at them, which is now the proof that the shipped file equals today's expression.
RELEASED = taxonomy_render.RELEASED_VOCABULARIES
V1_0_TSV = RELEASED / "v1.0_taxpath.tsv"
V1_2_TSV = RELEASED / "v1.2_taxpath.tsv"
SAMPLES_JSON = Path(root) / "samples.json"

# The six sources curated in the table ahead of their arrival in the published artifact.
# Derived below from samples.json rather than restated, but named here for the message.
PENDING_HINT = "frepj, daplankton, tara_pacific_{bongo,decknet,hsn,manta}"


def _published_dataset_names() -> set:
    """The 15 ``dataset`` values present in the published planktonzilla-17M.

    Read from ``samples.json`` — a scan of the frozen Hub artifact, i.e. observed data —
    the same source ``tests/test_dataset_licenses.py`` uses, so "v1.0" means one thing in
    the repository rather than two hard-coded lists that can drift apart.
    """
    return {row["dataset"] for row in json.loads(SAMPLES_JSON.read_text())["counts"]}


def _read_vocabulary(path: Path) -> list:
    """Read a committed ``class_id<TAB>tax_label`` fixture back into an ordered name list.

    The class id is not ignored: it is asserted to be the row's own position, so a fixture
    whose ids and order disagree cannot silently pass as the expectation.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "class_id\ttax_label", f"{path.name}: unexpected header {lines[0]!r}"
    assert lines[-1] == "", f"{path.name}: must end with a trailing newline"

    names = []
    for position, line in enumerate(lines[1:-1]):
        class_id, _, name = line.partition("\t")
        assert int(class_id) == position, f"{path.name}: row {position} carries class_id {class_id}"
        names.append(name)
    return names


def _class_rows() -> list:
    """One row per class directory, in the shape ``build_only_plankton`` sees.

    Built through ``build_taxonomy_lookup`` — the reader of record — rather than by parsing
    the CSV again, so the projection is pinned against the SAME null and type handling the
    published build uses. Two of its behaviours matter here and are load-bearing: blank
    cells become ``None`` (not ``""``), and ``plankton`` arrives as a real ``bool``.
    """
    lookup = build_taxonomy_lookup(str(constants.DEFAULT_TAXONOMY_CSV_FILENAME))
    return [{"dataset": dataset, **columns} for (dataset, _raw_label), columns in lookup.items()]


def _project(rows) -> list:
    """The vocabulary projection, transcribed from ``build_only_plankton``.

    Kept as a local transcription rather than a call so the two derivations stay
    independent; ``test_the_shipped_builder_reproduces_the_projection`` asserts they agree.
    """
    kept = [row for row in rows if row["plankton"] is True and row["Kingdom"] != ""]
    return sorted({" ".join(row[col] for col in TAXONOMY_COLS if row[col] not in ("", None)) for row in kept})


def _first_divergence(actual, expected) -> str:
    """Name the first class id where two vocabularies differ, not just that they do."""
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            return f"first divergence at class id {index}: got {got!r}, expected {want!r}"
    longer, label = (actual, "actual") if len(actual) > len(expected) else (expected, "expected")
    shared = min(len(actual), len(expected))
    return f"identical up to class id {shared}; {label} continues with {longer[shared]!r}"


@pytest.fixture(scope="module")
def rows():
    return _class_rows()


@pytest.fixture(scope="module")
def published():
    return _published_dataset_names()


def test_the_v1_2_vocabulary_matches_the_committed_snapshot(rows):
    """All 21 registered sources: 850 names, in order."""
    expected = _read_vocabulary(V1_2_TSV)
    actual = _project(rows)

    assert len(expected) == 850, "the fixture itself changed size"
    assert actual == expected, _first_divergence(actual, expected)


def test_the_v1_0_vocabulary_matches_the_committed_snapshot(rows, published):
    """The 15 published sources: 599 names, in order — what the released models encode."""
    expected = _read_vocabulary(V1_0_TSV)
    actual = _project([row for row in rows if row["dataset"] in published])

    assert len(published) == 15, f"samples.json no longer describes 15 sources: {sorted(published)}"
    assert len(expected) == 599, "the fixture itself changed size"
    assert actual == expected, _first_divergence(actual, expected)


def test_the_shipped_builder_reproduces_the_projection(rows):
    """The transcription above is not a second opinion — run the real builder and compare.

    A synthetic in-memory dataset carrying one row per class directory has exactly the same
    DISTINCT taxonomy strings as the 17.4 M-row published set, so it produces the same
    ``ClassLabel`` names while running in a second. ``image`` rides along as a string: the
    builder only passes it through (``gen_planktonzilla_only_plankton.py:106``).
    """
    dataset = Dataset.from_dict(
        {
            "image": [""] * len(rows),
            "dataset": [row["dataset"] for row in rows],
            "plankton": [row["plankton"] for row in rows],
            **{rank: [row[rank] for row in rows] for rank in TAXONOMY_COLS},
        }
    )

    built = build_only_plankton(dataset, num_proc=1)
    actual = built.features["label"].names
    expected = _read_vocabulary(V1_2_TSV)

    assert actual == expected, _first_divergence(actual, expected)


def test_the_empty_string_class_sits_at_index_zero(rows, published):
    """KI: five plankton rows carry no Kingdom at all, and they form class id 0.

    Not a rounding detail — index 0 is the id every untrained head predicts first, and the
    class it names is "no taxonomy at all". Pinned with the rows that cause it so a curation
    fix to any of the five shows up here as the vocabulary change it really is.
    """
    no_kingdom = [row for row in rows if row["plankton"] is True and row["Kingdom"] is None]

    assert len(no_kingdom) == 5
    by_dataset = {}
    for row in no_kingdom:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
    assert by_dataset == {"global_uvp5": 1, "whoi": 2, "zoolake": 1, "tara_pacific_bongo": 1}

    # Four of the five are in v1.0, so the empty class exists in BOTH vocabularies.
    assert sum(count for name, count in by_dataset.items() if name in published) == 4
    assert _read_vocabulary(V1_2_TSV)[0] == ""
    assert _read_vocabulary(V1_0_TSV)[0] == ""


def test_the_kingdom_filter_is_a_no_op_under_published_semantics(rows):
    """``x["Kingdom"] != ""`` is documented as "with a Kingdom assigned". It is not.

    ``build_taxonomy_lookup`` maps a blank cell to ``None``, and ``None != ""`` is true, so
    the filter admits exactly the rows it reads as excluding. That is the whole origin of
    the empty-string class above. Pinned as behaviour, NOT as a defect to fix here: changing
    it would drop class 0 and renumber all 849 names after it.
    """
    plankton_rows = [row for row in rows if row["plankton"] is True]
    kept = [row for row in plankton_rows if row["Kingdom"] != ""]

    assert kept == plankton_rows, "the Kingdom filter now excludes something — the vocabulary just moved"
    assert any(row["Kingdom"] is None for row in kept)


def test_the_repeated_token_strings_are_present(rows):
    """Six names repeat a token. Four are KI-8 rank-slot contamination; two are tautonyms.

    The distinction is the point. ``porpita porpita`` and ``eudactylota eudactylota`` are
    correct binomials and ``docs/TAXONOMY_REPRESENTATION.md`` §3 lists tautonyms among the
    conventions a representation change must NOT "fix". The other four are the same name
    occupying two rank slots of one row. Both kinds are frozen, for opposite reasons.
    """
    repeated = [name for name in _project(rows) if len(name.split()) != len(set(name.split()))]

    assert repeated == [
        "animalia cnidaria hydrozoa anthoathecata porpitidae porpita porpita",
        "animalia rotifera eurotatoria ploima euchlanidae eudactylota eudactylota",
        "chromista cryptophyta cryptophyta kathablepharidacea katablepharidaceae katablepharis remigera",
        "chromista heterokontophyta bacillariophyceae bacillariophyceae bacillariophyceae neomoelleria cornuta",
        "chromista myzozoa dinophyceae dinophyceae amphidomataceae azadinium caudatum",
        "chromista ochrophyta dictyochophyceae florenciellales florenciellales pseudochattonella farcimen",
    ]


def test_landing_the_pending_sources_renumbers_591_of_599_class_ids(rows, published):
    """The crosswalk requirement, measured.

    The six pending sources are additive in the TABLE and catastrophic in the VOCABULARY:
    because class ids are alphabetical positions, inserting 251 names re-points almost every
    existing one. A model trained on v1.0 and an ``id2label`` built after the v1.2 push agree
    on 8 of 599 classes.

    This is the number that makes "released vocabularies are frozen snapshots, never
    recomputed" a requirement rather than a preference. It is pinned so that any future claim
    of a compatible vocabulary has to move it first.
    """
    v1_2 = _project(rows)
    v1_0 = _project([row for row in rows if row["dataset"] in published])
    position_in_v1_2 = {name: index for index, name in enumerate(v1_2)}

    moved = [name for index, name in enumerate(v1_0) if position_in_v1_2[name] != index]

    assert set(v1_0) <= set(v1_2), "a v1.0 class name vanished from the 21-source vocabulary"
    assert len(moved) == 591, f"{len(moved)} of {len(v1_0)} moved; pending sources are {PENDING_HINT}"


def test_the_vocabularies_are_sorted_deduplicated_and_tsv_safe():
    """The fixtures are what ``sorted(set(...))`` produces, and survive a TSV round trip."""
    for path in (V1_0_TSV, V1_2_TSV):
        names = _read_vocabulary(path)
        assert names == sorted(set(names)), f"{path.name} is not sorted(set(...))"
        assert not [n for n in names if "\t" in n or n != n.strip(" ")], f"{path.name} has a name a TSV cannot round-trip"


# Step 8 — the vocabulary as a shipped, frozen artifact
def test_a_released_vocabulary_is_read_not_recomputed():
    """The distinction step 8 exists for. Both tags load, in class-id order, at their pinned sizes."""
    assert len(taxonomy_render.released_vocabulary("v1.0")) == 599
    assert len(taxonomy_render.released_vocabulary("v1.2")) == 850
    assert taxonomy_render.released_vocabulary("v1.0")[0] == "", "the empty-string class left index 0"


def test_an_unpublished_tag_names_the_tags_that_exist():
    """A typo in a release tag must not fall back to computing — that is the renumbering hazard."""
    from planktonzilla.planktonzilla_dataset.taxonomy import TaxonomyError

    with pytest.raises(TaxonomyError, match=r"published tags are \['v1.0', 'v1.2'\]"):
        taxonomy_render.released_vocabulary("v1.1")


def test_a_shuffled_class_id_is_refused_at_read(tmp_path, monkeypatch):
    """The ids ARE the order. A file whose ids are unique but shuffled encodes every class wrongly.

    Uniqueness — what a primary key would check — passes on exactly that file, which is why the
    rule is positional and why it is asserted here rather than left to the schema.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy import TaxonomyError
    from planktonzilla.planktonzilla_dataset.taxonomy.model import read_tsv, write_tsv

    monkeypatch.setattr(taxonomy_render, "RELEASED_VOCABULARIES", tmp_path)
    rows = read_tsv(V1_0_TSV)
    write_tsv(tmp_path / "v9.9_taxpath.tsv", ("class_id", "tax_label"), [rows[1], rows[0], *rows[2:]])

    with pytest.raises(TaxonomyError, match="sits at position"):
        taxonomy_render.released_vocabulary("v9.9")


def test_a_label_the_release_does_not_publish_is_named_rather_than_encoded():
    """The check that turns a silent renumbering into a stopped build.

    Landing the six pending sources adds names v1.0 never published. Encoding them against v1.0
    would shift 591 of its 599 ids; this says which labels are new and stops.
    """
    v1_2_only = set(taxonomy_render.released_vocabulary("v1.2")) - set(taxonomy_render.released_vocabulary("v1.0"))
    assert v1_2_only, "the two releases would have to differ for this to mean anything"

    unknown = taxonomy_render.unknown_labels(sorted(v1_2_only)[:3], "v1.0")

    assert unknown == sorted(v1_2_only)[:3]
    assert taxonomy_render.unknown_labels(taxonomy_render.released_vocabulary("v1.0"), "v1.0") == []


def test_the_builder_refuses_to_renumber_a_released_vocabulary():
    """``build_only_plankton`` with a tag it cannot satisfy stops, and says why, before encoding.

    Exercised on the real check rather than on a dataset, because building one costs minutes and
    the assertion is about the refusal, not about datasets.
    """
    from planktonzilla.planktonzilla_dataset.taxonomy import TaxonomyError

    with pytest.raises(TaxonomyError, match="no released label vocabulary"):
        taxonomy_render.unknown_labels(["animalia"], "v0.1")
