"""
(c) Inria

Semantic consistency of the taxonomy's external identifiers: does each id actually NAME
the taxon its row describes?

`tests/test_taxonomy_validation.py` enforces the table's internal contract — an id is
well-formed, unique per label, and stable. None of that can tell a right identifier from
a plausible-looking wrong one; the failure mode is real, and this repository has already
been bitten by it (`resolve_frepj_ids.verify_ncbi_lineage` blanked nine frepj rows whose
bare-epithet search had resolved a copepod to the hydrozoan *Sarsia*). That guard ran
once, at fill time, over 229 draft rows. These tests apply the same idea to every row and
every registry, continuously.

**Network-free, like every other taxonomy test.** They read a committed snapshot of what
WoRMS / NCBI Taxonomy / Wikidata / the GBIF backbone say about each identifier and label
(`tests/fixtures/taxonomy/external_id_snapshot.tsv`), produced on demand by
`planktonzilla/planktonzilla_dataset/utils/verify_taxon_ids.py`:

    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids

Refresh the snapshot when identifiers change, and read the diff — a registry that has
since re-circumscribed a taxon shows up there rather than as a surprise in CI.

The scoring is deliberately generous about NAMES and strict about ORGANISMS: orthographic
variants, accepted-vs-synonym pairs, and the coarse-rank propagation KI-13 documents all
pass. A `contradiction` — the registry's organism shares no name and no lineage with the
row's — is the only failing verdict, and requires positive lineage evidence, so a
registry that returns a bare name can never manufacture one.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
from collections import Counter

import pytest

from planktonzilla.planktonzilla_dataset.utils import verify_taxon_ids as verifier

_CSV_PATH = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"
_SNAPSHOT_PATH = root / "tests" / "fixtures" / "taxonomy" / "external_id_snapshot.tsv"

# Identifiers a registry no longer resolves. Empty on purpose: the one the checker found
# — NCBI 941245 on the four `asterolamprales` rows, retired upstream with no replacement
# node (a name search lands on *Coscinodiscales*, a different order) — was blanked rather
# than kept, so nothing in the table now points at an id nobody recognizes. An entry here
# needs a reason for why a dead pointer is worth shipping.
ACCEPTED_UNKNOWN_IDS: dict[tuple[str, str], str] = {}

# Registry records that describe something other than the row's taxon, each checked by
# hand and kept. Keyed by (proposed_label, id column) -> why it stays. Anything NOT listed
# fails `test_no_identifier_names_a_different_organism`.
ACCEPTED_CONTRADICTIONS: dict[tuple[str, str], str] = {
    ("kapelodinium vestifici", "wikidata_ID"): (
        "Q25364681 is Torodinium. This is the single genuine wikidata collision KNOWN_ISSUES "
        "KI-13 already records (the QID is stamped on two distinct taxa); correcting it changes "
        "a published column and stays with that open entry."
    ),
    ("odontella sinensis", "NCBI_ID"): (
        "NCBI 1514140 is *Trieres chinensis* — the same organism after Odontella was split into "
        "Trieres, with the epithet corrected sinensis->chinensis in the same revision. The epithet "
        "change is why the recombination rule cannot see it. Adopting NCBI's combination is a "
        "taxonomic-currency decision for the maintainer, not an id repair; the table keeps its own "
        "naming, as it does for the two sibling Odontella rows the checker reports as recombinations."
    ),
    ("radiozoa", "NCBI_ID"): (
        "NCBI 543769 is the clade *Rhizaria*, which CONTAINS Radiozoa — NCBI has no Radiozoa node "
        "at all, and no Chromista kingdom, so nothing in its lineage can overlap this table's "
        "chromista/radiozoa. Coarser than the taxon rather than wrong: KI-13's documented "
        "coarse-propagation bucket, seen across two classification systems. Affects 7 rows."
    ),
}

# Ranked labels the GBIF backbone will not match, pinned as a set so a NEW one — the
# signature of a typo or a fabricated name entering a rank column — fails the suite.
# Four causes, all benign and each checked by hand: informal EcoTaxa morphotypes
# registered in `constants.SUB_RANK_LABELS`; high-rank names GBIF declines to match when
# its own backbone disagrees with this table's rank vocabulary (the KI-31 orders,
# `hexapoda` as a Class); genus homonyms it refuses to disambiguate even given the
# lineage hints; and species its backbone simply does not carry, where it answers with
# the genus at `matchType=HIGHERRANK` — which the checker refuses rather than accept as a
# name match (`rhinomonas nottbeckii`, whose genus lineage GBIF confirms exactly).
GBIF_UNMATCHED_LABELS = frozenset(
    {
        "achelata", "actinosphaerium nucleofilum", "anabaenopsis", "animalia", "arthropoda",
        "asteromphalus", "biddulphia", "bosmina", "camptocercus", "centropyxis discoides",
        "ceratoperidinium margalefii", "ceriantharia", "chaetoceros inter ciliate",
        "chaetoceros inter. calothrix", "chroococcus", "chydorus", "cirripedia",
        "cladopyxis quadrispina", "climacodium crocosphaera", "coscinodiscids", "coscinodiscus",
        "cyanophyceae", "diacyclops biscuspidatus", "diaphanosoma nipponica", "diatoma",
        "dictyocysta", "dinobryon coalescens", "echidnophaga", "eucyclops", "eutintinnus",
        "fragilaria", "gymnodinium", "haslea silbo", "helix", "hemiaulus", "hexapoda", "hydra",
        "leydigia cliata", "macrothrix", "melosira", "microstellaria", "neobrightwellia alternans",
        "nodularia", "obelia", "odontella sp.", "oltmannsiellopsis viridis", "pennales",
        "phyllodocidae", "rhinomonas nottbeckii", "rotifera", "spirorbis", "synedra",
        "teleostei", "thecofilosea",
        "thecosomata", "thecostraca", "trichodesmium", "tripos geniculatus", "volvox", "vorticella",
    }
)


@pytest.fixture(scope="module")
def rows():
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def snapshot():
    assert _SNAPSHOT_PATH.is_file(), (
        f"missing {_SNAPSHOT_PATH.relative_to(root)} — regenerate it with "
        "`python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids`"
    )
    return verifier.load_snapshot(_SNAPSHOT_PATH)


@pytest.fixture(scope="module")
def findings(rows, snapshot):
    return verifier.score_table(rows, snapshot)


def _describe(finding):
    return (
        f"row {finding['row']} {finding['dataset']}/{finding['raw_label']!r} "
        f"[{finding['label']}] {finding['column']}={finding['value']}: {finding['detail']}"
    )


# --- The snapshot itself ---------------------------------------------------------------


def test_every_identifier_in_the_table_has_a_snapshot_entry(rows, snapshot):
    """A stale snapshot must not silently stop checking a newly added identifier."""
    missing = []
    for index, row in enumerate(rows):
        for column, source in verifier.SOURCE_FOR_COLUMN.items():
            value = row[column].strip()
            if value and (source, verifier.snapshot_key(column, value)) not in snapshot:
                missing.append((index, column, value))
    assert missing == [], (
        f"{len(missing)} identifier(s) absent from the snapshot, e.g. {missing[:5]} — refresh it with "
        "`python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids`"
    )


def test_the_snapshot_has_no_stale_entries(rows, snapshot):
    """Every snapshot row still corresponds to something the table uses."""
    live = {(source, verifier.snapshot_key(column, row[column]))
            for row in rows
            for column, source in verifier.SOURCE_FOR_COLUMN.items()
            if row[column].strip()}
    live |= {(verifier.GBIF_SOURCE, row["proposed_label"].strip().lower()) for row in rows}
    stale = sorted(key for key in snapshot if key not in live)
    assert stale == [], f"{len(stale)} snapshot entry/entries no longer used by the table: {stale[:5]}"


def test_the_registries_were_actually_reachable(findings):
    """A snapshot taken while a registry was down would silently check nothing.

    `unchecked` is the honest verdict for a failed lookup, so it must stay a small
    minority — if it dominates, the snapshot needs re-fetching, not trusting.
    """
    counts = Counter(finding["verdict"] for finding in findings)
    unchecked = counts[verifier.VERDICT_UNCHECKED]
    assert unchecked < len(findings) * 0.2, f"{unchecked}/{len(findings)} identifier checks are unchecked: {counts}"


# --- The checks that matter ------------------------------------------------------------


def test_no_identifier_names_a_different_organism(findings):
    """THE check: an id whose registry record shares no name and no lineage with its row.

    This is the wrong-taxon class — a homonym collision, a copy-paste from a neighbouring
    row, an id that drifted when a registry re-circumscribed a taxon.
    """
    contradictions = [
        finding
        for finding in findings
        if finding["verdict"] == verifier.VERDICT_CONTRADICTION
        and (finding["label"], finding["column"]) not in ACCEPTED_CONTRADICTIONS
    ]
    assert contradictions == [], "\n".join(
        [f"{len(contradictions)} identifier(s) name a different organism:"] + [_describe(f) for f in contradictions]
    )


def test_every_identifier_still_resolves(findings):
    """An id no registry recognizes cannot be verified by anyone downstream either."""
    unknown = [
        finding
        for finding in findings
        if finding["verdict"] == verifier.VERDICT_UNKNOWN
        and (verifier.SOURCE_FOR_COLUMN[finding["column"]], verifier.snapshot_key(finding["column"], finding["value"]))
        not in ACCEPTED_UNKNOWN_IDS
    ]
    assert unknown == [], "\n".join(
        [f"{len(unknown)} identifier(s) no longer resolve at their registry:"] + [_describe(f) for f in unknown]
    )


def test_labels_are_names_the_gbif_backbone_recognizes(rows, snapshot):
    """Every ranked label is checked against the GBIF backbone, in OUR lineage.

    The query carries the row's Kingdom..Family as hints, so this asks the strong
    question — is this name a taxon in this branch — rather than the weak one. It is what
    catches a fabricated or badly misspelled rank value, the failure the `Florenciellaceae`
    repair ran into where an obvious-looking family name had never been published.
    """
    findings = verifier.score_labels_against_gbif(rows, snapshot)
    contradictions = [f for f in findings if f["verdict"] == verifier.VERDICT_CONTRADICTION]
    assert contradictions == [], "\n".join(
        ["labels whose GBIF match is a different organism:"] + [_describe(f) for f in contradictions]
    )

    unmatched = {f["label"] for f in findings if f["verdict"] == verifier.VERDICT_UNKNOWN}
    appeared = unmatched - GBIF_UNMATCHED_LABELS
    assert appeared == set(), (
        f"{len(appeared)} label(s) the GBIF backbone stopped recognizing — a typo or a fabricated name in a rank "
        f"column looks exactly like this: {sorted(appeared)}"
    )
    resolved = GBIF_UNMATCHED_LABELS - unmatched
    assert resolved == set(), (
        f"{len(resolved)} label(s) GBIF now matches; drop them from GBIF_UNMATCHED_LABELS: {sorted(resolved)}"
    )


# --- The scoring itself, so the checks above cannot rot into vacuous passes -------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("globorotalidae", "Globorotaliidae", True),  # orthographic variant
        ("kryptoperidinium triquetrum", "Kryptoperidinium triquetrum", True),
        ("tripos", "Tripos Bory, 1823", True),  # authority stripped
        ("ctenophora", "Cladoceramus", False),  # unrelated names
        ("daphnia pulex", "Bivalvia", False),
    ],
)
def test_name_similarity_separates_variants_from_different_names(left, right, expected):
    assert (verifier.name_similarity(left, right) >= verifier.NEAR_MATCH_RATIO) is expected


def test_scoring_recognizes_the_documented_relationships():
    """The four non-defect verdicts, each on a case the table actually contains."""
    species_row = {
        "Dataset": "frepj", "Raw_Labels": "x", "Kingdom": "animalia", "Phylum": "arthropoda",
        "Class": "branchiopoda", "Order": "anomopoda", "Family": "daphniidae", "Genus": "daphnia",
        "Species": "pulex", "proposed_label": "daphnia pulex",
    }
    exact = verifier.score_row_against_record(
        species_row, {"status": "ok", "name": "daphnia pulex", "rank": "species", "lineage": set()}
    )
    assert exact.verdict == verifier.VERDICT_EXACT

    # KI-13's coarse propagation: the genus taxid stamped on a species row.
    ancestor = verifier.score_row_against_record(
        species_row, {"status": "ok", "name": "daphnia", "rank": "genus", "lineage": set()}
    )
    assert ancestor.verdict == verifier.VERDICT_ANCESTOR

    # An accepted-name/synonym pair: different name, same branch.
    lineage = verifier.score_row_against_record(
        species_row,
        {"status": "ok", "name": "daphnia obtusa", "rank": "species", "lineage": {"daphniidae", "arthropoda"}},
    )
    assert lineage.verdict == verifier.VERDICT_LINEAGE

    # The defect this suite exists for.
    contradiction = verifier.score_row_against_record(
        species_row,
        {"status": "ok", "name": "cladoceramus", "rank": "genus", "lineage": {"mollusca", "inoceramidae"}},
    )
    assert contradiction.verdict == verifier.VERDICT_CONTRADICTION
    assert contradiction.is_defect


def test_a_bare_name_mismatch_never_manufactures_a_contradiction():
    """Without a lineage there is no evidence of a different organism — only a report."""
    row = {
        "Dataset": "d", "Raw_Labels": "x", "Kingdom": "animalia", "Phylum": "cnidaria", "Class": "", "Order": "",
        "Family": "", "Genus": "", "Species": "", "proposed_label": "ctenophora",
    }
    verdict = verifier.score_row_against_record(row, {"status": "ok", "name": "something else", "rank": "", "lineage": set()})
    assert verdict.verdict == verifier.VERDICT_NAME_MISMATCH
    assert not verdict.is_defect


def test_a_rankless_bucket_row_is_never_contradicted():
    """`artefact` / `other` / `egg` rows assert no organism for an id to contradict."""
    row = {rank: "" for rank in ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")}
    row |= {"Dataset": "d", "Raw_Labels": "x", "proposed_label": "detritus"}
    record = {"status": "ok", "name": "daphnia", "rank": "genus", "lineage": {"arthropoda"}}
    assert not verifier.score_row_against_record(row, record).is_defect


def test_a_wrong_identifier_in_the_real_table_would_be_caught(rows, snapshot):
    """Falsification: the suite passing must mean something.

    Two identifiers already in the snapshot are moved onto rows they do not describe —
    the exact shape of the KI-29 defect (a mollusc's AphiaID on a crustacean row) and a
    cross-kingdom taxid. Both must turn `test_no_identifier_names_a_different_organism`
    red. The control confirms the checker still tolerates the coarse-but-correct case it
    is designed to allow, so it is not simply flagging everything.
    """
    baseline = len([f for f in verifier.score_table(rows, snapshot) if f["verdict"] == verifier.VERDICT_CONTRADICTION])

    def contradictions_after(label, column, value):
        mutated = [dict(row) for row in rows]
        for row in mutated:
            if row["proposed_label"] == label:
                row[column] = value
        return len([f for f in verifier.score_table(mutated, snapshot) if f["verdict"] == verifier.VERDICT_CONTRADICTION])

    # Bivalvia's AphiaID on the copepod order rows — shares only the kingdom.
    assert contradictions_after("calanoida", "aphia_ID", "105.0") > baseline
    # Daphnia's taxid on a diatom genus.
    assert contradictions_after("chaetoceros", "NCBI_ID", "6668.0") > baseline
    # Control: a genus taxid on its own species row is the documented KI-13 bucket.
    assert contradictions_after("daphnia pulex", "NCBI_ID", "6668.0") == baseline


def test_an_identifier_absent_from_the_snapshot_is_never_silently_accepted(rows, snapshot):
    """The other half of the falsification: a NEWLY WRONG id has no snapshot entry.

    Scoring alone would call it `unchecked` and pass, so coverage is what catches it —
    `test_every_identifier_in_the_table_has_a_snapshot_entry` goes red and asks for a
    refresh, after which the scoring check sees the record and judges it.
    """
    mutated = [dict(row) for row in rows]
    for row in mutated:
        if row["proposed_label"] == "acantharia":
            row["NCBI_ID"] = "65575.0"  # one digit off; never fetched, so never verified
    uncovered = [
        (index, column)
        for index, row in enumerate(mutated)
        for column, source in verifier.SOURCE_FOR_COLUMN.items()
        if row[column].strip() and (source, verifier.snapshot_key(column, row[column])) not in snapshot
    ]
    assert uncovered, "a typo'd identifier must show up as missing snapshot coverage"


def test_a_failed_lookup_is_never_read_as_a_defect():
    row = {rank: "" for rank in ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")}
    row |= {"Dataset": "d", "Raw_Labels": "x", "proposed_label": "daphnia", "Genus": "daphnia", "Phylum": "arthropoda"}
    assert verifier.score_row_against_record(row, None).verdict == verifier.VERDICT_UNCHECKED
    error = verifier.score_row_against_record(row, {"status": "error", "name": "", "rank": "", "lineage": set()})
    assert error.verdict == verifier.VERDICT_UNCHECKED
    assert not error.is_defect
