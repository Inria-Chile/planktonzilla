"""
(c) Inria

Records, constants and the in-memory store of the normalised taxonomy package.

This module is the bottom of the taxonomy package: it knows the shape of every relation and
nothing about how they are read (:mod:`loader`), rendered (:mod:`render`) or generated
(:mod:`migrate`). Everything above imports its column tuples from here, so the schema is
declared once.

:class:`TaxonomyStore` is deliberately able to hold LESS than the full model. A store built
from the normalised package carries the taxon tree, the identifier index and the release
layer; a store built from a legacy wide CSV carries only its rows. Both serve the legacy views
every current consumer needs — ``rows()``, ``lookup()``, ``frame()``, the coverage helpers —
which is what lets step 3 move the readers onto one API before the wide CSV is retired, and
what keeps the ten fixture-writing test modules working untouched.
"""

from dataclasses import dataclass, field
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants

LEGACY_RANKS = tuple(constants.TAXONOMY_RANKS)
LEGACY_ID_COLUMNS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "BOLD_ID", "ecotaxa_ID")
LEGACY_HEADER = (
    "Dataset",
    "Raw_Labels",
    *LEGACY_RANKS,
    "proposed_label",
    "plankton",
    "living",
    "root_class",
    "qualifier",
    *LEGACY_ID_COLUMNS,
)

# The 16 columns build_taxonomy_lookup returns, in its order. NOT the CSV header order and NOT
# constants.CONSOLIDATED_COLUMNS order — all three differ, and conflating them silently swaps
# id values between columns.
LOOKUP_COLUMNS = (
    *LEGACY_RANKS,
    *constants.EXTRA_COLS,
    *constants.ID_STR_COLS,
    *constants.ID_NUM_COLS,
)

TAXON_PREFIX = "pzt"
TAXON_COLUMNS = (
    "taxonID",
    "parentNameUsageID",
    "taxonRank",
    "scientificName",
    "taxonomicStatus",
    "acceptedNameUsageID",
    "nameAccordingTo",
    "kind",
    "taxonRemarks",
)
IDENTIFIER_COLUMNS = (
    "subject_id",
    "predicate_id",
    "object_id",
    "mapping_justification",
    "mapping_date",
    "author_id",
    "confidence",
    "comment",
)
MAPPING_COLUMNS = (
    "datasetID",
    "verbatimIdentification",
    "taxonID",
    "concept",
    "root_class",
    "qualifier",
    "plankton",
    "status",
    "method",
    "donor",
    "identifiedBy",
    "dateIdentified",
    "identificationReferences",
    "confidence",
    "remarks",
)
ROW_ORDER_COLUMNS = ("sequence", "datasetID", "verbatimIdentification")
# The tombstone ledger. An id present in the previous commit and absent now must appear here,
# so a published pin never dangles: `merge` writes the row, the validator enforces the rule.
MERGED_COLUMNS = ("retired_id", "replacement_id", "date", "reason")
OVERRIDE_COLUMNS = (
    "datasetID",
    "verbatimIdentification",
    *LEGACY_ID_COLUMNS,
    "root_class",
    "qualifier",
    "plankton",
    "reason",
)

# One authority per legacy ID column. `ecotaxa_legacy` is a DIFFERENT id space from today's
# EcoTaxa taxon ids and is never backfilled, so it gets its own scope rather than sharing one.
AUTHORITIES = (
    # authority, legacy column, CURIE prefix, scope, multi-valued
    ("wikidata", "wikidata_ID", "wikidata", "all life, crowd-curated", "false"),
    ("worms", "aphia_ID", "worms", "marine taxa; blank for freshwater is expected", "false"),
    ("ncbi", "NCBI_ID", "ncbi", "sequenced taxa; no Chromista/Myzozoa", "false"),
    ("bold", "BOLD_ID", "bold", "barcoded taxa", "false"),
    ("ecotaxa_legacy", "ecotaxa_ID", "ecotaxa_legacy", "legacy id space, no longer resolvable; NEVER backfill", "true"),
)
PREFIX_TO_LEGACY_COLUMN = {prefix: legacy for _name, legacy, prefix, _scope, _multi in AUTHORITIES}
MULTI_VALUED_COLUMNS = frozenset(legacy for _n, legacy, _p, _s, multi in AUTHORITIES if multi == "true")

# The seven legacy slots, as a rank vocabulary. `legacy_slot` is what makes project7() a lookup
# rather than a branch: a rank without one contributes no cell to the wide CSV, which is how the
# 12 `unranked` concept nodes stay distinct without widening the frozen columns.
RANK_VOCABULARY = (
    *((rank.lower(), str(ordinal), rank) for ordinal, rank in enumerate(LEGACY_RANKS, start=1)),
    ("unranked", "", ""),
)

# `qualifier` is blank on 253 rows today. Blank is a value, not an absence, so it is named.
UNQUALIFIED = "unqualified"

# The one predicate the identifier table carries, and the justification every migrated row was
# given. Named here so the migration and the write API cannot drift on the bytes they mint.
EXACT_MATCH = "skos:exactMatch"
UNSPECIFIED_MATCHING = "semapv:UnspecifiedMatching"

# What a lineage-less concept records about itself. Not an apology — it is the difference between
# "the source gave no lineage" and "the lineage is empty", which a reader cannot otherwise tell.
BUCKET_REMARK = "no lineage in the source table; sub-classification is a curation act"

# What a concept finer than any legacy rank records about itself — 12 such nodes exist today
# (`brachyura` under `order=decapoda`). Shared so the migration and the write API mint the same
# bytes for the same node.
FINER_REMARK = "finer than its deepest legacy rank; true rank unassigned"


def lineage_of(row) -> tuple:
    """A wide row's rank path as ``((rank, name), ...)`` — the inverse of ``render.project7``.

    Lowercase ranks, blank cells dropped. Species carry the binomial rather than the epithet the
    legacy column holds: the epithet alone is not a node identity, since 19 of them occur under
    more than one genus.
    """
    path = []
    for rank in LEGACY_RANKS:
        value = row[rank]
        if not value:
            continue
        name = f"{row['Genus']} {value}" if rank == "Species" and row["Genus"] else value
        path.append((rank.lower(), name))
    return tuple(path)


def collation_key(label: str) -> str:
    """The sort key the frozen prefix obeys: case-folded, hyphens ignored.

    One normalisation absorbs both documented exceptions (``Eukaryota``, ``pseudo-nitzschia``), and
    the frozen 1,485-row prefix has zero descents under it. That is what makes it usable as the
    placement rule for rows mapped after the freeze rather than as a new convention.
    """
    return label.lower().replace("-", "")


class TaxonomyError(RuntimeError):
    """Any refusal from the taxonomy package: a malformed store, or a render that cannot hold."""


@dataclass(frozen=True)
class Taxon:
    """One node of the taxon table — a tree node, or a lineage-less bucket."""

    taxon_id: str
    parent_id: str
    rank: str
    scientific_name: str
    status: str
    accepted_id: str
    according_to: str
    kind: str
    remarks: str

    @classmethod
    def from_row(cls, row) -> "Taxon":
        return cls(
            taxon_id=row["taxonID"],
            parent_id=row["parentNameUsageID"],
            rank=row["taxonRank"],
            scientific_name=row["scientificName"],
            status=row["taxonomicStatus"],
            accepted_id=row["acceptedNameUsageID"],
            according_to=row["nameAccordingTo"],
            kind=row["kind"],
            remarks=row["taxonRemarks"],
        )


@dataclass(frozen=True)
class Mapping:
    """One source class directory, mapped onto a concept."""

    dataset: str
    verbatim: str
    taxon_id: str
    concept: str
    root_class: str
    qualifier: str
    plankton: bool
    status: str

    @property
    def key(self) -> tuple:
        return (self.dataset, self.verbatim)

    @classmethod
    def from_row(cls, row) -> "Mapping":
        plankton = row["plankton"]
        if plankton not in ("true", "false"):
            # Excel and LibreOffice write TRUE for true. Coercing here would silently invert the
            # only boolean the published dataset carries, with no failing check once the curator
            # re-renders — measured on design A's prototype for all 120 zooscan rows.
            raise TaxonomyError(
                f"{row['datasetID']}/{row['verbatimIdentification']}: plankton is {plankton!r}, "
                f"expected 'true' or 'false'. Run `fmt` rather than hand-editing booleans."
            )
        return cls(
            dataset=row["datasetID"],
            verbatim=row["verbatimIdentification"],
            taxon_id=row["taxonID"],
            concept=row["concept"],
            root_class=row["root_class"],
            qualifier=row["qualifier"],
            plankton=plankton == "true",
            status=row["status"],
        )


@dataclass
class TaxonomyStore:
    """The loaded taxonomy, from either backend.

    ``taxa`` / ``identifiers`` / ``row_order`` / ``overrides`` are populated only by the package
    backend. ``legacy_rows`` is the common denominator both backends provide and every current
    consumer actually reads; the package backend computes it from the model (see
    :func:`render.legacy_rows`), never by parsing rendered bytes.
    """

    source: Path | None = None
    taxa: dict = field(default_factory=dict)
    mappings: dict = field(default_factory=dict)
    identifiers: dict = field(default_factory=dict)
    row_order: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    vocab: dict = field(default_factory=dict)
    is_package: bool = False
    _legacy_rows: list | None = None
    _legacy_bytes: bytes | None = None
    _lookup: dict | None = None

    # Legacy views
    def rows(self) -> list:
        """The wide rows as dicts of strings — what ``csv.DictReader`` yields today.

        Serves ``sankey``, ``verify_taxonomy_ids`` and ``verify_label_consistency`` unchanged.
        """
        if self._legacy_rows is None:
            from planktonzilla.planktonzilla_dataset.taxonomy import render

            self._legacy_rows = render.legacy_rows(self)
        return self._legacy_rows

    def frame(self):
        """The wide rows as an all-string polars frame — ``pl.read_csv(..., infer_schema_length=0)``."""
        import polars as pl

        return pl.DataFrame(self.rows(), schema={column: pl.Utf8 for column in self.legacy_header()}, orient="row")

    def legacy_header(self) -> tuple:
        """The wide columns this store can produce, in file order.

        A store loaded from an 18-column fixture reports 18, which is why the header is read off
        the rows rather than assumed: the loader must keep accepting what the tests already write.
        """
        if self._legacy_rows is not None and self._legacy_rows:
            return tuple(self._legacy_rows[0])
        return LEGACY_HEADER

    def lookup(self) -> dict:
        """``(Dataset, Raw_Labels) -> {16 columns}``, value- and type-identical to the legacy reader.

        ``plankton`` is a real ``bool``; every other value is ``str`` or ``None``; the three
        numeric id columns are decimal-free strings. Pinned against ``build_taxonomy_lookup`` by
        ``tests/test_taxonomy_render_golden.py``.

        Cached. It is a pure function of the rows and callers treat it as a value rather than a
        call — read inside a nested loop it is otherwise rebuilt tens of thousands of times.
        """
        if self._lookup is None:
            from planktonzilla.planktonzilla_dataset.taxonomy import render

            self._lookup = render.build_lookup(self.rows())
        return self._lookup

    def render_wide_csv(self) -> bytes:
        """The 19-column CSV, byte-exact.

        A store loaded FROM a wide CSV returns that file's own bytes: it is a passthrough, not a
        re-serialisation, so loading and rendering a legacy file can never perturb it.
        """
        if self._legacy_bytes is not None:
            return self._legacy_bytes

        from planktonzilla.planktonzilla_dataset.taxonomy import render

        return render.render_wide_csv(self)

    # Coverage helpers, for pz_planktonzilla pre-flight
    def datasets(self) -> list:
        """Every source with at least one row, sorted."""
        return sorted({row["Dataset"] for row in self.rows()})

    def labels_for(self, dataset: str) -> list:
        """The byte-exact class-directory names mapped for one source, sorted."""
        return sorted(row["Raw_Labels"] for row in self.rows() if row["Dataset"] == dataset)

    def has(self, dataset: str, raw_label: str) -> bool:
        """Whether one ``(source, class directory)`` pair is mapped.

        The check ``check_taxonomy_csv`` needs: a miss is what makes the published build emit
        sixteen nulls for every image of that class rather than failing.
        """
        return (dataset, raw_label) in self.lookup()

    # Tree navigation, package backend only
    def taxon(self, taxon_id: str) -> Taxon:
        try:
            return self.taxa[taxon_id]
        except KeyError:
            raise TaxonomyError(f"unknown taxonID {taxon_id!r}") from None

    def ancestors(self, taxon_id: str) -> list:
        """The node and its ancestors, root first. Raises on a cycle or a dangling parent."""
        chain, seen, current = [], set(), taxon_id
        while current:
            if current in seen:
                raise TaxonomyError(f"parent cycle through {current!r}")
            seen.add(current)
            node = self.taxon(current)
            chain.append(node)
            current = node.parent_id
        chain.reverse()
        return chain


def read_tsv(path: Path) -> list:
    """Read a canonical TSV. The inverse of :func:`write_tsv`."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines[-1] != "":
        raise TaxonomyError(f"{Path(path).name}: missing trailing newline")
    columns = lines[0].split("\t")
    return [dict(zip(columns, line.split("\t"))) for line in lines[1:-1]]


def write_tsv(path: Path, columns, table) -> None:
    """Write a canonical TSV: tab-separated, LF, no quoting, trailing newline.

    A tab or newline inside a value would make the file unparseable, so it is refused rather
    than escaped — no value in this data has ever contained either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)]
    for row in table:
        values = [row[column] for column in columns]
        for column, value in zip(columns, values):
            if "\t" in value or "\n" in value or "\r" in value:
                raise TaxonomyError(f"{path.name}: {column}={value!r} contains a tab or newline")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
