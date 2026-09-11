"""
(c) Inria

One-shot migration: the 19-column ``planktonzilla_taxonomy.csv`` into a normalised table
package, plus the verification that the package renders it back byte-for-byte.

Step 1 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``. What it produces, and why each file exists,
is in that plan; what matters *here* is the two properties the plan makes this step stand or
fall on, because both have already broken a prototype:

**Identifier minting is idempotent.** Ids are seeded from the committed ``taxon.tsv`` and minted
only for nodes absent from it. The panel's own migration minted in sorted-path order with no
seed, so inserting a single row re-pointed 1342 of 1351 ids while every foreign key still
resolved — a silent, total corruption of identity that no integrity check can see. The seed is
reconstructed from the file itself (walk ``parentNameUsageID`` to a root, join rank=name) rather
than from a stored key column, so there is no second copy of the identity to drift.

**The package is verified by rendering, not by inspection.** ``--verify`` rebuilds the 19-column
CSV from the written files and compares bytes. The refuters found design A's prototype package
had been hand-finished after its generator ran: re-running the generator produced a different
concept table and an exporter that could not run on its output. A migration that cannot re-derive
its input is not evidence of anything, so ``main`` verifies by default and a failed verification
is a non-zero exit.

Modelling decisions, all mechanical — this script makes no taxonomic judgements:

- A concept (today's ``proposed_label``) with a lineage becomes, or hangs from, a node of the
  rank tree; a concept with no lineage becomes a node with ``kind=bucket``. ``kind`` is derived
  from the presence of a lineage and NOTHING else, in particular never from ``root_class``:
  ``mix``, ``other`` and ``unknown`` each appear under two or three root classes, and a
  ``kind`` that implied living-ness would have to pick one. That is the defect design D shipped,
  where 45 ``root_class=living`` mappings ended up pointing at concepts typed non-living.
- The 12 concepts that are not the deepest node of their own lineage (``brachyura`` under
  ``decapoda``, ``alciopini`` under ``phyllodocidae``, ...) become real nodes with rank
  ``unranked``, parented to that deepest node. Their true ranks — infraorder, tribe, subclass —
  are a curation act, not a migration one. Without this the concept-to-node relation is not
  injective: 10 nodes are named by two or three concepts each.
- Species are stored as binomial nodes; the legacy epithet-only ``Species`` cell is derived by
  removing the genus prefix. All 364 species nodes have a Genus parent, tautonyms included.
- Sub-classifying the 43 buckets (morphotype / material / informal) and giving ``Eukaryota`` its
  true rank are deliberately left open. They are curation, and this script is re-runnable.

Usage:
    python -m planktonzilla.planktonzilla_dataset.taxonomy.migrate           # generate + verify
    python -m planktonzilla.planktonzilla_dataset.taxonomy.migrate --verify  # verify only
"""

import argparse
import csv
import hashlib
import io
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

PACKAGE_DIR = Path(__file__).parent / "data"

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
_LEGACY_COLUMN_AUTHORITY = {legacy: (name, prefix) for name, legacy, prefix, _scope, _multi in AUTHORITIES}

# The seven legacy slots, as a rank vocabulary. `legacy_slot` is what makes project7() a lookup
# rather than a branch: a rank without one contributes no cell to the wide CSV, which is how the
# 12 `unranked` concept nodes stay distinct without widening the frozen columns.
RANK_VOCABULARY = (
    *((rank.lower(), str(ordinal), rank) for ordinal, rank in enumerate(LEGACY_RANKS, start=1)),
    ("unranked", "", ""),
)

# `qualifier` is blank on 253 rows today. Blank is a value, not an absence, so it is named.
UNQUALIFIED = "unqualified"


class MigrationError(RuntimeError):
    """Raised when the package cannot be built from, or cannot reproduce, the legacy CSV."""


# Reading the legacy CSV
def read_legacy_rows(csv_path: Path) -> list:
    """Every row of the wide CSV, as strings, in physical file order."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != LEGACY_HEADER:
        raise MigrationError(f"unexpected legacy header: {tuple(rows[0])}")
    return rows


def lineage_of(row) -> tuple:
    """The row's rank path as ``((rank, name), ...)``, lowercase ranks, blanks dropped.

    Species carry the binomial rather than the epithet: the epithet alone is not a node
    identity (19 epithets occur under more than one genus).
    """
    path = []
    for rank in LEGACY_RANKS:
        value = row[rank]
        if not value:
            continue
        name = f"{row['Genus']} {value}" if rank == "Species" and row["Genus"] else value
        path.append((rank.lower(), name))
    return tuple(path)


def natural_key(path) -> str:
    """The identity of a node, independent of any minted id.

    Used to seed ids across re-runs. Reconstructed from ``taxon.tsv`` by walking parents, so the
    file never stores a second copy of it.
    """
    return "/".join(f"{rank}={name}" for rank, name in path)


# Building the tables
def build_taxa(rows) -> tuple:
    """Return ``(nodes, concept_to_key)``: every node's path, and each concept's node.

    ``nodes`` maps natural key -> ``(path, kind, remarks)``. Three kinds of node are created,
    in this order of precedence: a rank-path node for every prefix of every lineage; an
    ``unranked`` child for a concept that names something finer than its deepest node; a
    parentless ``bucket`` for a concept with no lineage at all.
    """
    nodes = {}
    concept_to_key = {}

    for row in rows:
        path = lineage_of(row)
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            nodes.setdefault(natural_key(prefix), (prefix, "taxon", ""))

    for row in rows:
        label = row["proposed_label"]
        if label in concept_to_key:
            continue
        path = lineage_of(row)

        if not path:
            key = f"bucket={label}"
            remark = "no lineage in the source table; sub-classification is a curation act"
            nodes[key] = ((("unranked", label),), "bucket", remark)
            concept_to_key[label] = key
            continue

        if path[-1][1] == label:
            concept_to_key[label] = natural_key(path)
            continue

        # The concept names something finer than the deepest rank the legacy columns can hold.
        finer = (*path, ("unranked", label))
        key = natural_key(finer)
        nodes[key] = (finer, "taxon", "finer than its deepest legacy rank; true rank unassigned")
        concept_to_key[label] = key

    return nodes, concept_to_key


def seed_ids(package_dir: Path) -> dict:
    """Reconstruct ``{natural key: taxonID}`` from a previously written ``taxon.tsv``.

    The key is rebuilt by walking ``parentNameUsageID`` to a root and joining ``rank=name``, so
    a re-run reuses an id for a node whose identity is unchanged even if its row moved, its
    remarks changed, or new siblings appeared around it.
    """
    path = package_dir / "taxon.tsv"
    if not path.exists():
        return {}

    by_id = {row["taxonID"]: row for row in read_tsv(path)}

    def key_of(taxon_id, seen=None):
        seen = seen or set()
        if taxon_id in seen:
            raise MigrationError(f"parent cycle through {taxon_id} in the committed taxon.tsv")
        row = by_id[taxon_id]
        segment = f"{row['taxonRank']}={row['scientificName']}"
        if row["kind"] == "bucket":
            return f"bucket={row['scientificName']}"
        if not row["parentNameUsageID"]:
            return segment
        return key_of(row["parentNameUsageID"], seen | {taxon_id}) + "/" + segment

    return {key_of(taxon_id): taxon_id for taxon_id in by_id}


def assign_ids(nodes, seeded) -> dict:
    """``{natural key: taxonID}`` for every node: seeded where known, minted in path order after.

    Minting starts above the highest seeded number rather than renumbering, so an id is never
    reused for a different node even after one is retired.

    Raises:
        MigrationError: If a node the committed package already has an id for has disappeared.
            Dropping it would retire an identity silently, which is the same class of loss this
            step exists to prevent — a retirement belongs in ``merged.tsv``, written by the
            ``rename`` / ``merge`` commands of step 6, not inferred from an absence.
    """
    retired = sorted(key for key in seeded if key not in nodes)
    if retired:
        shown = ", ".join(f"{seeded[key]} ({key})" for key in retired[:5])
        more = f" (+{len(retired) - 5} more)" if len(retired) > 5 else ""
        raise MigrationError(
            f"{len(retired)} node(s) with a committed id are absent from this run: {shown}{more}. "
            f"Retiring an identifier is a versioned act: record it in merged.tsv rather than "
            f"letting the migration drop it."
        )

    assigned = {key: seeded[key] for key in nodes if key in seeded}
    next_number = 1 + max((int(value.split(":")[1]) for value in assigned.values()), default=0)

    for key in sorted(key for key in nodes if key not in assigned):
        assigned[key] = f"{TAXON_PREFIX}:{next_number:06d}"
        next_number += 1

    return assigned


def taxon_table(nodes, ids) -> list:
    """``taxon.tsv`` rows, sorted by natural key so a genus sits above its species."""
    table = []
    for key in sorted(nodes):
        path, kind, remarks = nodes[key]
        rank, name = path[-1]
        parent_key = natural_key(path[:-1])
        table.append(
            {
                "taxonID": ids[key],
                "parentNameUsageID": ids[parent_key] if len(path) > 1 else "",
                "taxonRank": rank,
                "scientificName": name,
                "taxonomicStatus": "accepted",
                "acceptedNameUsageID": "",
                "nameAccordingTo": "",
                "kind": kind,
                "taxonRemarks": remarks,
            }
        )
    return table


def canonical_id_tuple(rows) -> dict:
    """``{concept: (id, ...)}`` taken from the concept's FIRST row in physical file order.

    Frequency would be the tempting rule and is the wrong one: design E picked by frequency,
    which ties on real data, and reordering two ``branchiopoda`` rows silently stripped that
    taxon of all four authority ids with every check still green. First-in-file-order cannot tie.
    """
    canonical = {}
    for row in rows:
        canonical.setdefault(row["proposed_label"], tuple(row[column] for column in LEGACY_ID_COLUMNS))
    return canonical


def identifier_table(canonical, concept_to_key, ids) -> list:
    """One external id per line, SSSOM-shaped, with ``;``-joined legacy lists exploded.

    Sorted by (subject, predicate, object). All 132 multi-valued ``ecotaxa_ID`` cells are already
    in string-sorted order in the committed file, so the canonical sort reproduces the legacy
    join order without storing it anywhere — verified by ``--verify``, not assumed.
    """
    table = []
    for concept, values in canonical.items():
        subject = ids[concept_to_key[concept]]
        for column, value in zip(LEGACY_ID_COLUMNS, values):
            if not value:
                continue
            _authority, prefix = _LEGACY_COLUMN_AUTHORITY[column]
            for item in value.split(";"):
                if not item:
                    continue
                table.append(
                    {
                        "subject_id": subject,
                        "predicate_id": "skos:exactMatch",
                        "object_id": f"{prefix}:{_render_legacy_id(item)}",
                        "mapping_justification": "semapv:UnspecifiedMatching",
                        "mapping_date": "",
                        "author_id": "",
                        "confidence": "",
                        "comment": "",
                    }
                )
    table.sort(key=lambda row: (row["subject_id"], row["predicate_id"], row["object_id"]))
    return table


def _render_legacy_id(value: str) -> str:
    """``135336.0`` -> ``135336``. The float form is a CSV-typing artefact, not an id."""
    return value.removesuffix(".0")


def mapping_tables(rows, concept_to_key, ids) -> dict:
    """``{dataset: [row, ...]}``, each file sorted by the byte-exact source label."""
    tables = {}
    for row in rows:
        label = row["proposed_label"]
        tables.setdefault(row["Dataset"], []).append(
            {
                "datasetID": row["Dataset"],
                "verbatimIdentification": row["Raw_Labels"],
                "taxonID": ids[concept_to_key[label]],
                "concept": label,
                "root_class": row["root_class"],
                "qualifier": row["qualifier"] or UNQUALIFIED,
                "plankton": "true" if row["plankton"] == "True" else "false",
                "status": "accepted",
                "method": "",
                "donor": "",
                "identifiedBy": "",
                "dateIdentified": "",
                "identificationReferences": "",
                "confidence": "",
                "remarks": "",
            }
        )
    for table in tables.values():
        table.sort(key=lambda row: row["verbatimIdentification"])
    return tables


def override_table(rows, canonical) -> list:
    """The mappings whose id tuple differs from their concept's canonical one.

    Thirteen rows today, all FREPJ, from ids blanked per row instead of per taxon. The three
    trailing columns are empty and exist so a curator can pin a ``root_class`` / ``qualifier`` /
    ``plankton`` correction without turning the byte gate red — design A's pin layer could not,
    and one qualifier edit moved a line of its render with nowhere to record it.
    """
    table = []
    for row in rows:
        values = tuple(row[column] for column in LEGACY_ID_COLUMNS)
        if values == canonical[row["proposed_label"]]:
            continue
        table.append(
            {
                "datasetID": row["Dataset"],
                "verbatimIdentification": row["Raw_Labels"],
                **{column: value for column, value in zip(LEGACY_ID_COLUMNS, values)},
                "root_class": "",
                "qualifier": "",
                "plankton": "",
                "reason": f"ids blanked per row rather than per taxon; diverges from {row['proposed_label']}",
            }
        )
    return table


def row_order_table(rows) -> list:
    """The physical order of the wide CSV, stored because it is not fully derivable.

    The label-level order IS derivable — the frozen 1485-row prefix is non-descending under
    ``proposed_label.lower().replace("-", "")``, with zero descents — and
    :func:`check_prefix_collation` asserts exactly that, so a corrupted manifest is a failing
    check rather than a silent reordering. What no key recovers is the order of rows WITHIN a
    label: 56 of the 240 multi-row prefix labels are not in ``(Dataset, Raw_Labels)`` order.
    """
    return [
        {"sequence": str(index), "datasetID": row["Dataset"], "verbatimIdentification": row["Raw_Labels"]}
        for index, row in enumerate(rows)
    ]


def collation_key(label: str) -> str:
    """The sort key the frozen prefix obeys: case-folded, hyphens ignored."""
    return label.lower().replace("-", "")


def check_prefix_collation(rows, prefix_length: int = 1485) -> None:
    """The stored prefix order must be non-descending under :func:`collation_key`.

    Raises:
        MigrationError: naming the first descent.
    """
    keys = [collation_key(row["proposed_label"]) for row in rows[:prefix_length]]
    for index in range(len(keys) - 1):
        if keys[index] > keys[index + 1]:
            raise MigrationError(
                f"frozen prefix is not in collation order at row {index}: "
                f"{rows[index]['proposed_label']!r} sorts after {rows[index + 1]['proposed_label']!r}"
            )


# Writing
def write_tsv(path: Path, columns, table) -> None:
    """Write a canonical TSV: tab-separated, LF, no quoting, trailing newline.

    A tab or newline inside a value would make the file unparseable, so it is refused rather
    than escaped — no value in this data has ever contained either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)]
    for row in table:
        values = [row[column] for column in columns]
        for column, value in zip(columns, values):
            if "\t" in value or "\n" in value or "\r" in value:
                raise MigrationError(f"{path.name}: {column}={value!r} contains a tab or newline")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def read_tsv(path: Path) -> list:
    """Read a canonical TSV back. The inverse of :func:`write_tsv`."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines[-1] != "":
        raise MigrationError(f"{path.name}: missing trailing newline")
    columns = lines[0].split("\t")
    return [dict(zip(columns, line.split("\t"))) for line in lines[1:-1]]


def write_package(package_dir: Path, rows) -> dict:
    """Generate every file of the package. Returns the tables, for the caller to verify."""
    check_prefix_collation(rows)

    nodes, concept_to_key = build_taxa(rows)
    ids = assign_ids(nodes, seed_ids(package_dir))
    canonical = canonical_id_tuple(rows)

    taxa = taxon_table(nodes, ids)
    identifiers = identifier_table(canonical, concept_to_key, ids)
    mappings = mapping_tables(rows, concept_to_key, ids)
    overrides = override_table(rows, canonical)
    order = row_order_table(rows)

    write_tsv(package_dir / "taxon.tsv", TAXON_COLUMNS, taxa)
    write_tsv(package_dir / "identifier.tsv", IDENTIFIER_COLUMNS, identifiers)
    for dataset, table in mappings.items():
        write_tsv(package_dir / "mappings" / f"{dataset}.tsv", MAPPING_COLUMNS, table)

    write_tsv(
        package_dir / "vocab" / "rank.tsv",
        ("rank", "ordinal", "legacy_slot"),
        [{"rank": rank, "ordinal": ordinal, "legacy_slot": slot} for rank, ordinal, slot in RANK_VOCABULARY],
    )
    write_tsv(
        package_dir / "vocab" / "root_class.tsv",
        ("root_class", "living"),
        [
            {"root_class": name, "living": "true" if name == "living" else "false"}
            for name in sorted({row["root_class"] for row in rows})
        ],
    )
    write_tsv(
        package_dir / "vocab" / "qualifier.tsv",
        ("qualifier", "legacy_value"),
        [
            {"qualifier": name, "legacy_value": "" if name == UNQUALIFIED else name}
            for name in sorted({row["qualifier"] or UNQUALIFIED for row in rows})
        ],
    )
    write_tsv(
        package_dir / "vocab" / "authority.tsv",
        ("authority", "legacy_column", "prefix", "scope", "multi_valued"),
        [
            {"authority": name, "legacy_column": legacy, "prefix": prefix, "scope": scope, "multi_valued": multi}
            for name, legacy, prefix, scope, multi in AUTHORITIES
        ],
    )
    write_tsv(
        package_dir / "vocab" / "kind.tsv",
        ("kind", "definition"),
        [
            {"kind": "taxon", "definition": "a node of the phylogenetic tree; says nothing about whether a mapping is living"},
            {
                "kind": "bucket",
                "definition": "a named class with no lineage; NEVER implies non-living (see mix, other, unknown)",
            },
        ],
    )
    write_tsv(package_dir / "merged.tsv", ("retired_id", "replacement_id", "date", "reason"), [])

    release = package_dir / "release" / "v1.0"
    write_tsv(release / "legacy_row_order.tsv", ROW_ORDER_COLUMNS, order)
    write_tsv(release / "legacy_overrides.tsv", OVERRIDE_COLUMNS, overrides)
    digest = hashlib.sha256(
        "\n".join(f"{row['datasetID']}\t{row['verbatimIdentification']}" for row in order).encode("utf-8")
    ).hexdigest()
    (release / "sha256").write_text(f"{digest}\n", encoding="utf-8", newline="")

    return {"taxa": taxa, "identifiers": identifiers, "mappings": mappings, "overrides": overrides, "order": order}


# Verification
def render_legacy_csv(package_dir: Path) -> bytes:
    """Rebuild the 19-column CSV from the written package.

    Deliberately reads the FILES rather than the in-memory tables: a package that only renders
    from the objects that produced it proves nothing about what was committed.
    """
    taxa = {row["taxonID"]: row for row in read_tsv(package_dir / "taxon.tsv")}
    legacy_slot = {row["rank"]: row["legacy_slot"] for row in read_tsv(package_dir / "vocab" / "rank.tsv")}
    blank_qualifier = {row["qualifier"]: row["legacy_value"] for row in read_tsv(package_dir / "vocab" / "qualifier.tsv")}
    prefix_to_column = {prefix: legacy for _name, legacy, prefix, _scope, _multi in AUTHORITIES}

    ids_by_subject = {}
    for row in read_tsv(package_dir / "identifier.tsv"):
        prefix, _, value = row["object_id"].partition(":")
        ids_by_subject.setdefault(row["subject_id"], {}).setdefault(prefix_to_column[prefix], []).append(value)

    mappings = {}
    for path in (package_dir / "mappings").glob("*.tsv"):
        for row in read_tsv(path):
            if row["datasetID"] != path.stem:
                raise MigrationError(f"{path.name} holds a row for {row['datasetID']}")
            mappings[(row["datasetID"], row["verbatimIdentification"])] = row

    overrides = {
        (row["datasetID"], row["verbatimIdentification"]): row
        for row in read_tsv(package_dir / "release" / "v1.0" / "legacy_overrides.tsv")
    }

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(LEGACY_HEADER)

    for entry in read_tsv(package_dir / "release" / "v1.0" / "legacy_row_order.tsv"):
        key = (entry["datasetID"], entry["verbatimIdentification"])
        mapping = mappings[key]
        slots = project7(taxa, mapping["taxonID"], legacy_slot)
        override = overrides.get(key)
        ids = ids_by_subject.get(mapping["taxonID"], {})

        cells = {
            "Dataset": mapping["datasetID"],
            "Raw_Labels": mapping["verbatimIdentification"],
            **slots,
            "proposed_label": mapping["concept"],
            "plankton": "True" if mapping["plankton"] == "true" else "False",
            "living": "True" if mapping["root_class"] == "living" else "False",
            "root_class": mapping["root_class"],
            "qualifier": blank_qualifier[mapping["qualifier"]],
        }
        for column in LEGACY_ID_COLUMNS:
            if override is not None:
                cells[column] = override[column]
            else:
                cells[column] = _legacy_id_cell(column, ids.get(column, []))
        writer.writerow([cells[column] for column in LEGACY_HEADER])

    return buffer.getvalue().encode("utf-8")


_MULTI_VALUED = {legacy for _n, legacy, _p, _s, multi in AUTHORITIES if multi == "true"}


def _legacy_id_cell(column: str, values) -> str:
    """Render an id list back into its legacy cell form, float suffix included.

    Raises:
        MigrationError: If a single-valued authority carries more than one id. A second
            ``exact`` id is schema-legal in any SSSOM-shaped table and would otherwise be
            dropped here silently, with the survivor decided by sort order rather than by the
            model — one of the over-claims the refuters found in every design.
    """
    if not values:
        return ""
    if column in _MULTI_VALUED:
        return ";".join(values)
    if len(values) > 1:
        raise MigrationError(f"{column} is single-valued but carries {len(values)} ids: {values}")
    return values[0] if column == "wikidata_ID" else f"{values[0]}.0"


def project7(taxa, taxon_id, legacy_slot) -> dict:
    """Fill the seven legacy slots from the ancestors whose rank HAS a slot.

    A rank with no ``legacy_slot`` contributes no cell, which is what keeps the 12 ``unranked``
    concept nodes distinct without widening the frozen columns, and what renders a ``bucket``
    as seven blanks. The Species cell is the epithet, derived by removing the genus prefix.
    """
    slots = dict.fromkeys(LEGACY_RANKS, "")
    genus = ""
    chain = []
    current = taxon_id
    while current:
        row = taxa[current]
        chain.append(row)
        current = row["parentNameUsageID"]

    for row in reversed(chain):
        slot = legacy_slot.get(row["taxonRank"], "")
        if not slot:
            continue
        if slot == "Genus":
            genus = row["scientificName"]
        if slot != "Species":
            slots[slot] = row["scientificName"]
            continue
        if not genus:
            raise MigrationError(f"species {row['scientificName']!r} has no genus ancestor; the epithet is underivable")
        slots[slot] = row["scientificName"].removeprefix(f"{genus} ")
    return slots


def verify(package_dir: Path, csv_path: Path) -> None:
    """Render the package back and compare against the committed CSV, byte for byte."""
    rendered = render_legacy_csv(package_dir)
    committed = csv_path.read_bytes()
    if rendered == committed:
        logger.info(f"verified: the package renders «{csv_path.name}» byte-for-byte ({len(committed)} bytes).")
        return

    rendered_lines = rendered.decode("utf-8").split("\n")
    committed_lines = committed.decode("utf-8").split("\n")
    for index, (got, want) in enumerate(zip(rendered_lines, committed_lines)):
        if got != want:
            raise MigrationError(f"render differs from the committed CSV at line {index + 1}:\n  got:  {got}\n  want: {want}")
    raise MigrationError(f"render has {len(rendered_lines)} lines, the committed CSV has {len(committed_lines)}")


def main(argv=None) -> int:
    """Generate the package from the committed CSV and verify it renders that CSV back."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=constants.DEFAULT_TAXONOMY_CSV_FILENAME)
    parser.add_argument("--package", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--verify", action="store_true", help="Verify an existing package; generate nothing.")
    args = parser.parse_args(argv)

    if not args.verify:
        rows = read_legacy_rows(args.csv)
        tables = write_package(args.package, rows)
        logger.info(
            f"wrote {len(tables['taxa'])} taxa, {len(tables['identifiers'])} identifiers, "
            f"{len(tables['mappings'])} mapping files ({sum(len(t) for t in tables['mappings'].values())} rows), "
            f"{len(tables['overrides'])} legacy id override(s) to «{args.package}»."
        )

    verify(args.package, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
