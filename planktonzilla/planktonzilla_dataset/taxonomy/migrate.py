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
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, render
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    AUTHORITIES,
    BUCKET_REMARK,
    EXACT_MATCH,
    FINER_REMARK,
    IDENTIFIER_COLUMNS,
    LEGACY_HEADER,
    LEGACY_ID_COLUMNS,
    MAPPING_COLUMNS,
    MERGED_COLUMNS,
    OVERRIDE_COLUMNS,
    RANK_VOCABULARY,
    ROW_ORDER_COLUMNS,
    TAXON_COLUMNS,
    TAXON_PREFIX,
    UNQUALIFIED,
    UNSPECIFIED_MATCHING,
    TaxonomyError,
    collation_key,
    lineage_of,
    read_tsv,
    write_tsv,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

PACKAGE_DIR = loader.PACKAGE_DIR

# The migration's own failures are taxonomy failures; the name is kept so the step-1 gate and
# any runbook that catches it still do.
MigrationError = TaxonomyError

_LEGACY_COLUMN_AUTHORITY = {legacy: (name, prefix) for name, legacy, prefix, _scope, _multi in AUTHORITIES}


# Reading the legacy CSV
def read_legacy_rows(csv_path: Path) -> list:
    """Every row of the wide CSV, as strings, in physical file order."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != LEGACY_HEADER:
        raise MigrationError(f"unexpected legacy header: {tuple(rows[0])}")
    return rows


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
            remark = BUCKET_REMARK
            nodes[key] = ((("unranked", label),), "bucket", remark)
            concept_to_key[label] = key
            continue

        if path[-1][1] == label:
            concept_to_key[label] = natural_key(path)
            continue

        # The concept names something finer than the deepest rank the legacy columns can hold.
        finer = (*path, ("unranked", label))
        key = natural_key(finer)
        nodes[key] = (finer, "taxon", FINER_REMARK)
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
                        "predicate_id": EXACT_MATCH,
                        "object_id": f"{prefix}:{_render_legacy_id(item)}",
                        "mapping_justification": UNSPECIFIED_MATCHING,
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
    write_tsv(package_dir / "merged.tsv", MERGED_COLUMNS, [])

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
    """Render the 19-column CSV out of the written package.

    Goes through the shipped loader and renderer rather than a second implementation: a
    migration verified by its own private copy of the projection proves only that the copy
    agrees with itself. Reads the FILES, so what is checked is what was committed.
    """
    return render.render_wide_csv(loader.load_taxonomy(package_dir))


def verify(package_dir: Path, csv_path: Path) -> None:
    """Render the package back and compare against the committed CSV, byte for byte."""
    rendered = render_legacy_csv(package_dir)
    committed = Path(csv_path).read_bytes()
    if rendered == committed:
        logger.info(f"verified: the package renders «{Path(csv_path).name}» byte-for-byte ({len(committed)} bytes).")
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
