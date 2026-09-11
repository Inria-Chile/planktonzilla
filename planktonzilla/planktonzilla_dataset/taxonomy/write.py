"""
(c) Inria

The write side: one API that every curation path goes through.

Step 6 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``, and the step the plan schedules alone. Until
this landed, three builders wrote the taxonomy by locating their own block in the byte stream and
re-serialising the file around it, which is how one of them came to destroy 644 rows on a no-op
re-run. Their curation logic — rule tables, label grammars, donor resolution, ~2,600 lines of it —
is source knowledge and stayed exactly where it is; porting all three deleted 150 lines of write
path and added 102, because :func:`records_from_wide_rows` takes the 19-column rows they already
produce.

Five properties, each answering a specific way the old write path went wrong:

**Per-source partition.** :func:`upsert_source` writes ``mappings/<dataset>.tsv`` and nothing else
under ``mappings/``. A builder for one source is physically unable to touch another's rows — not
guarded against it, unable. That is the failure mode of ``build_frepj_taxonomy.write_csv``, which
copied a byte prefix and dropped everything after it.

**Dry run by default.** Every entry point returns a :class:`ChangeSet` and writes nothing unless
``apply=True``. The change set is cell-level: table, key, column, before, after. A curator sees what
would move before it moves.

**Ids are never blanked by omission.** A record that says nothing about an authority leaves that
authority alone; a record that contradicts one is refused rather than doubled. Clearing an id needs
:func:`clear_id`, which is explicit and records a reason. Blanking by omission is how the 13 legacy
overrides came to exist — ids cleared per row instead of per taxon (commit 7262085).

**Canonical serialisation.** :func:`fmt` produces the same bytes whether a table was written by a
builder, edited by hand, or left unsorted by a ``merge=union`` resolution. Without it the union
merge PR 4 relies on leaves a ledger permanently out of order.

**All or nothing.** A builder that curates several sources (:func:`upsert_wide_rows`) either writes
all of them or none: a failure on the fourth rolls the first three back, files it created included.
Half a curation with nothing naming it is the state a curator cannot reason about.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, render
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    BUCKET_REMARK,
    EXACT_MATCH,
    FINER_REMARK,
    IDENTIFIER_COLUMNS,
    LEGACY_ID_COLUMNS,
    MAPPING_COLUMNS,
    MERGED_COLUMNS,
    MULTI_VALUED_COLUMNS,
    PREFIX_TO_LEGACY_COLUMN,
    TAXON_COLUMNS,
    TAXON_PREFIX,
    UNQUALIFIED,
    UNSPECIFIED_MATCHING,
    TaxonomyError,
    lineage_of,
    read_tsv,
    write_tsv,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

LEGACY_COLUMN_TO_PREFIX = {legacy: prefix for prefix, legacy in PREFIX_TO_LEGACY_COLUMN.items()}

# Columns a record may set directly. Everything else about a mapping row is derived.
PROVENANCE_COLUMNS = (
    "method",
    "donor",
    "identifiedBy",
    "dateIdentified",
    "identificationReferences",
    "confidence",
    "remarks",
)


@dataclass(frozen=True)
class Change:
    """One cell that would move, named the way a curator would name it."""

    table: str
    key: str
    column: str
    before: str
    after: str

    def describe(self) -> str:
        kind = "add" if self.before == "" else ("clear" if self.after == "" else "set")
        return f"{kind:5s} {self.table}[{self.key}].{self.column}: {self.before!r} -> {self.after!r}"


@dataclass
class ChangeSet:
    """What a write would do, and — once applied — what it did."""

    changes: list = field(default_factory=list)
    applied: bool = False

    def __bool__(self) -> bool:
        return bool(self.changes)

    def __len__(self) -> int:
        return len(self.changes)

    def by_table(self) -> dict:
        out = {}
        for change in self.changes:
            out.setdefault(change.table, []).append(change)
        return out

    def removals(self) -> list:
        """The changes that delete a whole row. What ``allow_delete`` gates and a reviewer reads first."""
        return [change for change in self.changes if change.column == "*" and change.after == ""]

    def describe(self, limit: int = 40) -> str:
        if not self.changes:
            return "no changes"
        head = "\n".join(f"  {change.describe()}" for change in self.changes[:limit])
        more = f"\n  … and {len(self.changes) - limit} more" if len(self.changes) > limit else ""
        verb = "applied" if self.applied else "would apply"
        return f"{verb} {len(self.changes)} change(s):\n{head}{more}"


def _diff_tables(table: str, key_columns, before, after) -> list:
    """The cell-level difference between two row lists, keyed rather than positional.

    Positional diffing would report a whole file as changed when one row is inserted, which is the
    diff problem the wide CSV has and the reason a keyed diff is worth the few lines.
    """

    def keyed(rows):
        return {"|".join(row[column] for column in key_columns): row for row in rows}

    old, new = keyed(before), keyed(after)
    changes = []
    for key in sorted(set(old) | set(new)):
        if key not in new:
            changes.append(Change(table, key, "*", "row", ""))
        elif key not in old:
            changes.append(Change(table, key, "*", "", "row"))
        else:
            changes += [
                Change(table, key, column, old[key].get(column, ""), new[key][column])
                for column in new[key]
                if old[key].get(column, "") != new[key][column]
            ]
    return changes


def _natural_key(lineage) -> str:
    """``rank=name/rank=name/…`` — a node's identity by where it sits, not by the id it was given.

    The key the migration mints against, so a builder re-run resolves to the id already committed
    instead of allocating a second one for the same concept.
    """
    return "/".join(f"{rank}={name}" for rank, name in lineage)


def _path_of(row, by_id) -> str:
    """The natural key of a row already in ``taxon.tsv``."""
    if row["kind"] == "bucket":
        return f"bucket={row['scientificName']}"
    segments, current, seen = [], row, set()
    while current:
        if current["taxonID"] in seen:
            raise TaxonomyError(f"parent cycle through {current['taxonID']!r} in taxon.tsv")
        seen.add(current["taxonID"])
        segments.append(f"{current['taxonRank']}={current['scientificName']}")
        current = by_id.get(current["parentNameUsageID"])
    return "/".join(reversed(segments))


def _index_by_key(taxa) -> dict:
    """``natural key -> row`` for every committed node, resolving each parent chain once."""
    by_id = {row["taxonID"]: row for row in taxa}
    return {_path_of(row, by_id): row for row in taxa}


def _sorted_taxa(taxa) -> list:
    """Canonical order: by the path a node sits at, so a genus sits above its own species."""
    by_id = {row["taxonID"]: row for row in taxa}
    return sorted(taxa, key=lambda row: _path_of(row, by_id))


def _taxon_row(taxon_id, parent, rank, name, kind, remarks) -> dict:
    return {
        "taxonID": taxon_id,
        "parentNameUsageID": parent,
        "taxonRank": rank,
        "scientificName": name,
        "taxonomicStatus": "accepted",
        "acceptedNameUsageID": "",
        "nameAccordingTo": "",
        "kind": kind,
        "taxonRemarks": remarks,
    }


@dataclass
class _Minter:
    """Allocates taxon ids, and never allocates one it does not use.

    An id burned on a concept that turned out to exist is not cosmetic: ids are the identity this
    whole representation rests on, they are published in ``taxonID``, and a gap left by a dry run
    that was never applied is indistinguishable from a retirement with no tombstone. So the counter
    only moves inside :meth:`mint`.
    """

    by_key: dict
    rows: list
    next_id: int

    @classmethod
    def over(cls, taxa) -> "_Minter":
        highest = max((int(row["taxonID"].split(":")[1]) for row in taxa), default=0)
        return cls(by_key=_index_by_key(taxa), rows=list(taxa), next_id=highest + 1)

    def mint(self, key, parent, rank, name, kind, remarks) -> str:
        taxon_id = f"{TAXON_PREFIX}:{self.next_id:06d}"
        self.next_id += 1
        row = _taxon_row(taxon_id, parent, rank, name, kind, remarks)
        self.by_key[key] = row
        self.rows.append(row)
        return taxon_id

    def resolve(self, lineage, concept) -> str:
        """The id of the concept, minting whatever part of its chain is missing.

        Ancestors are minted before descendants, so ids read down the tree rather than in whatever
        order records happened to arrive. A chain that already exists mints nothing and allocates
        nothing — re-running a builder on unchanged input must be a no-op, which is exactly the
        property the byte-splicing write path never had.
        """
        if not lineage:
            key = f"bucket={concept}"
            existing = self.by_key.get(key)
            return existing["taxonID"] if existing else self.mint(key, "", "unranked", concept, "bucket", BUCKET_REMARK)

        parent = ""
        for depth in range(1, len(lineage) + 1):
            key = _natural_key(lineage[:depth])
            rank, name = lineage[depth - 1]
            existing = self.by_key.get(key)
            if existing:
                parent = existing["taxonID"]
                continue
            # A node the legacy ranks cannot hold says so, in the same words the migration uses.
            remarks = FINER_REMARK if rank == "unranked" else ""
            parent = self.mint(key, parent, rank, name, "taxon", remarks)
        return parent


def records_from_wide_rows(rows) -> list:
    """Records from 19-column legacy rows — the shape all three builders already produce.

    This is what makes porting a builder a small diff. Their ~2,600 lines of curation — rule tables,
    label grammars, donor resolution, reconciliation reports — produce wide rows and go on producing
    them; only the write at the end changes, from splicing bytes into the master CSV to calling
    :func:`upsert_source` with these.

    The rank columns invert through :func:`model.lineage_of`, and a ``proposed_label`` that names
    something finer than the deepest rank becomes an ``unranked`` child of it — the rule that holds
    the 12 such concepts in the committed table. The id cells go through as they are; shedding the
    float suffix the CSV carries is :func:`_add_ids`'s job, so every entry point sheds it.
    """
    records = []
    for row in rows:
        lineage = lineage_of(row)
        concept = row["proposed_label"]
        if lineage and lineage[-1][1] != concept:
            lineage = (*lineage, ("unranked", concept))
        records.append(
            {
                "verbatim": row["Raw_Labels"],
                "concept": concept,
                "lineage": lineage,
                "root_class": row["root_class"],
                "qualifier": row["qualifier"] or UNQUALIFIED,
                "plankton": row["plankton"] == "True",
                "ids": {column: row.get(column, "") for column in LEGACY_ID_COLUMNS},
            }
        )
    return records


def _record_lineage(dataset, record) -> tuple:
    """``(lineage, concept)`` from a record, with the record's own claims checked against each other.

    ``lineage`` is the FULL chain including the concept's own node, each step ``(rank, name)``; the
    last step's rank is ``unranked`` for a concept that is a refinement rather than a rank (12 such
    nodes exist today — ``brachyura`` under ``order=decapoda``, and the like). An empty lineage is a
    bucket, and ``concept`` names it.

    ``concept`` is required and asserted, never inferred: it is what the mapping file carries beside
    the opaque id, and the loader hard-errors when the two disagree. A builder that has drifted on
    the point should be told here rather than committing a package that will not load.
    """
    verbatim = record.get("verbatim")
    if not verbatim:
        raise TaxonomyError(f"{dataset}: a record has no 'verbatim' (the class directory, byte-exact)")
    concept = record.get("concept")
    if not concept:
        raise TaxonomyError(f"{dataset}/{verbatim}: a record has no 'concept'")

    lineage = tuple(tuple(step) for step in record.get("lineage") or ())
    for step in lineage:
        if len(step) != 2 or not all(step):
            raise TaxonomyError(f"{dataset}/{verbatim}: lineage step {step!r} is not a non-empty (rank, name) pair")
    if lineage and lineage[-1][1] != concept:
        raise TaxonomyError(
            f"{dataset}/{verbatim}: concept is {concept!r} but the lineage ends at {lineage[-1][1]!r}. "
            f"The lineage carries the concept's own node; append ('unranked', {concept!r}) for a refinement."
        )
    return lineage, concept


def _mapping_row(dataset, verbatim, taxon_id, concept, record) -> dict:
    plankton = record.get("plankton")
    if not isinstance(plankton, bool):
        # Not coerced. `plankton` is the only boolean the published dataset carries, and a truthy
        # string ("false" among them) would invert it with nothing downstream to catch it.
        raise TaxonomyError(f"{dataset}/{verbatim}: plankton must be a bool, got {plankton!r}")
    root_class = record.get("root_class")
    if not root_class:
        raise TaxonomyError(f"{dataset}/{verbatim}: a record has no 'root_class'")

    row = dict.fromkeys(MAPPING_COLUMNS, "")
    row.update(
        datasetID=dataset,
        verbatimIdentification=verbatim,
        taxonID=taxon_id,
        concept=concept,
        root_class=root_class,
        qualifier=record.get("qualifier") or UNQUALIFIED,
        plankton="true" if plankton else "false",
        status=record.get("status") or "accepted",
    )
    for column in PROVENANCE_COLUMNS:
        if record.get(column):
            row[column] = str(record[column])
    return row


def _bare_id(dataset, verbatim, column, value) -> str:
    """``135336.0`` -> ``"135336"`` for the three numeric authorities. A CURIE is not a float.

    The wide CSV serialises those three through a float column, so every value read back out of it
    — and out of the Markdown summaries derived from it — carries the artefact. Stripping it here
    rather than at each caller is what stops ``worms:106265`` and ``worms:106265.0`` being two ids
    for one concept.
    """
    if column not in constants.ID_NUM_COLS:
        return value
    bare = render._as_decimal_free_string(value)
    if bare is None:
        raise TaxonomyError(f"{dataset}/{verbatim}: {column} is numeric but the record gives {value!r}")
    return bare


def _held_index(id_rows) -> dict:
    """``{(subject, prefix): object_id}`` for the ids already present.

    Only consulted for single-valued authorities, so keeping one entry per ``(subject, prefix)`` is
    the whole truth for the pairs that matter. It exists because the alternative — scanning the
    identifier table per record per authority — is tens of millions of comparisons on a full
    re-upsert of the committed table, and grows with the square of it.
    """
    index = {}
    for subject, object_id in id_rows:
        index[(subject, object_id.split(":", 1)[0])] = object_id
    return index


def _add_ids(id_rows, held, dataset, verbatim, subject, ids) -> None:
    """Add the ids a record names. An authority it does not name is left alone.

    Args:
        id_rows: ``{(subject, object_id): row}``, mutated in place.
        held: The :func:`_held_index` over ``id_rows``, kept in step with it.

    Raises:
        TaxonomyError: If a single-valued authority already carries a DIFFERENT id for this concept.
            Keeping both is schema-legal and is what a plain upsert would do, but it renders as
            ``"aphia_ID is single-valued but carries 2 ids"`` two steps later, with the survivor
            decided by sort order. Correcting an id is a deliberate act: ``clear_id`` then re-add.
    """
    for legacy_column, value in (ids or {}).items():
        prefix = LEGACY_COLUMN_TO_PREFIX.get(legacy_column)
        if prefix is None:
            raise TaxonomyError(f"{dataset}/{verbatim}: unknown authority column {legacy_column!r}")
        values = [item.strip() for item in str(value).split(";") if item.strip()] if value else []
        if not values:
            # Silence about an authority is silence, not an instruction to clear it.
            continue
        values = [_bare_id(dataset, verbatim, legacy_column, item) for item in values]
        if legacy_column not in MULTI_VALUED_COLUMNS:
            if len(values) > 1:
                raise TaxonomyError(f"{dataset}/{verbatim}: {legacy_column} is single-valued but the record gives {values}")
            already = held.get((subject, prefix))
            if already is not None and already != f"{prefix}:{values[0]}":
                raise TaxonomyError(
                    f"{dataset}/{verbatim}: {subject} already holds {already} for {legacy_column}; the record says "
                    f"{values[0]!r}. Correcting an id is deliberate: clear_id() first, then re-run."
                )
        for item in values:
            held[(subject, prefix)] = f"{prefix}:{item}"
            id_rows.setdefault(
                (subject, f"{prefix}:{item}"),
                {
                    "subject_id": subject,
                    "predicate_id": EXACT_MATCH,
                    "object_id": f"{prefix}:{item}",
                    "mapping_justification": UNSPECIFIED_MATCHING,
                    "mapping_date": "",
                    "author_id": "",
                    "confidence": "",
                    "comment": "",
                },
            )


def upsert_source(package_dir, dataset: str, records, *, apply: bool = False, allow_delete: bool = False) -> ChangeSet:
    """Write one source's mappings, minting taxa and identifiers as the records require.

    This is the entry point every builder ports onto. A record describes one class directory the way
    a curator thinks about it; the store's bookkeeping is this function's problem:

        {"verbatim": "Bosminidae,Bosmina,Bosmina fatalis",    the class dir, byte-exact
         "concept": "bosmina fatalis",                        the display name
         "lineage": [("kingdom", "animalia"), …,              the full chain, concept included;
                     ("species", "bosmina fatalis")],         omit entirely for a bucket
         "root_class": "living", "qualifier": "full_body", "plankton": True,
         "ids": {"aphia_ID": "234567", …},                    optional, per authority
         "status": "accepted",                                "draft" for a partial curation
         "method": "…", "donor": "…"}                         optional provenance

    ``records`` is the source's mapping file in full, so a class directory that disappears upstream
    disappears here — but only with ``allow_delete``, because the failure this whole API exists to
    prevent looked exactly like an accidental omission.

    Args:
        package_dir: The package to write into.
        dataset: The source. Only ``mappings/<dataset>.tsv`` is written under ``mappings/``.
        records: The class directories, in any order; the file is written in canonical order.
        apply: Write. Defaults to False, so the caller sees the change set first.
        allow_delete: Permit the write to drop rows this source has but ``records`` does not.

    Returns:
        A :class:`ChangeSet`, cell by cell.

    Raises:
        TaxonomyError: If a record is malformed, contradicts an id already held, names a class
            directory twice, or would drop rows without ``allow_delete``.
    """
    package_dir = Path(package_dir)
    taxa = read_tsv(package_dir / "taxon.tsv")
    identifiers = read_tsv(package_dir / "identifier.tsv")
    mapping_path = package_dir / "mappings" / f"{dataset}.tsv"
    existing_mappings = read_tsv(mapping_path) if mapping_path.exists() else []

    minter = _Minter.over(taxa)
    id_rows = {(row["subject_id"], row["object_id"]): row for row in identifiers}
    held = _held_index(id_rows)
    new_mappings = {}

    for record in records:
        lineage, concept = _record_lineage(dataset, record)
        verbatim = record["verbatim"]
        if verbatim in new_mappings:
            # The per-file primary key. Left to the last-wins default this is a row silently lost,
            # and the two records that collided are by construction the ones that disagree.
            raise TaxonomyError(f"{dataset}: two records both map {verbatim!r}")
        taxon_id = minter.resolve(lineage, concept)
        new_mappings[verbatim] = _mapping_row(dataset, verbatim, taxon_id, concept, record)
        _add_ids(id_rows, held, dataset, verbatim, taxon_id, record.get("ids"))

    new_taxa = _sorted_taxa(minter.rows)
    mapping_rows = [new_mappings[verbatim] for verbatim in sorted(new_mappings)]
    new_identifiers = sorted(id_rows.values(), key=lambda row: (row["subject_id"], row["predicate_id"], row["object_id"]))

    changeset = ChangeSet(
        changes=[
            *_diff_tables("taxon", ("taxonID",), taxa, new_taxa),
            *_diff_tables("identifier", ("subject_id", "object_id"), identifiers, new_identifiers),
            *_diff_tables(f"mappings/{dataset}", ("verbatimIdentification",), existing_mappings, mapping_rows),
        ]
    )

    dropped = changeset.removals()
    if dropped and not allow_delete:
        shown = ", ".join(change.key for change in dropped[:5])
        more = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
        raise TaxonomyError(
            f"{dataset}: these records would drop {len(dropped)} committed row(s): {shown}{more}. "
            f"Pass allow_delete=True if the class directories really are gone."
        )

    if apply:
        write_tsv(package_dir / "taxon.tsv", TAXON_COLUMNS, new_taxa)
        write_tsv(package_dir / "identifier.tsv", IDENTIFIER_COLUMNS, new_identifiers)
        write_tsv(mapping_path, MAPPING_COLUMNS, mapping_rows)
        loader.cache_clear()
        changeset.applied = True
        logger.info(f"upsert_source({dataset}): {len(changeset)} change(s) written.")

    return changeset


def upsert_wide_rows(package_dir, rows, *, apply: bool = False, allow_delete: bool = False) -> ChangeSet:
    """:func:`upsert_source` for a builder that curates 19-column rows, for one source or several.

    Rows are grouped by their own ``Dataset`` column, so a builder that curates four sources — Tara
    Pacific does — writes four mapping files and still cannot reach a fifth. That grouping is the
    whole safety property, expressed as a partition rather than as a check: the failure this
    replaces was a builder writing a byte range that happened to contain other sources' rows.

    Every group is validated before any is written, and the whole write is rolled back if a later
    group raises — a malformed record in the fourth source must not leave the first three applied.
    A dry run over the groups is not sufficient on its own: two groups of the same run can conflict
    with each other (both naming a different aphia id for one shared concept) while each agrees
    with what is committed, and that only surfaces once the first has been written.
    """
    package_dir = Path(package_dir)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Dataset"], []).append(row)

    changes = ChangeSet()
    for dataset, group in sorted(grouped.items()):
        changes.changes += upsert_source(package_dir, dataset, records_from_wide_rows(group), allow_delete=allow_delete).changes
    if not apply:
        return changes

    changes = ChangeSet(applied=True)
    snapshot = _snapshot(package_dir)
    try:
        for dataset, group in sorted(grouped.items()):
            changes.changes += upsert_source(
                package_dir, dataset, records_from_wide_rows(group), apply=True, allow_delete=allow_delete
            ).changes
    except Exception:
        _restore(snapshot)
        loader.cache_clear()
        raise
    return changes


def _snapshot(package_dir) -> tuple:
    """``(package_dir, {path: bytes})`` for every file a write can touch."""
    paths = [package_dir / "taxon.tsv", package_dir / "identifier.tsv", *(package_dir / "mappings").glob("*.tsv")]
    return package_dir, {path: path.read_bytes() for path in paths if path.exists()}


def _restore(snapshot) -> None:
    """Put every snapshotted file back and delete the mapping files the write created.

    The second half is the one that is easy to forget: a run that adds two sources and fails on the
    second must not leave the first source's file behind, half a curation with nothing naming it.
    """
    package_dir, blobs = snapshot
    for path, blob in blobs.items():
        if path.read_bytes() != blob:
            path.write_bytes(blob)
    for path in (package_dir / "mappings").glob("*.tsv"):
        if path not in blobs:
            path.unlink()


def add_ids(package_dir, dataset: str, ids_by_verbatim, *, apply: bool = False) -> ChangeSet:
    """Add identifiers to the concepts one source's class directories map to.

    The port target for ``resolve_frepj_ids.backfill_csv``, which rewrote four id cells per ROW by
    splitting each line on its last five commas. Ids are a property of the concept, not of the row
    that happens to name it, so they go on the taxon — which is also why two rows mapping the same
    concept can no longer disagree about its aphia id.

    Args:
        package_dir: The package to write into.
        dataset: The source whose class directories ``ids_by_verbatim`` is keyed by.
        ids_by_verbatim: ``{verbatimIdentification: {legacy column: value}}``. A column left out, or
            given empty, is left alone — :func:`clear_id` is the only way to blank one.
        apply: Write. Defaults to False.

    Raises:
        TaxonomyError: If a key names a class directory this source does not map, or a value
            contradicts an id the concept already holds.
    """
    package_dir = Path(package_dir)
    mapping_path = package_dir / "mappings" / f"{dataset}.tsv"
    if not mapping_path.exists():
        raise TaxonomyError(f"{dataset} maps nothing; there is no mappings/{dataset}.tsv")
    taxon_of = {row["verbatimIdentification"]: row["taxonID"] for row in read_tsv(mapping_path)}

    unknown = sorted(set(ids_by_verbatim) - set(taxon_of))
    if unknown:
        more = f" (+{len(unknown) - 3} more)" if len(unknown) > 3 else ""
        raise TaxonomyError(f"{dataset} does not map {len(unknown)} of these class directories: {unknown[:3]}{more}")

    identifiers = read_tsv(package_dir / "identifier.tsv")
    id_rows = {(row["subject_id"], row["object_id"]): row for row in identifiers}
    held = _held_index(id_rows)
    for verbatim, ids in ids_by_verbatim.items():
        _add_ids(id_rows, held, dataset, verbatim, taxon_of[verbatim], ids)

    updated = sorted(id_rows.values(), key=lambda row: (row["subject_id"], row["predicate_id"], row["object_id"]))
    changeset = ChangeSet(changes=_diff_tables("identifier", ("subject_id", "object_id"), identifiers, updated))
    if apply and changeset:
        write_tsv(package_dir / "identifier.tsv", IDENTIFIER_COLUMNS, updated)
        loader.cache_clear()
        changeset.applied = True
        logger.info(f"add_ids({dataset}): {len(changeset)} change(s) written.")
    return changeset


def clear_id(package_dir, taxon_id: str, authority: str, reason: str, *, apply: bool = False) -> ChangeSet:
    """Remove one authority's ids from one concept, with a reason. The only way to blank an id.

    Separate from :func:`upsert_source` on purpose. Clearing by omission — a record that simply
    stops mentioning an authority — is what produced the 13 per-row legacy overrides, so it takes a
    deliberate call that records why.
    """
    package_dir = Path(package_dir)
    if not reason:
        raise TaxonomyError("clearing an id needs a reason")
    identifiers = read_tsv(package_dir / "identifier.tsv")
    prefix = LEGACY_COLUMN_TO_PREFIX.get(authority, authority)
    if prefix not in PREFIX_TO_LEGACY_COLUMN:
        raise TaxonomyError(f"unknown authority {authority!r}")
    kept = [row for row in identifiers if not (row["subject_id"] == taxon_id and row["object_id"].startswith(f"{prefix}:"))]

    changeset = ChangeSet(changes=_diff_tables("identifier", ("subject_id", "object_id"), identifiers, kept))
    if apply and changeset:
        write_tsv(package_dir / "identifier.tsv", IDENTIFIER_COLUMNS, kept)
        loader.cache_clear()
        changeset.applied = True
        logger.info(f"clear_id({taxon_id}, {authority}): {reason}")
    return changeset


def rename(package_dir, taxon_id: str, new_name: str, *, apply: bool = False) -> ChangeSet:
    """Rename a node, keeping its id. The id is the identity; the name never was.

    Every mapping carries the display name beside the id and the loader hard-errors when the two
    disagree, so a rename that stopped at ``taxon.tsv`` would leave the package unloadable. The
    mapping files move with it.
    """
    package_dir = Path(package_dir)
    if not new_name:
        raise TaxonomyError("a rename needs a name")
    taxa = read_tsv(package_dir / "taxon.tsv")
    if taxon_id not in {row["taxonID"] for row in taxa}:
        raise TaxonomyError(f"unknown taxonID {taxon_id!r}")

    updated = [{**row, "scientificName": new_name} if row["taxonID"] == taxon_id else dict(row) for row in taxa]
    changes = _diff_tables("taxon", ("taxonID",), taxa, updated)

    mapping_writes = {}
    for path in sorted((package_dir / "mappings").glob("*.tsv")):
        rows = read_tsv(path)
        renamed = [{**row, "concept": new_name} if row["taxonID"] == taxon_id else dict(row) for row in rows]
        if renamed != rows:
            mapping_writes[path] = renamed
            changes += _diff_tables(f"mappings/{path.stem}", ("verbatimIdentification",), rows, renamed)

    changeset = ChangeSet(changes=changes)
    if apply and changeset:
        write_tsv(package_dir / "taxon.tsv", TAXON_COLUMNS, _sorted_taxa(updated))
        for path, rows in mapping_writes.items():
            write_tsv(path, MAPPING_COLUMNS, rows)
        loader.cache_clear()
        changeset.applied = True
    return changeset


def merge(package_dir, retired_id: str, into_id: str, reason: str, *, apply: bool = False) -> ChangeSet:
    """Retire one concept into another, leaving a tombstone rather than a hole.

    An id that simply disappears is an identity lost silently — the migration refuses to do it and
    so does this. ``merged.tsv`` is the NCBI ``merged.dmp`` precedent: a retired id keeps resolving,
    to its replacement.
    """
    package_dir = Path(package_dir)
    if not reason:
        raise TaxonomyError("retiring an id needs a reason")
    if retired_id == into_id:
        raise TaxonomyError(f"cannot retire {retired_id} into itself")
    taxa = read_tsv(package_dir / "taxon.tsv")
    tombstones = read_tsv(package_dir / "merged.tsv")
    retired = {row["retired_id"]: row for row in tombstones}
    by_id = {row["taxonID"]: row for row in taxa}
    for candidate in (retired_id, into_id):
        if candidate in by_id:
            continue
        if candidate in retired:
            # The point of the ledger: a retired id still resolves, so the answer is where it went
            # rather than "no such thing".
            raise TaxonomyError(
                f"{candidate} was already retired into {retired[candidate]['replacement_id']} "
                f"on {retired[candidate]['date']} ({retired[candidate]['reason']})"
            )
        raise TaxonomyError(f"unknown taxonID {candidate!r}")
    if any(row["parentNameUsageID"] == retired_id for row in taxa):
        raise TaxonomyError(f"{retired_id} still has children; re-parent them before retiring it")

    kept_taxa = [row for row in taxa if row["taxonID"] != retired_id]
    survivor = by_id[into_id]["scientificName"]
    changes = _diff_tables("taxon", ("taxonID",), taxa, kept_taxa)

    mapping_writes = {}
    for path in sorted((package_dir / "mappings").glob("*.tsv")):
        rows = read_tsv(path)
        moved = [
            {**row, "taxonID": into_id, "concept": survivor} if row["taxonID"] == retired_id else dict(row) for row in rows
        ]
        if moved != rows:
            mapping_writes[path] = moved
            changes += _diff_tables(f"mappings/{path.stem}", ("verbatimIdentification",), rows, moved)

    # The retired concept's ids go with it rather than moving: an aphia id asserted of one concept
    # is not evidence about another, and re-asserting it here would launder a curator's judgement
    # into the identifier table with `semapv:UnspecifiedMatching` as its only justification.
    identifiers = read_tsv(package_dir / "identifier.tsv")
    kept_ids = [row for row in identifiers if row["subject_id"] != retired_id]
    changes += _diff_tables("identifier", ("subject_id", "object_id"), identifiers, kept_ids)

    tombstone = {"retired_id": retired_id, "replacement_id": into_id, "date": _today(), "reason": reason}
    new_tombstones = sorted([*tombstones, tombstone], key=lambda row: row["retired_id"])
    changes += _diff_tables("merged", ("retired_id",), tombstones, new_tombstones)

    changeset = ChangeSet(changes=changes)
    if apply:
        write_tsv(package_dir / "taxon.tsv", TAXON_COLUMNS, kept_taxa)
        write_tsv(package_dir / "identifier.tsv", IDENTIFIER_COLUMNS, kept_ids)
        write_tsv(package_dir / "merged.tsv", MERGED_COLUMNS, new_tombstones)
        for path, rows in mapping_writes.items():
            write_tsv(path, MAPPING_COLUMNS, rows)
        loader.cache_clear()
        changeset.applied = True
        logger.info(f"merge({retired_id} -> {into_id}): {reason}")
    return changeset


def _today() -> str:
    """UTC, so two curators on two continents write the same tombstone date for the same act."""
    return datetime.now(timezone.utc).date().isoformat()


def fmt(package_dir, *, apply: bool = False) -> ChangeSet:
    """Canonical form: sorted rows, normalised booleans, one spelling of every table.

    The counterpart to PR 4's ``merge=union``. A union resolution keeps both sides' lines and leaves
    the result unsorted; a spreadsheet round trip writes ``TRUE`` for ``true``. Both are cosmetic
    until they are not — the second silently inverts the only boolean the published dataset carries.
    """
    package_dir = Path(package_dir)
    changes, writes = [], {}

    taxa = read_tsv(package_dir / "taxon.tsv")
    ordered = _sorted_taxa(taxa)
    if [row["taxonID"] for row in ordered] != [row["taxonID"] for row in taxa]:
        changes.append(Change("taxon", "*", "order", "unsorted", "canonical"))
        writes[package_dir / "taxon.tsv"] = (TAXON_COLUMNS, ordered)

    identifiers = read_tsv(package_dir / "identifier.tsv")
    ordered_ids = sorted(identifiers, key=lambda row: (row["subject_id"], row["predicate_id"], row["object_id"]))
    if ordered_ids != identifiers:
        changes.append(Change("identifier", "*", "order", "unsorted", "canonical"))
        writes[package_dir / "identifier.tsv"] = (IDENTIFIER_COLUMNS, ordered_ids)

    tombstones = read_tsv(package_dir / "merged.tsv")
    ordered_tombstones = sorted(tombstones, key=lambda row: row["retired_id"])
    if ordered_tombstones != tombstones:
        changes.append(Change("merged", "*", "order", "unsorted", "canonical"))
        writes[package_dir / "merged.tsv"] = (MERGED_COLUMNS, ordered_tombstones)

    for path in sorted((package_dir / "mappings").glob("*.tsv")):
        rows = read_tsv(path)
        fixed = []
        for row in rows:
            row = dict(row)
            lowered = row["plankton"].strip().lower()
            if lowered in ("true", "false") and row["plankton"] != lowered:
                changes.append(
                    Change(f"mappings/{path.stem}", row["verbatimIdentification"], "plankton", row["plankton"], lowered)
                )
                row["plankton"] = lowered
            fixed.append(row)
        fixed.sort(key=lambda row: row["verbatimIdentification"])
        if fixed == rows:
            continue
        if [row["verbatimIdentification"] for row in fixed] != [row["verbatimIdentification"] for row in rows]:
            changes.append(Change(f"mappings/{path.stem}", "*", "order", "unsorted", "canonical"))
        writes[path] = (MAPPING_COLUMNS, fixed)

    changeset = ChangeSet(changes=changes)
    if apply and writes:
        for path, (columns, rows) in writes.items():
            write_tsv(path, columns, rows)
        loader.cache_clear()
        changeset.applied = True
    return changeset


def diff_published(package_dir, before: bytes) -> ChangeSet:
    """What a write moved in the PUBLISHED columns, rendered before and after.

    The question a reviewer actually has — "which published cells did this change?" — answered on
    the legacy view rather than on the store, because that is the view the dataset is built from.
    """
    after = render.render_wide_csv(loader.load_taxonomy(Path(package_dir)))
    if after == before:
        return ChangeSet()

    def keyed(raw):
        return {(row["Dataset"], row["Raw_Labels"]): row for row in csv.DictReader(io.StringIO(raw.decode("utf-8")))}

    old, new = keyed(before), keyed(after)
    changes = []
    for key in sorted(set(old) | set(new)):
        label = "/".join(key)
        if key not in new:
            changes.append(Change("published", label, "*", "row", ""))
        elif key not in old:
            changes.append(Change("published", label, "*", "", "row"))
        else:
            changes += [
                Change("published", label, column, old[key][column], new[key][column])
                for column in new[key]
                if old[key][column] != new[key][column]
            ]
    return ChangeSet(changes=changes)


__all__ = [
    "Change",
    "ChangeSet",
    "add_ids",
    "clear_id",
    "diff_published",
    "fmt",
    "merge",
    "records_from_wide_rows",
    "rename",
    "upsert_source",
    "upsert_wide_rows",
]
