"""
(c) Inria

``load_taxonomy()`` — the one entry point every consumer moves onto in step 3.

Two backends, chosen by what ``source`` points at:

``a directory``  the normalised package: taxon tree, identifier index, per-source mapping files,
                 vocabularies and the frozen-release layer.
``a file``       a legacy wide CSV. Not a courtesy — it is what keeps the Hydra key
                 ``taxonomy_csv_path`` and the ten fixture-writing test modules working
                 untouched. Those fixtures carry 18 columns in a different order, capitalised
                 rank values and ``root_class: zoo``; all of it must keep loading.
``None``         the package bundled in the wheel.

Both produce a :class:`TaxonomyStore` serving the same legacy views, so a caller never has to
know which one it got.
"""

import csv
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    LEGACY_RANKS,
    PREFIX_TO_LEGACY_COLUMN,
    Mapping,
    Taxon,
    TaxonomyError,
    TaxonomyStore,
    read_tsv,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

PACKAGE_DIR = Path(__file__).parent / "data"


def load_taxonomy(source=None) -> TaxonomyStore:
    """Load the taxonomy from the package directory, a legacy wide CSV, or the bundled package.

    Args:
        source: A directory (package backend), a file (wide-CSV backend), or ``None`` for the
            bundled package.

    Returns:
        A store serving ``rows()``, ``lookup()``, ``frame()``, ``render_wide_csv()`` and the
        coverage helpers, whichever backend was used.

    Raises:
        TaxonomyError: If ``source`` does not exist, or the package is internally inconsistent.
    """
    path = Path(source) if source is not None else PACKAGE_DIR
    if not path.exists():
        raise TaxonomyError(f"no taxonomy at «{path}»")
    return _load_package(path) if path.is_dir() else _load_wide_csv(path)


def _load_package(package_dir: Path) -> TaxonomyStore:
    """Read the normalised package into a store, checking what the shape itself can check."""
    store = TaxonomyStore(source=package_dir, is_package=True)

    store.taxa = {}
    for row in read_tsv(package_dir / "taxon.tsv"):
        taxon = Taxon.from_row(row)
        if taxon.taxon_id in store.taxa:
            raise TaxonomyError(f"duplicate taxonID {taxon.taxon_id!r} in taxon.tsv")
        store.taxa[taxon.taxon_id] = taxon

    for taxon in store.taxa.values():
        if taxon.parent_id and taxon.parent_id not in store.taxa:
            raise TaxonomyError(f"{taxon.taxon_id} has a dangling parent {taxon.parent_id!r}")

    store.vocab = {
        # rank -> legacy slot. project7 reads this, so it is a lookup rather than a table.
        "rank": {row["rank"]: row["legacy_slot"] for row in read_tsv(package_dir / "vocab" / "rank.tsv")},
        # qualifier -> the cell the legacy CSV carries (blank for `unqualified`).
        "qualifier": {row["qualifier"]: row["legacy_value"] for row in read_tsv(package_dir / "vocab" / "qualifier.tsv")},
        "root_class": {
            row["root_class"]: row["living"] == "true" for row in read_tsv(package_dir / "vocab" / "root_class.tsv")
        },
    }

    store.mappings = {}
    for path in sorted((package_dir / "mappings").glob("*.tsv")):
        for row in read_tsv(path):
            # dataset == file stem. Without it a foreign row passes the per-file primary key and
            # the exporter keeps whichever file the glob happens to visit last.
            if row["datasetID"] != path.stem:
                raise TaxonomyError(f"{path.name} holds a row for {row['datasetID']!r}")
            mapping = Mapping.from_row(row)
            if mapping.key in store.mappings:
                raise TaxonomyError(f"duplicate mapping {mapping.key} in {path.name}")
            if mapping.taxon_id not in store.taxa:
                raise TaxonomyError(f"{path.name}: {mapping.verbatim!r} points at unknown {mapping.taxon_id!r}")
            # The display name beside the opaque id, asserted equal rather than trusted: a blank
            # id resolves by name, but an id and a name that disagree is a hard error, never an
            # overwrite.
            named = store.taxa[mapping.taxon_id].scientific_name
            if mapping.concept != named:
                raise TaxonomyError(
                    f"{path.name}: {mapping.verbatim!r} names concept {mapping.concept!r} but {mapping.taxon_id} is {named!r}"
                )
            store.mappings[mapping.key] = mapping

    store.identifiers = {}
    for row in read_tsv(package_dir / "identifier.tsv"):
        prefix, separator, value = row["object_id"].partition(":")
        if separator != ":" or prefix not in PREFIX_TO_LEGACY_COLUMN:
            raise TaxonomyError(f"identifier.tsv: unrecognised CURIE {row['object_id']!r}")
        if row["subject_id"] not in store.taxa:
            raise TaxonomyError(f"identifier.tsv: unknown subject {row['subject_id']!r}")
        store.identifiers.setdefault(row["subject_id"], {}).setdefault(PREFIX_TO_LEGACY_COLUMN[prefix], []).append(value)

    release = package_dir / "release" / "v1.0"
    store.row_order = [(row["datasetID"], row["verbatimIdentification"]) for row in read_tsv(release / "legacy_row_order.tsv")]
    store.overrides = {
        (row["datasetID"], row["verbatimIdentification"]): row for row in read_tsv(release / "legacy_overrides.tsv")
    }

    # `draft` rows are excluded on purpose, so a partially mapped source can be committed: the
    # renderer skips them and they are not in any release's row order. Design B could not commit
    # one — 150 blank rows produced 150 errors and a renderer crash.
    accepted = [mapping for mapping in store.mappings.values() if mapping.status != "draft"]
    if len(store.row_order) != len(accepted):
        raise TaxonomyError(
            f"the row-order manifest has {len(store.row_order)} keys but the mapping files have "
            f"{len(accepted)} accepted rows ({len(store.mappings) - len(accepted)} draft)"
        )

    return store


def _load_wide_csv(csv_path: Path) -> TaxonomyStore:
    """Read a legacy wide CSV into a store.

    Column-order agnostic and tolerant of absent columns, because the committed file and the
    test fixtures agree on neither. ``rows()`` yields exactly what ``csv.DictReader`` would, and
    ``render_wide_csv()`` returns the file's own bytes rather than re-serialising it.
    """
    raw = csv_path.read_bytes()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    missing = [column for column in ("Dataset", "Raw_Labels") if rows and column not in rows[0]]
    if missing:
        raise TaxonomyError(f"«{csv_path}» has no {missing} column(s); it is not a taxonomy CSV")

    seen = {}
    for row in rows:
        key = (row["Dataset"], row["Raw_Labels"])
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        # Warn and keep last-wins, matching the legacy reader: a CSV that grows a duplicate says
        # so instead of quietly picking a winner.
        shown = ", ".join(f"{dataset}/{label}" for dataset, label in duplicates[:10])
        more = f" (+{len(duplicates) - 10} more)" if len(duplicates) > 10 else ""
        logger.warning(f"Taxonomy CSV has {len(duplicates)} duplicate (Dataset, Raw_Labels) keys; keeping last: {shown}{more}")

    return TaxonomyStore(source=csv_path, is_package=False, _legacy_rows=rows, _legacy_bytes=raw)


def build_taxonomy_lookup(csv_path) -> dict:
    """The legacy reader's signature, served by the loader.

    Kept so ``generate_planktonzilla.build_taxonomy_lookup`` can become a one-line alias in
    step 3 without any caller changing its import.
    """
    return load_taxonomy(csv_path).lookup()


def default_source():
    """The wide CSV while it is still the source of record; the package once it is not.

    One place to flip when step 8's retirement switch is thrown, rather than a default spread
    across four Hydra configs.
    """
    return constants.DEFAULT_TAXONOMY_CSV_FILENAME


# Re-exported so a caller needs one import rather than three.
__all__ = ["LEGACY_RANKS", "PACKAGE_DIR", "TaxonomyError", "TaxonomyStore", "build_taxonomy_lookup", "load_taxonomy"]
