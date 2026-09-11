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
import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    PREFIX_TO_LEGACY_COLUMN,
    Mapping,
    Taxon,
    TaxonomyError,
    TaxonomyStore,
    collation_key,
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
    return load_cached(str(path.resolve()), _fingerprint(path))


def _fingerprint(path: Path) -> tuple:
    """``(size, mtime_ns)`` for a file; the same over every file for a package directory.

    Part of the cache key, so a taxonomy rewritten at a path already read is re-read rather than
    served stale. Without it the cache is a correctness bug and not merely a stale value: routing
    EVERY reader through one cache — as step 5 does — widens the blast radius of the legacy
    reader's own staleness from the generation path to the verifiers and the builders, which write
    a table and read it back in the same process. Found by a test doing exactly that.
    """
    if path.is_dir():
        return tuple(
            (str(child.relative_to(path)), child.stat().st_size, child.stat().st_mtime_ns)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        )
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def load_cached(resolved: str, fingerprint=None) -> TaxonomyStore:
    """The cached body of :func:`load_taxonomy`, keyed by resolved path AND content fingerprint.

    Caching is not an optimisation detail here: a published build constructs one redefiner per
    source, and without it each would re-read and re-derive the whole taxonomy. The legacy reader
    cached for exactly this reason, and dropping it while switching the readers over would trade a
    silent 21x regression for a tidier signature.

    The store it hands back is shared, which is the same contract the legacy cache had — it
    returned one shared dict. Callers treat it as a value; its own lazy caches are idempotent.
    A path whose contents change under the cache needs :func:`cache_clear`, as it always did.
    """
    path = Path(resolved)
    return _load_package(path) if path.is_dir() else _load_wide_csv(path)


def cache_clear() -> None:
    """Forget every loaded store. Needed when a test rewrites a taxonomy at a path already read."""
    load_cached.cache_clear()


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
    frozen = [(row["datasetID"], row["verbatimIdentification"]) for row in read_tsv(release / "legacy_row_order.tsv")]
    _check_release_pin(release, frozen)
    store.row_order = _effective_order(frozen, store.mappings)
    store.overrides = {
        (row["datasetID"], row["verbatimIdentification"]): row for row in read_tsv(release / "legacy_overrides.tsv")
    }

    return store


def _check_release_pin(release: Path, frozen) -> None:
    """A frozen release's order is frozen; its pin proves the bytes have not drifted.

    Deliberately NOT a hash of the render: design B gated on that and adding one source turned the
    gate red. This hashes the manifest's own keys, so the package may grow while the release it
    froze stays provably the one that was published.
    """
    digest = hashlib.sha256("\n".join(f"{dataset}\t{verbatim}" for dataset, verbatim in frozen).encode("utf-8")).hexdigest()
    recorded = (release / "sha256").read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise TaxonomyError(
            f"{release.name}/legacy_row_order.tsv has drifted: it hashes to {digest[:12]}… but "
            f"{release.name}/sha256 pins {recorded[:12]}…. A frozen release's order is not editable — "
            f"add rows to the mapping files instead; they render after the frozen block."
        )


def _effective_order(frozen, mappings) -> list:
    """The frozen release's order, then everything mapped since, by the rule the frozen block follows.

    A release freezes the physical order of the rows it published, and nothing more. Rows mapped
    afterwards are not in its manifest and must still render — otherwise ``write.upsert_source``
    would leave the package unloadable between adding a source and cutting the next release, which
    is every consumer broken for the length of a curation. So they follow the frozen block, sorted
    by the collation key the prefix already obeys (plan §10.6).

    Appending rather than interleaving is what the wide CSV itself did for its three post-freeze
    blocks, and it is what keeps the 1,486-line sha pin meaningful: a new source can never push a
    frozen row down the file.

    `draft` rows are excluded on purpose, so a partially mapped source can be committed. Design B
    could not commit one — 150 blank rows produced 150 errors and a renderer crash.
    """
    if len(set(frozen)) != len(frozen):
        repeated = sorted(key for key, count in Counter(frozen).items() if count > 1)
        raise TaxonomyError(f"the row-order manifest names {len(repeated)} key(s) twice, e.g. {repeated[0]}")

    for key in frozen:
        mapping = mappings.get(key)
        if mapping is None:
            raise TaxonomyError(f"the row-order manifest names {key}, which no mapping file carries")
        if mapping.status == "draft":
            # A row the release published cannot be walked back to draft: that is a published row
            # vanishing from the render with nothing to say it went.
            raise TaxonomyError(f"{key} is published in the frozen row order but its mapping is {mapping.status!r}")

    pinned = set(frozen)
    appended = sorted(
        (mapping for mapping in mappings.values() if mapping.status != "draft" and mapping.key not in pinned),
        key=lambda mapping: (collation_key(mapping.concept), mapping.dataset, mapping.verbatim),
    )
    return [*frozen, *(mapping.key for mapping in appended)]


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
__all__ = [
    "PACKAGE_DIR",
    "TaxonomyError",
    "TaxonomyStore",
    "build_taxonomy_lookup",
    "cache_clear",
    "load_cached",
    "load_taxonomy",
]
