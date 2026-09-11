"""
(c) Inria

The normalised taxonomy table package.

``load_taxonomy()`` is the single entry point on the read side. It accepts the package directory, a
legacy wide CSV, or nothing (the bundled package), and returns a :class:`TaxonomyStore` serving the
same legacy views either way — ``rows()``, ``lookup()``, ``frame()``, ``render_wide_csv()``, the
published projection, the label vocabularies and the coverage helpers. Every CSV reader in the
repository goes through it (plan PR 5).

:mod:`write` is the single entry point on the write side (plan PR 6): per-source, keyed, dry-run by
default, and physically unable to touch a source it was not asked about.

Following ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``: :mod:`migrate` generates ``data/`` from the
committed wide CSV and verifies it renders that CSV back byte-for-byte; :mod:`loader`, :mod:`model`
and :mod:`render` are the read path; :mod:`validate` is the schema of record.
``planktonzilla_taxonomy.csv`` stays the source of record until the retirement switch in PR 9.
"""

from planktonzilla.planktonzilla_dataset.taxonomy.loader import (
    PACKAGE_DIR,
    build_taxonomy_lookup,
    load_taxonomy,
)
from planktonzilla.planktonzilla_dataset.taxonomy.model import (
    Mapping,
    Taxon,
    TaxonomyError,
    TaxonomyStore,
)
from planktonzilla.planktonzilla_dataset.taxonomy.write import (
    Change,
    ChangeSet,
    clear_id,
    diff_published,
    fmt,
    upsert_source,
)

__all__ = [
    "PACKAGE_DIR",
    "Change",
    "ChangeSet",
    "Mapping",
    "Taxon",
    "TaxonomyError",
    "TaxonomyStore",
    "build_taxonomy_lookup",
    "clear_id",
    "diff_published",
    "fmt",
    "load_taxonomy",
    "upsert_source",
]
