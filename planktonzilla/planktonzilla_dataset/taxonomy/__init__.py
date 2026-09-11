"""
(c) Inria

The normalised taxonomy table package.

``load_taxonomy()`` is the single entry point. It accepts the package directory, a legacy wide
CSV, or nothing (the bundled package), and returns a :class:`TaxonomyStore` serving the same
legacy views either way — ``rows()``, ``lookup()``, ``frame()``, ``render_wide_csv()``, the
published projection, the label vocabularies and the coverage helpers.

Steps 1 and 2 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``: :mod:`migrate` generates ``data/``
from the committed wide CSV and verifies it renders that CSV back byte-for-byte;
:mod:`loader`, :mod:`model` and :mod:`render` are the shipped read path. Nothing in the
repository reads this package yet — moving the nine current CSV readers onto it is step 3, and
``planktonzilla_taxonomy.csv`` stays the source of record until then.
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

__all__ = [
    "PACKAGE_DIR",
    "Mapping",
    "Taxon",
    "TaxonomyError",
    "TaxonomyStore",
    "build_taxonomy_lookup",
    "load_taxonomy",
]
