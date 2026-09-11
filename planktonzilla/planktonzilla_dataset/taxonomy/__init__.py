"""
(c) Inria

The normalised taxonomy table package.

Step 1 of ``docs/TAXONOMY_IMPLEMENTATION_PLAN.md``: :mod:`migrate` generates ``data/`` from
the committed wide CSV, and verifies it by rendering the CSV back byte-for-byte. Nothing in
the repository reads ``data/`` yet — the loader, the model and the shipped renderer are step 2,
and until they land ``planktonzilla_taxonomy.csv`` remains the single source every consumer
reads.
"""
