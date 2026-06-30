"""
(c) Inria

The planktonzilla explorer package.

A light package marker for the interactive taxonomy/geo explorer. The two
load-bearing submodules are kept dependency-light on purpose:

* ``shapes`` — pure polars data-shaping transforms over the frozen taxonomy CSV
  (no gradio/plotly/huggingface_hub at module scope).
* ``data_access`` — the SINGLE network boundary, behind an injectable loader seam
  (the real HF read lazy-imports its heavy deps inside the function body).

The view phases (11/12/13) and the Space composition (Phase 14) build on these.
This ``__init__`` does NOT import the submodules eagerly, so importing the
package never pulls in polars/huggingface_hub/pyarrow until a submodule is used.
"""
