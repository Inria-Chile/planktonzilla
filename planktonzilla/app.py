"""
(c) Inria

Root Hugging Face Space entry point for the planktonzilla explorer (Phase 14, SPACE-01/02).

This is the capstone composition (D1): a single tabbed ``gr.Blocks`` that ties the three
existing explorer views together into one runnable Space —

* **Sankey** — ``planktonzilla.explorer.sankey.render()`` (taxonomy flow diagram),
* **Hierarchy** — ``planktonzilla.explorer.hierarchy.render()`` (sunburst / icicle),
* **Map** — ``planktonzilla.explorer.geomap.render()`` (token-free natural-earth sampling map),
* **About** — provenance + the INFERRED-LOCATION CAVEAT (D2) so users never mistake inferred
  points for ground truth.

Design (D1): ``app.py`` lives UNDER ``planktonzilla/`` (imported as ``planktonzilla.app``), so the
Phase 9 dependency-isolation guard (which scans every ``*.py`` under ``planktonzilla/``) scans this
file too. It therefore keeps its ``gradio`` import FUNCTION-LOCAL (inside ``build_demo()`` and
``main()``) — mirroring the view modules — so NO module-scope viz import trips the guard. The view
modules keep their gradio/plotly imports FUNCTION-LOCAL as well; this file stays lean and only
composes their ``render()`` fragments.

Loading state (D3): the Map tab's first load runs the live 17M-row geo read via
``data_access.load_geo`` (in-process ``lru_cache``, once per session). The view fragment's own
``fragment.load(...)`` populates the plot after that cached read; this file adds a visible
first-load cue (a Markdown note above the map + ``show_progress`` on the fragment's load events)
so the cold-start latency is not a blank screen.
"""

from __future__ import annotations

from planktonzilla.explorer import geomap, hierarchy, sankey

# --------------------------------------------------------------------------- #
# About panel (D2, SPACE-02): data provenance + the inferred-location caveat.
# Mirrored in deploy/README.md (Plan 03) so the caveat is visible on the Space
# landing page too. Worded so users NEVER mistake inferred points for ground truth.
# --------------------------------------------------------------------------- #
ABOUT_MARKDOWN = """\
## About this explorer

An interactive explorer over the **planktonzilla** master composite dataset — the frozen,
published taxonomy crosswalk and per-dataset sampling data behind the
[project-oceania](https://huggingface.co/project-oceania) models and datasets.

### Data provenance

The taxonomy crosswalk is assembled from these external sources:

- **NCBI Entrez** — taxonomic identifiers and COX1 sequence lookups
- **Wikidata** — cross-referenced taxon identifiers
- **WHOI** — Woods Hole plankton imagery / metadata
- **EcoTaxa** — annotated plankton image collections
- **ZooLake** — lake plankton imaging (Eawag)

These feed the **frozen taxonomy crosswalk**
(`planktonzilla_taxonomy.csv`) that the Sankey and Hierarchy views render. The Map view
overlays per-source sampling locations.

### ⚠️ Inferred-location caveat — read before trusting the map

Sampling locations on the **Map** tab come in three confidence grades. **Inferred points are
dataset-level estimates, NOT ground truth** — do not treat them as precise collection sites:

- **Measured** — real per-sample GPS coordinates from the source dataset. Each measured dataset
  is shown as **one count-weighted centroid marker** summarising all its sampling sites (so a
  cruise track is a single representative point, not tens of thousands of markers).
- **Inferred — high confidence** — no per-sample GPS was published, but the source paper /
  archive documents a specific site, so the dataset-level location is a well-supported estimate.
- **Inferred — low confidence** — the source documents only a broad region; the plotted point is
  a coarse proxy near that region.

Some datasets are **excluded from the map entirely** because they have no real collection site
(marked `na` / "do not plot"): e.g. **lensless** (lab-cultured *Carolina Biological* protozoa,
imaged at IBM Almaden — not field-collected) and **planktoscope** (a diffuse multi-campaign
reference library with no single location).

Finally, one correction: the **zoolake** dataset is plotted at **Lake Greifensee, Switzerland**
— NOT Lake Zurich, which the project's earlier README incorrectly reported.
"""

# First-load cue for the Map tab (D3): the live read can be slow on a cold cpu-basic Space.
MAP_LOADING_NOTE = (
    "> ⏳ **First map load fetches live sampling coordinates** and may take a moment on cold "
    "start. The result is cached for the rest of your session, so subsequent loads are instant."
)


def build_demo():
    """Compose the four-tab explorer ``gr.Blocks`` (Sankey / Hierarchy / Map / About).

    Builds a single ``gr.Blocks`` with a ``gr.Tabs()`` holding exactly four ``gr.Tab``s in order:
    "Sankey", "Hierarchy", "Map", "About". The first three tabs embed the existing view
    ``render()`` fragments (each returns its own ``gr.Blocks``); the About tab is a ``gr.Markdown``
    carrying the D2 provenance + inferred-location caveat. Above the Map fragment a Markdown note
    (``MAP_LOADING_NOTE``) provides the D3 first-load cue for the live geo read.

    Does NOT launch — returns the built Blocks so the composition smoke test can introspect the
    tab tree offline (SPACE-04). ``main()`` below launches it for local dev.

    Returns:
        A ``gr.Blocks`` with four tabs; the Map tab shows a first-load loading affordance.
    """
    import gradio as gr

    with gr.Blocks(title="planktonzilla explorer") as demo:
        gr.Markdown("# planktonzilla explorer")
        with gr.Tabs():
            with gr.Tab("Sankey"):
                sankey.render()
            with gr.Tab("Hierarchy"):
                hierarchy.render()
            with gr.Tab("Map"):
                gr.Markdown(MAP_LOADING_NOTE)
                geomap.render()
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MARKDOWN)
    return demo


def main() -> None:
    """Build and launch the four-tab explorer Space for local dev.

    Keeps its ``gradio`` import FUNCTION-LOCAL (D1) so no viz import lands at module scope; the
    root Space runtime imports ``planktonzilla.app`` and calls ``build_demo()`` directly, so this
    launch path is only exercised for local ``python -m planktonzilla.app`` runs.
    """
    import gradio as gr  # noqa: F401  -- keep the launch path's viz import function-local (D1)

    build_demo().launch()


if __name__ == "__main__":
    main()
