---
title: planktonzilla explorer
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
python_version: "3.12"
pinned: false
short_description: "Interactive Sankey / Hierarchy / Map explorer over the planktonzilla taxonomy & sampling data. The map distinguishes Measured from Inferred locations — inferred points are dataset-level estimates, NOT ground truth."
---

# planktonzilla explorer

An interactive explorer over the **planktonzilla** master composite dataset — the frozen,
published taxonomy crosswalk and per-dataset sampling data behind the
[project-oceania](https://huggingface.co/project-oceania) models and datasets.

It is a single tabbed Gradio app with four tabs:

- **Sankey** — taxonomy flow diagram (Kingdom → … → Species) over the frozen crosswalk.
- **Hierarchy** — sunburst / icicle of the same taxonomy, with search and color toggles.
- **Map** — a token-free natural-earth world map of source-dataset sampling sites.
- **About** — data provenance and the inferred-location caveat.

## Data provenance

The taxonomy crosswalk (`planktonzilla_taxonomy.csv`) is assembled from these external sources:

- **NCBI Entrez** — taxonomic identifiers and COX1 sequence lookups
- **Wikidata** — cross-referenced taxon identifiers
- **WHOI** — Woods Hole plankton imagery / metadata
- **EcoTaxa** — annotated plankton image collections
- **ZooLake** — lake plankton imaging (Eawag)

## ⚠️ Inferred-location caveat — read before trusting the map

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
