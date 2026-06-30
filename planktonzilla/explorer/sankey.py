"""
(c) Inria

In-Space interactive Sankey view over the taxonomy crosswalk (Phase 11, SANKEY-01..05).

This module is the first of the three explorer views. It builds a Plotly ``go.Sankey``
FRESH from the Phase 10 canonical ``shapes.build_sankey_index`` (D1) and exposes:

* ``make_sankey_figure`` — the figure builder (``go.Sankey``) with value-sorted nodes,
  per-stage "Other (n taxa)" long-tail aggregation, and root_class lineage link coloring.
* ``render`` — a ``gr.Blocks`` fragment: a CheckboxGroup of stages (pick + order), one-click
  preset buttons, a per-source-dataset Dropdown filter, a ``gr.Plot``, and HTML + PNG export.
* ``main`` — a standalone local Gradio app (``python -m planktonzilla.explorer.sankey``).

Design (load-bearing — D4): gradio, plotly, AND kaleido are imported INSIDE function
bodies, NEVER at module scope. The Phase 9 dependency-isolation guard
(``tests/test_dependency_isolation.py``) fails the build on any module-scope viz import
under ``planktonzilla/``. The module's pure helpers (Other-aggregation, lineage tally,
dataset filtering) import cleanly in the CORE env with gradio/plotly/kaleido ABSENT.

The standalone ``generate_sankey.py`` HTML CLI is a SECOND, independent front-end and is
NEVER modified (D5) — its self-contained HTML output stays byte-identical (zero-drift).
The Sankey semantics ported here (Other-aggregation, value-sort, lineage coloring) mirror
that file's JS ``build()`` / ``draw()`` but are computed entirely Python-side.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from planktonzilla.explorer import shapes
from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_TAXONOMY_CSV_FILENAME,
    EXTRA_COLS,  # noqa: F401  -- re-exported for the view phases' / app.py convenience
    TAXONOMY_RANKS,  # noqa: F401  -- re-exported for stage-choice composition
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Curated stage order + friendly labels for the taxonomy crosswalk — ported from
# generate_sankey.py's PREFERRED list (kept in sync, NOT imported — D1).
PREFERRED: list[tuple[str, str]] = [
    ("Dataset", "Source dataset"),
    ("root_class", "Root class"),
    ("plankton", "Plankton?"),
    ("living", "Living?"),
    ("Kingdom", "Kingdom"),
    ("Phylum", "Phylum"),
    ("Class", "Class"),
    ("Order", "Order"),
    ("Family", "Family"),
    ("Genus", "Genus"),
    ("Species", "Species"),
    ("qualifier", "Qualifier"),
    ("proposed_label", "Proposed label"),
]
PREFERRED_LABELS: dict[str, str] = dict(PREFERRED)

# Default stages shown on first load — matches generate_sankey.py's default (Linnaean order).
DEFAULT_STAGES = ["Dataset", "Kingdom", "Phylum", "Class"]

# One-click presets (key -> stage list) — ported from generate_sankey.py [data-preset] buttons.
PRESETS: dict[str, list[str]] = {
    "Source → taxonomy": ["Dataset", "Kingdom", "Phylum", "Class"],
    "Linnaean ranks": ["Kingdom", "Phylum", "Class", "Order", "Family"],
    "Source → category": ["Dataset", "root_class", "plankton", "living"],
    "Lower ranks → label": ["Class", "Order", "Family", "Genus", "proposed_label"],
}

# The per-source-dataset filter column and its "no filter" sentinel.
DATASET_COLUMN = "Dataset"
ALL_DATASETS = "All"

# Node palette per stage (ported from generate_sankey.py STAGE_COLORS).
STAGE_COLORS = ["#c9191e", "#2b5c8a", "#3f8e6b", "#c77f2e", "#6b5b95", "#3f8e9e", "#a23b72", "#7a8290", "#b0563f"]
# Categorical palette for lineage (root_class) link coloring (ported from LINEAGE_PALETTE).
LINEAGE_PALETTE = ["#c9191e", "#2b5c8a", "#3f8e6b", "#c77f2e", "#6b5b95", "#a23b72"]
LINEAGE_UNKNOWN = "Unknown"

# Shared muted gray for "(blank)" and "Other (...)" nodes; the Other-node label prefix.
MUTED = "#aab3bf"
OTHER_PREFIX = "Other ("  # full label = "Other (n taxa)"


# --------------------------------------------------------------------------- #
# Pure helpers — polars / stdlib only. NO gradio / plotly / kaleido here.
# --------------------------------------------------------------------------- #
def distinct_datasets(df: pl.DataFrame, *, column: str = DATASET_COLUMN) -> list[str]:
    """Return the dataset Dropdown choices: ``"All"`` plus sorted distinct dataset values.

    Args:
        df: The taxonomy frame (typically from ``shapes.load_taxonomy``).
        column: The source-dataset column name. Defaults to ``"Dataset"``.

    Returns:
        ``["All", <sorted non-empty distinct dataset values>...]``. If ``column`` is
        absent the result is just ``["All"]``.
    """
    if column not in df.columns:
        return [ALL_DATASETS]
    vals = df.get_column(column).str.strip_chars().unique().to_list()
    return [ALL_DATASETS, *sorted(v for v in vals if v)]


def filter_by_dataset(df: pl.DataFrame, dataset: str | None, *, column: str = DATASET_COLUMN) -> pl.DataFrame:
    """Filter crosswalk rows to a single source dataset BEFORE ``build_sankey_index`` (SANKEY-02).

    ``"All"``/``None``/empty (or a missing column) is a no-op that returns the frame
    unchanged. Otherwise rows are kept where ``column`` equals ``dataset`` exactly
    (the frame is already stripped by ``load_taxonomy``).

    Args:
        df: The taxonomy frame.
        dataset: A dataset value, ``"All"``, ``None``, or "".
        column: The source-dataset column name. Defaults to ``"Dataset"``.

    Returns:
        The filtered (or unchanged) frame. The input is never mutated.
    """
    if not dataset or dataset == ALL_DATASETS or column not in df.columns:
        return df
    return df.filter(pl.col(column).str.strip_chars() == dataset)


def link_lineage(df: pl.DataFrame, stages: list[str], *, drop_blank: bool = False) -> dict[tuple[int, int], str]:
    """Compute the majority ``root_class`` of the records flowing through each raw link.

    Mirrors generate_sankey.py's ``build()`` ``linkLineage``: walk every row's
    consecutive-stage transitions, tally ``root_class`` per ``(source_node, target_node)``
    link (blank/missing root_class -> ``"Unknown"``), then return the argmax class per
    link. Link node identity matches ``build_sankey_index`` exactly (``(stage_index,
    value)`` with blanks as ``"(blank)"``), so the returned keys align with the index's
    raw ``(source, target)`` node-id pairs.

    Args:
        df: The (already dataset-filtered) taxonomy frame.
        stages: Ordered stage columns (same list passed to ``build_sankey_index``).
        drop_blank: Mirror of ``build_sankey_index``'s ``drop_blank`` so the lineage
            tally walks exactly the same rows.

    Returns:
        ``{(source_node_id, target_node_id): majority_root_class}``. Empty if
        ``root_class`` is absent (callers fall back to per-stage node tint).
    """
    if "root_class" not in df.columns or len(stages) < 2:
        return {}
    missing = [s for s in stages if s not in df.columns]
    if missing:
        return {}

    node_index: dict[tuple[int, str], int] = {}

    def nid(stage_i: int, value: str) -> int:
        key = (stage_i, value)
        idx = node_index.get(key)
        if idx is None:
            idx = len(node_index)
            node_index[key] = idx
        return idx

    columns = [df.get_column(s).to_list() for s in stages]
    rc_col = df.get_column("root_class").to_list()
    tally: dict[tuple[int, int], dict[str, int]] = {}
    n_rows = df.height
    for row_i in range(n_rows):
        vals = [columns[s_i][row_i] or shapes.BLANK for s_i in range(len(stages))]
        if drop_blank and shapes.BLANK in vals:
            continue
        rc = (rc_col[row_i] or "").strip() or LINEAGE_UNKNOWN
        for i in range(len(stages) - 1):
            s = nid(i, vals[i])
            t = nid(i + 1, vals[i + 1])
            per_link = tally.setdefault((s, t), {})
            per_link[rc] = per_link.get(rc, 0) + 1

    majority: dict[tuple[int, int], str] = {}
    for link_key, counts in tally.items():
        majority[link_key] = max(counts.items(), key=lambda kv: kv[1])[0]
    return majority


def aggregate_other(index: dict, *, min_flow: int = 1, root_class_by_link: dict | None = None) -> dict:
    """Collapse per-stage long-tail nodes into one muted "Other (n taxa)" node per stage.

    Pure port of generate_sankey.py's ``build()`` aggregation pass. A node's flow is the
    max of its incident in/out link values. Nodes whose flow is below ``min_flow`` collapse
    into a single ``"Other (n taxa)"`` node at the same stage; collapsed links are rerouted
    and their values summed so flow is CONSERVED (nothing is dropped). With ``min_flow <= 1``
    no node collapses and the index is returned with the metadata fields added.

    The returned index extends ``build_sankey_index``'s shape with two per-node fields:
    ``other`` (bool: this is an aggregated Other node) and the same ``stage``/``label``;
    and a per-link ``link_class`` list (majority root_class) when ``root_class_by_link`` is
    given. ``used_rows`` is carried through unchanged.

    Args:
        index: A ``shapes.build_sankey_index`` result (nodes/source/target/value/used_rows).
        min_flow: Nodes whose incident flow is below this collapse into Other. ``<= 1`` keeps all.
        root_class_by_link: Optional ``{(raw_source, raw_target): root_class}`` from
            ``link_lineage`` used to color the aggregated links by majority lineage.

    Returns:
        ``{"nodes": [{"stage", "label", "other"}...], "source", "target", "value",
        "link_class", "used_rows"}``. Total link value is unchanged by aggregation.
    """
    raw_nodes = index["nodes"]
    raw_source = index["source"]
    raw_target = index["target"]
    raw_value = index["value"]

    n_raw = len(raw_nodes)
    in_flow = [0] * n_raw
    out_flow = [0] * n_raw
    for s, t, v in zip(raw_source, raw_target, raw_value, strict=True):
        out_flow[s] += v
        in_flow[t] += v
    node_flow = [max(in_flow[i], out_flow[i]) for i in range(n_raw)]

    # Decide which raw nodes collapse into their stage's Other node.
    collapse = [False] * n_raw
    other_taxa: dict[int, int] = {}  # stage -> count of distinct taxa folded into Other
    if min_flow > 1:
        for i in range(n_raw):
            if node_flow[i] < min_flow:
                collapse[i] = True
                stage = raw_nodes[i]["stage"]
                other_taxa[stage] = other_taxa.get(stage, 0) + 1

    # Build the aggregated node list: surviving raw nodes + one Other per stage that needs it.
    nodes: list[dict] = []
    remap = [-1] * n_raw
    for i in range(n_raw):
        if collapse[i]:
            continue
        remap[i] = len(nodes)
        nodes.append({"stage": raw_nodes[i]["stage"], "label": raw_nodes[i]["label"], "other": False})
    other_node: dict[int, int] = {}  # stage -> aggregated node index
    for stage, n in other_taxa.items():
        other_node[stage] = len(nodes)
        nodes.append({"stage": stage, "label": f"{OTHER_PREFIX}{n} taxa)", "other": True})

    def final_id(raw_id: int) -> int:
        return other_node[raw_nodes[raw_id]["stage"]] if collapse[raw_id] else remap[raw_id]

    # Reroute every link through final_id, summing values and merging lineage tallies.
    agg_val: dict[tuple[int, int], int] = {}
    agg_lineage: dict[tuple[int, int], dict[str, int]] = {}
    for idx, (rs, rt, v) in enumerate(zip(raw_source, raw_target, raw_value, strict=True)):
        fs, ft = final_id(rs), final_id(rt)
        fk = (fs, ft)
        agg_val[fk] = agg_val.get(fk, 0) + v
        if root_class_by_link is not None:
            rc = root_class_by_link.get((rs, rt), LINEAGE_UNKNOWN)
            dst = agg_lineage.setdefault(fk, {})
            dst[rc] = dst.get(rc, 0) + v

    source: list[int] = []
    target: list[int] = []
    value: list[int] = []
    link_class: list[str] = []
    for (fs, ft), v in agg_val.items():
        source.append(fs)
        target.append(ft)
        value.append(v)
        if root_class_by_link is not None:
            counts = agg_lineage.get((fs, ft), {})
            link_class.append(max(counts.items(), key=lambda kv: kv[1])[0] if counts else LINEAGE_UNKNOWN)
        else:
            link_class.append(LINEAGE_UNKNOWN)

    return {
        "nodes": nodes,
        "source": source,
        "target": target,
        "value": value,
        "link_class": link_class,
        "used_rows": index.get("used_rows", 0),
    }


def lineage_colors(index: dict) -> dict[str, str]:
    """Map each distinct ``link_class`` in an aggregated index to a stable hex color.

    Mirrors generate_sankey.py's ``buildLineageColors``: known classes (sorted) get the
    categorical palette in order; ``"Unknown"`` always maps to the muted gray.

    Args:
        index: An ``aggregate_other`` result carrying ``link_class``.

    Returns:
        ``{root_class: "#rrggbb"}`` covering every class present in ``link_class``.
    """
    seen: list[str] = []
    for rc in index.get("link_class", []):
        if rc not in seen:
            seen.append(rc)
    known = sorted(k for k in seen if k != LINEAGE_UNKNOWN)
    colors: dict[str, str] = {}
    for i, k in enumerate(known):
        colors[k] = LINEAGE_PALETTE[i % len(LINEAGE_PALETTE)]
    if LINEAGE_UNKNOWN in seen:
        colors[LINEAGE_UNKNOWN] = MUTED
    return colors


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(...)`` string (mirrors generate_sankey.py hexA)."""
    n = int(hex_color.lstrip("#"), 16)
    return f"rgba({(n >> 16) & 255},{(n >> 8) & 255},{n & 255},{alpha})"


# --------------------------------------------------------------------------- #
# Lazy figure builder — plotly imported INSIDE the function body (D4).
# --------------------------------------------------------------------------- #
def make_sankey_figure(index: dict, stages: list[str], *, root_class_by_link: dict | None = None):
    """Build a Plotly ``go.Sankey`` figure from an aggregated shapes index (SANKEY-01).

    The ``index`` is expected to be an ``aggregate_other`` result (carrying ``other`` per
    node + ``link_class`` per link); a plain ``build_sankey_index`` result also works (it
    is aggregated with ``min_flow=1`` first, a no-op). Nodes are value-sorted per stage
    (heaviest at top; "Other (...)"/``(blank)`` pinned to the bottom) via node x/y, echoing
    generate_sankey.py's ``draw()``. Node hovertemplate shows ``"%{label}<br>%{value} rows"``;
    link hovertemplate shows ``"source → target<br>value rows"``; flow widths equal link
    values (= label-row counts). Links are colored by their majority ``root_class`` (lineage);
    if no lineage is present they fall back to a per-stage node tint.

    plotly is imported INSIDE this function (D4) so the module loads in the core env.

    Args:
        index: An ``aggregate_other`` (or ``build_sankey_index``) result.
        stages: The ordered stage columns (for x positions + column annotations).
        root_class_by_link: Unused here (lineage already baked into ``index["link_class"]``);
            accepted for call-site symmetry.

    Returns:
        A ``plotly.graph_objects.Figure`` with one ``go.Sankey`` trace.
    """
    import plotly.graph_objects as go

    # Accept a plain build_sankey_index result by aggregating with a no-op threshold.
    if index.get("nodes") and "other" not in index["nodes"][0]:
        index = aggregate_other(index, min_flow=1)

    nodes = index["nodes"]
    source = index["source"]
    target = index["target"]
    value = index["value"]
    link_class = index.get("link_class", [LINEAGE_UNKNOWN] * len(source))
    n_nodes = len(nodes)

    # Per-node total flow (max of incident in/out) for sizing/sorting the aggregated nodes.
    in_flow = [0] * n_nodes
    out_flow = [0] * n_nodes
    for s, t, v in zip(source, target, value, strict=True):
        out_flow[s] += v
        in_flow[t] += v
    flow = [max(in_flow[i], out_flow[i]) for i in range(n_nodes)]

    # Group node indices by stage; order each stage by value DESC (Other/blank pinned bottom).
    by_stage: dict[int, list[int]] = {}
    for i, n in enumerate(nodes):
        by_stage.setdefault(n["stage"], []).append(i)

    def _pinned(i: int) -> bool:
        return bool(nodes[i].get("other")) or nodes[i]["label"] == shapes.BLANK

    ys = [0.5] * n_nodes
    for idxs in by_stage.values():
        idxs.sort(key=lambda i: (1 if _pinned(i) else 0, -flow[i]))
        total = sum(flow[i] for i in idxs) or 1
        cum = 0
        for i in idxs:
            ys[i] = min(0.98, max(0.02, (cum + flow[i] / 2) / total))
            cum += flow[i]

    n_stage = len(stages)
    xs = [(min(0.96, max(0.04, n["stage"] / (n_stage - 1))) if n_stage > 1 else 0.5) for n in nodes]

    node_colors = [
        (MUTED if (n.get("other") or n["label"] == shapes.BLANK) else STAGE_COLORS[n["stage"] % len(STAGE_COLORS)])
        for n in nodes
    ]
    node_labels = [n["label"] for n in nodes]

    colors = lineage_colors(index)
    link_colors = [_hex_rgba(colors.get(rc, MUTED), 0.45) for rc in link_class]

    sankey = go.Sankey(
        arrangement="fixed",
        node={
            "pad": 12,
            "thickness": 16,
            "line": {"color": "#ffffff", "width": 0.8},
            "label": node_labels,
            "color": node_colors,
            "x": xs,
            "y": ys,
            "hovertemplate": "%{label}<br>%{value} rows<extra></extra>",
        },
        link={
            "source": source,
            "target": target,
            "value": value,
            "color": link_colors,
            "hovertemplate": "%{source.label} → %{target.label}<br>%{value} rows<extra></extra>",
        },
    )

    annotations = [
        {
            "x": i / (n_stage - 1) if n_stage > 1 else 0.5,
            "y": 1.04,
            "xref": "paper",
            "yref": "paper",
            "text": f"<b>{PREFERRED_LABELS.get(k, k)}</b>",
            "showarrow": False,
            "font": {"color": "#374151", "size": 11},
            "xanchor": "left" if i == 0 else ("right" if i == n_stage - 1 else "center"),
        }
        for i, k in enumerate(stages)
    ]

    # Scale height to the busiest visible stage so dense ranks get vertical room.
    max_nodes_in_stage = max((len(v) for v in by_stage.values()), default=1)
    height = max(600, max_nodes_in_stage * 22)

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1a1a1a", "size": 11},
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        annotations=annotations,
    )
    return fig


def build_figure(
    df: pl.DataFrame,
    stages: list[str],
    *,
    dataset: str | None = None,
    drop_blank: bool = False,
    min_flow: int = 1,
):
    """End-to-end figure: dataset-filter -> build_sankey_index -> aggregate_other -> figure.

    Convenience wrapper used by ``render``'s callbacks and the tests. Filters the frame to
    ``dataset`` (SANKEY-02), builds the canonical index (D1), computes per-link majority
    lineage, aggregates the long tail into "Other (n taxa)" (SANKEY-01), and renders the
    ``go.Sankey``. Returns ``None`` if fewer than two stages are selected (gradio shows a
    blank plot rather than raising).

    Args:
        df: The taxonomy frame.
        stages: Ordered selected stage columns.
        dataset: Optional source-dataset filter ("All"/None = no filter).
        drop_blank: Drop rows blank in any selected stage.
        min_flow: Long-tail aggregation threshold (nodes below collapse into Other).

    Returns:
        A ``plotly.graph_objects.Figure`` or ``None`` when ``< 2`` stages are selected.
    """
    if not stages or len(stages) < 2:
        return None
    sub = filter_by_dataset(df, dataset)
    index = shapes.build_sankey_index(sub, list(stages), drop_blank=drop_blank)
    lineage = link_lineage(sub, list(stages), drop_blank=drop_blank)
    aggregated = aggregate_other(index, min_flow=int(min_flow), root_class_by_link=lineage)
    return make_sankey_figure(aggregated, list(stages))


# --------------------------------------------------------------------------- #
# Lazy export helpers — plotly / kaleido pulled in INSIDE the function (D4, SANKEY-04).
# --------------------------------------------------------------------------- #
def export_html(fig) -> str:
    """Write the current figure to a standalone HTML file and return its path (SANKEY-04).

    Uses ``fig.to_html(include_plotlyjs="cdn", full_html=True)`` so the page is a complete,
    self-contained document loading plotly.js from the CDN. The path is created with
    ``tempfile`` (no user-controlled string interpolation into the path — T-11-03).

    Args:
        fig: A ``plotly.graph_objects.Figure``.

    Returns:
        The absolute path to the written ``.html`` file.

    Raises:
        ValueError: If ``fig`` is ``None`` (nothing to export).
    """
    if fig is None:
        raise ValueError("no figure to export")
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    fd, path = tempfile.mkstemp(prefix="planktonzilla_sankey_", suffix=".html")
    Path(path).write_text(html, encoding="utf-8")
    import os

    os.close(fd)
    logger.info("Exported Sankey HTML to «%s» (%d bytes).", path, len(html))
    return path


def export_png(fig) -> str:
    """Write the current figure to a static PNG file and return its path (SANKEY-04).

    Uses ``fig.write_image(path)`` which pulls in ``kaleido`` (the isolated explorer-group
    static-export backend) transitively. The path is created with ``tempfile`` (T-11-03).

    Args:
        fig: A ``plotly.graph_objects.Figure``.

    Returns:
        The absolute path to the written ``.png`` file.

    Raises:
        ValueError: If ``fig`` is ``None`` (nothing to export).
    """
    if fig is None:
        raise ValueError("no figure to export")
    fd, path = tempfile.mkstemp(prefix="planktonzilla_sankey_", suffix=".png")
    import os

    os.close(fd)
    fig.write_image(path)  # requires kaleido (explorer group only)
    logger.info("Exported Sankey PNG to «%s».", path)
    return path


# --------------------------------------------------------------------------- #
# Lazy UI fragment — gradio imported INSIDE the function body (D4).
# --------------------------------------------------------------------------- #
def render(df: pl.DataFrame | None = None):
    """Build the Sankey ``gr.Blocks`` fragment (CheckboxGroup + presets + filter + export).

    Wires (D2/SANKEY-02/03/04):

    * a ``gr.CheckboxGroup`` of available stages (pick + order; default ``DEFAULT_STAGES``),
    * one-click preset ``gr.Button``s (``PRESETS``) that SET the CheckboxGroup value,
    * a ``gr.Dropdown`` of distinct ``Dataset`` values + ``"All"`` that filters rows BEFORE
      ``build_sankey_index``,
    * a ``gr.Slider`` for the "Other (n taxa)" aggregation threshold and a "drop blank" checkbox,
    * a ``gr.Plot`` output, and
    * HTML + PNG ``gr.DownloadButton`` export controls.

    gradio is imported INSIDE this function (D4). The frozen taxonomy CSV is loaded lazily
    when ``df`` is not supplied (network-free; pure shapes).

    Args:
        df: Optional pre-loaded taxonomy frame. Loaded from the frozen CSV when ``None``.

    Returns:
        A ``gr.Blocks`` fragment ready to be composed into the Space (Phase 14) or launched
        standalone by ``main``.
    """
    import gradio as gr

    if df is None:
        df = shapes.load_taxonomy(DEFAULT_TAXONOMY_CSV_FILENAME)

    available_stages = [k for k, _ in PREFERRED if k in df.columns]
    # Any extra columns not in PREFERRED are still selectable (label = column name).
    available_stages += [c for c in df.columns if c not in available_stages]
    stage_default = [s for s in DEFAULT_STAGES if s in df.columns]
    dataset_choices = distinct_datasets(df)

    def _update(stages, dataset, drop_blank, min_flow):
        return build_figure(df, list(stages or []), dataset=dataset, drop_blank=drop_blank, min_flow=int(min_flow))

    def _export_html(stages, dataset, drop_blank, min_flow):
        fig = build_figure(df, list(stages or []), dataset=dataset, drop_blank=drop_blank, min_flow=int(min_flow))
        return export_html(fig) if fig is not None else None

    def _export_png(stages, dataset, drop_blank, min_flow):
        fig = build_figure(df, list(stages or []), dataset=dataset, drop_blank=drop_blank, min_flow=int(min_flow))
        return export_png(fig) if fig is not None else None

    with gr.Blocks() as fragment:
        gr.Markdown("## Taxonomy Sankey\nFlow width = number of label rows. Hover a node/link for counts.")
        with gr.Row():
            with gr.Column(scale=1):
                stage_in = gr.CheckboxGroup(
                    choices=available_stages,
                    value=stage_default,
                    label="Flow stages (selection order = left→right stage order)",
                )
                gr.Markdown("**Presets**")
                preset_buttons = [gr.Button(preset_name, size="sm") for preset_name in PRESETS]
                dataset_in = gr.Dropdown(choices=dataset_choices, value=ALL_DATASETS, label="Source dataset filter")
                drop_blank_in = gr.Checkbox(value=False, label="Drop rows blank in any selected stage")
                min_flow_in = gr.Slider(minimum=1, maximum=40, step=1, value=1, label="Aggregate below N taxa into «Other»")
                with gr.Row():
                    html_btn = gr.DownloadButton("Export HTML", size="sm")
                    png_btn = gr.DownloadButton("Export PNG", size="sm")
            with gr.Column(scale=4):
                plot = gr.Plot(label="Sankey")

        controls = [stage_in, dataset_in, drop_blank_in, min_flow_in]
        for control in controls:
            control.change(_update, inputs=controls, outputs=plot)

        # Preset buttons SET the CheckboxGroup value (then the .change above redraws).
        for button, preset_name in zip(preset_buttons, PRESETS, strict=True):
            preset_stages = [s for s in PRESETS[preset_name] if s in df.columns]
            button.click(lambda value=preset_stages: gr.update(value=value), inputs=None, outputs=stage_in)

        html_btn.click(_export_html, inputs=controls, outputs=html_btn)
        png_btn.click(_export_png, inputs=controls, outputs=png_btn)

        fragment.load(_update, inputs=controls, outputs=plot)

    return fragment


def main(argv: list[str] | None = None) -> int:
    """Launch the standalone local Sankey app (``python -m planktonzilla.explorer.sankey``).

    Wraps ``render()`` in a ``gr.Blocks`` and ``.launch()``-es it for local development. No
    core console script imports viz at module scope (D4); this entry point is invoked under
    the explorer group only.

    Args:
        argv: Unused; accepted for CLI-symmetry with the other entry points.

    Returns:
        Process exit code (0 on a clean launch).
    """
    import gradio as gr

    with gr.Blocks(title="planktonzilla — taxonomy Sankey") as app:
        render()
    app.launch()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
