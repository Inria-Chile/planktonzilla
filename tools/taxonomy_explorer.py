#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["gradio", "plotly", "polars"]
# ///
"""
(c) Inria

Gradio Blocks taxonomy explorer for the planktonzilla taxonomy crosswalk.

Ports the standalone ``tools/generate_sankey.py`` HTML generator into an
interactive multi-tab Gradio app over ``planktonzilla_taxonomy.csv`` — the CSV
that routes each source ``Dataset``/``Raw_Labels`` row through the Linnaean ranks
(``Kingdom -> Phylum -> ... -> Species``) down to a unified ``proposed_label``.

The app exposes three tabs sharing global filters (plankton / Kingdom / Phylum):

1. Sankey: ordered-stage flow over any CSV columns (default Dataset -> Order).
2. Hierarchy: sunburst (or icicle) over Kingdom -> Species.
3. Audit: a searchable dataframe of the curated columns.

Design (load-bearing): the pure data-shaping layer imports ONLY polars at module
scope. ``gradio`` and ``plotly`` are lazily imported INSIDE the figure / UI
builder functions so this module imports cleanly in an environment that has
neither library (e.g. the project test env), which lets the test suite exercise
the pure layer directly.

Run the full app (resolves gradio/plotly via PEP 723 inline metadata)::

    uv run tools/taxonomy_explorer.py

CI-style verification without launching a server::

    uv run tools/taxonomy_explorer.py --smoke
    uv run tools/taxonomy_explorer.py --smoke --csv path/to/other.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Column groupings — copied verbatim from
# planktonzilla/planktonzilla_dataset/constants.py (do NOT import that package;
# this tool is standalone and must not pull in the planktonzilla deps).
# --------------------------------------------------------------------------- #
TAXONOMY_RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
EXTRA_COLS = ("proposed_label", "plankton", "root_class", "qualifier")
ID_STR_COLS = ("wikidata_ID", "ecotaxa_ID")
ID_NUM_COLS = ("aphia_ID", "NCBI_ID", "BOLD_ID")

# Curated stage order + friendly labels for the taxonomy crosswalk — copied from
# tools/generate_sankey.py's PREFERRED list (kept in sync, not imported).
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

# Requirements ask the Sankey tab to default to a 5-stage view.
DEFAULT_STAGES = ["Dataset", "Kingdom", "Phylum", "Class", "Order"]

# String sentinel for a blank cell in a Sankey stage (mirrors generate_sankey.py JS).
BLANK = "(blank)"

# Link colour palette per source stage (mirrors generate_sankey.py STAGE_COLORS).
STAGE_COLORS = ["#3fb6b2", "#6c8cff", "#e0879b", "#caa45a", "#7bcf7b", "#b99be0", "#5ec8e0", "#e09a6c", "#9aa7b8"]


# --------------------------------------------------------------------------- #
# Pure data-shaping layer — polars only. No gradio / plotly / pandas here.
# --------------------------------------------------------------------------- #
def load_taxonomy(csv_path: str | Path) -> pl.DataFrame:
    """Read the taxonomy CSV as all-string columns with blanks normalized to "".

    Every column is read as Utf8; nulls and surrounding whitespace are stripped so
    a blank cell is always the empty string "" (never None). This makes downstream
    blank-detection and the string-valued ``plankton``/``living`` columns ("True"/
    "False") robust to stray whitespace.
    """
    df = pl.read_csv(
        Path(csv_path),
        infer_schema_length=0,  # force every column to Utf8
        null_values=[],
    )
    return df.with_columns(pl.all().cast(pl.Utf8).fill_null("").str.strip_chars())


def _matches(column_expr: pl.Expr, value: str) -> pl.Expr:
    """Case-insensitive, whitespace-tolerant equality against a literal value."""
    return column_expr.str.strip_chars().str.to_lowercase() == value.strip().lower()


def apply_filters(
    df: pl.DataFrame,
    *,
    plankton: str = "All",
    kingdom: str = "All",
    phylum: str = "All",
) -> pl.DataFrame:
    """Filter rows by the three global filters.

    - ``plankton``: "All" is a no-op; "plankton"/"True" keeps rows whose plankton
      column equals the string "True"; "non-plankton"/"False" keeps "False".
      Comparison is case-insensitive and whitespace-tolerant.
    - ``kingdom`` / ``phylum``: "All" is a no-op; otherwise keep rows whose column
      equals the given value (case-insensitive on the lowercase taxonomy data).
    """
    out = df
    plankton_norm = plankton.strip().lower()
    if plankton_norm in {"plankton", "true"}:
        out = out.filter(_matches(pl.col("plankton"), "True"))
    elif plankton_norm in {"non-plankton", "false"}:
        out = out.filter(_matches(pl.col("plankton"), "False"))
    # "all" (or anything else) -> no-op.

    if kingdom.strip().lower() != "all":
        out = out.filter(_matches(pl.col("Kingdom"), kingdom))
    if phylum.strip().lower() != "all":
        out = out.filter(_matches(pl.col("Phylum"), phylum))
    return out


def distinct_values(df: pl.DataFrame, column: str) -> list[str]:
    """Return sorted, non-empty distinct values of a column (for dropdowns)."""
    if column not in df.columns:
        return []
    vals = df.get_column(column).str.strip_chars().unique().to_list()
    return sorted(v for v in vals if v)


def build_sankey_index(
    df: pl.DataFrame,
    stages: list[str],
    *,
    drop_blank: bool = False,
    min_flow: int = 1,
) -> dict:
    """Build a well-formed Sankey node/link index over ordered ``stages``.

    Mirrors generate_sankey.py's JS ``build()``:

    - A node identity is ``(stage_index, value)``; a blank cell renders as
      ``"(blank)"``.
    - If ``drop_blank`` is true, any row that is blank in ANY selected stage is
      skipped entirely.
    - Consecutive-stage transitions are aggregated into link counts.
    - Links whose count is below ``min_flow`` are dropped.

    Returns ``{"nodes": [{"stage": int, "label": str}, ...], "source": [int...],
    "target": [int...], "value": [int...], "used_rows": int}``. Every link
    source/target is a valid index into ``nodes`` and there are no orphan nodes.

    Raises ValueError if fewer than two stages are given.
    """
    if len(stages) < 2:
        raise ValueError("build_sankey_index requires at least 2 stages")
    missing = [s for s in stages if s not in df.columns]
    if missing:
        raise ValueError(f"stage column(s) not in dataframe: {missing}")

    node_index: dict[tuple[int, str], int] = {}
    nodes: list[dict] = []

    def nid(stage_i: int, value: str) -> int:
        key = (stage_i, value)
        idx = node_index.get(key)
        if idx is None:
            idx = len(nodes)
            node_index[key] = idx
            nodes.append({"stage": stage_i, "label": value})
        return idx

    link_counts: dict[tuple[int, int], int] = {}
    used_rows = 0
    columns = [df.get_column(s).to_list() for s in stages]
    n_rows = df.height
    for row_i in range(n_rows):
        vals = [columns[s_i][row_i] or BLANK for s_i in range(len(stages))]
        if drop_blank and BLANK in vals:
            continue
        used_rows += 1
        for i in range(len(stages) - 1):
            s = nid(i, vals[i])
            t = nid(i + 1, vals[i + 1])
            link_counts[(s, t)] = link_counts.get((s, t), 0) + 1

    source: list[int] = []
    target: list[int] = []
    value: list[int] = []
    for (s, t), count in link_counts.items():
        if count < min_flow:
            continue
        source.append(s)
        target.append(t)
        value.append(count)

    return {
        "nodes": nodes,
        "source": source,
        "target": target,
        "value": value,
        "used_rows": used_rows,
    }


def build_hierarchy_table(df: pl.DataFrame, *, ranks: tuple[str, ...] = TAXONOMY_RANKS) -> dict:
    """Build a ragged-depth-safe path+value table for a sunburst / icicle.

    For each row, walk ``ranks`` left to right, stopping at the first blank rank
    (ragged tree). Each visited prefix becomes a node whose id is the path joined
    with "/" (e.g. ``"animalia/cnidaria/hydrozoa"``). A node's value is the count
    of rows whose path is exactly that prefix (i.e. how many rows pass through it).
    Parents wire each node to its immediate prefix; root-level nodes have parent "".

    Returns ``{"ids": [...], "labels": [...], "parents": [...], "values": [...]}``.
    No orphans: every non-root parent id is guaranteed to appear in ``ids``.
    """
    present = [r for r in ranks if r in df.columns]
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    parents: dict[str, str] = {}

    columns = [df.get_column(r).to_list() for r in present]
    n_rows = df.height
    for row_i in range(n_rows):
        prefix_parts: list[str] = []
        parent_id = ""
        for col_i in range(len(present)):
            value = (columns[col_i][row_i] or "").strip()
            if not value:
                break  # ragged: stop at first blank rank
            prefix_parts.append(value)
            node_id = "/".join(prefix_parts)
            if node_id not in counts:
                counts[node_id] = 0
                labels[node_id] = value
                parents[node_id] = parent_id
            counts[node_id] += 1
            parent_id = node_id

    ids = list(counts.keys())
    return {
        "ids": ids,
        "labels": [labels[i] for i in ids],
        "parents": [parents[i] for i in ids],
        "values": [counts[i] for i in ids],
    }


def table_columns() -> list[str]:
    """Return the ordered column list for the audit table."""
    return ["Dataset", "Raw_Labels", *TAXONOMY_RANKS, "proposed_label", "plankton", *ID_STR_COLS, *ID_NUM_COLS]


def build_table_rows(df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
    """Return ``(columns, rows)`` for the audit dataframe over curated columns."""
    columns = table_columns()
    present = [c for c in columns if c in df.columns]
    sub = df.select(present)
    rows = [[str(cell) for cell in record] for record in sub.iter_rows()]
    return present, rows


def search_table(rows: list[list[str]], query: str) -> list[list[str]]:
    """Case-insensitive substring filter across all cells; empty query keeps all."""
    needle = (query or "").strip().lower()
    if not needle:
        return rows
    return [row for row in rows if any(needle in str(cell).lower() for cell in row)]


# --------------------------------------------------------------------------- #
# Lazy figure builders — plotly imported INSIDE the function body.
# --------------------------------------------------------------------------- #
def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(...)`` string (mirrors generate_sankey hexA)."""
    n = int(hex_color.lstrip("#"), 16)
    return f"rgba({(n >> 16) & 255},{(n >> 8) & 255},{n & 255},{alpha})"


def make_sankey_figure(index: dict, stages: list[str]):
    """Build a plotly ``go.Sankey`` figure from a sankey index.

    Nodes are coloured per stage (blank nodes muted); links inherit a translucent
    tint of their source node colour, echoing generate_sankey.py's draw().
    """
    import plotly.graph_objects as go

    nodes = index["nodes"]
    node_labels = [n["label"] for n in nodes]
    node_colors = ["#52607a" if n["label"] == BLANK else STAGE_COLORS[n["stage"] % len(STAGE_COLORS)] for n in nodes]
    link_colors = [_hex_rgba(node_colors[s] if node_colors[s].startswith("#") else "#52607a", 0.35) for s in index["source"]]

    n_stage = len(stages)
    xs = [min(0.96, max(0.04, n["stage"] / (n_stage - 1))) if n_stage > 1 else 0.5 for n in nodes]

    sankey = go.Sankey(
        arrangement="snap",
        node={
            "pad": 12,
            "thickness": 16,
            "line": {"color": "#0f1419", "width": 0.5},
            "label": node_labels,
            "color": node_colors,
            "x": xs,
            "hovertemplate": "%{label}<br>%{value} rows<extra></extra>",
        },
        link={
            "source": index["source"],
            "target": index["target"],
            "value": index["value"],
            "color": link_colors,
            "hovertemplate": "%{source.label} → %{target.label}<br>%{value} rows<extra></extra>",
        },
    )
    annotations = [
        {
            "x": i / (n_stage - 1) if n_stage > 1 else 0.5,
            "y": 1.06,
            "xref": "paper",
            "yref": "paper",
            "text": f"<b>{PREFERRED_LABELS.get(k, k)}</b>",
            "showarrow": False,
            "font": {"color": "#8b98a9", "size": 11},
            "xanchor": "left" if i == 0 else ("right" if i == n_stage - 1 else "center"),
        }
        for i, k in enumerate(stages)
    ]
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3", "size": 11},
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        annotations=annotations,
    )
    return fig


def make_hierarchy_figure(table: dict, *, icicle: bool = False):
    """Build a plotly ``go.Sunburst`` (or ``go.Icicle``) from a hierarchy table."""
    import plotly.graph_objects as go

    common = {
        "ids": table["ids"],
        "labels": table["labels"],
        "parents": table["parents"],
        "values": table["values"],
        "branchvalues": "total",
        "hovertemplate": "%{label}<br>%{value} rows<extra></extra>",
    }
    trace = go.Icicle(**common) if icicle else go.Sunburst(**common)
    fig = go.Figure(data=[trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3", "size": 11},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    return fig


# --------------------------------------------------------------------------- #
# Lazy UI builder — gradio imported INSIDE the function body.
# --------------------------------------------------------------------------- #
def build_app(df: pl.DataFrame):
    """Build the Gradio Blocks app with three tabs and shared global filters."""
    import gradio as gr

    all_columns = list(df.columns)
    kingdom_choices = ["All", *distinct_values(df, "Kingdom")]
    phylum_choices = ["All", *distinct_values(df, "Phylum")]
    stage_default = [s for s in DEFAULT_STAGES if s in all_columns]

    def _filtered(plankton: str, kingdom: str, phylum: str) -> pl.DataFrame:
        return apply_filters(df, plankton=plankton, kingdom=kingdom, phylum=phylum)

    def _update_sankey(plankton, kingdom, phylum, stages, drop_blank, min_flow):
        sub = _filtered(plankton, kingdom, phylum)
        if not stages or len(stages) < 2:
            return None
        index = build_sankey_index(sub, list(stages), drop_blank=drop_blank, min_flow=int(min_flow))
        return make_sankey_figure(index, list(stages))

    def _update_hierarchy(plankton, kingdom, phylum, icicle):
        sub = _filtered(plankton, kingdom, phylum)
        table = build_hierarchy_table(sub)
        return make_hierarchy_figure(table, icicle=icicle)

    def _update_table(plankton, kingdom, phylum, query):
        sub = _filtered(plankton, kingdom, phylum)
        _, rows = build_table_rows(sub)
        return search_table(rows, query)

    with gr.Blocks(title="planktonzilla taxonomy explorer") as app:
        gr.Markdown("# planktonzilla taxonomy explorer\nFlow width / node value = number of label rows.")
        with gr.Row():
            plankton_in = gr.Dropdown(choices=["All", "plankton only", "non-plankton"], value="All", label="Plankton filter")
            kingdom_in = gr.Dropdown(choices=kingdom_choices, value="All", label="Kingdom focus")
            phylum_in = gr.Dropdown(choices=phylum_choices, value="All", label="Phylum focus")

        def _plankton_arg(label: str) -> str:
            return {"plankton only": "plankton", "non-plankton": "non-plankton"}.get(label, "All")

        with gr.Tab("Sankey"):
            stage_in = gr.Dropdown(
                choices=all_columns, value=stage_default, multiselect=True, label="Flow stages (selection order = stage order)"
            )
            drop_blank_in = gr.Checkbox(value=False, label="Drop rows blank in any selected stage")
            min_flow_in = gr.Slider(minimum=1, maximum=40, step=1, value=1, label="Minimum flow size")
            sankey_plot = gr.Plot(label="Sankey")

            def _sankey_cb(plankton, kingdom, phylum, stages, drop_blank, min_flow):
                return _update_sankey(_plankton_arg(plankton), kingdom, phylum, stages, drop_blank, min_flow)

            sankey_inputs = [plankton_in, kingdom_in, phylum_in, stage_in, drop_blank_in, min_flow_in]
            for control in sankey_inputs:
                control.change(_sankey_cb, inputs=sankey_inputs, outputs=sankey_plot)
            app.load(_sankey_cb, inputs=sankey_inputs, outputs=sankey_plot)

        with gr.Tab("Hierarchy"):
            icicle_in = gr.Checkbox(value=False, label="Icicle layout (off = sunburst)")
            hierarchy_plot = gr.Plot(label="Kingdom → Species hierarchy")

            def _hierarchy_cb(plankton, kingdom, phylum, icicle):
                return _update_hierarchy(_plankton_arg(plankton), kingdom, phylum, icicle)

            hierarchy_inputs = [plankton_in, kingdom_in, phylum_in, icicle_in]
            for control in hierarchy_inputs:
                control.change(_hierarchy_cb, inputs=hierarchy_inputs, outputs=hierarchy_plot)
            app.load(_hierarchy_cb, inputs=hierarchy_inputs, outputs=hierarchy_plot)

        with gr.Tab("Audit"):
            search_in = gr.Textbox(label="Search (substring across all cells)", placeholder="e.g. cnidaria")
            audit_df = gr.Dataframe(headers=table_columns(), label="Audit table", wrap=True)

            def _table_cb(plankton, kingdom, phylum, query):
                return _update_table(_plankton_arg(plankton), kingdom, phylum, query)

            table_inputs = [plankton_in, kingdom_in, phylum_in, search_in]
            for control in table_inputs:
                control.change(_table_cb, inputs=table_inputs, outputs=audit_df)
            app.load(_table_cb, inputs=table_inputs, outputs=audit_df)

    return app


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def default_csv() -> Path:
    """Locate the bundled taxonomy CSV relative to this script (repo/tools/).

    Mirrors tools/generate_sankey.py's resolution (do NOT import that module).
    """
    repo = Path(__file__).resolve().parent.parent
    return repo / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"


def _run_smoke(csv_path: Path) -> int:
    """Load the CSV, run every pure transform AND build every figure, print counts."""
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    df = load_taxonomy(csv_path)
    filtered = apply_filters(df, plankton="All", kingdom="All", phylum="All")
    stages = [s for s in DEFAULT_STAGES if s in df.columns]
    index = build_sankey_index(filtered, stages)
    table = build_hierarchy_table(filtered)
    columns, rows = build_table_rows(filtered)

    # Build every figure too — proves the lazy plotly path resolves end to end.
    make_sankey_figure(index, stages)
    make_hierarchy_figure(table, icicle=False)
    make_hierarchy_figure(table, icicle=True)

    print(f"csv               : {csv_path}")
    print(f"rows              : {df.height}")
    print(f"filtered rows     : {filtered.height}")
    print(f"sankey stages     : {stages}")
    print(f"sankey nodes      : {len(index['nodes'])}")
    print(f"sankey links      : {len(index['source'])}")
    print(f"sankey used_rows  : {index['used_rows']}")
    print(f"hierarchy nodes   : {len(table['ids'])}")
    print(f"audit columns     : {len(columns)}")
    print(f"audit rows        : {len(rows)}")
    print("smoke OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="taxonomy_explorer.py",
        description="Interactive Gradio explorer for the planktonzilla taxonomy crosswalk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=default_csv(), help="input CSV (default: bundled taxonomy CSV)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run all transforms + build all figures, print counts, exit (no server)",
    )
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = parser.parse_args(argv)

    if args.smoke:
        return _run_smoke(args.csv)

    if not args.csv.exists():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    df = load_taxonomy(args.csv)
    app = build_app(df)
    app.launch(share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
