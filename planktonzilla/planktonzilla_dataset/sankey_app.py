"""
(c) Inria

pz_sankey_app — an interactive, server-side Plotly ``go.Sankey`` Gradio 6 explorer for the
planktonzilla taxonomy.

This is a NEW, parallel deliverable that reuses the converging data model of ``build_sankey.py``
(the frozen self-contained HTML emitter) but renders it live: columns flow
``source_dataset -> root_class -> Kingdom..Species / (unclassified) -> proposed_label`` as a
drill-down Sankey with a JS double-click zoom bridge and the full Inria visual identity.

Everything downstream of ``row_path`` is expressed as RIBBONS (a full column path plus a weight):
focus re-roots them, threshold pooling TRUNCATES them at sub-threshold edges, and a single
``_accumulate`` pass then derives node values and edge values from whatever ribbons survived. Flow
therefore conserves at every node by construction, including where the DAG converges (one child
reached from several parents) — the case that defeats popping edges out of a finished graph.

Layering (dependency-isolation guard, Phase 9): the pure data core — ``row_path`` / ``build_graph``
/ ``breadcrumb`` / ``_read_rows`` — imports NOTHING from ``gradio`` / ``plotly`` and therefore runs
in the frozen core env without the ``explorer`` group installed. The figure (``make_figure``) and
app (``build_app`` / ``main``) layers keep their ``import plotly`` / ``import gradio`` FUNCTION-LOCAL
so no module-scope viz import trips ``tests/test_dependency_isolation.py``.

The command is available as ``pz_sankey_app`` and via
``python -m planktonzilla.planktonzilla_dataset.sankey_app``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from pathlib import Path

from planktonzilla.planktonzilla_dataset.build_sankey import (
    DEFAULT_LOGO_URL,
    fetch_fonts,
    fetch_logo,
    load_sample_counts,
    scan_dataset,
)
from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
    DEFAULT_TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
)

# --------------------------------------------------------------- column vocabulary
# Fixed ordered column-key vocabulary; drives arrangement="fixed" x-positions and the
# CheckboxGroup collapse/relink. "group" is the (unclassified) living-fallback bucket.
SOURCE_COL = "source_dataset"
ROOT_COL = "root_class"
GROUP_COL = "group"
LEAF_COL = "proposed_label"
ALL_COLUMNS: tuple[str, ...] = (SOURCE_COL, ROOT_COL, *TAXONOMY_RANKS, GROUP_COL, LEAF_COL)
_COLUMN_INDEX: dict[str, int] = {col: i for i, col in enumerate(ALL_COLUMNS)}

# Sentinel first-token for pooled "+N other" node keys, so they never collide with a real
# (column_key, label) node identity (no real column is "\x00other").
_OTHER = "\x00other"

# A node identity in the converging DAG is a (column_key, label) tuple.
ColumnKey = tuple[str, str]


@dataclass
class Node:
    """One Sankey node: its column, display label, flow value, and the pooled-"other" flag."""

    col: str
    label: str
    value: float
    is_other: bool = False


@dataclass
class Link:
    """A ribbon between two nodes, referenced by their index in ``Graph.nodes``."""

    src: int
    tgt: int
    value: float


@dataclass
class Graph:
    """A converging Sankey DAG: nodes plus index-referenced links."""

    nodes: list[Node]
    links: list[Link]


def _read_rows(csv_path: Path | str = DEFAULT_TAXONOMY_CSV_FILENAME) -> list[dict]:
    """Read the taxonomy-mapping CSV into a list of row dicts (viz-free, local file only)."""
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _norm_columns(columns_enabled: object) -> set[str]:
    """Normalize the enabled-columns argument to a membership set (None -> every column)."""
    if columns_enabled is None:
        return set(ALL_COLUMNS)
    return set(columns_enabled)  # type: ignore[arg-type]


def _count_key(row: dict) -> tuple[str, str, str]:
    """The (dataset, proposed_label_lower, root_class) key that ``load_sample_counts`` uses."""
    return row["Dataset"].strip(), row["proposed_label"].strip().lower(), row["root_class"].strip()


def row_path(row: dict, columns_enabled: object = None) -> list[ColumnKey]:
    """Ordered (column_key, label) path for one CSV row through the converging column model.

    * living row with >=1 filled rank: source_dataset -> root_class(living) -> filled ranks (in order).
    * living row with ALL ranks blank: FALLBACK mirroring ``build_sankey._living_path`` exactly —
      source_dataset -> root_class(living) -> group("(unclassified)") -> proposed_label. Without this
      the ~31 all-blank living rows would dead-end on root_class:living, breaking conservation.
    * non-living row: source_dataset -> root_class -> proposed_label.

    A column key absent from ``columns_enabled`` is omitted (collapse), so the surviving neighbours
    become adjacent (relink). ``columns_enabled=None`` keeps every column.
    """
    enabled = _norm_columns(columns_enabled)
    src = row["Dataset"].strip()
    rc = row["root_class"].strip()
    path: list[ColumnKey] = [(SOURCE_COL, src), (ROOT_COL, rc)]
    if rc == "living":
        ranks = [(rk, row[rk].strip()) for rk in TAXONOMY_RANKS if row[rk].strip()]
        if ranks:
            path.extend(ranks)
        else:
            label = row["proposed_label"].strip() or "(unlabeled)"
            path.append((GROUP_COL, "(unclassified)"))
            path.append((LEAF_COL, label))
    else:
        label = row["proposed_label"].strip() or "(unlabeled)"
        path.append((LEAF_COL, label))
    return [key for key in path if key[0] in enabled]


def _ribbons_images(paths: list[tuple[dict, list[ColumnKey]]], counts: Counter) -> list[tuple[list[ColumnKey], float]]:
    """Images metric: link value = additive per-(dataset, label, root_class) image count.

    Each count key is attributed to exactly ONE ribbon (the first row that owns it, mirroring
    ``build_sankey``'s ``setdefault``), so a label shared by several CSV rows is not double-counted.
    Zero-count ribbons are dropped (the Sankey shows only flows that carry images).
    """
    ribbons: list[tuple[list[ColumnKey], float]] = []
    seen: set[tuple[str, str, str]] = set()
    for row, path in paths:
        key = _count_key(row)
        if key in seen:
            continue
        seen.add(key)
        weight = float(counts.get(key, 0.0))
        if weight > 0.0:
            ribbons.append((path, weight))
    return ribbons


def _ribbons_taxa(paths: list[tuple[dict, list[ColumnKey]]]) -> list[tuple[list[ColumnKey], float]]:
    """Distinct-taxa metric via FRACTIONAL 1/N ATTRIBUTION.

    ``go.Sankey`` nodes have no size attribute, so a "distinct taxa" count must be encoded in the
    link values. Each distinct leaf taxon carries a mass of 1 split evenly across the N source
    datasets it appears in: every link of a given source ribbon carries ``1/N``. Flow therefore
    conserves at every node and the shared distinct leaf node value equals exactly ``1.0``.
    """
    leaf_datasets: dict[ColumnKey, set[str]] = defaultdict(set)
    representative: dict[tuple[ColumnKey, str], list[ColumnKey]] = {}
    for row, path in paths:
        leaf = path[-1]
        dataset = row["Dataset"].strip()
        leaf_datasets[leaf].add(dataset)
        representative.setdefault((leaf, dataset), path)
    ribbons: list[tuple[list[ColumnKey], float]] = []
    for (leaf, _dataset), path in representative.items():
        ribbons.append((path, 1.0 / len(leaf_datasets[leaf])))
    return ribbons


def _focus_ribbons(ribbons: list[tuple[list[ColumnKey], float]], focus_key: ColumnKey) -> list[tuple[list[ColumnKey], float]]:
    """Restrict to ribbons passing through ``focus_key`` and re-root each at it (its descendants)."""
    focused: list[tuple[list[ColumnKey], float]] = []
    for path, weight in ribbons:
        if focus_key in path:
            focused.append((path[path.index(focus_key) :], weight))
    return focused


def _accumulate(
    ribbons: list[tuple[list[ColumnKey], float]],
) -> tuple[dict[ColumnKey, float], dict[tuple[ColumnKey, ColumnKey], float], list[ColumnKey], list[ColumnKey]]:
    """Sum ribbon weights into node values (flow-through) and edge values; return roots too."""
    node_value: dict[ColumnKey, float] = defaultdict(float)
    edge_value: dict[tuple[ColumnKey, ColumnKey], float] = defaultdict(float)
    order: list[ColumnKey] = []
    seen: set[ColumnKey] = set()
    targets: set[ColumnKey] = set()
    for path, weight in ribbons:
        for key in path:
            if key not in seen:
                seen.add(key)
                order.append(key)
            node_value[key] += weight
        for src, tgt in pairwise(path):
            edge_value[(src, tgt)] += weight
            targets.add(tgt)
    roots = [key for key in order if key not in targets]
    return node_value, edge_value, order, roots


def _pool_ribbons(
    ribbons: list[tuple[list[ColumnKey], float]],
    threshold: float,
) -> tuple[list[tuple[list[ColumnKey], float]], dict[ColumnKey, ColumnKey]]:
    """Pool small siblings by TRUNCATING RIBBONS, so conservation holds by construction.

    Popping sub-threshold EDGES out of an already-accumulated graph is unsound on a converging DAG:
    a child reachable from two parents keeps the node value contributed by BOTH while only the
    surviving parent still feeds it, so ``incoming < value``. Instead this works on the ribbon set
    itself — a ribbon crossing a sub-threshold edge is cut at that edge and re-pointed at a
    synthetic "+N other" key, dropping its tail. ``_accumulate`` then re-derives values AND edges
    from the surviving ribbons, so every node's value is exactly what flows into it.

    Pooling is per-parent AND per-CHILD-COLUMN: a parent whose small children span two columns gets
    one "+N other" in EACH of those columns (never one bucket parked in the left-most). Levels are
    processed by ascending column index so a truncation at one depth feeds the flows of the next; a
    ``row_path`` visits each column at most once, so a ribbon contributes at most one edge per level.

    Returns the rewritten ribbons plus ``other_meta``: synthetic key -> ``(display_column, label)``,
    exactly the shape ``_materialize`` consumes. Synthetic keys carry the ``_OTHER`` sentinel in
    slot 0, so they never match a real column, are never re-pooled, and terminate the ribbon.
    """
    other_meta: dict[ColumnKey, ColumnKey] = {}
    current = ribbons
    for col in ALL_COLUMNS:
        flow: dict[tuple[ColumnKey, ColumnKey], float] = defaultdict(float)
        for path, weight in current:
            for src, tgt in pairwise(path):
                if tgt[0] == col:
                    flow[(src, tgt)] += weight
        small = {edge for edge, value in flow.items() if value < threshold}  # STRICT: at-threshold is kept
        if not small:
            continue

        pooled_children: dict[ColumnKey, list[ColumnKey]] = defaultdict(list)
        for parent, child in small:
            pooled_children[parent].append(child)
        other_key_for: dict[ColumnKey, ColumnKey] = {}
        for parent, kids in pooled_children.items():
            # "\x00" joiner: no real column key or label can contain a NUL, so no collision.
            key: ColumnKey = (_OTHER, f"{parent[0]}\x00{parent[1]}\x00{col}")
            other_key_for[parent] = key
            other_meta[key] = (col, f"+{len(kids)} other")

        rebuilt: list[tuple[list[ColumnKey], float]] = []
        for path, weight in current:
            truncated = path
            for i, (src, tgt) in enumerate(pairwise(path)):
                if (src, tgt) in small:
                    truncated = [*path[: i + 1], other_key_for[src]]
                    break
            rebuilt.append((truncated, weight))
        current = rebuilt
    return current, other_meta


def _materialize(
    node_value: dict[ColumnKey, float],
    edge_value: dict[tuple[ColumnKey, ColumnKey], float],
    order: list[ColumnKey],
    other_meta: dict[ColumnKey, ColumnKey],
) -> Graph:
    """Turn the value/edge dicts into an index-referenced ``Graph``, columns left-to-right."""

    def col_label(key: ColumnKey) -> ColumnKey:
        return other_meta[key] if key in other_meta else key

    ordered = sorted(
        order,
        key=lambda key: (_COLUMN_INDEX.get(col_label(key)[0], len(ALL_COLUMNS)), -node_value[key], col_label(key)[1]),
    )
    index = {key: i for i, key in enumerate(ordered)}
    nodes = [
        Node(col=col_label(key)[0], label=col_label(key)[1], value=node_value[key], is_other=key in other_meta)
        for key in ordered
    ]
    links = [Link(src=index[src], tgt=index[tgt], value=value) for (src, tgt), value in edge_value.items()]
    return Graph(nodes=nodes, links=links)


def build_graph(
    rows: list[dict],
    counts: Counter | None = None,
    *,
    columns_enabled: object = None,
    size_metric: str = "images",
    min_threshold: float = 0.0,
    focus_key: ColumnKey | None = None,
) -> Graph:
    """Aggregate CSV rows into a converging Sankey ``Graph``.

    ``size_metric="images"`` sizes links by additive per-class image counts from ``counts``;
    ``size_metric="taxa"`` sizes them by fractional 1/N distinct-taxa attribution (see
    ``_ribbons_taxa``). ``columns_enabled`` collapses columns, ``min_threshold`` TRUNCATES ribbons
    at sub-threshold edges into a gray "+N other" node (see ``_pool_ribbons``), and ``focus_key``
    restricts to a node and its descendants.

    Focus, then pooling, then ONE ``_accumulate`` pass: node values and edge values are always
    derived from the SAME surviving ribbon set, so flow conserves at every node under every
    combination of the four knobs.
    """
    counts = counts or Counter()
    paths = [(row, path) for row in rows if (path := row_path(row, columns_enabled))]

    ribbons = _ribbons_images(paths, counts) if size_metric == "images" else _ribbons_taxa(paths)
    if focus_key is not None:
        ribbons = _focus_ribbons(ribbons, focus_key)

    other_meta: dict[ColumnKey, ColumnKey] = {}
    if min_threshold and min_threshold > 0:
        ribbons, other_meta = _pool_ribbons(ribbons, float(min_threshold))

    node_value, edge_value, order, _roots = _accumulate(ribbons)
    return _materialize(node_value, edge_value, order, other_meta)


def breadcrumb(
    graph_or_rows: Graph | list[dict],
    focus_key: ColumnKey | None,
    columns_enabled: object = None,
) -> list[ColumnKey]:
    """Ancestor chain from the root column down to ``focus_key`` (inclusive) for the Back affordance.

    Accepts either a built ``Graph`` (walks a real left-to-right path up to a root) or the raw rows
    (finds the first row whose path passes through ``focus_key``). Returns ``[]`` when ``focus_key``
    is None.

    ``columns_enabled`` MUST mirror the CheckboxGroup used to build the view: the rows branch feeds
    it to ``row_path`` so Zoom-out can never target an ancestor that the current view collapsed
    away (which would re-root on a key no ribbon carries and blank the chart).
    """
    if focus_key is None:
        return []
    if isinstance(graph_or_rows, Graph):
        # ALL in-edge parents per node (a converging DAG has many), not just the first seen.
        parents: dict[ColumnKey, list[ColumnKey]] = defaultdict(list)
        for link in graph_or_rows.links:
            src = graph_or_rows.nodes[link.src]
            tgt = graph_or_rows.nodes[link.tgt]
            parents[(tgt.col, tgt.label)].append((src.col, src.label))
        chain = [focus_key]
        cur = focus_key
        seen = {cur}
        while True:
            # A breadcrumb is a LEFT-TO-RIGHT path: only accept a parent strictly left of `cur`,
            # take the left-most such candidate, label as the deterministic tiebreak.
            cur_index = _COLUMN_INDEX.get(cur[0], len(ALL_COLUMNS))
            candidates = [
                key
                for key in parents.get(cur, [])
                if key not in seen and _COLUMN_INDEX.get(key[0], len(ALL_COLUMNS)) < cur_index
            ]
            if not candidates:
                break
            cur = min(candidates, key=lambda key: (_COLUMN_INDEX.get(key[0], len(ALL_COLUMNS)), key[1]))
            seen.add(cur)  # cycle guard (indices already decrease strictly, but keep it explicit)
            chain.append(cur)
        chain.reverse()
        return chain
    for row in graph_or_rows:
        path = row_path(row, columns_enabled)
        if focus_key in path:
            return path[: path.index(focus_key) + 1]
    return [focus_key]


# ===================================================================== figure layer
# Inria data ramp (charter §8): fixed order Bleu mat -> Bleu canard -> Violet -> Framboise, then
# their 70% / 50% tints. Rouge #C9191E is RESERVED (a signal, never a data node); "+N other" pooled
# nodes use the reserved gray. plotly needs concrete hexes, so the tints are precomputed here.
ROUGE_RESERVED = "#c9191e"  # NEVER assigned to a data node
DATA_OTHER_COLOR = "#aab3bf"
INRIA_DATA_RAMP_BASE: tuple[str, ...] = ("#27348b", "#1067a3", "#534b9a", "#a60f79")

_COLUMN_TITLES = {
    "source_dataset": "Source dataset",
    "root_class": "Root class",
    "group": "Group",
    "proposed_label": "Label",
}


def _tint(hex_color: str, weight: float) -> str:
    """Mix ``hex_color`` toward white by ``weight`` (1.0 = full brand, 0.5 = 50% tint)."""
    mixed = tuple(round(int(hex_color[i : i + 2], 16) * weight + 255 * (1 - weight)) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _shade(hex_color: str, weight: float) -> str:
    """Mix ``hex_color`` toward BLACK by ``weight`` (1.0 = full brand, 0.3 = near-black shade)."""
    mixed = tuple(round(int(hex_color[i : i + 2], 16) * weight) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _color_ramp(hex_color: str) -> tuple[str, ...]:
    """The 11 Tailwind-shaped stops (c50..c950) for a brand hue, centred on the hue itself at c500.

    ``gr.themes.Color`` takes those eleven stops POSITIONALLY, so a Gradio theme can be built from
    the immutable Inria Palette 2024 hexes instead of a Tailwind hue name. Lighter stops are tints
    (toward white), darker stops are shades (toward black) — the hue is never re-hued.
    """
    return (
        *(_tint(hex_color, w) for w in (0.08, 0.16, 0.30, 0.50, 0.75)),
        hex_color,
        *(_shade(hex_color, w) for w in (0.85, 0.70, 0.55, 0.42, 0.30)),
    )


def _rgba(hex_color: str, alpha: float) -> str:
    """Return ``hex_color`` as an ``rgba(...)`` string with the given alpha (for translucent links)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


INRIA_DATA_RAMP: tuple[str, ...] = (
    *INRIA_DATA_RAMP_BASE,
    *(_tint(c, 0.7) for c in INRIA_DATA_RAMP_BASE),
    *(_tint(c, 0.5) for c in INRIA_DATA_RAMP_BASE),
)


def _column_title(col: str) -> str:
    """Human-readable header for a column key (ranks keep their own name)."""
    return _COLUMN_TITLES.get(col, col)


def _stack_y(graph: Graph, present: list[str]) -> dict[int, float]:
    """Per-column y in (0,1): stack each column's nodes by descending value, centers value-weighted."""
    band_lo, band_hi = 0.03, 0.97
    y_pos: dict[int, float] = {}
    for col in present:
        col_nodes = sorted(
            (i for i, n in enumerate(graph.nodes) if n.col == col),
            key=lambda i: (-graph.nodes[i].value, graph.nodes[i].label),
        )
        total = sum(graph.nodes[i].value for i in col_nodes) or 1.0
        cursor = 0.0
        for i in col_nodes:
            frac = graph.nodes[i].value / total
            y_pos[i] = band_lo + (band_hi - band_lo) * (cursor + frac / 2)
            cursor += frac
    return y_pos


def make_figure(graph: Graph, *, theme: str = "light", size_metric: str = "images"):
    """Build a fixed-column ``go.Sankey`` figure for ``graph`` (plotly imported FUNCTION-LOCAL).

    Columns are pinned with ``arrangement="fixed"``: each node's column maps to an x evenly spaced
    in 0.02..0.98 (in the fixed column order, over the columns present), with a value-stacked y kept
    off the 0/1 edges (RESEARCH section 3). Node colors follow the Inria data ramp; ``is_other``
    nodes are gray; Rouge is never used. ``theme`` ("light"/"dark") tracks the page background/ink.
    """
    import plotly.graph_objects as go

    ink = "#171a1d" if theme != "dark" else "#e9edf0"
    paper = "#ffffff" if theme != "dark" else "#121517"
    metric_word = "images" if size_metric == "images" else "distinct taxa"

    present = [c for c in ALL_COLUMNS if any(n.col == c for n in graph.nodes)]
    if len(present) <= 1:
        col_x = {c: 0.5 for c in present}
    else:
        col_x = {c: 0.02 + 0.96 * i / (len(present) - 1) for i, c in enumerate(present)}
    col_color = {c: INRIA_DATA_RAMP[i % len(INRIA_DATA_RAMP)] for i, c in enumerate(present)}

    y_pos = _stack_y(graph, present)
    xs = [col_x[n.col] for n in graph.nodes]
    ys = [y_pos[i] for i in range(len(graph.nodes))]
    colors = [DATA_OTHER_COLOR if n.is_other else col_color[n.col] for n in graph.nodes]
    labels = [n.label for n in graph.nodes]
    columns = [_column_title(n.col) for n in graph.nodes]

    sankey = go.Sankey(
        arrangement="fixed",
        node=dict(
            label=labels,
            x=xs,
            y=ys,
            color=colors,
            customdata=columns,
            pad=14,
            thickness=16,
            line=dict(color=paper, width=0.5),
            hovertemplate="<b>%{label}</b><br>%{customdata}<br>%{value:.3g} " + metric_word + "<extra></extra>",
        ),
        link=dict(
            source=[link.src for link in graph.links],
            target=[link.tgt for link in graph.links],
            value=[link.value for link in graph.links],
            color=[_rgba(colors[link.src], 0.32) for link in graph.links],
            hovertemplate="%{source.label} → %{target.label}<br>%{value:.3g} " + metric_word + "<extra></extra>",
        ),
        # NOTE: sankey node labels render on the PAPER, not on the colored node, so for WCAG AA they
        # track the theme ink (dark on light / near-white on dark) rather than always-white.
        textfont=dict(color=ink, family="Inria Sans, Tahoma, sans-serif", size=12),
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=paper,
        font=dict(family="Inria Sans, Tahoma, sans-serif", color=ink, size=13),
        margin=dict(t=54, l=12, r=12, b=12),
    )
    for col in present:
        fig.add_annotation(
            x=col_x[col],
            y=1.045,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"<b>{_column_title(col)}</b>",
            font=dict(family="Inria Sans, Tahoma, sans-serif", size=12, color=ink),
            xanchor="center",
            yanchor="bottom",
        )
    return fig


# ============================================================== Inria style constants
# Double-click zoom bridge (RESEARCH section 2). gr.Plot has NO server-side click event, so this
# client-side listener attaches to the rendered plotly div, debounces plotly_click to detect a
# double-click on the SAME node index (~350ms), then writes that index into the click sink's
# <input> and dispatches 'input' + 'change' so the svelte binding observes the write and the
# component's .change() fires the Python re-root callback.
#
# SELF-HEALING: every server-side rebuild REPLACES the plotly div, which drops any handler bound to
# the old one. A one-shot attach therefore works exactly until the first control change. Instead a
# MutationObserver on the stable #pz_sankey wrapper re-scans on every subtree mutation and binds
# each fresh .js-plotly-plot once (guarded by a __pzBound marker), with an immediate scan plus a
# bounded poll covering the case where the wrapper itself is not mounted yet.
#
# LINK GUARD: plotly fires the SAME plotly_click binder for sankey LINK paths. Link points expose
# `source`/`target`; node points do not — so a link click must be dropped, or it would zoom to an
# arbitrary node index. Node index is `pointNumber` (`index` only as a fallback).
#
# SINGLE-CLICK FALLBACK: if a browser/Plotly build never delivers the paired click (RESEARCH A1/A2),
# drop the `now - last < 350 && idx === lastIdx` guard so a single click zooms instead.
BRIDGE_JS = r"""
() => {
  const bind = (gd) => {
    if (!gd || !gd.on || gd.__pzBound) return;               // bind each plotly div exactly once
    gd.__pzBound = true;
    let last = 0, lastIdx = -1;
    gd.on('plotly_click', (ev) => {
      const pt = ev && ev.points && ev.points[0];
      if (!pt) return;
      if (pt.source !== undefined || pt.target !== undefined) return;   // a LINK, not a node
      const idx = pt.pointNumber !== undefined ? pt.pointNumber : pt.index;
      if (idx === undefined || idx === null) return;
      const now = Date.now();
      if (now - last < 350 && idx === lastIdx) {              // debounced double-click
        const inp = document.querySelector('#pz_click input, #pz_click textarea');
        if (inp) {
          inp.value = String(idx);
          inp.dispatchEvent(new Event('input', {bubbles: true}));   // svelte binding
          inp.dispatchEvent(new Event('change', {bubbles: true}));  // -> gr.Number.change
        }
      }
      last = now; lastIdx = idx;
    });
  };
  const scan = () => document.querySelectorAll('#pz_sankey .js-plotly-plot').forEach(bind);
  const host = document.querySelector('#pz_sankey') || document.body;
  new MutationObserver(scan).observe(host, {childList: true, subtree: true});
  scan();
  let tries = 0;
  const poll = setInterval(() => { scan(); if (++tries > 40) clearInterval(poll); }, 300);
}
"""

# Token-driven Inria palette + dot-grid motif + blanc-tournant, light AND dark (charter §1/§2/§5).
# Brand hexes are immutable across themes; only neutrals re-tune, and BOTH neutral sets are complete
# (charter §2: a partial dark override is a bug). Three dark signals are emitted: the OS media query,
# Gradio's own `.dark` CLASS (what the app actually toggles — a `:root[data-theme]` selector alone
# never matches under Gradio 6), and `:root[data-theme="dark"]` for an external stamp.
INRIA_CSS = """
:root{
  --rouge:#c9191e; --framboise:#a60f79; --violet:#534b9a; --bleu-mat:#27348b; --bleu-canard:#1067a3;
  --data-other:#aab3bf;
  --page:#ffffff; --panel:#ffffff; --sunken:#f4f6f8;
  --ink:#171a1d; --ink-2:#3f474e; --ink-muted:#5c666f;
  --hair:#e0e5ea; --border-strong:#b9c1ca;
}
@media (prefers-color-scheme:dark){:root{
  --page:#121517; --panel:#1a1e21; --sunken:#0e1113;
  --ink:#e9edf0; --ink-2:#b7c0c8; --ink-muted:#8a939c;
  --hair:#2a3034; --border-strong:#414b52;
}}
.dark, :root[data-theme="dark"]{
  --page:#121517; --panel:#1a1e21; --sunken:#0e1113;
  --ink:#e9edf0; --ink-2:#b7c0c8; --ink-muted:#8a939c;
  --hair:#2a3034; --border-strong:#414b52;
}
:root[data-theme="light"]{
  --page:#ffffff; --panel:#ffffff; --sunken:#f4f6f8;
  --ink:#171a1d; --ink-2:#3f474e; --ink-muted:#5c666f;
  --hair:#e0e5ea; --border-strong:#b9c1ca;
}
/* blanc tournant — the white margin frames the WHOLE app, so the lockup and the content below it
   share one 1200px measure instead of the header being padded twice (charter §5). */
.gradio-container{
  background:var(--page); color:var(--ink);
  max-width:1200px; margin:0 auto; padding:24px;
}
@media (min-width:768px){.gradio-container{padding:48px;}}
@media (min-width:1024px){.gradio-container{padding:64px;}}
.pz-frame{position:relative; margin-bottom:16px;}
/* signature dot-grid motif — a corner tile only, NEVER tiled behind body text */
.inria-motif{
  background-image:radial-gradient(currentColor 1px, transparent 1.5px);
  background-size:16px 16px; color:var(--bleu-mat); opacity:.08;
  width:240px; height:240px; pointer-events:none;
}
@media (prefers-color-scheme:dark){.inria-motif{opacity:.11;}}
.dark .inria-motif{opacity:.11;}
.pz-header{display:flex; align-items:center; gap:16px; flex-wrap:wrap;}
.pz-rf{display:flex; align-items:center; gap:8px; color:var(--ink);}
.pz-rf-label{font-weight:700; font-size:11px; line-height:1.05; letter-spacing:.03em;}
.pz-rf-devise{font-size:9px; font-style:italic; font-weight:400;}
.pz-rule{width:1px; height:38px; background:var(--hair);}
/* NON-PRODUCTION facsimile — shown only if the official Inria SVG fails to load. Serif + Rouge. */
.pz-inria-fallback{font-family:"Inria Serif",Georgia,serif; color:var(--rouge); font-size:28px; font-style:italic;}
"""

# The double-click sink must be RENDERED (Gradio 6 never mounts a visible=False component into the
# DOM, leaving BRIDGE_JS with no <input> to write), so it is hidden GEOMETRICALLY instead.
# `display:none` is deliberately NOT used: a display-none input is skipped by the browser's event
# plumbing, and the programmatic 'input'/'change' dispatch would never reach the svelte binding.
CLICK_SINK_CSS = "#pz_click{position:absolute!important;width:0;height:0;opacity:0;pointer-events:none;overflow:hidden;}"
INRIA_CSS += f"{CLICK_SINK_CSS}\n"

# Sole-emitter (EMITTER_MODE=sole) lockup: the République Française bloc-marque on the LEFT (State
# colors #000091 / #E1000F / #FFFFFF — deliberately OUTSIDE the Inria 5-hue palette), a thin rule,
# then the official Inria red-script wordmark. onerror reveals the flagged Serif-Rouge facsimile.
HEADER_HTML = f"""
<div class="pz-header" role="banner">
  <div class="pz-rf" aria-label="République Française">
    <!-- NON-PRODUCTION facsimile of the controlled DSFR bloc-marque (charter §4): reproduce the
         official asset, never restyle it — swap this hand-drawn tricolour for the real SVG. -->
    <svg width="42" height="38" viewBox="0 0 42 38" role="img" aria-hidden="true">
      <rect x="0" y="0" width="14" height="38" fill="#000091"></rect>
      <rect x="14" y="0" width="14" height="38" fill="#ffffff"></rect>
      <rect x="28" y="0" width="14" height="38" fill="#e1000f"></rect>
    </svg>
    <div>
      <div class="pz-rf-label">RÉPUBLIQUE<br>FRANÇAISE</div>
      <div class="pz-rf-devise">Liberté · Égalité · Fraternité</div>
    </div>
  </div>
  <div class="pz-rule"></div>
  <img class="pz-inria-logo" alt="Inria" height="34" src="{DEFAULT_LOGO_URL}"
       onerror="this.style.display='none'; this.nextElementSibling.style.display='inline';">
  <span class="pz-inria-fallback" style="display:none;">Inria</span>
</div>
"""


# ======================================================================= gradio app
def _metric_key(metric: object) -> str:
    """Map a Size-by Radio choice ("Images"/"Taxa") to the build_graph metric key."""
    return "images" if str(metric).lower().startswith("image") else "taxa"


def _theme_key(theme: object) -> str:
    """Map a Theme Radio choice ("Light"/"Dark") to the make_figure theme key."""
    return "dark" if str(theme).lower().startswith("dark") else "light"


def _click_index(idx: object) -> int | None:
    """Coerce the click sink's value to a node index, or ``None`` when it is not one.

    The sink is a DOM-rendered, client-writable ``<input>``, so an arbitrary payload can be POSTed
    at the zoom callback. A bare ``int(idx)`` on ``None`` / NaN / infinity / a non-numeric string
    raises straight out of the event handler as a 500, so those are rejected here; the caller's
    ``0 <= i < len(nodes)`` bound check then handles out-of-range integers.
    """
    try:
        value = float(idx)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isfinite(value):
        return None
    return int(value)


def build_app(rows: list[dict] | None = None, counts: Counter | None = None, *, header_html: str = HEADER_HTML):
    """Compose the Sankey explorer ``gr.Blocks`` (gradio imported FUNCTION-LOCAL); does NOT launch.

    ``rows=None`` reads the bundled taxonomy CSV (local file, no network) so a no-arg
    ``build_app()`` smoke works offline. When ``counts`` is falsy the Size-by Radio offers "Taxa"
    only (gr.Radio has no per-choice disable, so "Images" is FILTERED OUT, not greyed) and the app
    defaults to the fractional-taxa metric. Controls rebuild the graph server-side and return a
    fresh figure; node double-click zoom is wired via the ``demo.load(js=BRIDGE_JS)`` bridge into a
    RENDERED-but-CSS-hidden ``gr.Number`` sink (see ``CLICK_SINK_CSS``). theme/css/head belong on
    ``.launch()`` (Gradio 6), applied by ``main()``.
    """
    import gradio as gr

    if rows is None:
        rows = _read_rows()
    has_counts = bool(counts)
    default_metric = "images" if has_counts else "taxa"
    all_columns = list(ALL_COLUMNS)

    def _rebuild(columns_enabled, metric, threshold, theme, focus):
        graph = build_graph(
            rows,
            counts,
            columns_enabled=columns_enabled or all_columns,
            size_metric=_metric_key(metric),
            min_threshold=threshold or 0.0,
            focus_key=focus,
        )
        crumb = breadcrumb(rows, focus, columns_enabled or all_columns)
        crumb_text = " / ".join(label for _col, label in crumb) if focus else "**planktonzilla-17M** taxonomy"
        return make_figure(graph, theme=_theme_key(theme), size_metric=_metric_key(metric)), crumb_text

    initial_graph = build_graph(rows, counts, columns_enabled=all_columns, size_metric=default_metric)

    with gr.Blocks(title="planktonzilla taxonomy Sankey", analytics_enabled=False) as demo:
        gr.HTML(
            f'<div class="pz-frame" style="position:relative;">{header_html}'
            '<div class="inria-motif" style="position:absolute; top:8px; right:8px;"></div></div>'
        )
        breadcrumb_md = gr.Markdown("**planktonzilla-17M** taxonomy")
        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(
                    value=make_figure(initial_graph, theme="light", size_metric=default_metric),
                    elem_id="pz_sankey",
                )
            with gr.Column(scale=1):
                back_btn = gr.Button("◄ Zoom out", size="sm")
                metric_radio = gr.Radio(
                    choices=["Images", "Taxa"] if has_counts else ["Taxa"],
                    value="Images" if has_counts else "Taxa",
                    label="Size by",
                )
                columns_group = gr.CheckboxGroup(choices=all_columns, value=all_columns, label="Columns")
                threshold_slider = gr.Slider(minimum=0, maximum=200, step=1, value=0, label="Pool below size")
                # "Light" matches the initial gr.Plot figure above; the radio is the single source
                # of truth for the figure's paper/ink from the first re-render onward.
                theme_radio = gr.Radio(choices=["Light", "Dark"], value="Light", label="Theme")
        focus_state = gr.State(None)
        # Rendered (not visible=False) so BRIDGE_JS has a real <input> to write, then hidden by
        # CLICK_SINK_CSS. Emitted here too, so a build_app() used without main()'s .launch(css=...)
        # still hides it. The -1 default is rejected by the bound check in _zoom_to.
        gr.HTML(f"<style>{CLICK_SINK_CSS}</style>")
        click_sink = gr.Number(value=-1, elem_id="pz_click", visible=True, container=False, show_label=False)

        controls = [columns_group, metric_radio, threshold_slider, theme_radio, focus_state]
        rebuild_outputs = [plot, breadcrumb_md]
        columns_group.change(_rebuild, controls, rebuild_outputs)
        metric_radio.change(_rebuild, controls, rebuild_outputs)
        threshold_slider.release(_rebuild, controls, rebuild_outputs)
        theme_radio.change(_rebuild, controls, rebuild_outputs)

        def _zoom_to(idx, columns_enabled, metric, threshold, theme, focus):
            graph = build_graph(
                rows,
                counts,
                columns_enabled=columns_enabled or all_columns,
                size_metric=_metric_key(metric),
                min_threshold=threshold or 0.0,
                focus_key=focus,
            )
            new_focus = focus
            i = _click_index(idx)
            # A pooled "+N other" node carries a synthetic _OTHER key that matches NO ribbon, so
            # re-rooting on its (col, label) yields an empty graph. Ignore the click instead: the
            # current view is returned unchanged rather than blanked.
            if i is not None and 0 <= i < len(graph.nodes) and not graph.nodes[i].is_other:
                node = graph.nodes[i]
                new_focus = (node.col, node.label)
            figure, crumb_text = _rebuild(columns_enabled, metric, threshold, theme, new_focus)
            return figure, crumb_text, new_focus

        click_sink.change(_zoom_to, [click_sink, *controls], [plot, breadcrumb_md, focus_state])

        def _zoom_out(columns_enabled, metric, threshold, theme, focus):
            new_focus = None
            if focus is not None:
                chain = breadcrumb(rows, focus, columns_enabled or all_columns)
                new_focus = chain[-2] if len(chain) >= 2 else None
            figure, crumb_text = _rebuild(columns_enabled, metric, threshold, theme, new_focus)
            return figure, crumb_text, new_focus

        back_btn.click(_zoom_out, controls, [plot, breadcrumb_md, focus_state])

        demo.load(js=BRIDGE_JS)  # attach the client-side double-click zoom bridge
    return demo


def main(argv: list[str] | None = None) -> None:
    """Build and launch the Sankey explorer for local dev (gradio imported FUNCTION-LOCAL).

    theme/css/head go on ``.launch()`` (Gradio 6), NOT the ``gr.Blocks`` constructor. Sample counts
    come from ``--samples-json`` (a prior scan) or, with ``--scan``, a live HF scan of
    ``--dataset-repo``; with neither, the Size-by Radio offers "Taxa" only.
    """
    import gradio as gr

    parser = argparse.ArgumentParser(prog="pz_sankey_app", description="Interactive planktonzilla taxonomy Sankey.")
    parser.add_argument("--samples-json", type=Path, default=None, help="precomputed per-class counts JSON to load")
    parser.add_argument("--scan", action="store_true", help="scan --dataset-repo live for image counts instead")
    parser.add_argument("--dataset-repo", default=DEFAULT_PLANKTONZILLA_DATASET_REPO_ID, help="HF dataset repo to scan")
    parser.add_argument("--workers", type=int, default=16, help="dataset shards read concurrently when --scan")
    parser.add_argument("--host", default="127.0.0.1", help="server bind host")
    parser.add_argument("--port", type=int, default=7860, help="server bind port")
    parser.add_argument("--share", action="store_true", help="open a public Gradio share tunnel")
    args = parser.parse_args(argv)

    if args.samples_json is not None:
        counts: Counter | None = load_sample_counts(args.samples_json)
    elif args.scan:
        counts = scan_dataset(args.dataset_repo, args.workers)
    else:
        counts = None

    # Charter palette, not Tailwind: "rose"/"fuchsia" are off-brand hues. Rouge #C9191E is the
    # view's ONE signal — primary button / active state only, never a surface flood (§0.1, §6) —
    # so it drives primary, with Bleu mat #27348B as the secondary accent.
    theme = gr.themes.Ocean(
        primary_hue=gr.themes.Color(*_color_ramp(ROUGE_RESERVED), name="inria-rouge"),
        secondary_hue=gr.themes.Color(*_color_ramp("#27348b"), name="inria-bleu-mat"),
        radius_size="md",
        font=[gr.themes.GoogleFont("Inria Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    fonts = fetch_fonts()
    head = f"<style>{fonts}</style>" if fonts else None
    logo_b64 = fetch_logo(DEFAULT_LOGO_URL)
    header = HEADER_HTML.replace(DEFAULT_LOGO_URL, f"data:image/svg+xml;base64,{logo_b64}") if logo_b64 else HEADER_HTML

    build_app(counts=counts, header_html=header).launch(
        theme=theme,
        css=INRIA_CSS,
        head=head,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
