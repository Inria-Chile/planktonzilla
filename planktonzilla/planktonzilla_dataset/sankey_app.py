"""
(c) Inria

pz_sankey_app — an interactive, server-side Plotly ``go.Sankey`` Gradio 6 explorer for the
planktonzilla taxonomy.

This is a NEW, parallel deliverable that reuses the converging data model of ``build_sankey.py``
(the frozen self-contained HTML emitter) but renders it live: columns flow
``source_dataset -> root_class -> Kingdom..Species / (unclassified) -> proposed_label`` as a
drill-down Sankey with a JS double-click zoom bridge and the full Inria visual identity.

Layering (dependency-isolation guard, Phase 9): the pure data core — ``row_path`` / ``build_graph``
/ ``breadcrumb`` / ``_read_rows`` — imports NOTHING from ``gradio`` / ``plotly`` and therefore runs
in the frozen core env without the ``explorer`` group installed. The figure (``make_figure``) and
app (``build_app`` / ``main``) layers keep their ``import plotly`` / ``import gradio`` FUNCTION-LOCAL
so no module-scope viz import trips ``tests/test_dependency_isolation.py``.

The command is available as ``pz_sankey_app`` and via
``python -m planktonzilla.planktonzilla_dataset.sankey_app``.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from planktonzilla.planktonzilla_dataset.build_sankey import DEFAULT_LOGO_URL
from planktonzilla.planktonzilla_dataset.constants import (
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


def _reachable(edges: dict[tuple[ColumnKey, ColumnKey], float], roots: list[ColumnKey]) -> set[ColumnKey]:
    """Nodes reachable from ``roots`` following ``edges`` (used to prune pooled-away subtrees)."""
    adjacency: dict[ColumnKey, list[ColumnKey]] = defaultdict(list)
    for src, tgt in edges:
        adjacency[src].append(tgt)
    seen = set(roots)
    stack = list(roots)
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _pool_below_threshold(
    node_value: dict[ColumnKey, float],
    edge_value: dict[tuple[ColumnKey, ColumnKey], float],
    order: list[ColumnKey],
    roots: list[ColumnKey],
    threshold: float,
) -> tuple[dict, dict, list, dict]:
    """Pool, within each parent, children whose incoming edge value < ``threshold`` into one gray
    "+N other" node (``is_other``), then prune the now-unreachable pooled subtrees. Conserves flow.
    """
    children: dict[ColumnKey, list[ColumnKey]] = defaultdict(list)
    for src, tgt in edge_value:
        children[src].append(tgt)

    new_edges = dict(edge_value)
    new_values = dict(node_value)
    other_meta: dict[ColumnKey, ColumnKey] = {}
    for idx, parent in enumerate(list(children)):
        small = [child for child in children[parent] if new_edges.get((parent, child), 0.0) < threshold]
        if not small:
            continue
        total = 0.0
        for child in small:
            total += new_edges.pop((parent, child))
        col = min((child[0] for child in small), key=lambda c: _COLUMN_INDEX.get(c, len(ALL_COLUMNS)))
        other_key: ColumnKey = (_OTHER, f"{parent[0]}:{parent[1]}#{idx}")
        new_edges[(parent, other_key)] = total
        new_values[other_key] = total
        other_meta[other_key] = (col, f"+{len(small)} other")
        order.append(other_key)

    kept = _reachable(new_edges, roots)
    kept_values = {key: value for key, value in new_values.items() if key in kept}
    kept_edges = {edge: value for edge, value in new_edges.items() if edge[0] in kept and edge[1] in kept}
    kept_order = [key for key in order if key in kept]
    kept_meta = {key: meta for key, meta in other_meta.items() if key in kept}
    return kept_values, kept_edges, kept_order, kept_meta


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
    ``_ribbons_taxa``). ``columns_enabled`` collapses columns, ``min_threshold`` pools small
    siblings into a gray "+N other" node, and ``focus_key`` restricts to a node and its descendants.
    Flow conserves at every node under every combination.
    """
    counts = counts or Counter()
    paths = [(row, path) for row in rows if (path := row_path(row, columns_enabled))]

    ribbons = _ribbons_images(paths, counts) if size_metric == "images" else _ribbons_taxa(paths)
    if focus_key is not None:
        ribbons = _focus_ribbons(ribbons, focus_key)

    node_value, edge_value, order, roots = _accumulate(ribbons)
    other_meta: dict[ColumnKey, ColumnKey] = {}
    if min_threshold and min_threshold > 0:
        node_value, edge_value, order, other_meta = _pool_below_threshold(
            node_value, edge_value, order, roots, float(min_threshold)
        )
    return _materialize(node_value, edge_value, order, other_meta)


def breadcrumb(graph_or_rows: Graph | list[dict], focus_key: ColumnKey | None) -> list[ColumnKey]:
    """Ancestor chain from the root column down to ``focus_key`` (inclusive) for the Back affordance.

    Accepts either a built ``Graph`` (walks incoming edges up to a root) or the raw rows (finds the
    first row whose path passes through ``focus_key``). Returns ``[]`` when ``focus_key`` is None.
    """
    if focus_key is None:
        return []
    if isinstance(graph_or_rows, Graph):
        parent: dict[ColumnKey, ColumnKey] = {}
        for link in graph_or_rows.links:
            src = graph_or_rows.nodes[link.src]
            tgt = graph_or_rows.nodes[link.tgt]
            parent.setdefault((tgt.col, tgt.label), (src.col, src.label))
        chain = [focus_key]
        cur = focus_key
        seen = {cur}
        while cur in parent:
            cur = parent[cur]
            if cur in seen:
                break
            seen.add(cur)
            chain.append(cur)
        chain.reverse()
        return chain
    for row in graph_or_rows:
        path = row_path(row)
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
# double-click on the SAME node index (~350ms), then writes that index into the hidden gr.Number's
# <input> and dispatches an 'input' event so its .change() fires the Python re-root callback.
# SINGLE-CLICK FALLBACK: if a browser/Plotly build never delivers the paired click (RESEARCH A1/A2),
# drop the `now - last < 350 && p.index === lastIdx` guard so a single click zooms instead.
BRIDGE_JS = r"""
() => {
  const attach = () => {
    const gd = document.querySelector('#pz_sankey .js-plotly-plot');
    if (!gd || !gd.on) { return setTimeout(attach, 300); }   // wait for Plotly.newPlot
    let last = 0, lastIdx = -1;
    gd.on('plotly_click', (e) => {
      const p = e.points && e.points[0];
      if (!p || p.index === undefined) return;               // p.index = clicked node index
      const now = Date.now();
      if (now - last < 350 && p.index === lastIdx) {          // debounced double-click
        const inp = document.querySelector('#pz_click input, #pz_click textarea');
        if (inp) {
          inp.value = String(p.index);
          inp.dispatchEvent(new Event('input', {bubbles: true}));  // -> gr.Number.change
        }
      }
      last = now; lastIdx = p.index;
    });
  };
  attach();
}
"""

# Token-driven Inria palette + dot-grid motif + blanc-tournant, light AND dark (charter §1/§2/§5).
# Brand hexes are immutable across themes; only neutrals re-tune. Both the media query and the
# data-theme overrides are emitted so a manual toggle wins over the OS preference.
INRIA_CSS = """
:root{
  --rouge:#c9191e; --framboise:#a60f79; --violet:#534b9a; --bleu-mat:#27348b; --bleu-canard:#1067a3;
  --data-other:#aab3bf;
  --page:#ffffff; --panel:#ffffff; --ink:#171a1d; --ink-muted:#5c666f; --hair:#e0e5ea;
}
@media (prefers-color-scheme:dark){:root{
  --page:#121517; --panel:#1a1e21; --ink:#e9edf0; --ink-muted:#8a939c; --hair:#2a3034;
}}
:root[data-theme="dark"]{
  --page:#121517; --panel:#1a1e21; --ink:#e9edf0; --ink-muted:#8a939c; --hair:#2a3034;
}
:root[data-theme="light"]{
  --page:#ffffff; --panel:#ffffff; --ink:#171a1d; --ink-muted:#5c666f; --hair:#e0e5ea;
}
.gradio-container{background:var(--page); color:var(--ink);}
/* blanc tournant — white margin framing the content zone, never bleeding to the viewport edge */
.pz-frame{padding:24px; background:var(--page);}
@media (min-width:768px){.pz-frame{padding:48px;}}
@media (min-width:1024px){.pz-frame{padding:64px; max-width:1200px; margin:0 auto;}}
/* signature dot-grid motif — a corner tile only, NEVER tiled behind body text */
.inria-motif{
  background-image:radial-gradient(currentColor 1px, transparent 1.5px);
  background-size:16px 16px; color:var(--bleu-mat); opacity:.08;
  width:240px; height:240px; pointer-events:none;
}
@media (prefers-color-scheme:dark){.inria-motif{opacity:.11;}}
.pz-header{display:flex; align-items:center; gap:16px; flex-wrap:wrap;}
.pz-rf{display:flex; align-items:center; gap:8px; color:var(--ink);}
.pz-rf-label{font-weight:700; font-size:11px; line-height:1.05; letter-spacing:.03em;}
.pz-rf-devise{font-size:9px; font-style:italic; font-weight:400;}
.pz-rule{width:1px; height:38px; background:var(--hair);}
/* NON-PRODUCTION facsimile — shown only if the official Inria SVG fails to load. Serif + Rouge. */
.pz-inria-fallback{font-family:"Inria Serif",Georgia,serif; color:var(--rouge); font-size:28px; font-style:italic;}
"""

# Sole-emitter (EMITTER_MODE=sole) lockup: the République Française bloc-marque on the LEFT (State
# colors #000091 / #E1000F / #FFFFFF — deliberately OUTSIDE the Inria 5-hue palette), a thin rule,
# then the official Inria red-script wordmark. onerror reveals the flagged Serif-Rouge facsimile.
HEADER_HTML = f"""
<div class="pz-header" role="banner">
  <div class="pz-rf" aria-label="République Française">
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
