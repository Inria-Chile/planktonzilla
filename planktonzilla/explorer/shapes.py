"""
(c) Inria

Pure polars data-shaping layer for the planktonzilla explorer (FND-05).

This module reshapes the FROZEN, committed taxonomy crosswalk
(``planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv`` — D4) and the
per-dataset geo frames into the structures every view phase (11/12/13) renders:

* ``load_taxonomy`` — read the frozen CSV as all-string columns (blanks -> "").
* ``build_sankey_index`` — ordered-stage Sankey node/link index.
* ``build_hierarchy_table`` — ragged Kingdom -> Species path/value roll-up.
* ``aggregate_geo`` — per-dataset geo aggregation, merging measured (live HF)
  and inferred (committed CSV) locations into one frame.

Design (load-bearing): this layer is PURE polars. Module-scope imports are ONLY
``polars``, stdlib, and the shared ``planktonzilla_dataset.constants`` (reused, not
copied — ROADMAP SC1). There are NO gradio/plotly/datasets/huggingface_hub/pyarrow
imports anywhere in this file: the Phase 9 dependency-isolation guard scans
``planktonzilla/`` for module-scope viz imports and rendering belongs to the view
phases. All network IO lives behind ``data_access.py``.
"""

from pathlib import Path

import polars as pl

from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_TAXONOMY_CSV_FILENAME,
    EXTRA_COLS,  # noqa: F401  -- re-exported for the view phases' convenience
    TAXONOMY_RANKS,
)

# String sentinel for a blank cell in a Sankey stage (mirrors generate_sankey.py).
BLANK = "(blank)"

# The casing of the live-HF geo columns (confirmed 2026-06-30): uppercase lat/lon
# grouped by dataset. The committed inferred-locations CSV uses lowercase columns;
# ``aggregate_geo`` reconciles the two without mutating either source.
GEO_DATASET_COL = "dataset"
GEO_LAT_COL = "Latitude"
GEO_LON_COL = "Longitude"

# The three confidence-graded geo categories carried by ``aggregate_geo`` (D4): measured rows
# (live per-sample GPS) are "measured"; inferred-CSV rows become "inferred-" + their confidence
# grade (high/low). The inferred ``confidence == "na"`` grade (planktoscope, lensless) is NOT a
# category — those rows are EXCLUDED entirely (they have no real collection site).
CATEGORY_MEASURED = "measured"
CATEGORY_INFERRED_HIGH = "inferred-high"
CATEGORY_INFERRED_LOW = "inferred-low"
GEO_CATEGORIES = (CATEGORY_MEASURED, CATEGORY_INFERRED_HIGH, CATEGORY_INFERRED_LOW)


def load_taxonomy(csv_path: str | Path = DEFAULT_TAXONOMY_CSV_FILENAME) -> pl.DataFrame:
    """Read the frozen taxonomy CSV as all-string columns with blanks normalized to "".

    Every column is read as Utf8 (``infer_schema_length=0``); nulls and surrounding
    whitespace are stripped so a blank cell is always the empty string "" (never
    None). This makes downstream blank-detection and the string-valued
    ``plankton``/``living`` columns ("True"/"False") robust to stray whitespace.

    Args:
        csv_path: Path to the taxonomy CSV. Defaults to the frozen committed CSV
            (``DEFAULT_TAXONOMY_CSV_FILENAME``) — the canonical source per D4.

    Returns:
        A polars DataFrame where every cell is a stripped string and blanks are "".
    """
    df = pl.read_csv(
        Path(csv_path),
        infer_schema_length=0,  # force every column to Utf8
        null_values=[],
    )
    return df.with_columns(pl.all().cast(pl.Utf8).fill_null("").str.strip_chars())


def build_sankey_index(
    df: pl.DataFrame,
    stages: list[str],
    *,
    drop_blank: bool = False,
    min_flow: int = 1,
) -> dict:
    """Build a well-formed Sankey node/link index over ordered ``stages``.

    Node identity is ``(stage_index, value)``; a blank cell renders as the
    ``"(blank)"`` sentinel. If ``drop_blank`` is true, any row that is blank in ANY
    selected stage is skipped entirely. Consecutive-stage transitions are
    aggregated into link counts, and links whose count is below ``min_flow`` are
    dropped (the long-tail aggregation knob).

    Args:
        df: The taxonomy frame (typically from ``load_taxonomy``).
        stages: Ordered list of column names; the selection order is the stage order.
        drop_blank: If true, drop rows blank in any selected stage.
        min_flow: Minimum link count to keep; links below this are dropped.

    Returns:
        ``{"nodes": [{"stage": int, "label": str}, ...], "source": [int...],
        "target": [int...], "value": [int...], "used_rows": int}``. Every link
        source/target is a valid index into ``nodes`` and there are no orphan nodes.

    Raises:
        ValueError: If fewer than two stages are given, or a stage column is missing.
    """
    if len(stages) < 2:
        raise ValueError("build_sankey_index requires at least 2 stages")
    missing = [s for s in stages if s not in df.columns]
    if missing:
        raise ValueError(f"stage column(s) not in dataframe: «{missing}»")

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
    with "/" (e.g. ``"animalia/cnidaria/hydrozoa"``). A node's value is the count of
    rows whose path passes through that prefix. Parents wire each node to its
    immediate prefix; root-level nodes have parent "".

    Args:
        df: The taxonomy frame (typically from ``load_taxonomy``).
        ranks: Ordered taxonomic ranks to walk. Defaults to ``TAXONOMY_RANKS``.

    Returns:
        ``{"ids": [...], "labels": [...], "parents": [...], "values": [...]}``.
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


def hierarchy_root_class(df: pl.DataFrame, *, ranks: tuple[str, ...] = TAXONOMY_RANKS) -> dict[str, str]:
    """Per-node majority ``root_class`` for the sunburst/icicle, via FULL-distribution argmax.

    Walks each row left-to-right over the present ``ranks`` EXACTLY as
    ``build_hierarchy_table`` does (stop at the first blank rank; node id is the "/"-joined
    prefix, e.g. ``"animalia/cnidaria/hydrozoa"``), so the returned keys align 1:1 with that
    function's ``ids``. For every visited node it accumulates the row's ``root_class`` into a
    per-node ``{root_class: count}`` tally over ALL rows passing through that prefix, THEN
    argmaxes ONCE.

    ⚠️ Correctness guardrail (the Sankey bug fixed in commit b92d7d0): the majority is the
    argmax of the node's FULL merged distribution — it is NOT computed by pre-argmaxing each
    child/row-group then merging (that lossy path credits all of a group's rows to one class
    and flips the winner when ≥2 sub-distributions merge, e.g. ``{X:5,Y:4}`` + ``{Y:3}`` would
    wrongly yield ``X`` instead of the merged ``{X:5,Y:7}`` -> ``Y``). The first-seen tie-break
    and the ``Unknown`` empty-map sentinel are delegated to ``sankey._argmax_key`` /
    ``LINEAGE_UNKNOWN`` (imported FUNCTION-LOCALLY so this pure layer keeps its viz-free,
    cycle-free module scope — the Phase 9 dependency-isolation guard stays green).

    Args:
        df: The taxonomy frame (typically from ``load_taxonomy``).
        ranks: Ordered taxonomic ranks to walk. Defaults to ``TAXONOMY_RANKS``.

    Returns:
        ``{node_id: majority_root_class}`` keyed identically to ``build_hierarchy_table``'s
        ``ids``. Blank/missing ``root_class`` cells fold into the ``Unknown`` sentinel. If the
        ``"root_class"`` column is absent from ``df`` the result is ``{}`` (the caller then
        falls back to rank/depth coloring).
    """
    from planktonzilla.explorer.sankey import LINEAGE_UNKNOWN, _argmax_key

    if "root_class" not in df.columns:
        return {}

    present = [r for r in ranks if r in df.columns]
    # Per-node {root_class: count} tally; plain dicts preserve first-seen insertion order so
    # _argmax_key's first-seen tie-break matches the Sankey semantics exactly.
    tallies: dict[str, dict[str, int]] = {}

    columns = [df.get_column(r).to_list() for r in present]
    rc_col = df.get_column("root_class").to_list()
    n_rows = df.height
    for row_i in range(n_rows):
        rc = (rc_col[row_i] or "").strip() or LINEAGE_UNKNOWN
        prefix_parts: list[str] = []
        for col_i in range(len(present)):
            value = (columns[col_i][row_i] or "").strip()
            if not value:
                break  # ragged: stop at first blank rank (parity with build_hierarchy_table)
            prefix_parts.append(value)
            node_id = "/".join(prefix_parts)
            tally = tallies.setdefault(node_id, {})
            tally[rc] = tally.get(rc, 0) + 1

    return {node_id: _argmax_key(tally) for node_id, tally in tallies.items()}


def _to_float_or_null(expr: pl.Expr) -> pl.Expr:
    """Cast a (possibly string, possibly blank) lat/lon column to Float64 or null.

    Blank/whitespace-only strings and unparseable values become null so they are
    dropped by the downstream ``drop_nulls``. ``strict=False`` keeps the cast from
    raising on already-blank cells produced by ``load_taxonomy``-style frames.
    """
    return expr.cast(pl.Utf8).str.strip_chars().replace("", None).cast(pl.Float64, strict=False)


def aggregate_geo(
    measured: pl.DataFrame,
    inferred: pl.DataFrame | None = None,
    *,
    round_decimals: int = 4,
) -> pl.DataFrame:
    """Aggregate per-dataset geo points, merging measured (live) + inferred (CSV).

    ``measured`` carries the live-HF casing ``{Latitude, Longitude, dataset}``. Rows
    with null/blank lat or lon are dropped, lat/lon are rounded to collapse
    near-duplicate points, and rows are counted per
    ``(dataset, Latitude, Longitude)``.

    If ``inferred`` is given (the committed CSV with LOWERCASE
    ``dataset,latitude,longitude,...,confidence``), its casing is normalized to match
    measured, its blank/na lat/lon rows are dropped, AND its ``confidence == "na"``
    rows (e.g. ``planktoscope``/``lensless`` — no real collection site) are EXCLUDED
    explicitly as belt-and-suspenders (D3/D4). The surviving inferred rows are
    concatenated under the same schema. A ``source`` column distinguishes the origin
    (``"measured"`` vs ``"inferred"``) and a ``count`` column carries the row count
    (inferred rows count as 1 per point). Neither input is mutated.

    Measured wins (dedup belt-and-suspenders): any inferred dataset that ALSO has
    measured coordinates is dropped from the inferred frame before concatenation. The
    inferred CSV exists only for datasets that LACK per-sample GPS; if a dataset turns
    up with real measured coordinates, the KNOWN measured location is authoritative and
    the inferred entry must not also appear (no double-plot, no conflicting category at
    two locations). In the current committed-data case there is NO overlap (the 9
    inferred datasets have no live GPS), so this is a no-op there.

    A 6th ``category`` column carries the confidence grade for legending (D4): measured
    rows -> ``"measured"``; inferred rows -> ``"inferred-high"`` / ``"inferred-low"``
    from their ``confidence``. The first five columns
    ``{dataset, Latitude, Longitude, count, source}`` keep their name/order so callers
    that select the legacy five by name are unaffected; ``category`` is appended last.

    Rendering/legending (styling the graded categories distinctly) is Phase 13; here we
    only produce the merged, category-graded aggregated frame.

    Args:
        measured: Live-HF geo frame with columns ``{Latitude, Longitude, dataset}``.
        inferred: Optional committed inferred-locations frame (lowercase columns,
            including a ``confidence`` grade of high/low/na).
        round_decimals: Decimal places for collapsing near-duplicate points.

    Returns:
        A polars DataFrame with columns
        ``{dataset, Latitude, Longitude, count, source, category}``, one row per
        distinct (dataset, rounded-lat, rounded-lon, source) point. ``category`` is
        one of ``GEO_CATEGORIES`` (measured | inferred-high | inferred-low);
        ``confidence == "na"`` and no-coord rows are excluded.
    """
    schema = [GEO_DATASET_COL, GEO_LAT_COL, GEO_LON_COL, "count", "source", "category"]

    measured_agg = (
        measured.select(
            pl.col(GEO_DATASET_COL).cast(pl.Utf8),
            _to_float_or_null(pl.col(GEO_LAT_COL)).round(round_decimals).alias(GEO_LAT_COL),
            _to_float_or_null(pl.col(GEO_LON_COL)).round(round_decimals).alias(GEO_LON_COL),
        )
        .drop_nulls([GEO_LAT_COL, GEO_LON_COL])
        .group_by([GEO_DATASET_COL, GEO_LAT_COL, GEO_LON_COL])
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.lit("measured").alias("source"),
            pl.lit(CATEGORY_MEASURED).alias("category"),
        )
        .select(schema)
    )

    if inferred is None:
        return measured_agg.sort([GEO_DATASET_COL, GEO_LAT_COL, GEO_LON_COL])

    # Normalize the inferred CSV's lowercase casing to match the measured schema, and
    # derive the confidence grade. Strip + lowercase ``confidence`` for robustness, then
    # EXCLUDE the "na" grade explicitly (belt-and-suspenders alongside the null-coord drop).
    inferred_norm = (
        inferred.with_columns(pl.col("confidence").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("confidence"))
        .filter(pl.col("confidence").is_in(["high", "low"]))
        .select(
            pl.col("dataset").cast(pl.Utf8).alias(GEO_DATASET_COL),
            _to_float_or_null(pl.col("latitude")).round(round_decimals).alias(GEO_LAT_COL),
            _to_float_or_null(pl.col("longitude")).round(round_decimals).alias(GEO_LON_COL),
            pl.col("confidence"),
        )
        .drop_nulls([GEO_LAT_COL, GEO_LON_COL])
        .with_columns(
            pl.lit(1, dtype=pl.UInt32).alias("count"),
            pl.lit("inferred").alias("source"),
            pl.when(pl.col("confidence") == "high")
            .then(pl.lit(CATEGORY_INFERRED_HIGH))
            .otherwise(pl.lit(CATEGORY_INFERRED_LOW))
            .alias("category"),
        )
        .select(schema)
    )

    # Measured wins: drop inferred rows for any dataset that already has measured
    # coordinates, so a dataset never plots twice at two locations with conflicting
    # categories. No-op when there is no overlap (the current committed-data case).
    measured_datasets = measured_agg.get_column(GEO_DATASET_COL).unique().to_list()
    inferred_norm = inferred_norm.filter(~pl.col(GEO_DATASET_COL).is_in(measured_datasets))

    merged = pl.concat([measured_agg, inferred_norm], how="vertical_relaxed")
    return merged.sort([GEO_DATASET_COL, "source", GEO_LAT_COL, GEO_LON_COL])
