#!/usr/bin/env python3
"""
(c) Inria

pz_build_sankey — regenerate the self-contained, Inria-styled taxonomic Sankey HTML.

Reads the planktonzilla taxonomy-mapping CSV and (by default) scans the published
HuggingFace image dataset for real per-class sample counts, then emits ONE
self-contained HTML file: the interactive Sankey with a *Size by* toggle (label
mappings / distinct taxa / dataset images) and an optional *Source dataset* level
inserted right after the root.

Nothing intermediate is committed. The two node trees, the per-class image counts,
the Inria Sans/Serif web fonts and the official Inria + Republique Francaise logo
are all produced or fetched at build time and embedded into the output. Only this
CLI and its ``templates/sankey_template.html`` live in the repo; when a new dataset
version is published, re-run this command to refresh the visualization.

Examples
--------
    # Defaults: bundled taxonomy CSV, scan project-oceania/planktonzilla-17M, write ./planktonzilla_sankey.html
    pz_build_sankey

    # Fast rebuild without the (minutes-long) dataset scan — mappings + taxa only
    pz_build_sankey --no-samples

    # Reuse a previously-scanned per-(dataset,proposed_label,root_class) counts JSON
    pz_build_sankey --samples-json counts.json --out flow.html

The same CLI is available via ``python -m planktonzilla.explorer.build_sankey``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
    DEFAULT_TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
)

logger = logging.getLogger("planktonzilla.explorer.build_sankey")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "sankey_template.html"
PLACEHOLDERS = ("__FONTS__", "__LOGO_B64__", "__TREE__", "__TREE_SRC__")
DEFAULT_LOGO_URL = "https://inria.fr/themes/custom/inria/logo/logo.svg"
GOOGLE_FONTS_CSS = (
    "https://fonts.googleapis.com/css2?family=Inria+Sans:ital,wght@0,300;0,400;0,700;1,400"
    "&family=Inria+Serif:ital,wght@0,400;0,700;1,400&display=swap"
)
# (family, style, weight) faces embedded from the latin subset
WANTED_FACES = {
    ("Inria Sans", "normal", "300"),
    ("Inria Sans", "normal", "400"),
    ("Inria Sans", "normal", "700"),
    ("Inria Sans", "italic", "400"),
    ("Inria Serif", "normal", "400"),
    ("Inria Serif", "normal", "700"),
    ("Inria Serif", "italic", "400"),
}
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


# ----------------------------------------------------------------- tree building
def _new_node(name: str, rank: str) -> dict:
    return {"n": name, "r": rank, "c": 0, "_ds": 0, "_kids": {}}


def _living_path(row: dict) -> list[tuple[str, str]]:
    """Ordered (rank, value) for the filled ranks; empty-lineage living rows go under (unclassified)."""
    path = [(rk, row[rk].strip()) for rk in TAXONOMY_RANKS if row[rk].strip()]
    if not path:
        lbl = row["proposed_label"].strip() or "(unlabeled)"
        path = [("group", "(unclassified)"), ("proposed_label", lbl)]
    return path


def _descend_taxonomy(node: dict, row: dict, rc: str) -> dict:
    """Walk/create the taxonomy path for one row under ``node``; return the terminal node."""
    if rc == "living":
        cur = node
        for rank, val in _living_path(row):
            key = (rank, val.lower())
            if key not in cur["_kids"]:
                cur["_kids"][key] = _new_node(val, rank)
            cur = cur["_kids"][key]
            cur["c"] += 1
        return cur
    lbl = row["proposed_label"].strip() or "(unlabeled)"
    key = lbl.lower()
    if key not in node["_kids"]:
        node["_kids"][key] = _new_node(lbl, "proposed_label")
    cur = node["_kids"][key]
    cur["c"] += 1
    return cur


def _attach_samples(label_to_node: dict, counts: Counter) -> int:
    """Add per-key image counts as ``_ds`` on the matching terminal nodes; return the mapped total."""
    mapped = 0
    for key, n in counts.items():
        node = label_to_node.get(key)
        if node is not None:
            node["_ds"] += n
            mapped += n
    return mapped


def _finalize(node: dict) -> dict:
    """Convert ``_kids`` dicts to sorted child lists and compute additive ``s`` (images) and ``t`` (leaf taxa)."""
    kids = list(node["_kids"].values())
    node.pop("_kids")
    direct_s = node.pop("_ds")
    if kids:
        for k in kids:
            _finalize(k)
        kids.sort(key=lambda x: (-x["c"], x["n"]))
        node["k"] = kids
        node["s"] = direct_s + sum(k["s"] for k in kids)
        node["t"] = sum(k["t"] for k in kids)
    else:
        node["s"] = direct_s
        node["t"] = 1
    return node


def build_tree(rows: list[dict], per_label: Counter) -> tuple[dict, int]:
    """Normal tree: root -> root_class -> taxonomy. ``per_label`` keyed by (proposed_label_lower, root_class)."""
    root = _new_node("planktonzilla-17M", "dataset")
    label_to_node: dict = {}
    for r in rows:
        rc = r["root_class"].strip()
        root["c"] += 1
        if rc not in root["_kids"]:
            root["_kids"][rc] = _new_node(rc, "root_class")
        node = root["_kids"][rc]
        node["c"] += 1
        cur = _descend_taxonomy(node, r, rc)
        label_to_node.setdefault((r["proposed_label"].strip().lower(), rc), cur)
    mapped = _attach_samples(label_to_node, per_label)
    _finalize(root)
    return root, mapped


def build_source_tree(rows: list[dict], per_source: Counter) -> tuple[dict, int]:
    """Source tree: root -> source -> root_class -> taxonomy. ``per_source`` keyed by (dataset, pl_lower, rc)."""
    root = _new_node("planktonzilla-17M", "dataset")
    label_to_node: dict = {}
    for r in rows:
        rc = r["root_class"].strip()
        src = r["Dataset"].strip()
        root["c"] += 1
        if src not in root["_kids"]:
            root["_kids"][src] = _new_node(src, "source")
        snode = root["_kids"][src]
        snode["c"] += 1
        rck = ("rootclass", rc)
        if rck not in snode["_kids"]:
            snode["_kids"][rck] = _new_node(rc, "root_class")
        rcnode = snode["_kids"][rck]
        rcnode["c"] += 1
        cur = _descend_taxonomy(rcnode, r, rc)
        label_to_node.setdefault((src, r["proposed_label"].strip().lower(), rc), cur)
    mapped = _attach_samples(label_to_node, per_source)
    _finalize(root)
    return root, mapped


def _tree_meta(root: dict, rows: list[dict], samples_available: bool, *, source: bool) -> dict:
    nodes = leaves = 0
    stack = [root]
    while stack:
        n = stack.pop()
        nodes += 1
        kids = n.get("k")
        if kids:
            stack.extend(kids)
        else:
            leaves += 1
    n_datasets = len({r["Dataset"].strip() for r in rows if r["Dataset"].strip()})
    return {
        "total_rows": len(rows),
        "n_datasets": n_datasets,
        "tree_nodes": nodes,
        "tree_leaves": leaves,
        "total_samples": root["s"],
        "total_taxa": root["t"],
        "samples_available": samples_available,
        "source_level": source,
    }


# --------------------------------------------------------------- dataset scan
def scan_dataset(repo_id: str, workers: int, retries: int = 4) -> Counter:
    """Aggregate per-(dataset, proposed_label_lower, root_class) image counts from the HF parquet shards.

    Only the three metadata columns are read (column projection), so the image bytes are never
    downloaded. Shards are discovered dynamically so a new dataset version with a different shard
    count still works.
    """
    import concurrent.futures as cf

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo_id}/**/*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet shards found for dataset {repo_id!r}")
    logger.info("Scanning %d parquet shards of %s for per-class image counts…", len(files), repo_id)

    def read_one(path: str) -> Counter:
        last: Exception | None = None
        for k in range(retries):
            try:
                with fs.open(path) as fh:
                    tbl = pq.read_table(fh, columns=["dataset", "proposed_label", "root_class"])
                ds = [(x or "").strip() for x in tbl.column("dataset").to_pylist()]
                pl = [(x or "").strip().lower() for x in tbl.column("proposed_label").to_pylist()]
                rc = [(x or "").strip() for x in tbl.column("root_class").to_pylist()]
                return Counter(zip(ds, pl, rc))
            except Exception as exc:
                last = exc
                time.sleep(2 * (k + 1))
        raise RuntimeError(f"failed to read {path}: {last}")

    agg: Counter = Counter()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(read_one, files):
            agg.update(part)
            done += 1
            if done % 20 == 0 or done == len(files):
                logger.info("  %d/%d shards scanned (%d rows so far)", done, len(files), sum(agg.values()))
    logger.info("Scan done: %d images across %d (dataset, label, root_class) classes.", sum(agg.values()), len(agg))
    return agg


def load_sample_counts(path: Path) -> Counter:
    """Load a precomputed per-(dataset, proposed_label, root_class) counts JSON (as written by a prior scan)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for row in doc.get("counts", []):
        key = (row["dataset"].strip(), row["proposed_label"].strip().lower(), row["root_class"].strip())
        counts[key] += int(row["n"])
    logger.info("Loaded %d classes (%d images) from %s.", len(counts), sum(counts.values()), path)
    return counts


# ----------------------------------------------------------------- asset fetch
def fetch_fonts() -> str:
    """Return an ``@font-face`` block (base64 data-URIs) for the Inria typefaces, or '' on failure."""
    import requests

    try:
        css = requests.get(GOOGLE_FONTS_CSS, headers={"User-Agent": _UA}, timeout=30).text
        faces, seen = [], set()
        for block in re.findall(r"@font-face\s*\{(.*?)\}", css, re.DOTALL):
            fam = re.search(r"font-family:\s*'([^']+)'", block)
            sty = re.search(r"font-style:\s*(\w+)", block)
            wgt = re.search(r"font-weight:\s*(\d+)", block)
            rng = re.search(r"unicode-range:\s*([^;]+)", block)
            src = re.search(r"url\((https://[^)]+\.woff2)\)", block)
            if not (fam and sty and wgt and src):
                continue
            key = (fam.group(1), sty.group(1), wgt.group(1))
            is_latin = bool(rng) and "U+0000-00FF" in rng.group(1)
            if key in WANTED_FACES and is_latin and key not in seen:
                data = requests.get(src.group(1), headers={"User-Agent": _UA}, timeout=30).content
                b64 = base64.b64encode(data).decode()
                faces.append(
                    f"@font-face{{font-family:'{key[0]}';font-style:{key[1]};font-weight:{key[2]};"
                    f"font-display:swap;src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
                )
                seen.add(key)
        logger.info("Embedded %d Inria font faces.", len(faces))
        return "\n".join(faces)
    except Exception as exc:
        logger.warning("Font fetch failed (%s); the viz will use its Georgia/Tahoma fallback stack.", exc)
        return ""


def fetch_logo(url: str) -> str:
    """Return the base64 of the logo SVG at ``url``, or '' on failure."""
    import requests

    try:
        svg = requests.get(url, headers={"User-Agent": _UA}, timeout=30).content
        logger.info("Embedded logo from %s (%d KB).", url, len(svg) // 1024)
        return base64.b64encode(svg).decode()
    except Exception as exc:
        logger.warning("Logo fetch failed (%s); the header lockup image will be blank.", exc)
        return ""


# ------------------------------------------------------------------- assemble
def assemble(template: str, data: dict, data_src: dict, fonts_css: str, logo_b64: str) -> str:
    """Substitute the four template placeholders and return the finished HTML."""

    def dump(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    html = template.replace("__FONTS__", fonts_css)
    html = html.replace("__LOGO_B64__", logo_b64)
    html = html.replace("__TREE_SRC__", dump(data_src))
    html = html.replace("__TREE__", dump(data))
    return html


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pz_build_sankey",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=Path(DEFAULT_TAXONOMY_CSV_FILENAME), help="taxonomy CSV (default: bundled)")
    ap.add_argument("--out", type=Path, default=Path("planktonzilla_sankey.html"), help="output HTML path")
    ap.add_argument(
        "--dataset-repo", default=DEFAULT_PLANKTONZILLA_DATASET_REPO_ID, help="HF dataset repo scanned for image counts"
    )
    ap.add_argument("--logo-url", default=DEFAULT_LOGO_URL, help="official Inria logo SVG URL to embed")
    ap.add_argument("--workers", type=int, default=16, help="concurrent parquet readers for the dataset scan")
    ap.add_argument(
        "--no-samples", action="store_true", help="skip the dataset scan (mappings + taxa only; the Images metric is hidden)"
    )
    ap.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="load precomputed per-(dataset,proposed_label,root_class) counts instead of scanning",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info("Read %d taxonomy rows from %s.", len(rows), args.csv)

    if args.no_samples:
        per_source: Counter = Counter()
    elif args.samples_json is not None:
        per_source = load_sample_counts(args.samples_json)
    else:
        per_source = scan_dataset(args.dataset_repo, args.workers)
    samples_available = bool(per_source)

    per_label: Counter = Counter()
    for (_ds, pl, rc), n in per_source.items():
        per_label[(pl, rc)] += n

    root, mapped = build_tree(rows, per_label)
    root_src, mapped_src = build_source_tree(rows, per_source)
    if samples_available:
        total = sum(per_source.values())
        logger.info("Images mapped onto trees: %d/%d (normal), %d/%d (source).", mapped, total, mapped_src, total)

    meta = _tree_meta(root, rows, samples_available, source=False)
    meta_src = _tree_meta(root_src, rows, samples_available, source=True)

    fonts_css = fetch_fonts()
    logo_b64 = fetch_logo(args.logo_url)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = assemble(template, {"meta": meta, "tree": root}, {"meta": meta_src, "tree": root_src}, fonts_css, logo_b64)

    leftover = [p for p in PLACEHOLDERS if p in html]
    if leftover:
        logger.error("Template placeholders were not all substituted: %s", leftover)
        return 1

    args.out.write_text(html, encoding="utf-8")
    logger.info(
        "Wrote %s — %d KB · %d rows · %d tree nodes · %d sources · %s images.",
        args.out,
        len(html.encode("utf-8")) // 1024,
        meta["total_rows"],
        meta["tree_nodes"],
        meta_src["n_datasets"],
        f"{meta['total_samples']:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
