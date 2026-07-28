#!/usr/bin/env python3
"""
(c) Inria

pz_sankey — emit a live, self-contained HTML Sankey of the planktonzilla label space.

The diagram runs ten fixed columns::

    Source dataset → root_class → Domain → Kingdom → Phylum → Class → Order → Family → Genus → Species

Rows whose ``root_class`` is **not** ``living`` have no Linnaean lineage, so their
``proposed_label`` is placed at the **Domain** column instead (``detritus``, ``fiber``,
``bubble``, …) and the ribbon ends there. Living rows carry a real domain — derived from
their Kingdom — and then fan out through the ranks, stopping at the deepest rank the
taxonomy actually fills (``Species`` is populated for only ~19 % of living labels, so a
ribbon that has no species simply *ends* at its genus rather than draining into a
meaningless "(blank)" node).

Everything downstream of the CSV is expressed as **ribbons**: one full column path plus a
weight. Column visibility, threshold pooling and focus are all ribbon operations, and node
and link values are re-derived from whichever ribbons survive. Flow therefore conserves at
every node by construction — including where the graph *converges*, which is the case that
defeats popping edges out of an already-built graph.

The emitted page is one file with no external requests: the renderer is a purpose-built SVG
Sankey (no plotting library), and the Inria typefaces and the official Inria + République
Française lockup are embedded as data-URIs at build time. Everything the page offers —
column visibility, colour dimension, the merge-threshold slider, focus, search — recomputes
in the browser, so the file stays live long after it is generated.

Examples
--------
    # Defaults: bundled taxonomy CSV + ./samples.json if present → planktonzilla_sankey_flow.html
    pz_sankey

    # Explicit counts file and output, then open it
    pz_sankey --samples-json samples.json --out flow.html --open

    # No image counts at all: ribbons are weighted by label mappings only
    pz_sankey --no-samples

    # Scan the published dataset for fresh per-class counts and cache them
    pz_sankey --dataset-repo project-oceania/planktonzilla-17M --save-samples samples.json

The same CLI is available via ``python -m planktonzilla.planktonzilla_dataset.sankey``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import webbrowser
from collections import Counter
from pathlib import Path

from planktonzilla.planktonzilla_dataset.build_sankey import (
    DEFAULT_LOGO_URL,
    fetch_fonts,
    fetch_logo,
    load_sample_counts,
    save_sample_counts,
    scan_dataset,
)
from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
    DEFAULT_TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "sankey_flow.html"
PLACEHOLDERS = ("__FONTS__", "__LOGO_B64__", "__PAYLOAD__")

# The ten Sankey columns, left to right. ``key`` is the machine name used in the payload,
# ``label`` is what the page prints in the column header and the visibility chips.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("dataset", "Source dataset"),
    ("root_class", "Root class"),
    ("domain", "Domain"),
    ("kingdom", "Kingdom"),
    ("phylum", "Phylum"),
    ("class", "Class"),
    ("order", "Order"),
    ("family", "Family"),
    ("genus", "Genus"),
    ("species", "Species"),
)
DOMAIN_COL = 2  # index of the Domain column — where non-living proposed_labels land

# Kingdom → biological domain. The taxonomy CSV uses the five-kingdom-plus-Bacteria scheme,
# so every kingdom it contains resolves to one of the three domains.
DOMAIN_BY_KINGDOM: dict[str, str] = {
    "bacteria": "Bacteria",
    "archaea": "Archaea",
    "animalia": "Eukaryota",
    "chromista": "Eukaryota",
    "plantae": "Eukaryota",
    "protozoa": "Eukaryota",
    "fungi": "Eukaryota",
}
# A living row with no Kingdom at all sometimes names its domain in proposed_label.
DOMAIN_ALIASES = {"eukaryota": "Eukaryota", "bacteria": "Bacteria", "archaea": "Archaea"}
UNCLASSIFIED = "(unclassified)"

ABSENT = -1  # path slot for "this ribbon has no value at this column and ends before it"


def _disp(value: str) -> str:
    """Capitalize the first character for display, leaving the rest of the string alone.

    Taxonomic names are conventionally capitalized (``Animalia``, ``Abylopsis tetragona``)
    while the CSV stores them lower-cased; the binomial's epithet must stay lower-case, so
    this deliberately is not ``str.title()``.
    """
    return value[:1].upper() + value[1:] if value else value


def _binomial(ranks: list[str]) -> list[str]:
    """Rewrite the Species slot from a bare epithet to the full binomial, in place.

    The CSV stores Species as the epithet alone (``tetragona``), which is both unreadable on
    its own and *unsafe as a node identity*: a node is keyed by (column, value), so two
    genera sharing an epithet — ``Chaetoceros socialis`` and ``Parvicorbicula socialis``, and
    six other pairs in this CSV — would converge into one bogus species node. Prefixing the
    genus makes the value unique and prints the name the way it is actually written.
    """
    genus, species = ranks[-2], ranks[-1]
    if species and genus:
        ranks[-1] = f"{genus} {species.lower()}"
    return ranks


def domain_of(kingdom: str, proposed_label: str) -> str:
    """Return the biological domain for a *living* row.

    Resolved from Kingdom where there is one; a living row with an empty Kingdom falls back
    to its ``proposed_label`` when that names a domain outright, and is otherwise reported
    honestly as ``(unclassified)`` rather than being guessed into Eukaryota.
    """
    king = kingdom.strip().lower()
    if king in DOMAIN_BY_KINGDOM:
        return DOMAIN_BY_KINGDOM[king]
    if king:  # a kingdom we do not know — surface it instead of silently dropping the row
        return _disp(king)
    return DOMAIN_ALIASES.get(proposed_label.strip().lower(), UNCLASSIFIED)


def build_lineage_index(rows: list[dict]) -> dict[tuple[str, str], tuple[str, ...]]:
    """Map ``(proposed_label_lower, root_class)`` to its eight column values (Domain → Species).

    Living rows contribute the derived domain plus their filled Linnaean ranks; every other
    root_class contributes its ``proposed_label`` at the Domain slot and nothing after it,
    which is what places non-living categories at Domain level in the diagram.

    Lineage is a function of ``(proposed_label, root_class)`` in this CSV — the first row
    for a key wins, so a duplicated mapping cannot fork a taxon into two lineages.
    """
    index: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in rows:
        rc = row["root_class"].strip()
        label = row["proposed_label"].strip()
        key = (label.lower(), rc)
        if key in index:
            continue
        if rc == "living":
            ranks = [_disp(row[rank].strip()) for rank in TAXONOMY_RANKS]
            index[key] = (domain_of(row["Kingdom"], label), *_binomial(ranks))
        else:
            index[key] = (_disp(label) or UNCLASSIFIED, *([""] * len(TAXONOMY_RANKS)))
    return index


def build_ribbons(rows: list[dict], counts: Counter) -> dict:
    """Build the page payload: column spec, per-column vocabularies and weighted ribbons.

    Every ribbon is one path across the ten columns plus two weights — ``img`` (dataset
    images, from ``counts``) and ``map`` (how many raw source labels the taxonomy routes
    through that path). Both are carried so the page can re-size the diagram without a
    rebuild. Paths are emitted as per-column vocabulary indices, with ``ABSENT`` marking the
    point where a ribbon stops.

    ``counts`` is keyed ``(dataset, proposed_label_lower, root_class)`` — exactly what
    ``build_sankey.load_sample_counts`` and ``scan_dataset`` produce.
    """
    lineage = build_lineage_index(rows)

    mappings: Counter = Counter()
    for row in rows:
        key = (row["Dataset"].strip(), row["proposed_label"].strip().lower(), row["root_class"].strip())
        mappings[key] += 1

    # Ribbons are the union of both weightings: a class present only in the CSV still shows
    # up (weight 0 images), and a class present only in the scan is never silently dropped.
    paths: dict[tuple[str, ...], list[int]] = {}
    unmapped: Counter = Counter()
    for key in sorted(set(mappings) | set(counts)):
        dataset, label, rc = key
        lin = lineage.get((label, rc))
        if lin is None:
            unmapped[key] += counts.get(key, 0)
            continue
        path = (dataset, _disp(rc), *lin)
        slot = paths.setdefault(path, [0, 0])
        slot[0] += int(counts.get(key, 0))
        slot[1] += int(mappings.get(key, 0))

    if unmapped:
        logger.warning(
            "%d (dataset, label, root_class) keys have no taxonomy row and were skipped (%s images).",
            len(unmapped),
            f"{sum(unmapped.values()):,}",
        )

    # Per-column string tables, ordered by descending image weight so the vocabularies are
    # stable and the heaviest values get the low indices.
    weight_by_value: list[Counter] = [Counter() for _ in COLUMNS]
    for path, (img, mps) in paths.items():
        for col, value in enumerate(path):
            if value:
                weight_by_value[col][value] += img or mps
    vocab: list[list[str]] = []
    lookup: list[dict[str, int]] = []
    for col in range(len(COLUMNS)):
        values = [v for v, _ in weight_by_value[col].most_common()]
        vocab.append(values)
        lookup.append({v: i for i, v in enumerate(values)})

    ribbons: list[list[int]] = []
    for path, (img, mps) in sorted(paths.items(), key=lambda kv: (-kv[1][0], kv[0])):
        encoded = [lookup[col].get(value, ABSENT) if value else ABSENT for col, value in enumerate(path)]
        ribbons.append([*encoded, img, mps])

    total_img = sum(r[len(COLUMNS)] for r in ribbons)
    total_map = sum(r[len(COLUMNS) + 1] for r in ribbons)
    meta = {
        "total_images": total_img,
        "total_mappings": total_map,
        "n_ribbons": len(ribbons),
        "n_datasets": len(vocab[0]),
        "n_taxa": sum(1 for r in ribbons if r[len(COLUMNS) - 1] != ABSENT),
        "samples_available": total_img > 0,
        "column_cardinality": [len(v) for v in vocab],
    }
    return {
        "cols": [{"key": k, "label": lab} for k, lab in COLUMNS],
        "vocab": vocab,
        "ribbons": ribbons,
        "meta": meta,
    }


def assemble(template: str, payload: dict, fonts_css: str, logo_b64: str) -> str:
    """Substitute the three template placeholders and return the finished page."""
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Defang any literal </script> in the data so it cannot break out of the JSON island.
    blob = blob.replace("</", "<\\/")
    html = template.replace("__FONTS__", fonts_css)
    html = html.replace("__LOGO_B64__", logo_b64)
    return html.replace("__PAYLOAD__", blob)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pz_sankey",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=Path(DEFAULT_TAXONOMY_CSV_FILENAME), help="taxonomy CSV (default: bundled)")
    ap.add_argument("--out", type=Path, default=Path("planktonzilla_sankey_flow.html"), help="output HTML path")
    ap.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="per-(dataset,proposed_label,root_class) counts JSON (default: ./samples.json when it exists)",
    )
    ap.add_argument(
        "--dataset-repo",
        default=None,
        help="scan this HF dataset for fresh image counts instead of reading a JSON "
        f"(e.g. {DEFAULT_PLANKTONZILLA_DATASET_REPO_ID})",
    )
    ap.add_argument("--save-samples", type=Path, default=None, help="write the scanned counts to this JSON for reuse")
    ap.add_argument("--workers", type=int, default=16, help="how many dataset shards are read concurrently in a scan")
    ap.add_argument("--no-samples", action="store_true", help="skip image counts entirely; weight ribbons by label mappings")
    ap.add_argument("--logo-url", default=DEFAULT_LOGO_URL, help="official Inria lockup SVG to embed")
    ap.add_argument("--no-assets", action="store_true", help="do not fetch fonts/logo (offline; uses the fallback stack)")
    ap.add_argument("--open", dest="open_browser", action="store_true", help="open the result in a browser when done")
    return ap


def resolve_counts(args: argparse.Namespace) -> Counter:
    """Decide where the per-class image counts come from, following the CLI's precedence.

    Explicit ``--no-samples`` wins, then an explicit ``--samples-json``, then an explicit
    ``--dataset-repo`` scan, and finally a ``./samples.json`` sitting next to the caller.
    With none of those the page is built from label mappings alone.
    """
    if args.no_samples:
        return Counter()
    if args.samples_json is not None:
        if not args.samples_json.exists():
            raise SystemExit(f"error: samples JSON not found: {args.samples_json}")
        return load_sample_counts(args.samples_json)
    if args.dataset_repo:
        return scan_dataset(args.dataset_repo, args.workers)
    fallback = Path("samples.json")
    if fallback.exists():
        logger.info("Using %s for image counts (pass --no-samples to ignore it).", fallback)
        return load_sample_counts(fallback)
    logger.warning("No image counts available — ribbons will be weighted by label mappings only.")
    return Counter()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    if not args.csv.exists():
        raise SystemExit(f"error: taxonomy CSV not found: {args.csv}")
    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info("Read %d taxonomy rows from %s.", len(rows), args.csv)

    counts = resolve_counts(args)
    if args.save_samples is not None:
        if counts:
            save_sample_counts(counts, args.save_samples)
        else:
            logger.warning("--save-samples ignored: there are no counts to save.")

    payload = build_ribbons(rows, counts)
    meta = payload["meta"]

    fonts_css = "" if args.no_assets else fetch_fonts()
    logo_b64 = "" if args.no_assets else fetch_logo(args.logo_url)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = assemble(template, payload, fonts_css, logo_b64)
    leftover = [p for p in PLACEHOLDERS if p in html]
    if leftover:
        logger.error("Template placeholders were not all substituted: %s", leftover)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    logger.info(
        "Wrote %s — %d KB · %d ribbons · %d datasets · %s images · %s label mappings.",
        args.out,
        len(html.encode("utf-8")) // 1024,
        meta["n_ribbons"],
        meta["n_datasets"],
        f"{meta['total_images']:,}",
        f"{meta['total_mappings']:,}",
    )
    if args.open_browser:
        webbrowser.open(args.out.resolve().as_uri())
        logger.info("Opened %s", args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
