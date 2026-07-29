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

The page names the dataset it describes and links it back to the Hub, and stamps its own
provenance — the dataset version (resolved from the Hub, or pinned) and the UTC build time —
so a downloaded copy still says which data it was built from and when.

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

    # Name a different dataset on the page, with its version pinned instead of read from the Hub
    pz_sankey --dataset-name org/plankton-9K --dataset-version v1.2

The same CLI is available via ``python -m planktonzilla.planktonzilla_dataset.sankey``.
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
import webbrowser
from collections import Counter
from datetime import UTC, datetime
from html import escape  # ``html`` itself is a local name throughout this module
from pathlib import Path

from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
    DEFAULT_TAXONOMY_CSV_FILENAME,
    TAXONOMY_RANKS,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "sankey_flow.html"
PLACEHOLDERS = ("__FONTS__", "__LOGO_B64__", "__PAYLOAD__", "__DATASET_NAME__", "__DATASET_REPO__", "__DATASET_URL__")
HF_DATASET_BASE_URL = "https://huggingface.co/datasets/"

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
    ``load_sample_counts`` and ``scan_dataset`` produce.
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


# --------------------------------------------------------------- dataset scan
def scan_dataset(repo_id: str, workers: int, retries: int = 4) -> Counter:
    """Aggregate per-(dataset, proposed_label_lower, root_class) image counts from the HF dataset.

    Driven by the HuggingFace ``datasets`` library in streaming mode with column projection, so only
    the three metadata columns are read — the image bytes are never downloaded. Each shard is handled
    as its own task: ``split_dataset_by_node(rank=i, world_size=num_shards)`` assigns exactly one
    whole, disjoint shard per rank, and the shards are read concurrently through a thread pool. The
    shard count is discovered from the dataset itself, so a new version with a different number of
    shards still works, and ``--workers`` keeps its "how many shards are read at once" meaning.

    ``datasets``/``huggingface_hub`` already auto-retry transient HTTP failures internally; the outer
    ``retries`` loop here only re-runs a shard whose stream still fails after those retries.
    """
    import concurrent.futures as cf

    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node

    base = (
        load_dataset(repo_id, split="train", streaming=True)
        .select_columns(["dataset", "proposed_label", "root_class"])
        .with_format("arrow")
    )
    n_shards = base.num_shards
    logger.info("Scanning %d dataset shards of %s for per-class image counts…", n_shards, repo_id)

    def read_shard(rank: int) -> Counter:
        last: Exception | None = None
        for k in range(retries):
            try:
                node = split_dataset_by_node(base, rank=rank, world_size=n_shards)
                part: Counter = Counter()
                for tbl in node.iter(batch_size=50_000):
                    ds = [(x or "").strip() for x in tbl.column("dataset").to_pylist()]
                    pl = [(x or "").strip().lower() for x in tbl.column("proposed_label").to_pylist()]
                    rc = [(x or "").strip() for x in tbl.column("root_class").to_pylist()]
                    part.update(zip(ds, pl, rc))
                return part
            except Exception as exc:
                last = exc
                time.sleep(2 * (k + 1))
        raise RuntimeError(f"failed to read shard {rank} of {repo_id!r}: {last}")

    agg: Counter = Counter()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, n_shards))) as ex:
        for part in ex.map(read_shard, range(n_shards)):
            agg.update(part)
            done += 1
            if done % 20 == 0 or done == n_shards:
                logger.info("  %d/%d shards scanned (%d rows so far)", done, n_shards, sum(agg.values()))
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


def save_sample_counts(counts: Counter, path: Path) -> None:
    """Write a per-(dataset, proposed_label, root_class) counts JSON — the inverse of load_sample_counts."""
    rows = sorted(counts.items(), key=lambda item: (item[0][0], item[0][2], item[0][1]))
    doc = {"counts": [{"dataset": ds, "proposed_label": pl, "root_class": rc, "n": int(n)} for (ds, pl, rc), n in rows]}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    logger.info("Saved %d classes (%d images) to %s.", len(counts), sum(counts.values()), path)


# ----------------------------------------------------------------- asset fetch
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


def fetch_dataset_metadata(repo_id: str) -> dict:
    """Return ``{'version', 'revision', 'modified'}`` for a Hub dataset, or ``{}`` if unreachable.

    A Hub dataset is versioned by commit, so the version reported here is the first of: an
    explicit ``version:`` in the dataset card, a tag pointing at the current commit, and
    otherwise the short commit sha. ``revision`` is always the full sha and ``modified`` the
    date the dataset was last written, both of which the page shows as the tile's tooltip.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id)
    except Exception as exc:
        logger.warning("Hub metadata for %s is unavailable (%s); the page will omit the dataset version.", repo_id, exc)
        return {}

    sha = info.sha or ""
    card = info.card_data.to_dict() if info.card_data else {}
    version = str(card.get("version") or "").strip()
    if not version:
        try:  # a release tag on the published commit is a better version than its sha
            refs = api.list_repo_refs(repo_id, repo_type="dataset")
            version = next((t.name for t in refs.tags if t.target_commit == sha), "")
        except Exception as exc:
            logger.debug("Could not list refs for %s (%s); falling back to the commit sha.", repo_id, exc)
    meta = {
        "version": version or sha[:7],
        "revision": sha,
        "modified": info.last_modified.date().isoformat() if info.last_modified else "",
    }
    logger.info("Hub metadata for %s: version %s (revision %s, last modified %s).", repo_id, *meta.values())
    return meta


# ------------------------------------------------------------------- provenance
def dataset_url(dataset: str) -> str:
    """Return the browsable URL for ``dataset``.

    A bare repo id (``org/name``) resolves to its page on the HuggingFace Hub; anything already
    absolute is passed through untouched, so a dataset published elsewhere can still be linked.
    """
    if dataset.startswith(("http://", "https://")):
        return dataset
    return f"{HF_DATASET_BASE_URL}{dataset}"


def dataset_name(dataset: str) -> str:
    """Return the bare dataset name — the last path segment of a repo id or URL, org stripped."""
    return dataset.rstrip("/").rsplit("/", 1)[-1]


def provenance(dataset_repo: str, *, version: str = "", offline: bool = False) -> dict:
    """Return the build-provenance block the page shows in its ``Dataset version`` / ``Generated`` tiles.

    ``generated_at`` is stamped locally in UTC to the second, so it is always present. The dataset
    version is resolved from the Hub (see ``fetch_dataset_metadata``) *unless* it is pinned with an
    explicit ``version`` or the build is ``offline`` — in both of those cases the Hub is never
    consulted, and the revision/last-modified detail is simply absent rather than guessed.
    """
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    meta = {"generated_at": stamp, "dataset_version": version, "dataset_revision": "", "dataset_modified": ""}
    if version or offline:
        return meta
    hub = fetch_dataset_metadata(dataset_repo)
    meta["dataset_version"] = hub.get("version", "")
    meta["dataset_revision"] = hub.get("revision", "")
    meta["dataset_modified"] = hub.get("modified", "")
    return meta


# ------------------------------------------------------------------- assemble
def assemble(
    template: str,
    payload: dict,
    fonts_css: str,
    logo_b64: str,
    dataset: str = DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
) -> str:
    """Substitute the template placeholders and return the finished page.

    The dataset id reaches the page as three tokens — its full repo id, its bare name (titles and
    download filenames) and its URL — each HTML-escaped, so a repo id that spells markup is printed
    rather than injected. ``__PAYLOAD__`` is substituted LAST so that data which happens to spell a
    placeholder is never mistaken for one.
    """
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Defang any literal </script> in the data so it cannot break out of the JSON island.
    blob = blob.replace("</", "<\\/")
    html = template.replace("__FONTS__", fonts_css)
    html = html.replace("__LOGO_B64__", logo_b64)
    html = html.replace("__DATASET_URL__", escape(dataset_url(dataset)))
    html = html.replace("__DATASET_REPO__", escape(dataset))
    html = html.replace("__DATASET_NAME__", escape(dataset_name(dataset)))
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
    ap.add_argument(
        "--dataset-name",
        default=None,
        help="dataset the page is about — repo id or URL (default: --dataset-repo, else the published dataset)",
    )
    ap.add_argument(
        "--dataset-version",
        default="",
        help="pin the version shown on the page instead of resolving it from the Hub",
    )
    ap.add_argument("--logo-url", default=DEFAULT_LOGO_URL, help="official Inria lockup SVG to embed")
    ap.add_argument(
        "--no-assets",
        action="store_true",
        help="build offline: no fonts/logo fetch (fallback stack) and no Hub lookup for the dataset version",
    )
    ap.add_argument("--open", dest="open_browser", action="store_true", help="open the result in a browser when done")
    return ap


def resolve_dataset_name(args: argparse.Namespace) -> str:
    """Decide which dataset the page names: an explicit ``--dataset-name`` wins, then the scanned
    ``--dataset-repo``, and otherwise the published planktonzilla dataset."""
    return args.dataset_name or args.dataset_repo or DEFAULT_PLANKTONZILLA_DATASET_REPO_ID


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

    dataset = resolve_dataset_name(args)
    payload = build_ribbons(rows, counts)
    payload["meta"].update(provenance(dataset, version=args.dataset_version, offline=args.no_assets))
    meta = payload["meta"]

    fonts_css = "" if args.no_assets else fetch_fonts()
    logo_b64 = "" if args.no_assets else fetch_logo(args.logo_url)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = assemble(template, payload, fonts_css, logo_b64, dataset)
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
