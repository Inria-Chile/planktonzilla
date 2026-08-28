"""
(c) Inria

Recorded Tara Pacific acquisition & layout facts for the four quantitative imaging
datasets of Mériguet et al. (2025, ESSD 17, 2761-2792), plus the committed
``ecotaxa_taxon_id -> class dir`` map that makes their import deterministic.

Why this module exists at all
-----------------------------
The four SEANOE deposits named by the paper (DOIs 10.17882/102694, /102697, /102336,
/102537) hold **no images**. Each one is an EcoTaxa TSV export: per-object metadata and
morphological features, one ``.tsv`` per sample inside a ``.zip``, verified by opening
every archive on 2026-08-26. The deposits' own abstracts say where the images are —
"All images and their taxonomic annotations are available in the open-access EcoTaxa
project at these links" — and then list the seven public EcoTaxa projects recorded in
:data:`SOURCES` below.

So, unlike every other source in this repository, a Tara Pacific import has no archive
to download. It reads a per-object MANIFEST from EcoTaxa's public read API and then
fetches each object's vignette from EcoTaxa's public vault. That work lives in
:mod:`planktonzilla.dataset_import.ecotaxa_client` (the network seam) and
:mod:`planktonzilla.dataset_import.tara_pacific_importer` (the importer); this module is
the network-free record of the facts both of them need, and the one place the
per-deposit citation/licence constants are written down.

One upstream defect is worth recording because it is what settled the design: the DeckNet
deposit's ``100% > 501 pixels`` archive (``.../00915/102697/data/114288.zip``, 281 669 134
bytes, the full length its own ``Content-Length`` advertises) is CORRUPT — its end-of-
central-directory offset overshoots the file by exactly 4 000 000 bytes, ``unzip`` reports
"missing 4000000 bytes in zipfile", and the entries straddling the gap cannot be read.
The EcoTaxa route is unaffected: it never touches that archive.

What is pinned here
-------------------
  * the seven EcoTaxa project ids, grouped into the four sources (ACQ-01),
  * the object counts each project reports and each deposit publishes (ACQ-02),
  * the frozen ``ecotaxa_taxon_id -> class dir`` map in ``tara_pacific_classes.tsv``,
    which is what makes ``Raw_Labels`` stable while EcoTaxa keeps renaming taxa (ACQ-03),
  * the CC BY 4.0 licence, the SEANOE DOI and the citation of each deposit (ACQ-04).

Nothing here downloads, writes or mutates anything: stdlib only, and the only file it
reads is the committed TSV beside it.
"""

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

# --- The committed class-dir map ------------------------------------------------------

# ``dataset``/``class_dir``/``ecotaxa_taxon_id``, one row per (source, taxon) pair. Shipped
# in the package rather than under tests/ because the IMPORTER reads it: a class dir is
# named from this map keyed by the manifest's ``obj.classif_id``, never from the live
# ``txo.display_name``.
#
# That indirection is the whole point. EcoTaxa renames taxa in place, so the display name
# of one taxon id is not stable over time — the SEANOE exports (2024) and the live API
# (2026-08-26) disagree on 12+ labels per source for the SAME taxa: `Copepoda<Maxillopoda`
# -> `Copepoda<Multicrustacea`, `detritus` -> `detritus<not-living`, `eudoxie<Diphyidae` ->
# `eudoxid<Diphyidae`, `Tintinnida` -> `Tintinnina<Choreotrichida`. Naming class dirs from
# the live value would silently change every `Raw_Labels` join key — and so every row's
# taxonomy — the next time upstream renames something. Pinning the spelling here turns that
# from silent corruption into a reported rename (see `reconcile_display_names`).
CLASSES_TSV = Path(__file__).parent / "tara_pacific_classes.tsv"

CLASSES_TSV_COLUMNS = ("dataset", "class_dir", "ecotaxa_taxon_id")


# --- Sources: which EcoTaxa projects make up each planktonzilla source (ACQ-01/02) ----

# name -> the facts a build needs. `projects` is ordered, and the manifest is written and
# read in that order, so a rebuild produces byte-identical files.
#
# `deposit_objects` is what the SEANOE deposit's abstract publishes; `ecotaxa_objects` is
# what the EcoTaxa projects reported on 2026-08-26. They are NOT expected to be equal —
# EcoTaxa is the live annotation database and the deposit is a 2024 snapshot of it, so
# objects get re-annotated and taxa merged in between. Both are recorded so a drift of a
# different KIND (a project emptied, halved, or swapped) is visible rather than inferred.
SOURCES = {
    "tara_pacific_bongo": {
        "projects": (11370, 11369),
        "deposit_doi": "10.17882/102694",
        "deposit_objects": 380769,
        "ecotaxa_objects": 380769,
        "n_class_dirs": 137,
        "instrument": "FlowCam",
        "net": "Bongo net",
    },
    "tara_pacific_decknet": {
        "projects": (11353, 11341),
        "deposit_doi": "10.17882/102697",
        "deposit_objects": 1581613,
        "ecotaxa_objects": 1581623,
        "n_class_dirs": 132,
        "instrument": "FlowCam",
        "net": "Deck net",
    },
    "tara_pacific_hsn": {
        "projects": (11292,),
        "deposit_doi": "10.17882/102336",
        "deposit_objects": 256352,
        "ecotaxa_objects": 256352,
        "n_class_dirs": 159,
        "instrument": "ZooScan",
        "net": "High-Speed Net",
    },
    "tara_pacific_manta": {
        "projects": (1344, 1345),
        "deposit_doi": "10.17882/102537",
        "deposit_objects": 137497,
        "ecotaxa_objects": 135876,
        "n_class_dirs": 172,
        "instrument": "ZooScan",
        "net": "Manta net",
    },
}

# Every EcoTaxa project the four sources draw from, in source order. Used by the tests and
# by the pre-flight report; a project appears exactly once.
ALL_PROJECTS = tuple(project for source in SOURCES.values() for project in source["projects"])


# --- Licence, provenance & citation (SEANOE deposit records — ACQ-04) -----------------

# All four deposits are CC BY 4.0, read from each SEANOE record on 2026-08-26. The slug is
# the one `configs/dataset_import/tara_pacific_*.yaml` declares and
# `constants.DATASET_LICENSES` records; tests/test_tara_pacific_layout.py pins the three
# against each other.
LICENSE = "cc-by-4.0"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

# The data paper all four deposits belong to (the GitHub issue this milestone answers).
PAPER_DOI = "10.5194/essd-17-2761-2025"
PAPER_URL = "https://essd.copernicus.org/articles/17/2761/2025/essd-17-2761-2025.html"

ECOTAXA_PROJECT_URL_TEMPLATE = "https://ecotaxa.obs-vlfr.fr/prj/{project_id}"
SEANOE_DOI_URL_TEMPLATE = "https://doi.org/{doi}"


def deposit_url(source_name: str) -> str:
    """The DOI URL of one source's SEANOE deposit (its ``source_url`` in the card)."""
    return SEANOE_DOI_URL_TEMPLATE.format(doi=SOURCES[source_name]["deposit_doi"])


def ecotaxa_project_urls(source_name: str) -> tuple[str, ...]:
    """The public EcoTaxa project pages one source's images come from."""
    return tuple(ECOTAXA_PROJECT_URL_TEMPLATE.format(project_id=p) for p in SOURCES[source_name]["projects"])


# --- Reading the committed class-dir map ----------------------------------------------


@lru_cache(maxsize=4)
def _load_class_map(path: str) -> dict:
    """Cached body of :func:`load_class_map`, keyed by resolved path string."""
    per_source: dict[str, dict[int, str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != CLASSES_TSV_COLUMNS:
            raise ValueError(f"«{path}» must have exactly the columns {CLASSES_TSV_COLUMNS}, got {reader.fieldnames}")
        for row in reader:
            taxon_id = int(row["ecotaxa_taxon_id"])
            class_dir = row["class_dir"]
            if not class_dir:
                raise ValueError(f"«{path}» has a blank class_dir for taxon {taxon_id}")
            bucket = per_source.setdefault(row["dataset"], {})
            if taxon_id in bucket and bucket[taxon_id] != class_dir:
                raise ValueError(
                    f"«{path}» maps taxon {taxon_id} of «{row['dataset']}» to both «{bucket[taxon_id]}» and «{class_dir}»"
                )
            bucket[taxon_id] = class_dir
    return per_source


def load_class_map(source_name: str, path: str | Path = CLASSES_TSV) -> dict[int, str]:
    """Return ``{ecotaxa_taxon_id: class_dir}`` for one source.

    Raises:
        KeyError: If ``source_name`` has no rows in the map — always a bug, since a
            source with no classes would import into an empty imagefolder.
    """
    per_source = _load_class_map(str(Path(path).resolve()))
    try:
        return dict(per_source[source_name])
    except KeyError:
        raise KeyError(
            f"«{source_name}» has no rows in {path}. Every Tara Pacific source must declare its "
            f"class dirs there; known sources: {sorted(per_source)}."
        ) from None


def class_dirs(source_name: str, path: str | Path = CLASSES_TSV) -> tuple[str, ...]:
    """The distinct class dirs of one source, sorted — the frozen ``Raw_Labels`` set."""
    return tuple(sorted(set(load_class_map(source_name, path).values())))


# --- Manifest -> imagefolder layout ---------------------------------------------------

# One image per object, named by the EcoTaxa object id. The id is the join key the
# redefiner reads back out of `original_path` (the EcoTaxaRedefiner idiom), and it is
# globally unique, so no two objects can collide inside a class dir.
IMAGE_SUFFIX = ".jpg"


def image_file_name(object_id: int | str) -> str:
    """The imagefolder file name for one EcoTaxa object (``<objid>.jpg``)."""
    return f"{object_id}{IMAGE_SUFFIX}"


def object_id_from_file_name(file_name: str) -> str:
    """The EcoTaxa object id encoded in an imagefolder file name, or ``""``.

    The inverse of :func:`image_file_name`, tolerant by design: it is applied to whatever
    ``original_path`` holds, and an unparseable name must yield a missing id rather than
    raise in the middle of a multi-hour redefine.
    """
    stem = Path(file_name).name
    stem = stem.removesuffix(IMAGE_SUFFIX)
    return stem if stem.isdigit() else ""


def reconcile_display_names(rows, class_map: dict[int, str]) -> tuple[dict[int, str], dict[int, str]]:
    """Compare a fetched manifest against the committed map, without deciding anything.

    Args:
        rows: Manifest rows, each carrying ``classif_id`` and ``display_name``.
        class_map: ``{taxon id: committed class dir}`` from :func:`load_class_map`.

    Returns:
        ``(renamed, unknown)``. ``renamed`` maps a known taxon id to the live display name
        that no longer equals the committed one — the class dir keeps the committed
        spelling, so this is a report, never a rewrite. ``unknown`` maps a taxon id absent
        from the map to its live display name: those objects have no ``Raw_Labels`` row and
        the importer refuses to guess one for them.
    """
    renamed: dict[int, str] = {}
    unknown: dict[int, str] = {}
    for row in rows:
        taxon_id = row["classif_id"]
        live = row["display_name"]
        committed = class_map.get(taxon_id)
        if committed is None:
            if taxon_id is not None:
                unknown.setdefault(taxon_id, live)
        elif live and live != committed:
            renamed.setdefault(taxon_id, live)
    return renamed, unknown


def class_counts(rows, class_map: dict[int, str]) -> Counter:
    """``{class_dir: n objects}`` for the rows the committed map can name."""
    counts: Counter = Counter()
    for row in rows:
        class_dir = class_map.get(row["classif_id"])
        if class_dir is not None:
            counts[class_dir] += 1
    return counts
