"""
(c) Inria

Recorded FREPJ-Z (Freshwater Plankton in Japanese Lakes and Reservoirs, I. Zooplankton)
acquisition & layout facts, plus the collision-safe two-magnification merge helper.

This module is the durable, importable record of the Phase-15 acquisition spike. The
constants below were captured from the figshare public API record (article 26891563)
and from a byte-accurate ZIP64 central-directory reconnaissance of the real 963 MB
``Zooplankton_images.zip`` archive (see ``15-RESEARCH.md``). They pin:

  * the download identity (file id / URL / md5) for supply-chain integrity (ACQ-01/T-15-01),
  * the frozen 229-class-dir contract (ACQ-02),
  * the expected baseline-JPEG image format (ACQ-03),
  * the CC BY 4.0 license + Otake et al. 2024 citation + both DOIs (ACQ-04),
  * and the both-magnification merge policy (ACQ-05).

Phase 16's ``frepj.yaml`` + ``FREPJDatasetImporter`` import or mirror these constants;
this module introduces zero new dependencies (stdlib ``pathlib`` / ``shutil`` only).

Zero behavioral drift: nothing here downloads, extracts, or mutates any frozen
artifact. The merge helper operates only on paths it is explicitly handed.
"""

import shutil
from pathlib import Path

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


# --- Archive identity (figshare file record for Zooplankton_images.zip) --------------

ARCHIVE_FILENAME = "Zooplankton_images.zip"
ARCHIVE_FILE_ID = 48928918
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/48928918"
ARCHIVE_MD5 = "5c8722abb72da5035f5ee70b1fc3d27c"
FIGSHARE_ARTICLE_ID = 26891563

# The zip FILE is Zooplankton_images.zip, but its internal top-level dir is plankton_images/.
ARCHIVE_ROOT = "plankton_images"

# The two magnification roots inside ARCHIVE_ROOT, each mapped to the filename prefix
# used when both are flattened into one class dir per taxon (ACQ-05 merge policy).
MAGNIFICATION_PREFIXES = (("images_40", "40_"), ("images_100", "100_"))

# Frozen count of class dirs (union of images_40/ + images_100/); see the committed
# tests/fixtures/frepj/frepj_class_dirs.tsv for the authoritative list (ACQ-02).
N_CLASS_DIRS = 229


# --- License, provenance & citation (figshare API record — ACQ-04) -------------------

LICENSE = "cc-by-4.0"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATA_DOI = "10.57400/data.bnmnsbot.26891563.v1"
PAPER_DOI = "10.50826/bnmnsbot.50.4_159"
SOURCE_URL = "https://jstagedata.jst.go.jp/articles/dataset/26891563"

HUMAN_READABLE_NAME = "FREPJ-Z: Freshwater Plankton in Japanese Lakes and Reservoirs (I. Zooplankton)"

CITATION_APA = (
    "Otake, Y., Osone, A., Makino, W., Ito, K., Aoki, T., Miura, K., Hayakawa, K., "
    "Yoshida, T., Ichise, S., Tuji, A., & Urabe, J. (2024). "
    "High-resolution Microscopic Image Dataset of Freshwater Plankton in Japanese Lakes "
    "and Reservoirs (FREPJ): I. Zooplankton. "
    "Bulletin of the National Museum of Nature and Science, Series B, 50(4), 159-164. "
    "https://doi.org/10.50826/bnmnsbot.50.4_159"
)

# Built via implicit string concatenation so no source line exceeds ruff's 128-char
# limit; the resulting text is byte-identical to the BibTeX recorded in 15-RESEARCH.md.
CITATION_BIBTEX = (
    "@article{dataset:frepj,\n"
    "  title   = {High-resolution Microscopic Image Dataset of Freshwater Plankton "
    "in Japanese Lakes and Reservoirs (FREPJ): I. Zooplankton},\n"
    "  author  = {Otake, Yurie and Osone, Aoi and Makino, Wataru and Ito, Koichi "
    "and Aoki, Takafumi and Miura, Kanta and Hayakawa, Kazuhide and Yoshida, Takehito "
    "and Ichise, Satoshi and Tuji, Akihiro and Urabe, Jotaro},\n"
    "  journal = {Bulletin of the National Museum of Nature and Science, Series B},\n"
    "  volume  = {50}, number = {4}, pages = {159--164}, year = {2024},\n"
    "  doi     = {10.50826/bnmnsbot.50.4_159}\n"
    "}\n"
)


# --- Expected image format (ACQ-03 reconnaissance profile) ----------------------------

EXPECTED_IMAGE_FORMAT = "JPEG"
EXPECTED_IMAGE_MODE = "RGB"
EXPECTED_IMAGE_BIT_DEPTH = 8

# Import-time normalization (ACQ-03, LOCKED by Plan 15-02's full-archive scan):
# The 18-image reconnaissance sample was uniformly baseline RGB JPEG, BUT the full
# 88,686-image scan found 5 outliers that are PNG-content / RGBA-mode files carrying a
# ``.jpg`` extension (all under images_40/; exact paths recorded in 15-02-SUMMARY.md).
# Because non-RGB modes are present, Phase 16's importer MUST ``.convert("RGB")`` every
# image before writing the imagefolder so the composite stays uniformly 3-channel.
IMPORT_NORMALIZATION = "RGB"


# --- Collision-safe both-magnification merge helper (ACQ-05 / seeds IMP-04) -----------


def frepj_merge_filename(prefix: str, original_name: str) -> str:
    """Return the magnification-prefixed filename used when merging the two roots.

    ``prefix`` is one of the values in :data:`MAGNIFICATION_PREFIXES` (``"40_"`` or
    ``"100_"``). Prefixing (rather than sub-foldering) keeps the taxon as a single
    class while resolving cross-magnification basename collisions (the reconnaissance
    found 985 real shared basenames across 20 shared taxa) and encodes the
    magnification as the Phase-17 geodata join key.
    """
    return f"{prefix}{original_name}"


def merge_two_magnification_roots(images_40_dir: str | Path, images_100_dir: str | Path, dest_dir: str | Path) -> int:
    """Flatten the two magnification roots into one class dir per taxon (ACQ-05).

    For each magnification root (``images_40`` → ``"40_"``, ``images_100`` → ``"100_"``),
    iterate its taxon dirs in :func:`sorted` order (determinism, IMP-03), create
    ``dest_dir/<taxon>`` exactly once per taxon (ONE class per taxon — magnification is
    per-image metadata, not a doubled class; ACQ-05 / milestone decision), and copy each
    ``*.jpg`` to ``dest_dir/<taxon>/<prefix><name>``.

    Copies are guarded by an explicit ``if not target.exists()`` check so no file is ever
    silently overwritten (IMP-01). Taxon dir names are copied VERBATIM — commas are
    filesystem-legal ``Class,Order,Family,Genus,Species`` tuples and are NEVER sanitized
    (T-15-02).

    Args:
        images_40_dir: Path to the ``images_40`` magnification root.
        images_100_dir: Path to the ``images_100`` magnification root.
        dest_dir: Destination root; created if absent.

    Returns:
        The number of image files copied.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    roots = (
        (Path(images_40_dir), MAGNIFICATION_PREFIXES[0][1]),
        (Path(images_100_dir), MAGNIFICATION_PREFIXES[1][1]),
    )
    for root, prefix in roots:
        if not root.is_dir():
            logger.info(f"Magnification root «{root}» absent; skipping.")
            continue
        for taxon_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # Taxon dir name copied verbatim — commas preserved, never sanitized.
            dest_taxon = dest_dir / taxon_dir.name
            dest_taxon.mkdir(exist_ok=True)
            for image in sorted(taxon_dir.glob("*.jpg")):
                target = dest_taxon / frepj_merge_filename(prefix, image.name)
                if not target.exists():
                    shutil.copy2(image, target)
                    copied += 1
                else:
                    logger.warning(f"Refusing to overwrite existing «{target}»; skipped.")
    return copied
