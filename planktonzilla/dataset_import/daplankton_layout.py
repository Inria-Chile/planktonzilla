"""
(c) Inria

Recorded DAPlankton (multi-instrument plankton benchmark, Batrakhanov et al. 2024)
acquisition & layout facts, plus the collision-safe five-domain merge helper.

Every constant below was captured from the live Fairdata/Etsin deposit and from a full
enumeration of the real archive's central directory on 2026-08-27 — not inferred from
the paper. The 2,739,982,748-byte Download-API package was fetched end to end and its
sha256 matched the checksum the API declares, then all 112,040 inner entries were
listed. What that enumeration pinned:

  * the download identity (Fairdata PID / DOI / landing page) for supply-chain integrity,
  * the DOUBLY-NESTED archive shape, which is the one thing that breaks a naive import,
  * the frozen 44-class-dir contract (see ``tests/fixtures/daplankton/daplankton_class_dirs.tsv``),
  * the two image formats actually present (CytoSense JPEG, IFCB/FlowCam PNG),
  * the CC BY 4.0 license + Batrakhanov et al. 2024 citation,
  * and the five-domain merge policy (see :data:`DOMAIN_PREFIXES`).

Nothing here downloads, extracts, or mutates anything; the merge helper operates only on
paths it is explicitly handed. ``DAPlanktonDatasetImporter`` imports these constants and
``daplankton.yaml`` mirrors the deposit-identity ones as literals — Hydra cannot read a
Python module — so the two are pinned together by
``tests/test_dataset_import_configs.py::test_daplankton_is_configured_for_fairdata``
rather than trusted to stay in step.
"""

import shutil
from pathlib import Path

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


# --- Deposit identity (Fairdata/Etsin record) ----------------------------------------

# The Etsin dataset UUID, which is what the Download API's `cr_id` contract wants and
# therefore what `dataset_import.fairdata_pid` must be set to. NOT the DOI form: the
# Metax record's `persistent_identifier` field holds
# "doi:10.23729/32583bd0-38cd-4532-a8d6-fc9dc5967dce", which is the CITATION identifier
# and 404s against the download service. Measured 2026-08-27: the UUID against
# etsin.fairdata.fi/api/download/requests?cr_id=... returns 200 with a ready package,
# the DOI form does not.
FAIRDATA_PID = "a53a55a9-a591-404a-a372-d657d7efb89f"

SOURCE_URL = f"https://etsin.fairdata.fi/dataset/{FAIRDATA_PID}"
DATA_DOI = "10.23729/32583bd0-38cd-4532-a8d6-fc9dc5967dce"
PAPER_ARXIV_ID = "2402.05615"

# The single IDA file of the deposit, and its sha256 as published by Metax
# (metax.fairdata.fi/v3/datasets/<pid>/files). Recorded for provenance, NOT verified at
# import time: it is the hash of the INNER archive, while what the Download API hands
# over is an on-demand *package* wrapping it. The package's own hash is a property of
# that packaging run and changes if Fairdata re-packages, so it is deliberately not
# pinned here — this one should not.
INNER_ARCHIVE_NAME = "DAPlankton.zip"
INNER_ARCHIVE_SHA256 = "765aa2204248454751f7f39f17a6b8067c34690fb82ebaf7ed8818a1f7b2abbf"
INNER_ARCHIVE_SIZE = 2763526670

# THE GOTCHA. The Download API package is a zip whose ONLY member is the IDA file, which
# is itself a zip. Extracting once yields a zip, not class folders, so the importer must
# unwrap twice. Verified: zipfile.ZipFile(package).namelist() == ["DAPlankton/DAPlankton.zip"].
ARCHIVE_ROOT = "DAPlankton"


# --- Internal layout (full central-directory enumeration) -----------------------------

# The five subset/instrument roots, each mapped to the filename prefix used when they are
# flattened into one class dir per taxon. Instruments: IFCB = Imaging FlowCytobot,
# CS = CytoSense flow cytometer, FC = FlowCam imaging microscope. FlowCam imaged the
# cultures only, so DAPlankton_sea has no FC root — hence five domains, not six.
#
# NOTE the casing: the subset directories on disk are lowercase-suffixed
# ("DAPlankton_lab" / "DAPlankton_sea") even though the paper, the Etsin description and
# the archive's own readme.md all write them DAPlankton_LAB / DAPlankton_SEA.
DOMAIN_PREFIXES = (
    (("DAPlankton_lab", "CS"), "lab_cs_"),
    (("DAPlankton_lab", "FC"), "lab_fc_"),
    (("DAPlankton_lab", "IFCB"), "lab_ifcb_"),
    (("DAPlankton_sea", "CS"), "sea_cs_"),
    (("DAPlankton_sea", "IFCB"), "sea_ifcb_"),
)

# Frozen counts from the enumeration; the authoritative per-class table is
# tests/fixtures/daplankton/daplankton_class_dirs.tsv.
#
# 15 lab classes + 31 sea classes = 46 class-domain label slots, but only 44 DISTINCT
# names: Aphanizomenon_flosaquae and Pseudopedinella_sp occur in both subsets. The merge
# below folds those two into one class dir each, which is why N_CLASS_DIRS is 44 and not 46.
N_CLASS_DIRS = 44
N_LAB_CLASSES = 15
N_SEA_CLASSES = 31
N_IMAGES = 111924
N_LAB_IMAGES = 47471
N_SEA_IMAGES = 64453

# Both are genuinely present and neither is a stray: every CytoSense folder is JPEG
# (26,018 files) and every IFCB/FlowCam folder is PNG (85,906 files). A glob for one
# extension silently drops a whole instrument, so the merge takes both.
IMAGE_SUFFIXES = (".jpg", ".png")


# --- License, provenance & citation ---------------------------------------------------

LICENSE = "cc-by-4.0"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

HUMAN_READABLE_NAME = "DAPlankton: Benchmark Dataset for Multi-instrument Plankton Recognition"

CITATION_APA = (
    "Batrakhanov, D., Eerola, T., Kraft, K., Haraguchi, L., Lensu, L., Suikkanen, S., "
    "Camarena-Gómez, M. T., Seppälä, J., & Kälviäinen, H. (2024). "
    "DAPlankton: Benchmark Dataset for Multi-instrument Plankton Recognition via "
    "Fine-grained Domain Adaptation [Data set]. LUT University. "
    f"https://doi.org/{DATA_DOI}"
)

# Implicit string concatenation so no line exceeds ruff's 128-char limit.
CITATION_BIBTEX = (
    "@inproceedings{dataset:daplankton,\n"
    "  title     = {DAPlankton: Benchmark Dataset for Multi-instrument Plankton "
    "Recognition via Fine-grained Domain Adaptation},\n"
    "  author    = {Batrakhanov, Daniel and Eerola, Tuomas and Kraft, Kaisa "
    "and Haraguchi, Lumi and Lensu, Lasse and Suikkanen, Sanna "
    "and Camarena-G{\\'o}mez, Mar{\\'i}a Teresa and Sepp{\\\"a}l{\\\"a}, Jukka "
    'and K{\\"a}lvi{\\"a}inen, Heikki},\n'
    "  booktitle = {IEEE International Conference on Image Processing (ICIP)},\n"
    "  year      = {2024},\n"
    f"  doi       = {{{DATA_DOI}}},\n"
    f"  eprint    = {{{PAPER_ARXIV_ID}}}\n"
    "}\n"
)


# --- Collision-safe five-domain merge helper -----------------------------------------


def daplankton_merge_filename(prefix: str, original_name: str) -> str:
    """Return the domain-prefixed filename used when merging the five domain roots.

    ``prefix`` is one of the values in :data:`DOMAIN_PREFIXES` (``"lab_cs_"``,
    ``"sea_ifcb_"``, …). Prefixing rather than sub-foldering keeps the taxon as a single
    class while making the subset and the imaging instrument recoverable from
    ``original_path`` in the consolidated dataset — the same policy FREPJ-Z applies to
    its two magnifications.

    It is also load-bearing, not cosmetic: image basenames repeat the class name plus a
    5-digit counter that RESTARTS in every class-domain folder, so
    ``Aphanizomenon_flosaquae00001`` exists five times over. Without the prefix the merge
    would collide on thousands of basenames.
    """
    return f"{prefix}{original_name}"


def merge_domain_roots(archive_root: str | Path, dest_dir: str | Path) -> int:
    """Flatten the five subset/instrument roots into one class dir per taxon.

    For each ``(subset, instrument)`` root in :data:`DOMAIN_PREFIXES`, iterate its class
    dirs in :func:`sorted` order (determinism), create ``dest_dir/<class>`` once per
    class — ONE class per taxon, since subset and instrument are per-image provenance
    rather than a five-fold multiplication of the label space — and copy every image to
    ``dest_dir/<class>/<prefix><name>``.

    Copies are guarded by an explicit ``if not target.exists()`` check, so no file is
    ever silently overwritten. A root that is absent is logged and skipped rather than
    raising: ``DAPlankton_sea/FC`` legitimately does not exist.

    Args:
        archive_root: The ``DAPlankton/`` directory holding the subset roots.
        dest_dir: Destination imagefolder root; created if absent.

    Returns:
        The number of image files COPIED BY THIS CALL — not the number present in
        ``dest_dir`` afterwards. The two differ on any re-run over a populated
        imagefolder, where every file is skipped and this returns 0 while the tree is
        complete, so a completeness check must count the destination instead.
    """
    archive_root = Path(archive_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for (subset, instrument), prefix in DOMAIN_PREFIXES:
        root = archive_root / subset / instrument
        if not root.is_dir():
            logger.info(f"Domain root «{root}» absent; skipping.")
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dest_class = dest_dir / class_dir.name
            dest_class.mkdir(exist_ok=True)
            images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            for image in images:
                target = dest_class / daplankton_merge_filename(prefix, image.name)
                if not target.exists():
                    shutil.copy2(image, target)
                    copied += 1
                else:
                    # DEBUG, not WARNING, and counted rather than enumerated: an already-present
                    # file is the NORMAL state of `refresh=rebuild`, which re-enters this function
                    # over a populated imagefolder (only `redownload` clears it first). At full
                    # scale that path skips all 111,924 files, and one warning record each buries
                    # every other line the run emits.
                    logger.debug(f"«{target}» already present; not overwritten.")
                    skipped += 1
    if skipped:
        logger.info(f"Left {skipped} already-present DAPlankton image(s) untouched; nothing was overwritten.")
    return copied


def find_archive_root(extraction_root: str | Path) -> Path:
    """Locate the directory holding the subset roots, wherever the double unwrap put it.

    Located rather than hard-coded, for the same reason :func:`find_class_root` exists:
    the package wraps the deposit, the deposit wraps the archive, and each unwrap adds a
    ``DAPlankton/`` level — a real extraction lands the subsets at
    ``<root>/DAPlankton/DAPlankton/`` — so any fixed path is one re-release away from
    silently matching nothing and producing an empty imagefolder.

    A directory qualifies when it holds at least one of the subset dirs named in
    :data:`DOMAIN_PREFIXES`. The shallowest match wins, so a nested duplicate cannot
    displace the real one.

    Args:
        extraction_root: Directory the archive was extracted into.

    Returns:
        The directory whose children are the ``DAPlankton_lab`` / ``DAPlankton_sea`` roots.

    Raises:
        FileNotFoundError: If no directory below ``extraction_root`` holds a subset root.
    """
    extraction_root = Path(extraction_root)
    subsets = {subset for (subset, _instrument), _prefix in DOMAIN_PREFIXES}

    candidates = [extraction_root, *(p for p in extraction_root.rglob("*") if p.is_dir())]
    matches = [p for p in candidates if any((p / subset).is_dir() for subset in subsets)]
    if not matches:
        raise FileNotFoundError(
            f"No DAPlankton subset root ({', '.join(sorted(subsets))}) found under «{extraction_root}». "
            "The archive is doubly nested — the Fairdata package holds "
            f"{ARCHIVE_ROOT}/{INNER_ARCHIVE_NAME}, which must itself be unzipped — so this usually means "
            "the second unwrap did not happen."
        )
    return min(matches, key=lambda p: (len(p.relative_to(extraction_root).parts), str(p)))
