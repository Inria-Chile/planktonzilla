"""
(c) Inria

Download, normalize, and push public plankton source datasets to HuggingFace Hub.

Defines the :class:`DatasetImporter` abstract base class and one concrete
subclass per source dataset (ZooLake, ZooScanNet, WHOI, EcoTaxa-derived sets,
etc.). The shared lifecycle is: download + extract the raw archive, normalize it
into a torchvision-style imagefolder layout (the per-source ``_prepare_imagefolder``
hook), load it as a HuggingFace ``imagefolder`` dataset, and — when gated on by
``push_to_hub`` — push the dataset and a rendered dataset card to the Hub.

This module also holds the dataset-card Jinja template and small image/zip/IO
helpers used during normalization.
"""

import concurrent.futures
import csv
import glob
import gzip
import importlib.metadata
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy2, copytree, move, rmtree
from typing import ClassVar, Dict, Final, Optional, Union
from zipfile import ZipFile, is_zipfile

import aiohttp
import numpy as np
import plotext as plt
import requests
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_dataset_builder,
)
from datasets.download import DownloadConfig, DownloadManager
from huggingface_hub import DatasetCard
from humanize import naturalsize
from PIL import Image
from rich import print as rich_print
from rich.markdown import Markdown
from tqdm import tqdm

import planktonzilla.dataset_import.public_data as public_data
from planktonzilla.dataset import compute_mean_and_std_dev
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

DATACARD_TEMPLATE = """
---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---
# Dataset *{{ pretty_name | default("Dataset Name", true) }}*
{{ dataset_description | default("[More Information Needed]", true) }}

- **Original dataset available online at:**  <{{ source_url | default("[More Information Needed]", true)}}>.
- **Original dataset license:** <{{ license | default("[More Information Needed]", true)}}>.

## Details

- **train split means (RGB):** {{ dataset_means | default("[More Information Needed]", true) }}
- **train split standard deviations (RGB):** {{ dataset_stds | default("[More Information Needed]", true) }}

{{ report_markdown | default("[More Information Needed]", true) }}

## Reference
{{ citation_apa | default("[More Information Needed]", true)}}

### BibTEX
```bibtex
{{ citation_bibtex | default("[More Information Needed]", true)}}
```

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("{{hf_org_name}}/{{hf_dataset_name}}")
```
"""


@lru_cache(maxsize=1)
def default_user_agent() -> str:
    """How this project identifies itself to the archives it downloads from.

    Identification, not impersonation: the string names the project, its version and its
    repository, so an archive maintainer seeing it in a log can tell who is fetching and
    where to complain. Nothing here pretends to be a browser.

    It exists because leaving it unset is not neutral — it means sending whatever the
    underlying library sends, and that is what gets refused. Measured against
    darchive.mblwhoilibrary.org (whoi's host) on 2026-08-04, one one-byte ranged GET per
    User-Agent:

        python-requests/2.34.2      -> connection dropped
        datasets/5.0.0; python/…    -> connection dropped
        aiohttp default / no UA     -> connection dropped
        Chrome 140                  -> connection dropped
        curl/8.7.1                  -> 206, application/zip, 1,158,978,503 bytes
        planktonzilla/<v> (+repo)   -> 206, application/zip

    So the host blocks known library User-Agents (and, oddly, a browser one) rather than
    automated access as such: identifying honestly is enough, and all nine whoi archives
    are reachable again.
    """
    try:
        version = importlib.metadata.version("planktonzilla")
    except importlib.metadata.PackageNotFoundError:
        # Running from a source tree that was never installed. The version is decoration
        # in this string; the identity and the contact URL are the parts that matter.
        version = "unknown"
    return f"planktonzilla/{version} (+https://github.com/Inria-Chile/planktonzilla)"


def is_dir_empty(dir: Path) -> bool:
    """Return True if ``dir`` is None, missing, or contains no entries."""
    if dir and dir.exists() and os.listdir(dir):
        return False
    return True


def strip_ansi_codes(text):
    """
    Removes ANSI escape sequences from a string.
    """
    reaesc = re.compile(r"\x1b[^m]*m")
    return reaesc.sub("", text)


def copytree_filtered(src: Path, dst: Path):
    """Copy a directory tree into ``dst``, skipping macOS junk (``._*``, ``.DS_Store``).

    Merges into ``dst`` if it already exists (``dirs_exist_ok=True``).
    """
    copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("._*", ".DS_Store"),
    )


def report_dataset_content(huggingface_dataset: Dataset | DatasetDict) -> str:
    """Build a Markdown report of per-class sample counts for each split.

    Renders an ANSI-free ``plotext`` bar chart of the label histogram per split,
    suitable for embedding in the dataset card. Accepts a single ``Dataset`` or a
    ``DatasetDict`` (one section per split).

    Returns:
        The Markdown report string.
    """

    def report_split(dataset: Dataset, split_name: str | None = None) -> str:
        class_idxs, class_counts = np.unique(dataset["label"], return_counts=True)

        content = []
        for class_idx in class_idxs:
            class_name = dataset.features["label"].int2str(int(class_idx))
            content += [f"{class_idx}: {class_name}"]

        # A bare Dataset has no split to name; only the DatasetDict branch passes one.
        title = f"Label histogram for {split_name} split " if split_name else "Label histogram "
        plt.simple_bar(content, class_counts.astype(int), title=title, width=83)
        plt.show()

        return strip_ansi_codes(plt.build())

    if isinstance(huggingface_dataset, DatasetDict):
        split_reports = []
        split_reports = [
            f"**Samples per class for split `{split}`**\n ```{report_split(huggingface_dataset[split], split)}```\n"
            for split in huggingface_dataset
        ]
        return "\n".join(split_reports)
    else:
        return report_split(huggingface_dataset) + "\n"


def unzip(zip_file: Path, output_dir: Path, show_progress: bool = True):
    """Unzips a zip file showing progress.

    Args:
        zip_file (Path): file to unzip
        output_dir (Path): where to put results
    """
    with ZipFile(zip_file, "r") as zip_ref:
        for file in tqdm(
            iterable=zip_ref.namelist(),
            total=len(zip_ref.namelist()),
            desc=f"Extracting {zip_file.name} ({naturalsize(os.stat(zip_file).st_size)})",
            leave=False,
            disable=not show_progress,
        ):
            zip_ref.extract(member=file, path=output_dir)


def cleanup_imagefolder_empty_dirs(imagefolder_dir: Path) -> None:
    """Delete empty subfolders as torchvision ImageFolder crashes if a folder is empty."""
    for class_dir in os.listdir(imagefolder_dir):
        dir = imagefolder_dir / class_dir
        if dir.is_dir() and not os.listdir(dir):
            shutil.rmtree(dir)


# Suffixes treated as images when locating class folders. Suffix-only on purpose:
# find_class_root walks the whole extracted tree, so opening every candidate with PIL
# (as is_valid_image_file does) would be far too slow. Readability is checked later,
# behind check_image_file_integrity.
IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".ppm", ".webp"})

# Depth cap for find_class_root, counted from the extraction root. Every bundled source
# nests its class folders at most 4 levels down — the deepest measured is SYKE ZooScan
# 2024 at 3 ("SYKE-plankton_ZooScan_2024/images/SYKE-plankton_ZooScan_2024/<class>",
# after its nested zip is unwrapped) — so 6 leaves headroom without letting a
# pathological tree turn the scan into a full-disk walk.
MAX_CLASS_ROOT_DEPTH: Final = 6


def _subdirectories(dir: Path) -> list[Path]:
    """Immediate subdirectories of ``dir``, sorted, skipping dot-directories."""
    return sorted(p for p in dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _holds_images(dir: Path) -> bool:
    """Return True if ``dir`` directly contains at least one image file."""
    return any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in dir.iterdir())


def find_class_root(extraction_root: Path) -> Path:
    """Locate the directory whose immediate subdirectories are the class folders.

    Most importers hard-code the path from the extraction root down to the class
    folders (e.g. ``ZooScanNet/imgs``), which means a re-release that adds or renames
    a wrapper directory breaks them. This finds it instead: it scans the tree and
    returns the directory with the most immediate subdirectories that directly hold
    images, which is what a torchvision-style ``<class>/<image>`` layout looks like.

    Use it when the archive's internal layout is not pinned by a checksum, so a
    wrapper folder appearing or disappearing upstream is not a silent breakage.

    Args:
        extraction_root: Directory the archive was extracted into.

    Returns:
        The directory holding the class folders (may be ``extraction_root`` itself).

    Raises:
        RuntimeError: If no directory in the tree has an image-bearing subdirectory.
    """
    best_dir, best_count = None, 0

    queue = [(extraction_root, 0)]
    while queue:
        dir, depth = queue.pop(0)
        subdirs = _subdirectories(dir)

        # Ties keep the shallowest candidate: BFS reaches it first and the comparison
        # is strict, so a nested duplicate of the same layout cannot displace it.
        count = sum(1 for sub in subdirs if _holds_images(sub))
        if count > best_count:
            best_dir, best_count = dir, count

        if depth < MAX_CLASS_ROOT_DEPTH:
            queue.extend((sub, depth + 1) for sub in subdirs)

    if best_dir is None:
        raise RuntimeError(
            f"No class folders found under {extraction_root}: no directory within "
            f"{MAX_CLASS_ROOT_DEPTH} levels has a subdirectory containing images "
            f"({', '.join(sorted(IMAGE_SUFFIXES))})."
        )

    logger.info(f"Located {best_count} class folders under {best_dir}.")
    return best_dir


# Class-folder depths the no-explicit-splits fallback will try, shallowest first.
# 1 is the flat `<class>/<image>` layout every URL-based importer produces; 2 is the
# split `<split>/<class>/<image>` layout LenslessDatasetImporter (train/ + test/) and
# ZooLakeDatasetImporter (train_split/ + val_split/ + test_split/) produce.
IMAGEFOLDER_CLASS_DEPTHS: Final = (1, 2)


def resolve_imagefolder_glob(imagefolder_dir: Path) -> str:
    """Return a fixed-depth glob that resolves to the image FILES under ``imagefolder_dir``.

    The fallback used when no canonical ``train``/``validation``/``test`` directory is
    found. It was a single hard-coded depth-2 glob (``*/*[!._]*``), which matches only
    the flat ``<class>/<image>`` layout. On a split layout that pattern matches the class
    **directories** instead of files; ``datasets`` keeps only ``type == "file"``, so zero
    files resolved and the load died with ``Instruction "train" corresponds to no data!``.
    That made ``lensless`` and ``zoolake`` — both active registry entries — unbuildable.

    Why a depth LADDER and not a recursive ``**`` glob: ``imagefolder`` infers the label
    from each file's parent directory and only emits a ``label`` column when the matched
    files sit at a uniform depth. A recursive glob also picks up any stray image sitting
    at the imagefolder root, which breaks that uniformity — the loader then silently drops
    ``label`` entirely and ``_taxonomy_row``'s ``class_names[example["label"]]`` raises.
    A fixed depth keeps the layout uniform and keeps root-level strays out.

    Depth 1 is tried first, so every flat source resolves exactly the pattern it always
    did. Only a layout that yields no files at depth 1 — i.e. one that raises today —
    falls through to depth 2. ``original_path`` is unaffected either way: the caller's
    single-split fallback keeps ``n_splits == 1``, so it stays the last two path chunks.

    When NO depth yields a file the shallowest pattern is returned anyway — deliberately,
    to keep this function output-preserving. That is the string the caller has always
    passed to ``load_dataset``, so an empty or unreadable imagefolder still fails exactly
    where and how it did before, rather than acquiring a new failure mode here. Raising
    instead would also break every caller that never resolves the pattern against a real
    filesystem, which is what the ``load_dataset``-monkeypatching Hydra tests do.

    Args:
        imagefolder_dir: Root of the imagefolder to probe.

    Returns:
        A glob string suitable for ``load_dataset("imagefolder", data_files=...)``.
    """
    patterns = [str(imagefolder_dir.joinpath(*["*"] * depth, "[!._]*")) for depth in IMAGEFOLDER_CLASS_DEPTHS]

    for depth, pattern in zip(IMAGEFOLDER_CLASS_DEPTHS, patterns):
        if any(Path(match).is_file() for match in glob.glob(pattern)):
            if depth != 1:
                logger.info(f"Imagefolder {imagefolder_dir} nests classes {depth} levels deep; using {pattern}.")
            return pattern

    logger.warning(
        f"No image files found under {imagefolder_dir} at any tried depth ({', '.join(patterns)}). "
        f"Falling back to {patterns[0]}; the loader will report the empty result. The imagefolder "
        f"is empty, or its class folders are nested deeper than {max(IMAGEFOLDER_CLASS_DEPTHS)} "
        f"levels — extend IMAGEFOLDER_CLASS_DEPTHS if so."
    )
    return patterns[0]


class FairdataResolutionError(RuntimeError):
    """The Fairdata Download API did not yield a usable download URL.

    Always carries the manual fallback, because that is what the caller has to do next.
    """


def _find_ready_package(payload) -> str | None:
    """Pull the name of a generated, ready-to-download package out of an API response.

    The live service returns a flat object::

        {"package": "<pid>_nhungsie.zip", "status": "SUCCESS", "size": 79363785,
         "checksum": "sha256:…", "generated": "…", "initiated": "…", "dataset": "<pid>"}

    which the ``"package" in payload`` branch below handles. The nested forms are kept
    because the endpoint is undocumented and its shape is the part most likely to move.
    Tolerant about SHAPE, strict about MEANING: only a package whose status says it is
    generated is accepted, and an unrecognised shape returns None so the caller raises
    with the manual fallback rather than guessing a package name.
    """
    if not isinstance(payload, dict):
        return None

    candidates = []
    for key in ("partial", "complete", "packages"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.extend(value.values())
        elif isinstance(value, list):
            candidates.extend(value)
    # A single top-level package object, rather than a collection of them.
    if "package" in payload:
        candidates.append(payload)

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        name = entry.get("package") or entry.get("filename") or entry.get("name")
        if not name:
            continue
        status = str(entry.get("status", "SUCCESS")).upper()
        if status in ("SUCCESS", "SUCCESSFUL", "COMPLETE", "COMPLETED", "READY", "GENERATED"):
            return name

    return None


def resolve_fairdata_download_url(
    pid: str,
    *,
    api_base: str = "https://etsin.fairdata.fi/api/download",
    timeout: int = 3600,
    poll_attempts: int = 60,
    poll_interval: int = 10,
    source_url: str | None = None,
    session=None,
    sleep=time.sleep,
) -> str:
    """Resolve a Fairdata dataset PID to a direct, authorized download URL.

    Fairdata serves no stable ``.zip`` link. A dataset is downloaded by asking the
    Download API to *package* it, waiting for that, then authorizing a single-use
    download. This walks that flow:

      1. ``GET  {api_base}/requests?cr_id=<pid>``  — reuse a ready package if there is one
      2. ``POST {api_base}/requests``              — otherwise ask for one
      3. poll (1) until a package reports SUCCESS
      4. ``POST {api_base}/authorize``             — exchange it for a single-use URL
      5. return that URL

    VERIFIED end to end against the live service on 2026-08-01 with
    ``6fa42787-9772-41a5-a6fc-0dde489ed908`` (SYKE ZooScan 2024): the flow resolved and
    downloaded the full 79,363,785-byte package, whose size and name matched what step
    (1) reported. Recorded shapes::

        GET  /requests?cr_id=<pid>
          -> {"package": "<pid>_nhungsie.zip", "status": "SUCCESS", "size": 79363785,
              "checksum": "sha256:…", "generated": "…", "initiated": "…",
              "dataset": "<pid>"}
        POST /requests  {"cr_id": "<pid>"}
          -> the same object plus "created": false when a package already exists
        POST /authorize {"cr_id": "<pid>", "package": "<name>"}
          -> {"url": "https://download.fairdata.fi:443/download?token=<jwt>"}

    Two details worth keeping: the query/body parameter is ``cr_id`` (``dataset`` is
    rejected as an unknown field), and ``authorize`` returns a ready ``url`` rather than
    a bare token — ``/download`` accepts only ``token``, so the URL must not be
    reassembled from parts.

    Args:
        pid: The dataset persistent identifier (the Etsin dataset id).
        api_base: Download API root. Overridable so a contract change, a mirror or a
            test double needs no code edit.
        timeout: Per-request timeout in seconds.
        poll_attempts: How many times to poll for package generation before giving up.
        poll_interval: Seconds between polls.
        source_url: Landing page, quoted in the error so the fallback is actionable.
        session: ``requests``-compatible session; defaults to a fresh one.
        sleep: Injected for tests, so polling does not actually wait.

    Returns:
        A single-use URL that downloads the packaged dataset.

    Raises:
        FairdataResolutionError: On any unexpected response, or if the package is not
            ready within ``poll_attempts * poll_interval`` seconds.
    """
    manual_hint = (
        f"Download the archive by hand from {source_url or 'the dataset landing page'} and point "
        f"dataset_import.manual_download_local_file_names at it."
    )
    requester = session or requests

    def _json(response, what):
        if not response.ok:
            raise FairdataResolutionError(
                f"Fairdata {what} returned HTTP {response.status_code} for dataset «{pid}». {manual_hint}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise FairdataResolutionError(f"Fairdata {what} returned a non-JSON body for «{pid}». {manual_hint}") from e

    try:
        status = _json(requester.get(f"{api_base}/requests", params={"cr_id": pid}, timeout=timeout), "request status")
        package = _find_ready_package(status)

        if package is None:
            # Asking for a package that already exists is harmless — the service returns
            # the existing one with "created": false — so this is safe to reach.
            created = _json(
                requester.post(f"{api_base}/requests", json={"cr_id": pid}, timeout=timeout),
                "package request",
            )
            package = _find_ready_package(created)

            for _ in range(poll_attempts):
                if package is not None:
                    break
                sleep(poll_interval)
                status = _json(
                    requester.get(f"{api_base}/requests", params={"cr_id": pid}, timeout=timeout),
                    "request status",
                )
                package = _find_ready_package(status)

        if package is None:
            raise FairdataResolutionError(
                f"Fairdata did not report a ready package for «{pid}» after "
                f"{poll_attempts * poll_interval}s. Large datasets can take longer — raise "
                f"dataset_import.fairdata_poll_attempts, or {manual_hint[0].lower() + manual_hint[1:]}"
            )

        authorized = _json(
            requester.post(f"{api_base}/authorize", json={"cr_id": pid, "package": package}, timeout=timeout),
            "authorization",
        )
        # The service hands back a complete single-use URL. Do NOT rebuild it from
        # parts: /download accepts only `token`, and rejects `dataset` and `file` as
        # unknown fields, so a reassembled URL is a 400.
        url = authorized.get("url") if isinstance(authorized, dict) else None
        if not url:
            raise FairdataResolutionError(
                f"Fairdata authorization returned no download url for «{pid}» (package {package!r}). {manual_hint}"
            )

    except requests.RequestException as e:
        raise FairdataResolutionError(f"Could not reach the Fairdata API for «{pid}»: {e}. {manual_hint}") from e

    return url


# --------------------------------------------------------------------- pre-flight ---
#
# Checking that a source COULD be imported, without importing it. The three questions a
# build actually fails on are: is the archive reachable, is the hand-downloaded file
# there, and is what comes back a file rather than a login page. All three are
# answerable in one request per target, which is what makes a whole-registry pre-flight
# cost seconds instead of the hours the real download takes.

# HTTP statuses that mean "this server does not do HEAD", not "this file is missing".
# Kept as a named constant because the fallback below is the difference between a
# correct pre-flight and one that reports half the registry as broken: darchive
# (WHOI) and dbarchive (JEDI) both answer a plain HEAD with a refusal and the file
# with a ranged GET.
HEAD_UNSUPPORTED_STATUSES: Final = frozenset({400, 401, 403, 405, 406, 501})

# Appended to the verdict when a host refuses the client rather than reporting a missing
# file. Deliberately NOT worked around with a spoofed User-Agent: the probe's job is to
# predict what the DOWNLOAD will do, and the download is an equally non-browser-like
# Python client (datasets' DownloadManager -> fsspec -> aiohttp).
#
# Measured against darchive.mblwhoilibrary.org (whoi) on 2026-08-04, because this is
# exactly the case that looks like a false alarm — the URL opens perfectly in a browser:
#
#     curl (its own UA)              -> 302 then 206, application/zip, 1,158,978,503 bytes
#     requests, UA python-requests   -> connection dropped
#     requests, UA datasets/5.0.0…   -> connection dropped
#     requests, no UA / Chrome UA    -> connection dropped
#     aiohttp, default UA            -> ServerDisconnectedError
#     fsspec .info()/.open()         -> FileNotFoundError   <- what a real run hits first
#
# So "it works in my browser" is not evidence the run would work, which is why the hint
# says so outright rather than leaving the user to conclude the pre-flight is broken.
BLOCKED_HINT: Final = (
    "the host may be down, slow (raise check_timeout), or refusing this client. A 403 or a dropped connection is what "
    "User-Agent, IP or region filtering looks like, and such a URL often still opens fine in a BROWSER — which is not "
    "evidence a run would work, because the download speaks the same non-browser Python client this probe does. Retry "
    "from another network, or point dataset_import.manual_download_local_file_names at a copy fetched by hand"
)


@dataclass(frozen=True)
class ProbeResult:
    """The verdict on ONE thing a real import would have to obtain.

    Attributes:
        kind: How it is obtained — ``url``, ``file`` (hand-downloaded), ``bundled``
            (shipped inside the package) or ``fairdata`` (packaged on demand by the
            Etsin Download API). ``none`` means the source declares no way to get its
            data at all, which is a configuration error rather than a reachability one.
        location: The URL, path or dataset PID that was checked.
        ok: Whether a real run could obtain it.
        detail: One line for the report, e.g. ``HTTP 200, application/zip, 492.4 MB``.
        size: Total bytes when the server (or the filesystem) discloses it, else None.
            Summed across a run to estimate how much disk the build needs.
        warning: Set when the target answered but the answer is suspect — an HTML body
            where an archive was expected, a package not yet generated. Not a failure:
            it is reported and the run continues.
    """

    kind: str
    location: str
    ok: bool
    detail: str
    size: Optional[int] = None
    warning: Optional[str] = None


def _as_uri_list(value) -> list[str]:
    """Normalise a ``download_uris`` value to a list of URLs.

    Sources declare it three ways — a bare string (zoolake), a list (whoi's nine
    release archives), and the empty string (sykezooscan2024, which resolves its
    download through Fairdata instead). Iterating the bare string would probe one URL
    per character, so the string case is special-cased rather than left to ``list()``.
    Hydra hands these over as ``ListConfig`` unless converted, so no ``isinstance(...,
    list)`` test is used.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _content_type(response) -> str:
    """The bare media type of a response, without parameters (``; charset=utf-8``)."""
    return (response.headers.get("Content-Type", "").split(";")[0].strip() or "unknown").lower()


def probe_url(url: str, *, timeout: int = 30, session=None, user_agent: Optional[str] = None) -> ProbeResult:
    """Check that ``url`` would download, without downloading it.

    HEAD first, then — if the server refuses it, answers with an error, or does not
    answer at all — a one-byte ranged GET. The fallback is not defensive programming: it
    is how the JEDI archive was verified by hand on 2026-08-01 ("HTTP 206 on a range
    request, application/zip"), and several of these data portals refuse or ignore a HEAD
    while serving the file perfectly to a GET. Reporting those as unreachable would make
    the pre-flight worse than useless — it would send someone hunting a download that
    works.

    A 2xx whose body is HTML is reported as ``ok`` with a warning rather than a failure:
    it is what a login wall, a "dataset has moved" notice or an anti-bot interstitial
    looks like, and only a human can tell which.

    Args:
        url: The URL a real run would fetch.
        timeout: Per-request timeout in seconds.
        session: ``requests``-compatible session; defaults to the module-level
            ``requests``. Injected by the caller so one connection pool (and one test
            double) serves a whole registry sweep.
        user_agent: Identify as this. Callers pass the importer's ``http_user_agent``,
            the same value the real download sends — a probe that identified itself
            differently from the downloader would be answering a different question, and
            on at least one of these hosts a different answer.

    Returns:
        A :class:`ProbeResult` of kind ``url``. Never raises for a network problem —
        an unreachable host is a verdict, not an exception, since the point is to
        report every target rather than stop at the first bad one.
    """
    requester = session or requests
    headers = {"User-Agent": user_agent} if user_agent else {}

    response, head_failure, used_ranged_get = None, None, False
    try:
        response = requester.head(url, headers=headers, allow_redirects=True, timeout=timeout)
    except requests.RequestException as e:
        # NOT a verdict, so it does not return here. Measured against the live hosts on
        # 2026-08-03: NCEI (planktonset1.0) times out on a HEAD and answers the one-byte
        # GET below in under a second, and darchive (WHOI) closes the HEAD connection
        # without answering. Taking a HEAD that RAISES as the answer called 10 of the 22
        # archives broken; the GET then reached all but the 9 that are genuinely refused.
        head_failure = e

    if response is None or response.status_code in HEAD_UNSUPPORTED_STATUSES or not response.ok:
        try:
            used_ranged_get = True
            response = requester.get(
                url,
                headers={**headers, "Range": "bytes=0-0"},
                stream=True,
                allow_redirects=True,
                timeout=timeout,
            )
            # Only the headers are wanted; closing here returns the connection to the
            # pool without pulling the body of a multi-gigabyte archive.
            response.close()
        except requests.RequestException as e:
            detail = f"unreachable — {type(e).__name__}: {e}"
            if head_failure is not None:
                detail += f" (HEAD failed too: {type(head_failure).__name__})"
            return ProbeResult(kind="url", location=url, ok=False, detail=f"{detail}. {BLOCKED_HINT}")

    length = response.headers.get("Content-Length", "")
    size = int(length) if length.isdigit() else None

    if response.status_code == 206:
        # On a ranged reply Content-Length is the length of the RANGE (one byte), not
        # of the file. The total is the part after the slash in Content-Range.
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        size = int(total) if total.isdigit() else None

    content_type = _content_type(response)
    ok = response.status_code in (200, 206)

    detail = f"HTTP {response.status_code}, {content_type}"
    if size is not None:
        detail += f", {naturalsize(size)}"
    if not ok and response.status_code in (401, 403):
        detail += f". {BLOCKED_HINT}"

    # Accumulated, not assigned: these conditions co-occur. planktonset1.0's NCEI endpoint
    # triggers BOTH of the new ones at once, and the HTML test double below discloses no
    # Content-Length either — so an implementation that overwrote would silently drop the
    # warning a caller was already relying on.
    warnings = []
    if ok and content_type.startswith("text/html"):
        warnings.append(
            "the server returned an HTML page, not a file — a login wall, a moved-dataset notice or an interstitial"
        )
    if ok and used_ranged_get and response.status_code == 200:
        # A server honouring Range answers 206. Answering 200 to `Range: bytes=0-0` means
        # it ignored the header and is streaming the whole file, so NOTHING downstream can
        # resume this download: every retry restarts at byte 0. Measured on NCEI
        # (planktonset1.0) 2026-08-27, where that turns a ~1h transfer into an all-or-
        # nothing one. Reported, not failed — the URL does serve the file.
        warnings.append(
            "the server ignored Range (answered 200, not 206), so an interrupted download cannot resume "
            "and every retry restarts from the beginning"
        )
    if ok and size is None:
        # No Content-Length (typically Transfer-Encoding: chunked) means no layer can tell
        # a complete download from one that stopped early, and the target contributes 0 to
        # the run's disk estimate.
        warnings.append(
            "the server disclosed no size, so a truncated download cannot be detected and this target adds "
            "nothing to the disk estimate"
        )

    return ProbeResult(
        kind="url", location=url, ok=ok, detail=detail, size=size, warning="; ".join(warnings) or None
    )


def probe_local_file(path, *, kind: str = "file") -> ProbeResult:
    """Check a file a real import would read from disk (hand-downloaded or bundled).

    A ``.zip`` is additionally opened for its central directory: an archive truncated by
    an interrupted download passes an existence check and then fails hours later inside
    ``extract``, which is exactly the failure this pre-flight exists to move forward.
    """
    path = Path(path)

    if not path.exists():
        return ProbeResult(kind=kind, location=str(path), ok=False, detail="not on disk")

    size = path.stat().st_size

    if path.suffix.lower() == ".zip" and not is_zipfile(path):
        return ProbeResult(
            kind=kind,
            location=str(path),
            ok=False,
            detail=f"{naturalsize(size)} on disk but NOT a readable zip (truncated or partially downloaded?)",
            size=size,
        )

    return ProbeResult(kind=kind, location=str(path), ok=True, detail=f"on disk, {naturalsize(size)}", size=size)


def probe_fairdata_package(pid: str, *, api_base: str, timeout: int = 30, session=None) -> ProbeResult:
    """Check the Fairdata Download API can serve ``pid``, WITHOUT asking it to package.

    Deliberately only step (1) of :func:`resolve_fairdata_download_url`'s flow — the
    read-only ``GET /requests``. The POST that asks the service to build a package is a
    side effect on someone else's infrastructure (and can occupy it for minutes on a
    dataset this size), so a pre-flight must not trigger it.

    The verdict mirrors the resolver's own contract exactly, which is what makes it
    trustworthy: ``_json`` there raises for ANY non-OK response, so any non-OK response
    here is a blocking failure — a real run would stop at the same request. Reporting a
    404 as "no package yet" instead looked reasonable and was wrong: probing the base URL
    this importer shipped with returned nginx's 404 page for every path, which that
    reading turned into a pass for a source that could not be downloaded at all.

    A 200 that simply names no ready package IS ``ok`` with a warning: that is the normal
    state for a dataset nobody has requested lately, and a real run would ask for one.
    """
    requester = session or requests
    url = f"{api_base}/requests"

    try:
        response = requester.get(url, params={"cr_id": pid}, timeout=timeout)
    except requests.RequestException as e:
        return ProbeResult(kind="fairdata", location=pid, ok=False, detail=f"unreachable — {type(e).__name__}: {e}")

    if not response.ok:
        detail = f"{url} returned HTTP {response.status_code}, which is what a real run's first request would get"
        if response.status_code == 404:
            detail += " — the dataset PID or fairdata_api_base is wrong, or the service moved"
        return ProbeResult(kind="fairdata", location=pid, ok=False, detail=detail)

    try:
        payload = response.json()
    except ValueError:
        return ProbeResult(kind="fairdata", location=pid, ok=False, detail=f"{url} returned a non-JSON body")

    package = _find_ready_package(payload)
    if package is None:
        return ProbeResult(
            kind="fairdata",
            location=pid,
            ok=True,
            detail=f"{url} reachable, no ready package reported",
            warning="a real run would ask the service to package this dataset first, which can take minutes",
        )

    size = payload.get("size") if isinstance(payload, dict) else None
    size = size if isinstance(size, int) else None
    detail = f"package {package} ready" + (f", {naturalsize(size)}" if size else "")
    return ProbeResult(kind="fairdata", location=pid, ok=True, detail=detail, size=size)


def is_valid_image_file(image_filename):
    """Return True if PIL can open and crop the file, i.e. it is a readable image.

    A crop is used rather than ``Image.verify()`` because ``verify`` misses some
    corruption cases; cropping forces the decoder to touch pixel data.
    """
    try:
        with Image.open(image_filename) as img:
            # img.verify() seems not to be enough to check all cases,
            # cropping image should do.
            img.crop((5, 5, 5, 5))
        return True
    except (IOError, SyntaxError):
        return False


@dataclass
class DatasetImporter:
    """Abstract base for importing a public plankton dataset into HuggingFace format.

    Each concrete subclass corresponds to one source dataset and need only implement
    :meth:`_prepare_imagefolder`, which normalizes that source's extracted layout into
    a torchvision-style imagefolder (``<root>/<split>/<class>/*`` or
    ``<root>/<class>/*``). The base class drives the shared lifecycle in
    :meth:`import_dataset`: download + extract, prepare the imagefolder, optionally
    validate image integrity, load it as a HuggingFace ``imagefolder`` dataset, push to
    the Hub (gated on ``push_to_hub``), then clean up intermediate files.

    Subclasses may also override :meth:`_download_and_extract` when the source is not a
    plain set of ``download_uris`` (e.g. a bundled zip).

    Key fields:
        data_dir: Root directory for raw downloads and the prepared imagefolder.
        human_readable_name: Display name used in the dataset card.
        download_uris: Source URLs to download and extract.
        push_to_hub: Master gate for any Hub push; when False the push is skipped.
        hf_dataset_name / hf_org_name / hf_token / hf_private: Hub target + auth.
        force_download / resume_download / force_imagefolder_preparation: Re-run gates
            controlling whether downloads and imagefolder preparation are redone.
        manual_download_local_file_names: Pre-downloaded archives to use instead of
            fetching ``download_uris``.
        check_image_file_integrity: When True, drop unreadable images before loading.
        cleanup_after_processing: When True, remove raw/intermediate files at the end.
        description / license / citation_* / source_url / paperswithcode_id / arxiv_id:
            Dataset-card metadata.

    Instance attributes set in ``__post_init__``: ``imagefolder_dir`` and ``raw_dir``
    (both derived from ``data_dir`` and the lowercased class name), plus
    ``extracted_dirs``, ``download_manager``, and ``hf_dataset`` placeholders.
    """

    data_dir: Path

    human_readable_name: str = None
    download_uris: list[str] = None

    push_to_hub: Optional[bool] = False
    hf_dataset_name: Optional[str] = None
    hf_private: Optional[bool] = True
    hf_token: str = None
    hf_org_name: str = None

    show_progress: Optional[bool] = True
    # null -> cpu_count(), resolved in __post_init__. Two reasons it is not simply
    # `= cpu_count()` here: a dataclass default is evaluated once at IMPORT time, so the
    # value would be baked in when the module is first loaded; and the field has to be
    # declared in configs/dataset_import/default.yaml to be overridable at all, where a
    # concrete default would then hard-code this machine's core count into the config.
    num_proc: Optional[int] = None

    # download-related configs
    force_download: Optional[bool] = False
    resume_download: Optional[bool] = True
    force_imagefolder_preparation: Optional[bool] = True
    max_download_retries: Optional[int] = 5
    http_timeout: Optional[int] = 3600
    # Sent on every download AND on every pre-flight probe, so the two cannot disagree
    # about what the server will do. null -> default_user_agent(); see it for why leaving
    # this unset is not the neutral option it looks like.
    http_user_agent: Optional[str] = None
    push_to_hub_retries: Optional[int] = 10
    check_image_file_integrity: Optional[bool] = False

    # if we have manually downloaded the files add the archives here
    manual_download_local_file_names: str | list[str] = None

    # Where a human obtains the archives named above, and anything they need to know to
    # do it (a login, a "request access" step, an unstable direct link). Purely
    # informational: it is what missing_manual_downloads() reports, so a missing file
    # produces instructions instead of an error from inside extract().
    manual_download_url: str = None
    manual_download_notes: str = None

    # Fairdata (Etsin) sources publish through a package API rather than a stable direct
    # URL. Set fairdata_pid to resolve the download automatically; leave it null to use
    # the manual route. See resolve_fairdata_download_url.
    fairdata_pid: str = None
    # The Etsin proxy, NOT https://download.fairdata.fi — which this defaulted to until
    # the download pre-flight probed it on 2026-08-04 and got nginx's 404 HTML page for
    # every path, while the value below returned the package JSON recorded in
    # resolve_fairdata_download_url's docstring. Since _download_and_extract passes this
    # field explicitly, the wrong host overrode the resolver's own (correct) default, so
    # a sykezooscan2024 import raised "Fairdata request status returned HTTP 404".
    # Keep it equal to resolve_fairdata_download_url's default; the two are pinned
    # together by tests/test_dataset_import_configs.py.
    fairdata_api_base: str = "https://etsin.fairdata.fi/api/download"
    fairdata_poll_attempts: int = 60
    fairdata_poll_interval: int = 10

    # The four Tara Pacific sources have no archive at all: their SEANOE deposits publish
    # EcoTaxa TSV exports (metadata, no vignettes) and point at public EcoTaxa projects for
    # the images. These four fields configure that walk, and live here — on the base
    # dataclass — for the same reason the fairdata_* block does: only fields declared on
    # the dataclass reach an importer through `hydra.utils.instantiate`, and the concrete
    # subclasses are not themselves dataclasses.
    #
    # ecotaxa_projects: null -> tara_pacific_layout.SOURCES[SOURCE_NAME]["projects"].
    ecotaxa_projects: list[int] = None
    ecotaxa_window_size: int = 10_000
    ecotaxa_image_workers: int = 8
    # How many vignettes may stay unfetched after their retries before the import refuses
    # to finish. 0 = none: an incomplete import is a failure, not a smaller dataset. The
    # fetch is resumable, so the remedy for a transient outage is to re-run.
    ecotaxa_max_missing_images: int = 0

    cleanup_after_processing: Optional[bool] = False

    description: str = ""
    license: str = None
    citation_bibtex: str = None
    citation_apa: str = None
    source_url: str = None
    image_url: str = None
    paperswithcode_id: str = None
    arxiv_id: str = None

    def _validate(self):
        """Validate that Hub credentials/target are present when ``push_to_hub`` is set.

        Raises:
            ValueError: If ``push_to_hub`` is True but ``hf_token`` or
                ``hf_dataset_name`` is missing.
        """
        if self.push_to_hub:
            if not self.hf_token:
                raise ValueError("push_to_hub=True but hf_token is not set.")
            if not self.hf_dataset_name:
                raise ValueError("push_to_hub=True but hgfc_dataset_name is not set.")

    def __post_init__(self):
        """Validate config and derive the per-class raw/imagefolder paths.

        ``imagefolder_dir`` and ``raw_dir`` are namespaced by the lowercased subclass
        name so that distinct importers sharing a ``data_dir`` do not collide.
        """
        self._validate()
        self.data_dir = Path(self.data_dir)
        self.http_user_agent = self.http_user_agent or default_user_agent()
        # `is None`, not `or`: 0 and -1 are values a caller can mean (map_nested reads
        # -1), and `or` would silently turn both into cpu_count().
        if self.num_proc is None:
            self.num_proc = cpu_count()

        self.imagefolder_dir = self.data_dir / f"{self.__class__.__name__.lower()}_imagefolder"
        self.raw_dir = self.data_dir / f"{self.__class__.__name__.lower()}_raw_download"
        self.extracted_dirs = None
        self.download_manager = None
        self.hf_dataset = None

    def manual_download_paths(self) -> list[Path]:
        """The archives this importer expects to have been fetched by hand, if any."""
        declared = self.manual_download_local_file_names
        if not declared:
            return []
        if isinstance(declared, str):
            declared = [declared]
        return [Path(path) for path in declared]

    def missing_manual_downloads(self) -> list[Path]:
        """Declared manual archives that are not on disk yet.

        Cheap and side-effect free, so a caller can pre-flight a whole build and report
        every missing archive at once rather than discovering them one failed source at
        a time, hours apart.
        """
        return [path for path in self.manual_download_paths() if not path.exists()]

    def manual_download_instructions(self) -> str:
        """Human-readable instructions for obtaining this importer's manual archives."""
        missing = self.missing_manual_downloads()
        if not missing:
            return ""

        lines = [f"«{self.human_readable_name or self.hf_dataset_name}» needs {len(missing)} file(s) downloaded by hand:"]
        lines.extend(f"  - {path}" for path in missing)
        if self.manual_download_url:
            lines.append(f"Get them from: {self.manual_download_url}")
        elif self.source_url:
            lines.append(f"Start from the dataset page: {self.source_url}")
        if self.manual_download_notes:
            lines.append(self.manual_download_notes.strip())
        lines.append("Create the parent directory if needed, then re-run. Nothing else is required.")
        return "\n".join(lines)

    # --- Sidecar inputs: what the REDEFINE step needs on every run ---------------------
    #
    # Everything above concerns the archive: fetched once, extracted once, turned into an
    # imagefolder once, then reused. A source can also have inputs OUTSIDE that lifecycle
    # which the redefine step joins on EVERY run, imagefolder reused or not — FREPJ's
    # md5-pinned geodata tables and its committed site crosswalk. The three hooks below
    # let a source declare, check and obtain them so the pipeline and the pre-flight treat
    # them like any other download, without knowing the source by name. An archive-only
    # source keeps the defaults: nothing declared, nothing missing, nothing to obtain.

    def sidecar_targets(self) -> list[tuple[str, str]]:
        """Build-time inputs outside the archive lifecycle, as ``(kind, location)``.

        Same shape as :meth:`download_targets`, which appends them — and
        :meth:`probe_downloads` guarantees them even under a subclass that overrides
        ``download_targets`` without ``super()`` — so the pre-flight probes them like any
        other download. Exactly two kinds are valid here: ``url`` for one
        :meth:`ensure_sidecars` fetches, ``bundled`` for one that ships with the package.
        ``[]`` for an archive-only source.
        """
        return []

    def missing_sidecars(self) -> list[Path]:
        """Sidecar files a real run would FETCH: not on disk, or on disk but failing their pin.

        Free — no network, no side effect. Not a failure: the run obtains them itself. The
        pre-flight uses this to know that a source will fetch even when its imagefolder is
        already built.
        """
        return []

    def ensure_sidecars(self) -> dict[str, Path]:
        """Obtain every sidecar target, verified — fetching only the misses.

        Returns ``{file name: path}``; raises with the exact remedy when it cannot. The one
        step that also runs when the imagefolder is reused: ``import_and_redefine_source``
        calls it before the imagefolder decision, and ``pz_planktonzilla`` calls it for every
        selected source before the first import. Default: nothing to obtain, ``{}``.
        """
        return {}

    def imagefolder_is_complete(self) -> bool:
        """Whether the imagefolder on disk holds EVERYTHING this source should import.

        The gate on rebuilding: :meth:`import_dataset` prepares the imagefolder when this
        is False, and ``generate_planktonzilla.import_and_redefine_source`` reuses it when
        it is True. Both used to test "is the directory non-empty?" inline, which this
        default reproduces exactly — for an archive-backed source the imagefolder is
        written in one pass out of an already-extracted archive, so non-empty and complete
        are the same answer.

        They are NOT the same answer for a source whose imagefolder is filled
        incrementally over hours, one network fetch per image, and can therefore be left
        genuinely half-built by an interruption
        (:class:`~planktonzilla.dataset_import.tara_pacific_importer.TaraPacificDatasetImporter`).
        There, "non-empty" would silently accept a fraction of the source as the whole of
        it — so it overrides this with a real count, and a partial imagefolder resumes
        instead of being published as if it were finished.
        """
        return not is_dir_empty(self.imagefolder_dir)

    def download_targets(self) -> list[tuple[str, str]]:
        """What a real import of this source would have to obtain, as ``(kind, location)``.

        The pre-flight counterpart of :meth:`_download_and_extract`, and it MIRRORS that
        method's precedence deliberately: a hand-downloaded archive shadows
        ``download_uris`` there, so it must shadow it here too, or the report describes a
        download the run would never perform.

        An empty list means the source declares no way to get its data — the same
        condition ``_download_and_extract`` raises on, surfaced before anything is
        fetched instead of at the source's turn in a multi-hour build.

        Subclasses that override ``_download_and_extract`` override this in step:
        :class:`LenslessDatasetImporter` (bundled zip) and
        :class:`SYKEZooScan2024DatasetImporter` (Fairdata). A source with a SECOND
        download outside the lifecycle — :class:`GlobalUVP5NetDatasetImporter`'s objects
        metadata, fetched from ``_prepare_imagefolder`` — adds it here as well, since a
        build stops just as dead on that one. A source with inputs it needs on EVERY run,
        imagefolder reused or not, declares them in :meth:`sidecar_targets` instead; they
        are appended here, so a hand-downloaded archive shadows the URL but never them.
        """
        if self.manual_download_local_file_names:
            return [("file", str(path)) for path in self.manual_download_paths()] + self.sidecar_targets()
        return [("url", uri) for uri in _as_uri_list(self.download_uris)] + self.sidecar_targets()

    def probe_downloads(self, *, timeout: int = 30, session=None) -> list[ProbeResult]:
        """Check every :meth:`download_targets` entry without downloading anything.

        Side-effect free and safe to run against the live services: URLs are probed with
        HEAD (falling back to a one-byte ranged GET), files are stat-ed, and the Fairdata
        API is only READ. Never raises for a network failure — each target's verdict is
        returned so one dead host does not hide the state of the other sources.

        Each URL is probed AS the download identifies itself (:attr:`http_user_agent`),
        since at least one of these hosts answers differently by User-Agent. The Fairdata
        request deliberately does not: ``resolve_fairdata_download_url`` sends no custom
        User-Agent either, and the probe's value is that it mirrors the real run.
        """
        targets = self.download_targets()
        # Guaranteed here, not only in download_targets(): a subclass that overrides that
        # method without calling super() (Lensless, SYKE ZooScan) would otherwise drop the
        # sidecar targets it later declares, and the pre-flight would never probe them.
        targets += [target for target in self.sidecar_targets() if target not in targets]

        if not targets:
            return [
                ProbeResult(
                    kind="none",
                    location=self.human_readable_name or self.hf_dataset_name or type(self).__name__,
                    ok=False,
                    detail=(
                        "nothing to fetch: neither download_uris nor manual_download_local_file_names is set, "
                        "so a real run would stop here"
                    ),
                )
            ]

        results = []
        for kind, location in targets:
            if kind == "url":
                results.append(probe_url(location, timeout=timeout, session=session, user_agent=self.http_user_agent))
            elif kind == "fairdata":
                results.append(
                    probe_fairdata_package(location, api_base=self.fairdata_api_base, timeout=timeout, session=session)
                )
            else:
                results.append(probe_local_file(location, kind=kind))
        return results

    def storage_options(self) -> dict:
        """fsspec/aiohttp options shared by every download this importer makes.

        The ``headers`` entry is the only route a User-Agent has into the real download:
        ``datasets`` builds one in ``get_from_cache`` and then calls
        ``fsspec_head``/``fsspec_get`` WITHOUT it (verified in datasets 5.0), so setting
        ``DownloadConfig.user_agent`` would change nothing on the wire. Going through
        ``client_kwargs`` also means the pre-flight and the download agree, because both
        read :attr:`http_user_agent`.
        """
        return {
            "client_kwargs": {
                "timeout": aiohttp.ClientTimeout(total=self.http_timeout),
                "headers": {"User-Agent": self.http_user_agent},
            }
        }

    def _downloadable_uris(self):
        """``download_uris`` in the shape ``DownloadManager.download`` actually accepts.

        Hydra hands a YAML list over as an OmegaConf ``ListConfig``, which is NOT a
        ``list`` subclass. ``datasets.map_nested`` dispatches on ``isinstance(x, list)``,
        so a ListConfig misses every sequence branch and lands in the singleton one: the
        whole container is then stringified to ``"['https://a', 'https://b']"``, a value
        with no URL scheme, which ``_download_single`` takes for a relative path and
        joins onto ``base_path``. The download therefore fails without ever naming a URL.

        This is the same normalisation ``probe_downloads`` has always applied via
        :func:`_as_uri_list`, which is precisely why the pre-flight could report all nine
        whoi archives reachable while the real import crashed on them: the two paths
        disagreed about what ``download_uris`` is. They now share one answer.

        The bare-string case is passed through UNCHANGED rather than wrapped in a list.
        ``DownloadManager.extract`` mirrors the structure it is given, so wrapping would
        make ``extracted_dirs`` a one-element list and break the
        ``Path(self.extracted_dirs)`` that opens nearly every ``_prepare_imagefolder``.
        Only :class:`WHOIPlanktonDatasetImporter` iterates it — and whoi is also the only
        source declaring a list.
        """
        if isinstance(self.download_uris, str):
            return self.download_uris
        return _as_uri_list(self.download_uris)

    def _download_and_extract(self):
        """Download ``download_uris`` (or use manual files) and extract them.

        Builds an aiohttp-backed ``DownloadManager`` honoring the download config
        fields (force/resume, retries, timeout, num_proc). When
        ``manual_download_local_file_names`` is set, those archives are used in place
        of fetching ``download_uris``. Sets ``self.download_manager`` and
        ``self.extracted_dirs`` as a side effect.
        """
        self.download_manager = DownloadManager(
            # str, not the Path: datasets joins this with anything it takes for a
            # relative path, and its is_remote_url() calls urlparse() on the result.
            # urlparse rejects a PosixPath with "'PosixPath' object has no attribute
            # 'decode'", which names neither the URL nor this argument. A str turns the
            # same mistake into a legible "Local file <path> doesn't exist".
            base_path=str(self.raw_dir),
            data_dir=self.raw_dir,
            download_config=DownloadConfig(
                cache_dir=self.raw_dir,
                force_download=self.force_download,
                # INERT against datasets 5.0.1 — passed for forward compatibility only, so
                # read neither as a guarantee of resume nor of retry. Verified 2026-08-27
                # against the installed package: `resume_download` appears exactly twice in
                # all of datasets (download/download_config.py:20 docstring, :56 field) and
                # nothing reads it, while the only `max_retries` reads
                # (utils/file_utils.py:838, :972) take config.STREAMING_*_MAX_RETRIES —
                # module constants on the streaming xopen path, not this field on the
                # DownloadManager.download path.
                #
                # This matters because it is why planktonset1.0 was misdiagnosed for so
                # long: its NCEI endpoint needs ~22s to emit a first byte, then streams a
                # multi-GB archive at ~0.6 MB/s while IGNORING Range (a ranged GET answers
                # 200, not 206) and disclosing no Content-Length. The config promised a
                # resumable, five-times-retried download; in reality every attempt restarts
                # at byte 0 and none is retried, so the leftover `.incomplete` is not
                # progress. A source that needs real resume must implement it itself — see
                # _fetch_single_use for the in-house pattern.
                resume_download=self.resume_download,
                max_retries=self.max_download_retries,
                # 1, NOT self.num_proc — for downloads only. self.num_proc still drives
                # the CPU-bound work (imagefolder preparation, the Hub push).
                #
                # datasets' map_nested spawns a multiprocessing Pool once num_proc > 1
                # and there are >= 2 URLs (parallel_min_length defaults to 2), and that
                # pool breaks downloads twice over:
                #
                #   - it fails. whoi's nine archives fetched by 12 processes from
                #     darchive.mblwhoilibrary.org — a host this project already sends a
                #     custom User-Agent to because it refuses library clients — died
                #     partway through. The same nine with num_proc=1 downloaded AND
                #     extracted completely, 36 GB, verified 2026-08-06.
                #   - it hides why. A worker's exception must be pickled back to the
                #     parent, and aiohttp errors carry their response headers as a
                #     CIMultiDictProxy, which has no pickle support. Pool then raises
                #     "MaybeEncodingError: can't pickle CIMultiDictProxy" INSTEAD of the
                #     real failure, discarding the status code, URL and reason — so
                #     every HTTP error arrives as the same unreadable message.
                #
                # Concurrency is not lost where it pays: _download_batched thread-maps
                # any batch of >= 16 files, in-process, so a many-URL source still
                # downloads concurrently — with no process boundary for an exception to
                # die at. whoi is the only source with more than one URL today.
                num_proc=1,
                disable_tqdm=not self.show_progress,
                storage_options=self.storage_options(),
            ),
        )
        if self.manual_download_local_file_names:
            # Checked here rather than left to extract(), which fails on a missing path
            # with an error that names neither the file wanted nor where to get it.
            missing = self.missing_manual_downloads()
            if missing:
                raise FileNotFoundError(self.manual_download_instructions())

            logger.info(f"Using manually downloaded file {self.manual_download_local_file_names}.")
            downloaded_paths = self.manual_download_local_file_names
        else:
            if not self.download_uris:
                raise ValueError(
                    f"«{self.human_readable_name or self.hf_dataset_name}» has neither download_uris nor "
                    f"manual_download_local_file_names set, so there is nothing to fetch. Give it a download URL, "
                    f"or point manual_download_local_file_names at an archive you downloaded yourself"
                    + (f" from {self.source_url}." if self.source_url else ".")
                )

            logger.info(f"Downloading files to {self.raw_dir}.")
            downloaded_paths = self.download_manager.download(self._downloadable_uris())

        logger.info("Extracting file(s).")
        self.extracted_dirs = self.download_manager.extract(downloaded_paths)

    def _prepare_imagefolder(self):
        """Normalize this source's extracted layout into an imagefolder (subclass hook).

        Concrete importers must override this to move/copy ``self.extracted_dirs`` into
        ``self.imagefolder_dir`` as ``<split>/<class>/*`` (or ``<class>/*`` when the
        source has no splits).

        Raises:
            NotImplementedError: Always, on the abstract base.
        """
        raise NotImplementedError()

    def update_dataset_metadata(self):
        """Render and push the dataset card for the already-published Hub dataset.

        Loads the existing card, refreshes metadata (description, citations, license,
        per-split label histogram, and train-split RGB mean/std), renders it through
        ``DATACARD_TEMPLATE``, and pushes the new card to the Hub. Loads the dataset
        from the Hub if not already cached on ``self.hf_dataset``. Side effects:
        network reads (card + dataset) and a Hub card push.
        """
        logger.info(f"Updating «{self.hf_org_name}/{self.hf_dataset_name}» card metadata.")
        card = DatasetCard.load(self.hf_org_name + "/" + self.hf_dataset_name)

        card.data.dataset_info["description"] = self.description
        card.data.dataset_info["dataset_name"] = self.human_readable_name
        card.data.dataset_info["citation"] = self.citation_bibtex
        card.data.dataset_info["homepage"] = self.source_url

        card.data["pretty_name"] = self.human_readable_name
        card.data["dataset_description"] = self.description
        card.data["license"] = self.license
        card.data["source_url"] = self.source_url

        if self.paperswithcode_id:
            card.data["paperswithcode_id"] = self.paperswithcode_id

        if self.arxiv_id:
            card.data["arxiv_id"] = self.arxiv_id

        card.data["citation_bibtex"] = self.citation_bibtex
        card.data["citation_apa"] = self.citation_apa
        card.data["task_categories"] = ["image-classification"]
        card.data["hf_dataset_name"] = self.hf_dataset_name
        card.data["hf_org_name"] = self.hf_org_name

        if not self.hf_dataset:
            self.hf_dataset = load_dataset(self.hf_org_name + "/" + self.hf_dataset_name)

        card.data["report_markdown"] = report_dataset_content(self.hf_dataset)

        means, stds = compute_mean_and_std_dev(self.hf_dataset["train"])
        card.data["dataset_means"] = "[" + ", ".join([str(item) for item in means]) + "]"
        card.data["dataset_stds"] = "[" + ", ".join([str(item) for item in stds]) + "]"

        new_card = DatasetCard.from_template(card.data, template_str=DATACARD_TEMPLATE)
        new_card.push_to_hub(self.hf_org_name + "/" + self.hf_dataset_name)

    def show_details(self):
        """Pretty-print the Hub dataset's builder info and rendered card to the terminal.

        Reads the dataset builder info and dataset card from the Hub (network reads).
        """
        builder = load_dataset_builder(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(builder.info)

        card = DatasetCard.load(self.hf_org_name + "/" + self.hf_dataset_name)
        rich_print(Markdown(card.text))

    def _push_to_hub(self):
        """Push the prepared dataset to the Hub, gated on ``push_to_hub``.

        No-op (logs a warning) when ``push_to_hub`` is False. Otherwise pushes
        ``self.hf_dataset`` with up to ``push_to_hub_retries`` attempts, then refreshes
        the dataset card via :meth:`update_dataset_metadata`. Logs an error if there is
        no dataset to push. Side effects: Hub dataset + card push.
        """
        if self.push_to_hub:
            if self.hf_dataset:
                logger.info(
                    f"Pushing «{self.human_readable_name}» to HuggingFace Hub as «{self.hf_org_name}/{self.hf_dataset_name}»."
                )
                last_error = None
                for attempt in range(self.push_to_hub_retries):
                    try:
                        self.hf_dataset.push_to_hub(
                            self.hf_org_name + "/" + self.hf_dataset_name,
                            token=self.hf_token,
                            private=self.hf_private,
                        )
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"Push to hub attempt {attempt + 1}/{self.push_to_hub_retries} failed, retrying. Cause: {e}."
                        )

                # Exhausting every retry used to fall through to update_dataset_metadata()
                # and return normally, so a push that never succeeded reported success and
                # the card was refreshed for a dataset that was never uploaded.
                if last_error is not None:
                    raise RuntimeError(
                        f"Failed to push «{self.hf_dataset_name}» to the HuggingFace Hub after "
                        f"{self.push_to_hub_retries} attempts. Last error: {last_error}"
                    ) from last_error

                self.update_dataset_metadata()
            else:
                logger.error("No dataset to push.")
        else:
            logger.warning("Skipping pushing dataset to HuggingFace Hub, set push_to_hub=True to change this.")

    def cleanup(self):
        """Remove raw downloads and extracted files, gated on ``cleanup_after_processing``.

        No-op (logs an info message) when ``cleanup_after_processing`` is False. The
        prepared imagefolder is intentionally kept (the removal is commented out) so it
        can be reused on a later run. Side effects: deletes ``raw_dir`` and extracted
        files from disk.
        """
        if self.cleanup_after_processing:
            logger.info("Removing downloaded and intermediate files.")
            if self.download_manager:
                self.download_manager.delete_extracted_files()

            if self.raw_dir and self.raw_dir.exists():
                rmtree(self.raw_dir, ignore_errors=True)

            # if self.imagefolder_dir and self.imagefolder_dir.exists():
            #    rmtree(self.imagefolder_dir, ignore_errors=True)
        else:
            logger.info("Keeping downloaded and intermediate files, set cleanup_after_processing=True to change this.")

    def import_dataset(self) -> Union[Dataset, DatasetDict]:
        """Run the full import lifecycle for this source dataset.

        Steps: (re)build the imagefolder when missing or
        ``force_imagefolder_preparation`` is set — downloading + extracting and calling
        the subclass :meth:`_prepare_imagefolder`; optionally drop unreadable images
        when ``check_image_file_integrity``; load the imagefolder as a HuggingFace
        ``imagefolder`` dataset (mapping ``train``/``validation``/``val``/``test``
        subdirectories to splits, or a single ``train`` split when none exist); then
        push to the Hub (gated on ``push_to_hub``) and clean up.

        Side effects: network downloads, extensive disk writes under ``data_dir``, and
        an optional Hub push. Sets ``self.hf_dataset``.

        Raises:
            RuntimeError: If extraction failed and no raw data is available to prepare
                the imagefolder.
        """
        # `.incomplete` and `.lock` do NOT count as raw data. A download that died
        # mid-stream leaves both behind, and counting them made this report a cache hit
        # that does not exist: planktonset1.0's raw_dir held nothing but a 116 MB
        # `.incomplete` and a 0-byte `.lock`, so the run announced it was "resolving
        # extracted paths from cache" and then refetched the whole archive from byte 0.
        # Anyone reading that log — including anyone debugging why this source keeps
        # failing — was told a resume was happening that datasets does not implement.
        raw_exists = self.raw_dir.exists() and any(
            not entry.endswith((".incomplete", ".lock")) for entry in os.listdir(self.raw_dir)
        )

        need_to_build_imagefolder = not self.imagefolder_is_complete() or self.force_imagefolder_preparation

        if need_to_build_imagefolder:
            if raw_exists:
                logger.info(f"Raw data already exists at {self.raw_dir}, resolving extracted paths from cache.")
            else:
                logger.info("Downloading and extracting dataset.")

            # A cached archive is reused rather than refetched; note that a PARTIAL one is
            # not resumed (see the DownloadConfig comment in _download_and_extract).
            self._download_and_extract()

            if getattr(self, "extracted_dirs", None) is None:
                raise RuntimeError("Cannot prepare imagefolder: extraction failed or raw data is unavailable.")

            logger.info(f"Preparing dataset as imagefolder in {self.imagefolder_dir}")
            # The subclass hook may assume the imagefolder ROOT exists. Most
            # implementations create it themselves as a side effect (copytree and
            # mkdir(parents=True) both create missing parents), but WHOI's per-class
            # `mkdir(exist_ok=True)` does not — on a data_dir that never held this
            # source, a from-scratch build raised FileNotFoundError here before copying
            # a single file.
            self.imagefolder_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_imagefolder()

            # A hook that copied NOTHING must fail here, loudly, naming itself.
            #
            # Every _prepare_imagefolder locates its class folders by walking a path it
            # believes the archive has, and `Path.glob` on a path that does not exist
            # returns an empty iterator rather than raising — so a layout that shifted by
            # one directory produces an EMPTY imagefolder and no error. That has now
            # happened three times: SYKE ZooScan (KI-22, which globbed PlanktonSet1's
            # accession path by mistake), whoi (fixed in 25d111f, whose nine archives nest
            # one wrapper deeper than assumed), and it stays latent in every importer that
            # still hard-codes a subpath. Left unchecked the run continues for hours and
            # dies inside load_dataset with `Instruction "train" corresponds to no data!`,
            # which names neither the source nor the reason — or worse, pushes a silently
            # short dataset to the Hub.
            #
            # A short-circuiting search, NOT a count: global_uvp5 (7.4M images) and whoi
            # (3.3M) would pay a full multi-million-entry rglob on every build to learn a
            # single bit. The count is computed only to enrich the failure message.
            if not any(path.suffix.lower() in IMAGE_SUFFIXES for path in self.imagefolder_dir.rglob("*")):
                raise RuntimeError(
                    f"«{self.human_readable_name or self.hf_dataset_name}»: "
                    f"{type(self).__name__}._prepare_imagefolder() produced no image files under "
                    f"{self.imagefolder_dir}. The extracted layout almost certainly differs from the path that "
                    f"method expects — check it against {self.extracted_dirs}, and prefer locating the class "
                    f"folders (find_class_root, or an rglob for the image files) over a hard-coded subpath."
                )

        else:
            logger.info(
                f"Using existing imagefolder at {self.imagefolder_dir}. Set force_imagefolder_preparation=True to rebuild."
            )

        if self.check_image_file_integrity:
            # rglob rather than a two-level listdir: importers that produce a split
            # layout (LenslessDatasetImporter's train/ + test/, ZooLakeDatasetImporter's
            # train_split/ ...) nest images one level deeper, and the flat version of
            # this walk handed those class DIRECTORIES to is_valid_image_file. That
            # returns False for a directory (IsADirectoryError subclasses OSError,
            # which IOError aliases), so the next line called os.remove on a directory
            # and raised uncaught — turning an opt-in integrity check into a crash on
            # exactly the layouts that need it most.
            candidates = [path for path in self.imagefolder_dir.rglob("*") if path.is_file()]

            for path in tqdm(
                candidates,
                desc="Validating images.",
                disable=not self.show_progress,
                leave=False,
            ):
                if not is_valid_image_file(path):
                    logger.warning(f"Invalid file {path} detected. Removing it from the dataset.")
                    os.remove(path)

            cleanup_imagefolder_empty_dirs(self.imagefolder_dir)

        logger.info(f"Loading imagefolder in {self.imagefolder_dir} as HuggingFace dataset.")

        root = Path(self.imagefolder_dir)

        split_aliases = {
            "train": ["train"],
            "validation": ["validation", "val"],
            "test": ["test"],
        }

        data_files = {}

        for canonical_split, aliases in split_aliases.items():
            for alias in aliases:
                split_path = root / alias
                if split_path.exists():
                    data_files[canonical_split] = str(split_path / "*/[!._]*")
                    break

        # No directory matched a canonical split alias, so take everything as train.
        # ZooLakeDatasetImporter reaches this branch even though it HAS splits: it names
        # them train_split/val_split/test_split, none of which is an alias above. The
        # resolver handles that by depth rather than by name — see resolve_imagefolder_glob.
        if not data_files:
            data_files = {"train": resolve_imagefolder_glob(root)}

        self.hf_dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            name=self.hf_dataset_name,
            save_infos=True,
            token=self.hf_token,
            num_proc=self.num_proc,
        )

        self._push_to_hub()
        self.cleanup()


class LenslessDatasetImporter(DatasetImporter):
    """Importer for the bundled lensless plankton dataset.

    Unlike URL-based sources, this dataset ships as a zip inside the
    ``planktonzilla.dataset_import.public_data`` package, so it overrides
    :meth:`_download_and_extract` to unzip locally. Its ``TRAIN_IMAGE`` / ``TEST_IMAGE``
    folders are renamed to the canonical ``train`` / ``test`` splits.
    """

    DATASET_FILENAME: Final[str] = "lensless_dataset"

    def download_targets(self) -> list[tuple[str, str]]:
        """The bundled zip: this source is never downloaded, so nothing is probed."""
        return [("bundled", str(Path(public_data.__path__[0]) / f"{self.DATASET_FILENAME}.zip"))]

    def _download_and_extract(self):
        """Unzip the bundled lensless dataset from ``public_data`` into ``raw_dir``."""
        dataset_path = Path(public_data.__path__[0])

        logger.info(f"Unzipping lensless zip {dataset_path / (self.DATASET_FILENAME + '.zip')}.")

        unzip(
            dataset_path / (self.DATASET_FILENAME + ".zip"),
            self.raw_dir,
            show_progress=self.show_progress,
        )
        self.extracted_dirs = self.raw_dir / self.DATASET_FILENAME

    def _prepare_imagefolder(self):
        if self.imagefolder_dir.exists():
            rmtree(self.imagefolder_dir, ignore_errors=True)
        self.imagefolder_dir.mkdir(exist_ok=True, parents=True)
        copytree(self.extracted_dirs, self.imagefolder_dir, dirs_exist_ok=True)
        (self.imagefolder_dir / "TRAIN_IMAGE").rename(self.imagefolder_dir / "train")
        (self.imagefolder_dir / "TEST_IMAGE").rename(self.imagefolder_dir / "test")


class ZooLakeDatasetImporter(DatasetImporter):
    """Importer for the ZooLake dataset (lake zooplankton, predefined splits).

    The source ships ``train``/``val``/``test`` filename manifests that index into a
    flat ``zooplankton_0p5x`` image tree; :meth:`_prepare_imagefolder` reads those
    manifests (see :attr:`SPLIT_NAMES`) and copies each listed image into the matching
    ``<split>/<class>`` folder, preserving the original split assignment.
    """

    SPLIT_NAMES: ClassVar[Dict[str, str]] = {
        "train_split": "train_filenames.txt",
        "val_split": "val_filenames.txt",
        "test_split": "test_filenames.txt",
    }

    def _prepare_imagefolder(self):
        for split_name in tqdm(
            list(self.SPLIT_NAMES),
            desc="Processing original split",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            with open(
                Path(self.extracted_dirs) / "data" / "zoolake_train_test_val_separated" / self.SPLIT_NAMES[split_name]
            ) as f:
                lines = f.readlines()
                for line in tqdm(
                    lines,
                    desc=f"Moving files in {split_name}",
                    leave=False,
                    position=1,
                    disable=not self.show_progress,
                ):
                    _, _, _, class_name, folder, file_name = line.strip().split("/")

                    source_img_file = Path(self.extracted_dirs) / "data" / "zooplankton_0p5x" / class_name / folder / file_name

                    target_folder = self.imagefolder_dir / split_name / class_name

                    if not (source_img_file).exists():
                        logger.warning(f"In split {split_name} (class {class_name}) file {source_img_file} does not exist.")
                        continue

                    target_folder.mkdir(exist_ok=True, parents=True)

                    if (target_folder / file_name).exists():
                        logger.warning(f"File name duplicate {file_name}, skipping.")
                    else:
                        copy2(
                            source_img_file,
                            target_folder,
                        )


class ZooScanNetDatasetImporter(DatasetImporter):
    """Importer for ZooScanNet (single, unsplit set of per-class folders).

    Copies each class folder under ``ZooScanNet/imgs`` into the imagefolder root.
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooScanNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class WHOIPlanktonDatasetImporter(DatasetImporter):
    """Importer for the WHOI plankton dataset (multiple per-release archives).

    Iterates the extracted release folders and copies each release's per-class ``.png``
    images into the imagefolder root (merging classes across releases), then deletes
    each consumed release folder from ``raw_dir`` to save space.
    """

    def _prepare_imagefolder(self):
        for release_folder in tqdm(
            self.extracted_dirs,
            desc="ImageFolder move progress",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            release_root = self.raw_dir / release_folder
            # Each release archive unpacks to a wrapper directory (observed: the release
            # year, e.g. `2006/<class>/*.png`) ABOVE the class dirs, not the class dirs
            # directly — a plain `release_root.glob("*")` picked up that wrapper as if it
            # were itself a (image-less) class, silently produced zero-file imagefolder
            # entries, and the eventual `load_dataset("imagefolder", ...)` died with
            # `Instruction "train" corresponds to no data!`. Locating class dirs by where
            # the `.png` files actually are — as `import_dataset`'s own integrity-check
            # rglob already does — is depth-agnostic and self-corrects regardless of how
            # many wrapper levels a given release's archive happens to add.
            class_dirs = sorted({img_file.parent for img_file in release_root.rglob("*.png")})
            for folder in tqdm(
                class_dirs,
                desc=f"Moving release {release_folder}",
                leave=False,
                position=1,
                disable=not self.show_progress,
            ):
                (self.imagefolder_dir / folder.name).mkdir(exist_ok=True)
                for img_file in folder.glob("*.png"):
                    try:
                        copy2(folder / img_file, self.imagefolder_dir / folder.name)
                    except OSError:
                        logger.debug(f"File {folder / img_file} already in {self.imagefolder_dir / folder.name}.")
            rmtree(self.raw_dir / release_folder, ignore_errors=True)


class JEDISystemsOceansCPICSDatasetImporter(DatasetImporter):
    """Importer for the JEDI Systems / Oceans CPICS validated dataset (nested zips).

    The ``CPICS_Validated`` payload contains nested per-release zips: this unzips each,
    deletes the now-redundant nested archive, fixes restrictive file permissions left by
    the inner zips, then moves each release's per-class ``.png`` images into the
    imagefolder root and removes the consumed release directory.
    """

    def _prepare_imagefolder(self) -> None:
        for zip_file in tqdm(
            sorted((Path(self.extracted_dirs) / "CPICS_Validated").glob("*.zip")),
            desc="Unzip progress",
            leave=False,
            disable=not self.show_progress,
        ):
            unzip(
                zip_file,
                Path(self.extracted_dirs) / "CPICS_Validated",
                show_progress=self.show_progress,
            )

            # nested zip files are an intermedite results, we delete them to save space
            Path(zip_file).unlink()

        # fixing file permissions issue in nested zips
        for file in (Path(self.extracted_dirs) / "CPICS_Validated").glob("*"):
            file.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)  # owner read/write/excecute

        for release_dir in tqdm(
            sorted([item for item in (Path(self.extracted_dirs) / "CPICS_Validated").glob("*") if item.is_dir()]),
            desc="ImageFolder preparation",
            leave=False,
            position=0,
            disable=not self.show_progress,
        ):
            for class_folder in tqdm(
                [item for item in release_dir.glob("*") if item.is_dir()],
                desc=f"Moving release {release_dir.name}",
                leave=False,
                position=1,
                disable=not self.show_progress,
            ):
                (self.imagefolder_dir / class_folder.name).mkdir(exist_ok=True, parents=True)
                for img_file in class_folder.glob("*.png"):
                    try:
                        move(
                            class_folder / img_file,
                            self.imagefolder_dir / class_folder.name,
                        )
                    except OSError:
                        logger.debug(f"File {class_folder / img_file} already in {self.imagefolder_dir / class_folder.name}.")
            rmtree(release_dir, ignore_errors=True)


class UVP6NetDatasetImporter(DatasetImporter):
    """Importer for UVP6Net; copies each class folder under ``imgs`` into the root."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ZooCAMNetDatasetImporter(DatasetImporter):
    """Importer for ZooCamNet; copies each class folder under ``ZooCamNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ZooCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class FlowCAMNetDatasetImporter(DatasetImporter):
    """Importer for FlowCamNet; copies each class folder under ``FlowCamNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "FlowCamNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class ISIISNetDatasetImporter(DatasetImporter):
    """Importer for ISIISNet; copies each class folder under ``ISIISNet/imgs``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "ISIISNet" / "imgs").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class PlanktoScopeDatasetImporter(DatasetImporter):
    """Importer for the PlanktoScope reference set.

    Copies each class folder under ``Planktoscope_reference/imgs``, skipping non-dirs
    and macOS junk entries (``._*``, ``.DS_Store``).
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "Planktoscope_reference" / "imgs").iterdir(),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            if (
                not plankton_class_dir.is_dir()
                or plankton_class_dir.name.startswith("._")
                or plankton_class_dir.name == ".DS_Store"
            ):
                continue

            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class GlobalUVP5NetDatasetImporter(DatasetImporter):
    """Importer for the Global UVP5 dataset, where labels come from a separate metadata file.

    The image archive has no class folders; instead an ``objects.tsv.gz`` (downloaded
    separately from :attr:`OBJECTS_URL`) maps each ``object_id`` to a ``taxon``.
    :meth:`_prepare_imagefolder` parses that mapping straight from the zip, creates one
    folder per taxon, then copies each image (named by object id) into its taxon folder
    using a bounded thread pool (network download + heavy disk I/O).
    """

    OBJECTS_URL = "https://www.seanoe.org/data/00964/107583/data/120871.zip"

    def download_targets(self) -> list[tuple[str, str]]:
        """The image archive AND the objects metadata zip.

        ``OBJECTS_URL`` is fetched from ``_prepare_imagefolder``, not from
        ``_download_and_extract``, so it is downloaded even by a ``refresh=rebuild`` that
        reuses the raw archive — and an import stops just as dead without it. A
        pre-flight that only looked at ``download_uris`` would miss it.
        """
        return [*super().download_targets(), ("url", self.OBJECTS_URL)]

    def _prepare_imagefolder(self):
        aux_dir = self.data_dir / "global_uvp5_aux"
        aux_dir.mkdir(parents=True, exist_ok=True)

        # --- Metadata (obj_id to taxo) ---
        dm = DownloadManager(
            # str base_path and num_proc=1 for the same reasons as the main archive
            # download in _download_and_extract; see the comments there. This one fetches
            # a single URL, so map_nested never reaches its parallel branch today — the
            # values match anyway so the two download sites cannot drift into disagreeing
            # about how a download is performed.
            base_path=str(aux_dir),
            data_dir=aux_dir,
            download_config=DownloadConfig(
                cache_dir=aux_dir,
                force_download=self.force_download,
                resume_download=self.resume_download,
                max_retries=self.max_download_retries,
                num_proc=1,
                disable_tqdm=not self.show_progress,
                # Same options as the main archive: this second download is a real one,
                # on the same host, and would be refused just the same without them.
                storage_options=self.storage_options(),
            ),
        )

        logger.info("Downloading objects metadata.")
        zip_path = dm.download(self.OBJECTS_URL)

        # --- Mapping ---
        mapping = {}
        logger.info("Parsing metadata directly from ZIP...")
        with ZipFile(zip_path, "r") as z:
            tsv_filename = next((name for name in z.namelist() if name.endswith("objects.tsv.gz")), None)
            if not tsv_filename:
                raise RuntimeError("objects.tsv.gz not found in zip")

            with z.open(tsv_filename) as gz_fileobj:
                with gzip.open(gz_fileobj, "rt", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader)

                    try:
                        obj_idx = header.index("object_id")
                        taxon_idx = header.index("taxon")
                    except ValueError:
                        raise RuntimeError("Columns 'object_id' or 'taxon' missing in TSV.")

                    for row in reader:
                        mapping[row[obj_idx]] = row[taxon_idx]

        logger.info("Creating target directories...")
        unique_taxa = set(mapping.values())
        for taxon in unique_taxa:
            (self.imagefolder_dir / taxon).mkdir(parents=True, exist_ok=True)

        images_root = Path(self.extracted_dirs) / "images"
        copy_tasks = []

        logger.info("Mapping files to their target directories...")

        try:
            sample_dirs = [entry.path for entry in os.scandir(images_root) if entry.is_dir()]
        except FileNotFoundError:
            raise RuntimeError(f"Directory not found: {images_root}. Check your extracted_dirs path.")

        for sample_dir_path in tqdm(sample_dirs, desc="Scanning directories", leave=False, disable=not self.show_progress):
            for entry in os.scandir(sample_dir_path):
                if not entry.is_file():
                    continue

                object_id = entry.name.rsplit(".", 1)[0]
                taxon = mapping.get(object_id)

                dst = self.imagefolder_dir / taxon / entry.name

                copy_tasks.append((entry.path, dst))

        # --- MultiThread ---
        def copy_worker(task):
            src, dst = task
            if not dst.exists():
                try:
                    copy2(src, dst)
                except OSError as e:
                    logger.warning(f"Failed to copy {src}: {e}")

        if copy_tasks:
            max_threads = min(16, self.num_proc)
            logger.info(f"Starting multi-threaded copy with {max_threads} workers...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                list(
                    tqdm(
                        executor.map(copy_worker, copy_tasks),
                        total=len(copy_tasks),
                        desc="Copying images",
                        disable=not self.show_progress,
                        leave=False,
                    )
                )
        else:
            logger.info("No new images to copy.")


class PlanktonSet1DatasetImporter(DatasetImporter):
    """Importer for PlanktonSet-1.

    Copies each class folder from the deeply nested
    ``0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014`` path, skipping non-dirs
    and dotfile/macOS junk entries.
    """

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "0127422" / "2.3" / "data" / "0-data" / "FINAL_Plankton_Segments_12082014").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            if (
                not plankton_class_dir.is_dir()
                or plankton_class_dir.name.startswith(".")
                or plankton_class_dir.name.startswith("._")
                or plankton_class_dir.name == ".DS_Store"
            ):
                continue

            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class SYKEIFCB2022DatasetImporter(DatasetImporter):
    """Importer for the SYKE IFCB 2022 set; copies class folders under ``labeled_20201020``."""

    def _prepare_imagefolder(self):
        for plankton_class_dir in tqdm(
            (Path(self.extracted_dirs) / "labeled_20201020").glob("*"),
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)


class SYKEZooScan2024DatasetImporter(DatasetImporter):
    """Importer for the SYKE ZooScan 2024 set.

    Copies each class folder from
    ``0127422/2.3/data/FINAL_Plankton_Segments_12082014``.

    The archive is published through Fairdata (Etsin) rather than at a stable direct
    URL, so ``download_uris`` cannot simply name a ``.zip``. Set ``fairdata_pid`` and
    this importer asks the Fairdata Download API to package the dataset and downloads
    the result; leave it unset and it falls back to the usual
    ``manual_download_local_file_names`` route, which now reports exactly which file is
    missing and where to get it.

    The package is a zip containing another zip, so the outer extraction must be
    unwrapped before the class folders are reachable — verified against the real
    archive on 2026-08-01::

        <package>.zip
          SYKE-plankton_ZooScan_2024/readme.md
          SYKE-plankton_ZooScan_2024/SYKE-plankton_ZooScan_2024.zip
            SYKE-plankton_ZooScan_2024/images/SYKE-plankton_ZooScan_2024/<class>/*.png
            SYKE-plankton_ZooScan_2024/class_splits/…
            __MACOSX/…                       (junk; copytree_filtered drops it)

    The 20 class folders that produces match the 20 ``sykezooscan2024`` ``Raw_Labels``
    in ``planktonzilla_taxonomy.csv`` exactly.
    """

    def download_targets(self) -> list[tuple[str, str]]:
        """Fairdata packages this dataset on demand, unless a manual copy shadows it.

        The precedence is the one ``_download_and_extract`` applies below — manual file,
        then PID, then ``download_uris`` (empty in the shipped config).
        """
        if self.fairdata_pid and not self.manual_download_local_file_names:
            return [("fairdata", self.fairdata_pid)]
        return super().download_targets()

    def _fetch_single_use(self, url: str) -> Path:
        """Download an authorized Fairdata URL exactly once, into ``raw_dir``.

        The URL authorises ONE download. Measured against the live service on
        2026-08-04, per token: HEAD then GET both answer 200, but a *completed* GET
        consumes it and the next GET is ``401 UNAUTHORIZED``.

        That is fatal to the usual path, because it makes two requests: ``datasets``
        calls ``fsspec_head`` (``HTTPFileSystem.info``, which falls back to a ranged GET
        when HEAD discloses no size — this service discloses none) and then
        ``fsspec_get``. The first consumes the token and the real download fails, which
        is what an import did until this method existed. So the package is fetched once,
        here, and handed to the ordinary hand-downloaded-archive path.

        Written to a ``.part`` file and renamed only on success: a half-downloaded
        package that looked complete would be reused by the next run and fail in
        ``extract`` instead, hours later and nowhere near the cause.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        target = self.raw_dir / f"{self.fairdata_pid}.zip"

        if target.exists() and not self.force_download:
            logger.info(f"Reusing the Fairdata package already at {target} (force_download=True to re-fetch).")
            return target

        partial = target.with_suffix(".part")
        logger.info(f"Downloading the Fairdata package to {target}.")

        with requests.get(
            url,
            stream=True,
            timeout=self.http_timeout,
            headers={"User-Agent": self.http_user_agent},
        ) as response:
            response.raise_for_status()
            declared = int(response.headers.get("Content-Length") or 0)
            with (
                open(partial, "wb") as handle,
                tqdm(
                    total=declared or None,
                    unit="B",
                    unit_scale=True,
                    desc="Fairdata package",
                    disable=not self.show_progress,
                    leave=False,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))

        written = partial.stat().st_size
        if declared and written != declared:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"The Fairdata package for «{self.fairdata_pid}» stopped at {naturalsize(written)} of "
                f"{naturalsize(declared)}. The download token is single-use, so retry from the start rather than "
                f"resuming — re-run the import."
            )

        partial.replace(target)
        logger.info(f"Fairdata package downloaded ({naturalsize(written)}).")
        return target

    def _download_and_extract(self):
        """Resolve the archive through the Fairdata API when a PID is configured."""
        if self.fairdata_pid and not self.manual_download_local_file_names:
            resolved = resolve_fairdata_download_url(
                self.fairdata_pid,
                api_base=self.fairdata_api_base,
                timeout=self.http_timeout,
                poll_attempts=self.fairdata_poll_attempts,
                poll_interval=self.fairdata_poll_interval,
                source_url=self.source_url,
            )
            logger.info(f"Fairdata resolved «{self.fairdata_pid}» to a download URL.")
            # NOT assigned to download_uris: that hands the single-use URL to
            # DownloadManager, which requests it twice. See _fetch_single_use.
            #
            # A bare string, NOT a one-element list: DownloadManager.extract mirrors the
            # structure it is given, so a list makes extracted_dirs a list, and every
            # _prepare_imagefolder starts with Path(self.extracted_dirs).
            self.manual_download_local_file_names = str(self._fetch_single_use(resolved))

        return super()._download_and_extract()

    def _prepare_imagefolder(self):
        root = Path(self.extracted_dirs)

        # Unwrap the nested archive. The previous implementation instead globbed
        # "0127422/2.3/data/FINAL_Plankton_Segments_12082014" — the NOAA accession path
        # belonging to PlanktonSet1, which does not exist anywhere in this archive, so
        # the loop silently iterated nothing and produced an EMPTY imagefolder.
        for nested in sorted(root.rglob("*.zip")):
            logger.info(f"Extracting nested archive {nested.name}.")
            unzip(nested, nested.parent, show_progress=self.show_progress)

        # Located rather than hard-coded: the class folders sit three levels down, under
        # images/<dataset name>/, and a re-release that renames a wrapper would break a
        # fixed path again.
        class_root = find_class_root(root)

        class_dirs = [dir for dir in _subdirectories(class_root) if _holds_images(dir)]
        for plankton_class_dir in tqdm(
            class_dirs,
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)

        cleanup_imagefolder_empty_dirs(self.imagefolder_dir)


class MedPlanktonSetDatasetImporter(DatasetImporter):
    """Importer for MedPlanktonSet — labeled IFCB images from the Mediterranean Sea.

    Source: ``IFCB_images.zip`` from Zenodo record 15471023 (see
    ``configs/dataset_import/medplanktonset.yaml``). The archive holds one folder per
    labeled class; ``planktonzilla_taxonomy.csv`` maps 139 ``medplanktonset``
    ``Raw_Labels`` (``Akashiwo_sanguinea``, ``Centric_diatoms``, ``Chaetoceros_spp``,
    …) onto the shared taxonomy, and those names are the class folder names.

    Unlike its sibling importers this one does not hard-code the path from the
    extraction root to the class folders — it locates them with
    :func:`find_class_root`, so a wrapper directory in the archive does not matter.

    .. warning::
       The archive's internal layout has NOT been verified against the real download:
       Zenodo was unreachable from the environment this importer was written in. The
       layout-independent scan is what makes that acceptable rather than a guess, but
       the first real run should be done with ``show_progress=true`` and its reported
       class count checked against the 139 ``medplanktonset`` rows in the taxonomy CSV.
       A mismatch means the scan found the wrong level, not that the CSV is wrong.
    """

    def _prepare_imagefolder(self):
        class_root = find_class_root(Path(self.extracted_dirs))

        # find_class_root scores a directory by how many subdirectories hold images, so
        # the winner may still have image-free siblings (a stray "metadata/" folder, a
        # "docs/"). Copying those would create class folders whose only members are
        # non-image files, which the imagefolder loader then tries to decode — so
        # select on image content here, the same test the scan itself used.
        class_dirs = [dir for dir in _subdirectories(class_root) if _holds_images(dir)]

        for plankton_class_dir in tqdm(
            class_dirs,
            desc="Progress",
            leave=False,
            disable=not self.show_progress,
        ):
            copytree_filtered(plankton_class_dir, self.imagefolder_dir / plankton_class_dir.name)

        # Belt and braces: a class folder whose every image was a macOS junk file is
        # dropped by copytree_filtered's ignore patterns and arrives empty.
        cleanup_imagefolder_empty_dirs(self.imagefolder_dir)
