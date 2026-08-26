"""
(c) Inria

Minimal read-only client for the public parts of EcoTaxa (https://ecotaxa.obs-vlfr.fr).

Two capabilities, both anonymous and both verified against the live service on
2026-08-26 for all seven Tara Pacific projects:

  * ``POST /api/object_set/{project_id}/query`` — a per-object MANIFEST. With
    ``fields=`` it returns exactly the columns a build needs (object id, original id,
    classification id + display name, lat/lon, date/time, depth range and the vault path
    of the visible image) for a whole project, ``window_size`` objects per request.
  * ``GET /vault/{file_name}`` — the vignette itself.

Everything else EcoTaxa can do needs an account: ``POST /api/object_set/export``, the
endpoint that would package a project into one archive, answers ``403 Not authenticated``.
That is why an import walks the manifest and fetches vignettes one by one instead of
downloading an archive — there is no anonymous archive to download.

Deliberately NOT a general EcoTaxa SDK: no writes, no classification, no login, no
project mutation. Every function here is a GET or a read-only POST, so running it against
the live service cannot change anything. The only dependency beyond the stdlib is
``requests``, already a project dependency, and every entry point takes an injectable
``session`` so tests drive it with a double instead of the network.
"""

import concurrent.futures
import csv
import os
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

API_BASE = "https://ecotaxa.obs-vlfr.fr/api"
VAULT_BASE = "https://ecotaxa.obs-vlfr.fr/vault"

# The `fields=` value sent to the query endpoint, and the manifest column each one fills.
# Order matters twice over: the endpoint returns `details` as a list in REQUEST order, and
# the manifest TSV is written in this order, so the two are defined together and can never
# drift apart. `img.file_name` is the vault path of the object's visible image; `obj.
# classif_id` is the EcoTaxa taxon id that `tara_pacific_layout` maps to a class dir.
MANIFEST_FIELD_SPECS = (
    ("obj.objid", "objid"),
    ("obj.orig_id", "orig_id"),
    ("obj.classif_id", "classif_id"),
    ("txo.display_name", "display_name"),
    ("obj.latitude", "latitude"),
    ("obj.longitude", "longitude"),
    ("obj.objdate", "objdate"),
    ("obj.objtime", "objtime"),
    ("obj.depth_min", "depth_min"),
    ("obj.depth_max", "depth_max"),
    ("img.file_name", "img_file_name"),
)

MANIFEST_FIELDS = ",".join(spec for spec, _ in MANIFEST_FIELD_SPECS)
MANIFEST_COLUMNS = tuple(column for _, column in MANIFEST_FIELD_SPECS)

# Integer manifest columns, parsed back to int on read; everything else stays text. The
# ids are the only values compared or used as dict keys, so they are the only ones whose
# type has to survive the round trip.
_INT_COLUMNS = ("objid", "classif_id")

# 10 000 objects/request. Measured against the live endpoint: the whole 1 057 984-object
# DeckNet project resolves in 106 requests at this size, and a window this large still
# answers well inside the default timeout. Larger windows buy little and make a retry
# after a mid-page failure re-fetch more.
DEFAULT_WINDOW_SIZE = 10_000

# Vignette fetches are small (a few kB each) and latency-bound, so a handful of threads is
# what turns a multi-hour walk into a bounded one. Kept deliberately low: this is somebody
# else's public service, and the ceiling is politeness, not throughput.
DEFAULT_IMAGE_WORKERS = 8


class EcoTaxaError(RuntimeError):
    """An EcoTaxa request failed in a way a retry did not fix."""


def project_url(project_id: int) -> str:
    """The read-only project record — what the pre-flight probes for reachability."""
    return f"{API_BASE}/projects/{project_id}"


def query_url(project_id: int, *, fields: str = MANIFEST_FIELDS, window_start: int = 0, window_size: int = 1) -> str:
    """The manifest query URL for one window of one project."""
    return (
        f"{API_BASE}/object_set/{project_id}/query"
        f"?fields={fields}&order_field=obj.objid&window_start={window_start}&window_size={window_size}"
    )


def vault_url(file_name: str) -> str:
    """The public URL of one vignette, from its manifest ``img_file_name``."""
    return f"{VAULT_BASE}/{file_name}"


def _request_json(url, *, session, user_agent, timeout, retries, method="post", json_body=None):
    """One request, retried with linear backoff; raises :class:`EcoTaxaError` when spent."""
    requester = session or requests
    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent

    last = None
    for attempt in range(1, retries + 1):
        try:
            if method == "post":
                response = requester.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                response = requester.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            last = f"HTTP {response.status_code}"
            # 4xx other than 429 will not become a 200 by waiting; fail immediately so the
            # caller sees "403 Not authenticated" rather than the same line five times.
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except (requests.RequestException, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(attempt)
    raise EcoTaxaError(f"{url} failed after {attempt} attempt(s): {last}")


def fetch_project_manifest(
    project_id: int,
    *,
    session=None,
    user_agent: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    timeout: int = 120,
    retries: int = 5,
    show_progress: bool = True,
) -> list[dict]:
    """Page one project's whole object manifest, ordered by object id.

    The endpoint reports ``total_ids`` for the (unfiltered) query on every page, so the
    walk is bounded by that rather than by "until a short page". Objects are ordered by
    ``obj.objid`` server-side, which is what makes a re-fetch byte-identical.

    Returns:
        One dict per object, keyed by :data:`MANIFEST_COLUMNS`.

    Raises:
        EcoTaxaError: If a window cannot be fetched, or if the service returns fewer rows
            than it promised — a silently short manifest would import a silently short
            dataset.
    """
    rows: list[dict] = []
    total = None
    progress = None

    while total is None or len(rows) < total:
        url = query_url(project_id, window_start=len(rows), window_size=window_size)
        payload = _request_json(
            url, session=session, user_agent=user_agent, timeout=timeout, retries=retries, json_body={"filters": {}}
        )

        if total is None:
            total = int(payload.get("total_ids", 0))
            progress = tqdm(
                total=total, desc=f"EcoTaxa manifest {project_id}", unit="obj", leave=False, disable=not show_progress
            )

        details = payload.get("details") or []
        if not details:
            break
        rows.extend(dict(zip(MANIFEST_COLUMNS, detail)) for detail in details)
        if progress is not None:
            progress.update(len(details))

    if progress is not None:
        progress.close()

    if total and len(rows) != total:
        raise EcoTaxaError(
            f"EcoTaxa project {project_id} promised {total} objects but returned {len(rows)}. "
            "Refusing a short manifest: it would import a silently incomplete dataset."
        )
    return rows


def write_manifest(rows, path: str | Path) -> Path:
    """Write a manifest to TSV atomically, in :data:`MANIFEST_COLUMNS` order.

    Written to a sibling ``.tmp`` and renamed, so an interrupted run never leaves a short
    manifest that the next run would take for a complete one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(["" if row.get(column) is None else row.get(column) for column in MANIFEST_COLUMNS])
    os.replace(tmp, path)
    return path


def read_manifest(path: str | Path) -> list[dict]:
    """Read a manifest TSV back, restoring the integer ids and blanks-as-None.

    Only :data:`_INT_COLUMNS` are typed: they are the ids used as dict keys and compared
    against the class map, so their type has to survive the round trip. Everything else
    comes back as TEXT — a coordinate the API returned as ``-6.8826`` reads back as
    ``"-6.8826"`` — which is exactly what the consumers want: the redefiner hands
    coordinates, dates and depths to the shared metadata path as strings anyway.

    Raises:
        ValueError: If the header is not exactly :data:`MANIFEST_COLUMNS` — a manifest
            written by an older field list must be re-fetched, not half-read.
    """
    path = Path(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"«{path}» has columns {reader.fieldnames}, expected {MANIFEST_COLUMNS}. "
                "Delete it and re-run to re-fetch the manifest."
            )
        for raw in reader:
            row = {column: (raw[column] or None) for column in MANIFEST_COLUMNS}
            for column in _INT_COLUMNS:
                row[column] = int(row[column]) if row[column] is not None else None
            rows.append(row)
    return rows


def _fetch_one_image(job, *, session, headers, timeout, retries):
    """Fetch one vignette to ``dest`` atomically.

    Returns ``(error or None, downloaded)``. The second element separates "already on
    disk" from "fetched just now", so a resumed run can report how much work it actually
    did instead of counting every skip as a download.
    """
    url, dest = job
    if dest.exists():
        return None, False

    last = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200 and response.content:
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Unique temp name per destination: two workers never share one, and a
                # killed run leaves a partial file rather than a truncated .jpg that the
                # next run's `dest.exists()` would accept as complete. Dot-prefixed on
                # purpose — `resolve_imagefolder_glob` matches `[!._]*`, so a leftover is
                # invisible to the imagefolder loader instead of being read as an image.
                tmp = dest.parent / f".{dest.name}.part"
                tmp.write_bytes(response.content)
                os.replace(tmp, dest)
                return None, True
            last = f"HTTP {response.status_code}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        except (requests.RequestException, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(attempt)
    return f"{url}: {last}", False


def download_vault_images(
    jobs,
    *,
    session=None,
    user_agent: Optional[str] = None,
    workers: int = DEFAULT_IMAGE_WORKERS,
    timeout: int = 60,
    retries: int = 3,
    show_progress: bool = True,
    desc: str = "EcoTaxa vignettes",
) -> tuple[int, list[str]]:
    """Fetch ``(url, destination Path)`` pairs concurrently, skipping what is on disk.

    Resumable by construction: an existing destination is left untouched, so a re-run
    after an interruption fetches only what is missing. Each file is written to a
    ``.part`` and renamed, so a destination that exists is always a complete file.

    A failing vignette is COLLECTED, not raised: one dead object must not abandon a
    million good ones. The caller decides what a non-empty failure list means.

    Returns:
        ``(n downloaded this call, n already on disk, failures)``. The first two are kept
        apart so a resumed run reports the work it did rather than counting every skip.
    """
    jobs = list(jobs)
    if not jobs:
        return 0, 0, []

    headers = {"User-Agent": user_agent} if user_agent else {}
    owns_session = session is None
    session = session or requests.Session()

    fetched, skipped, failures = 0, 0, []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(_fetch_one_image, job, session=session, headers=headers, timeout=timeout, retries=retries)
                for job in jobs
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="img",
                leave=False,
                disable=not show_progress,
            ):
                error, downloaded = future.result()
                if error is not None:
                    failures.append(error)
                elif downloaded:
                    fetched += 1
                else:
                    skipped += 1
    finally:
        if owns_session:
            session.close()

    if failures:
        logger.warning(f"{len(failures)} vignette(s) could not be fetched; first: {failures[0]}")
    return fetched, skipped, failures
