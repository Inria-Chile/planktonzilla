#!/usr/bin/env python
"""Mirror PlanktonSet-1's labeled image tree from NOAA's static FTP copy, once.

Why this exists
---------------
`planktonset1.0` is published two ways, and both are awkward:

- ``download_uris`` names NCEI's Archive Management System *download* endpoint, which
  does not serve a stored file — it tars the whole 0127422 accession on demand. Measured
  2026-08-27: ~22 s to first byte, ~0.6 MB/s (about an hour), ``Range`` IGNORED (a ranged
  GET answers 200, not 206), no ``Content-Length``, and a sustained ``503`` when busy —
  it was still 503 a day later. An hour-long transfer that cannot resume and cannot be
  size-checked fails often and loses everything each time.
- NOAA also mirrors the same accession statically, but publishes no archive there — only
  the extracted tree. Fetching it is resumable and verifiable, but it is ~60k round trips.

The data itself, though, is 121 classes / 60,736 JPEGs / 103.7 MB, and has not changed
since 2015. Re-fetching an immutable 104 MB dataset from a fragile generator on every
fresh machine is the actual problem. So: mirror it ONCE with this script, keep the
resulting archive somewhere you control, and point the importer at it forever after::

    uv run python scripts/mirror_planktonset1.py
    # then, on any machine, to build this source's imagefolder from the archive:
    uv run pz_import_dataset action=import dataset_import=planktonset1 \
      dataset_import.push_to_hub=false \
      dataset_import.manual_download_local_file_names=<abs path>/0127422.2.3.tar.gz

``action=import`` is not optional: the entry point defaults to ``action=show``, which only
queries the Hub and never imports anything.

Inside a full ``pz_planktonzilla`` run the same override goes through ``import_overrides``,
where BOTH the list and the element need quoting — the brackets are a glob to zsh, and
Hydra's override grammar rejects a bare ``=`` inside a list element::

    uv run pz_planktonzilla repo_id=<org>/<name> "sources=[planktonset1.0]" base=local \
      'import_overrides=["dataset_import.manual_download_local_file_names=<abs path>.tar.gz"]'

Note that ``import_overrides`` is appended to EVERY selected source's block, so pass it
only when ``sources`` is this one source. And once the imagefolder exists, ``refresh=reuse``
skips the import entirely and the override is never read — it matters on a fresh machine,
which is the whole point of keeping the archive.

FTP rather than the HTTPS form of the same mirror on purpose: ``ncei.noaa.gov/robots.txt``
says ``Disallow: /data*``, which covers the HTTPS mirror path, while the FTP tree is the
bulk route the accession landing page itself advertises.

Resumable by construction: a file already on disk at its expected size is skipped, so an
interrupted run is finished by re-running rather than restarted. Each file is written to a
``.part`` and renamed, so a file that exists is always complete.
"""

import argparse
import ftplib
import io
import json
import os
import queue
import sys
import tarfile
import threading
import time
from pathlib import Path

HOST = "ftp-oceans.ncei.noaa.gov"
ROOT = "/nodc/archive/arc0075/0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014"
SEGMENTS_DIRNAME = "FINAL_Plankton_Segments_12082014"

# Surveyed against the live mirror 2026-08-27. The run refuses to package anything that
# does not match these exactly, so a partial mirror can never be handed to the importer
# as if it were the whole source.
EXPECTED_CLASSES = 121
EXPECTED_FILES = 60_736
EXPECTED_BYTES = 108_723_866


def _connect(retries=5):
    """One anonymous FTP session, retried — the host drops connections under load."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            ftp = ftplib.FTP(HOST, timeout=120)
            ftp.login()
            ftp.set_pasv(True)
            return ftp
        except ftplib.all_errors as e:
            last = e
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"could not connect to {HOST} after {retries} attempts: {last}")


def _listdir(ftp, path):
    """``(dirs, {name: size})`` for one remote directory.

    Parsed from LIST because this server rejects MLSD and refuses a path argument to
    LIST, so the session must CWD first. The size column here is the real byte count —
    unlike the HTTPS mirror's Apache index, which rounds to two significant figures and
    is therefore useless as a verification source.
    """
    ftp.cwd(path)
    lines = []
    ftp.retrlines("LIST", lines.append)

    dirs, files = [], {}
    for line in lines:
        parts = line.split(maxsplit=8)
        if len(parts) < 9:
            continue
        name = parts[8]
        if line.startswith("d"):
            dirs.append(name)
        else:
            files[name] = int(parts[4])
    return dirs, files


def _survey(destination):
    """Build (or reuse) the per-class manifest of what the mirror holds."""
    cache = destination / ".manifest.json"
    if cache.exists():
        manifest = json.loads(cache.read_text())
        print(f"Reusing manifest for {len(manifest)} classes ({cache}).", flush=True)
        return manifest

    print("Surveying the mirror (one pass, ~2 min)…", flush=True)
    ftp = _connect()
    classes, _ = _listdir(ftp, ROOT)
    manifest = {}
    for i, klass in enumerate(classes, 1):
        _, files = _listdir(ftp, f"{ROOT}/{klass}")
        manifest[klass] = files
        if i % 20 == 0:
            print(f"  surveyed {i}/{len(classes)} classes", flush=True)
    ftp.quit()

    destination.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(manifest))
    return manifest


def _worker(jobs, out_root, counters, lock):
    """Drain per-class jobs over one long-lived connection, reconnecting as needed."""
    ftp = _connect()
    while True:
        try:
            klass, files = jobs.get_nowait()
        except queue.Empty:
            break

        target_dir = out_root / klass
        target_dir.mkdir(parents=True, exist_ok=True)

        for name, size in files.items():
            destination = target_dir / name
            # Resume: a file already at its declared size is complete, so skip it.
            if destination.exists() and destination.stat().st_size == size:
                with lock:
                    counters["skipped"] += 1
                continue

            for attempt in range(1, 4):
                try:
                    buffer = io.BytesIO()
                    ftp.cwd(f"{ROOT}/{klass}")
                    ftp.retrbinary(f"RETR {name}", buffer.write)
                    payload = buffer.getvalue()
                    if len(payload) != size:
                        raise OSError(f"got {len(payload)} bytes, expected {size}")

                    partial = target_dir / f".{name}.part"
                    partial.write_bytes(payload)
                    os.replace(partial, destination)
                    with lock:
                        counters["fetched"] += 1
                        counters["bytes"] += len(payload)
                    break
                except ftplib.all_errors as e:
                    if attempt == 3:
                        with lock:
                            counters["failed"] += 1
                            counters["failures"].append(f"{klass}/{name}: {e}")
                    else:
                        time.sleep(2 * attempt)
                        try:
                            ftp.quit()
                        except Exception:
                            pass
                        ftp = _connect()

        jobs.task_done()

    try:
        ftp.quit()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/manual_downloads/planktonset1"),
        help="where the tree and the packaged archive are written",
    )
    # 4, not more: an investigator stalled this host for ~10 minutes with 6 concurrent
    # streams, and the failure mode was a silent hang rather than an error.
    parser.add_argument("--workers", type=int, default=4, help="concurrent FTP connections (keep <= 4)")
    parser.add_argument("--no-archive", action="store_true", help="mirror only; do not package a .tar.gz")
    args = parser.parse_args()

    destination = args.destination.resolve()
    out_root = destination / SEGMENTS_DIRNAME
    manifest = _survey(destination)

    total_files = sum(len(files) for files in manifest.values())
    total_bytes = sum(sum(files.values()) for files in manifest.values())
    print(
        f"Mirror holds {len(manifest)} classes, {total_files} files, {total_bytes} bytes.",
        flush=True,
    )

    jobs = queue.Queue()
    for klass, files in sorted(manifest.items()):
        jobs.put((klass, files))

    counters = {"fetched": 0, "skipped": 0, "failed": 0, "bytes": 0, "failures": []}
    lock = threading.Lock()
    started = time.time()

    threads = [
        threading.Thread(target=_worker, args=(jobs, out_root, counters, lock), daemon=True)
        for _ in range(max(1, args.workers))
    ]
    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        time.sleep(30)
        with lock:
            done = counters["fetched"] + counters["skipped"]
        elapsed = time.time() - started
        rate = counters["fetched"] / elapsed if elapsed else 0
        remaining = (total_files - done) / rate / 60 if rate else float("inf")
        print(
            f"  {done}/{total_files} files ({counters['fetched']} fetched, {counters['skipped']} skipped, "
            f"{counters['failed']} failed) — {rate:.1f}/s, ~{remaining:.0f} min left",
            flush=True,
        )

    for thread in threads:
        thread.join()

    on_disk = sorted(p for p in out_root.rglob("*") if p.is_file() and not p.name.startswith("."))
    bytes_on_disk = sum(p.stat().st_size for p in on_disk)
    classes_on_disk = len([p for p in out_root.iterdir() if p.is_dir()])

    print(
        f"\nDone in {(time.time() - started) / 60:.1f} min: {classes_on_disk} classes, "
        f"{len(on_disk)} files, {bytes_on_disk} bytes.",
        flush=True,
    )
    if counters["failures"]:
        print(f"{len(counters['failures'])} failure(s); first few:", flush=True)
        for failure in counters["failures"][:5]:
            print(f"  {failure}", flush=True)

    # Refuse to package a partial mirror: an archive that LOOKS complete is exactly the
    # failure mode this whole exercise exists to remove.
    complete = classes_on_disk == EXPECTED_CLASSES and len(on_disk) == EXPECTED_FILES and bytes_on_disk == EXPECTED_BYTES
    if not complete:
        print(
            f"\nINCOMPLETE — expected {EXPECTED_CLASSES} classes / {EXPECTED_FILES} files / "
            f"{EXPECTED_BYTES} bytes. Re-run to fetch what is missing (finished files are skipped).",
            flush=True,
        )
        return 1

    print("Matches the surveyed mirror exactly.", flush=True)

    if args.no_archive:
        return 0

    archive = destination / "0127422.2.3.tar.gz"
    print(f"Packaging {archive}…", flush=True)
    tmp = archive.with_suffix(".tar.gz.part")
    with tarfile.open(tmp, "w:gz") as tar:
        # Arcname keeps the FINAL_Plankton_Segments_* directory at the archive root; the
        # importer locates that tree wherever it sits, so no accession wrapper is needed.
        tar.add(out_root, arcname=SEGMENTS_DIRNAME)
    os.replace(tmp, archive)

    print(
        f"\nWrote {archive} ({archive.stat().st_size} bytes).\n\n"
        f"Keep this somewhere you control, then build the imagefolder from it on any machine:\n"
        f"  uv run pz_import_dataset action=import dataset_import=planktonset1 \\\n"
        f"    dataset_import.push_to_hub=false \\\n"
        f"    dataset_import.manual_download_local_file_names={archive}\n\n"
        f"(action=import is required — the entry point defaults to action=show, which only\n"
        f"queries the Hub. Inside a full pz_planktonzilla run the override goes through\n"
        f"import_overrides, where the list AND the element both need quoting.)\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
