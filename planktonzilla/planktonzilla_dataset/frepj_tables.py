"""
(c) Inria

md5-verified fetch + pure CSV parsers for the three shipped FREPJ-Z sidecar tables.

The per-image ``(magnification, ID) -> (site_token, date)`` mapping the geodata
redefiner needs lives in ``Table_S3.csv`` (40x, ~62k rows) and ``Table_S4.csv``
(100x, ~26k rows), and the romanized-site -> lat/lon reference lives in
``Table_S1.csv`` — three separate figshare files that are NOT inside the image
zip. Rather than committing an ~2 MB derived index into VCS (the repo's Core
Value is lean, output-preserving code), this module fetches the three CSVs once
into a gitignored ``data/frepj_tables/`` directory via an md5-pinned
:func:`ensure_frepj_tables`, and exposes pure parsers over them.

Load-bearing source-data facts (pinned from ``15-RESEARCH.md`` — do NOT "fix"):

  * ``Table_S1.csv`` columns are ``site, North latitude, East latitude, date``.
    The ``East latitude`` column actually holds LONGITUDE — a typo in the source
    dataset. :func:`read_site_coordinates` reads it as longitude by design.
  * ``Table_S3.csv`` / ``Table_S4.csv`` headers carry trailing spaces on the
    ``Order `` / ``Others `` columns, and ``Table_S4`` rows carry stray trailing
    commas (extra empty columns). The parsers strip header whitespace and
    tolerate the ragged trailing columns.
  * ``ID`` is the integer filename stem; it is canonicalized via ``str(int(...))``
    so a zero-padded id joins the ``40_<ID>.jpg`` / ``100_<ID>.jpg`` merged
    filenames produced by ``frepj_layout``.

Zero behavioral drift: nothing here mutates any frozen artifact. Only network
path is :func:`ensure_frepj_tables`; every parser is offline and pure. No new
third-party dependency is introduced (polars/datasets are already vendored;
``hashlib``/``shutil``/``pathlib`` are stdlib).
"""

import hashlib
import shutil
from collections import defaultdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import polars as pl
from datasets.download import DownloadConfig, DownloadManager

from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


# --- Frozen figshare identities for the three sidecar tables --------------------------
# Mirrors tests/fixtures/frepj/frepj_figshare_manifest.json (article 26891563); pinned
# here so runtime never re-fetches the manifest. Only Table_S1/S3/S4 are geodata inputs.

FREPJ_TABLE_MANIFEST: list[dict] = [
    {
        "name": "Table_S1.csv",
        "file_id": 49092727,
        "url": "https://ndownloader.figshare.com/files/49092727",
        "md5": "15a9b8c25c483c0e42d3c637a955eb08",
        "size": 14088,
    },
    {
        "name": "Table_S3.csv",
        "file_id": 49092739,
        "url": "https://ndownloader.figshare.com/files/49092739",
        "md5": "3edfbeca846bfad2f324252e925a641c",
        "size": 6064306,
    },
    {
        "name": "Table_S4.csv",
        "file_id": 49092742,
        "url": "https://ndownloader.figshare.com/files/49092742",
        "md5": "3714ffbc3f99a221a0fbc70f8f3bdde6",
        "size": 2527004,
    },
]


# --- Path constants -------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).parent
# Repo-root ``data/`` is gitignored (see .gitignore ``/data/``); the 8.5 MB tables land
# here and are NEVER committed. PACKAGE_DIR.parents[1] == the repository root.
DEFAULT_TABLES_DIR = PACKAGE_DIR.parents[1] / "data" / "frepj_tables"
DEFAULT_CROSSWALK_PATH = PACKAGE_DIR / "frepj_site_crosswalk.csv"
DEFAULT_OVERRIDES_PATH = PACKAGE_DIR / "frepj_site_overrides.csv"

# The two magnification roots encoded in the merged filenames (see frepj_layout).
_MAGNIFICATION_PREFIXES = frozenset({"40", "100"})


# --- md5 helper -----------------------------------------------------------------------


def _md5(path: str | Path) -> str:
    """Return the hex md5 digest of a file (used for supply-chain verification)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --- Pure parsers ---------------------------------------------------------------------


def _read_csv(path: str | Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Read a sidecar CSV as all-string columns, tolerating ragged trailing commas.

    Returns the frame plus a mapping of whitespace-stripped column name -> actual
    column name so callers can select ``Order ``/``Others `` columns and ignore
    ``Table_S4``'s stray empty trailing columns.
    """
    df = pl.read_csv(path, infer_schema_length=0, truncate_ragged_lines=True)
    stripped = {c.strip(): c for c in df.columns}
    return df, stripped


def _parse_coord(raw: str | None, lo: float, hi: float) -> float | None:
    """Coerce a raw coordinate string to float, rejecting blanks / out-of-range to None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value < lo or value > hi:
        return None
    return value


# --- Duplicate-site conflict detection (CR-01, refined) --------------------------------
# Table_S1.csv carries one row per (site, sampling date), so a site sampled repeatedly
# appears more than once. Most repeat rows agree to within ordinary same-site
# measurement imprecision -- anywhere from a few metres up to a few km for a large
# reservoir surveyed from different shore points (e.g. ``Miharu Dam`` ~1.06 km,
# ``Lake Ashino`` ~5.5 km) -- but at least one site name is a genuine collision between
# two distinct real-world places (``Tsurugajo``, ~138 km apart; ``Kincho Dam``, an
# Okinawa/Hokkaido split ~2,360 km apart). ``_SITE_COORD_CONFLICT_KM`` separates the two:
# rows for the same site name that disagree by no more than this distance are folded to
# their CENTROID (the arithmetic mean of every valid row, more robust than picking any
# single row); rows that disagree by more are a genuine name collision and are never
# silently folded together (see 17-REVIEW.md CR-01 and its follow-up).
_EARTH_RADIUS_KM = 6371.0088
_SITE_COORD_CONFLICT_KM = 10.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two ``(lat, lon)`` points.

    Public (not underscore-prefixed) because :mod:`frepj_crosswalk` reuses it for its
    own fuzzy-match tie/conflict guard (WR-01) -- both modules share the same notion of
    "materially different real-world locations."
    """
    lat1, lon1 = radians(a[0]), radians(a[1])
    lat2, lon2 = radians(b[0]), radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(h))


def read_site_coordinates(s1_path: str | Path) -> dict[str, tuple[float | None, float | None]]:
    """Map each Table_S1 site name to ``(latitude, longitude)``.

    ``North latitude`` -> latitude and ``East latitude`` -> LONGITUDE (the ``East
    latitude`` header is a source typo; it holds longitude — see 15-RESEARCH.md).
    A row whose latitude falls outside ``[-90, 90]`` or longitude outside
    ``[-180, 180]`` (or is unparseable) contributes no coordinate for that row, never
    a guessed one; a site with no valid row resolves to ``(None, None)``.

    ``Table_S1.csv`` has one row per ``(site, date)``, so a repeatedly-sampled site has
    multiple rows. When every valid row for a site agrees to within
    :data:`_SITE_COORD_CONFLICT_KM` (ordinary same-site measurement imprecision across
    sampling dates/shore points — e.g. ``Miharu Dam`` ~1.06 km, ``Lake Ashino`` ~5.5 km),
    the site resolves to the CENTROID of its valid rows (the arithmetic mean of every
    ``(lat, lon)`` pair), which is more robust than last-write-wins. When two rows for
    the SAME site name disagree by MORE than that — a same-name collision describing two
    different real-world locations, e.g. ``Tsurugajo`` (~138 km apart) or ``Kincho Dam``
    (~2,360 km apart) — the site is treated as CONFLICTED/ambiguous and resolves to
    ``(None, None)`` rather than silently guessing (CR-01; never emit a wrong coordinate).
    """
    df, cols = _read_csv(s1_path)
    site_col = cols["site"]
    lat_col = cols["North latitude"]
    lon_col = cols["East latitude"]  # NB: source typo — this column is the longitude.

    valid_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen_sites: set[str] = set()
    for row in df.select(
        pl.col(site_col).alias("site"),
        pl.col(lat_col).alias("lat"),
        pl.col(lon_col).alias("lon"),
    ).iter_rows(named=True):
        site = str(row["site"]).strip()
        if not site:
            continue
        seen_sites.add(site)
        lat = _parse_coord(row["lat"], -90.0, 90.0)
        lon = _parse_coord(row["lon"], -180.0, 180.0)
        if lat is not None and lon is not None:
            valid_pairs[site].append((lat, lon))
        # A single bad axis contributes no coordinate for THIS row (never a
        # half-guessed point); it does not by itself null a site that has other
        # valid rows -- the per-site aggregation below decides the final value.

    coords: dict[str, tuple[float | None, float | None]] = {}
    for site in seen_sites:
        pairs = valid_pairs.get(site, [])
        if not pairs:
            coords[site] = (None, None)
            continue
        max_spread = max(
            (haversine_km(pairs[i], pairs[j]) for i in range(len(pairs)) for j in range(i + 1, len(pairs))),
            default=0.0,
        )
        if max_spread > _SITE_COORD_CONFLICT_KM:
            logger.warning(
                f"«{site}» has Table_S1 rows disagreeing by up to {max_spread:.1f} km "
                f"(> {_SITE_COORD_CONFLICT_KM:g} km tolerance) -- treating as a same-name "
                "collision and resolving to null rather than guessing (CR-01)."
            )
            coords[site] = (None, None)
        else:
            # Rows agree within tolerance (ordinary same-site measurement imprecision) --
            # resolve to their CENTROID rather than picking any single row.
            mean_lat = sum(p[0] for p in pairs) / len(pairs)
            mean_lon = sum(p[1] for p in pairs) / len(pairs)
            coords[site] = (mean_lat, mean_lon)
    return coords


def _canonical_id(raw: str | None) -> str | None:
    """Canonicalize an ``ID`` cell to ``str(int(...))`` or None when non-integer."""
    if raw is None:
        return None
    try:
        return str(int(str(raw).strip()))
    except ValueError:
        return None


def _index_one_table(
    path: str | Path,
    magnification: int,
    index: dict[tuple[int, str], tuple[str, str]],
) -> None:
    """Fold one sidecar table's rows into ``index`` keyed by ``(magnification, id)``."""
    df, cols = _read_csv(path)
    subset = df.select(
        pl.col(cols["ID"]).alias("id"),
        pl.col(cols["Sampling site"]).alias("site_token"),
        pl.col(cols["Sampling date"]).alias("date"),
    )
    for row in subset.iter_rows(named=True):
        canonical = _canonical_id(row["id"])
        if canonical is None:
            logger.debug(f"Skipping row with non-integer ID «{row['id']}» in {path}.")
            continue
        token = "" if row["site_token"] is None else str(row["site_token"]).strip()
        date = "" if row["date"] is None else str(row["date"]).strip()
        index[(magnification, canonical)] = (token, date)


def read_per_image_site_index(
    s3_path: str | Path,
    s4_path: str | Path,
) -> dict[tuple[int, str], tuple[str, str]]:
    """Build the per-image ``(magnification, id) -> (site_token, date)`` mapping.

    Table_S3 rows are keyed at magnification 40, Table_S4 rows at 100. Trailing-space
    headers and Table_S4's stray trailing-comma columns are tolerated; ``id`` is
    canonicalized via ``str(int(...))`` and non-integer ids are skipped (logged at
    debug), never raised.
    """
    index: dict[tuple[int, str], tuple[str, str]] = {}
    _index_one_table(s3_path, 40, index)
    _index_one_table(s4_path, 100, index)
    return index


def parse_frepj_filename(filename: str) -> tuple[int, str] | None:
    """Parse a merged FREPJ filename into ``(magnification, id)``.

    ``"40_123.jpg" -> (40, "123")``; ``"100_9.jpg" -> (100, "9")``. The ``id`` is
    canonicalized via ``str(int(...))`` so it joins the per-image index. A name
    whose prefix is not ``40``/``100`` or whose stem is non-numeric returns None
    and never raises.
    """
    stem = Path(str(filename)).stem
    prefix, separator, tail = stem.partition("_")
    if not separator or prefix not in _MAGNIFICATION_PREFIXES or not tail.isdigit():
        return None
    return (int(prefix), str(int(tail)))


def distinct_site_tokens(index: dict[tuple[int, str], tuple[str, str]]) -> set[str]:
    """Return the set of distinct sampling-site tokens present in ``index``."""
    return {token for token, _date in index.values()}


# --- md5-verified fetch ---------------------------------------------------------------


def ensure_frepj_tables(dest_dir: str | Path, manifest: list[dict] | None = None) -> dict[str, Path]:
    """Ensure the three sidecar tables exist locally, md5-verified, downloading only misses.

    For each manifest entry, if ``dest_dir/<name>`` already exists with the frozen
    md5, it is used as-is with NO download. Otherwise it is fetched (via the same
    ``datasets`` ``DownloadManager`` the importer uses) and md5-verified;
    a post-download mismatch raises ``ValueError`` and the bytes are never used
    (T-17-01). Returns ``{table_name: Path}``.
    """
    manifest = FREPJ_TABLE_MANIFEST if manifest is None else manifest
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    missing: list[dict] = []
    for entry in manifest:
        target = dest_dir / entry["name"]
        if target.exists() and _md5(target) == entry["md5"]:
            logger.info(f"«{entry['name']}» already present with matching md5; skipping download.")
            paths[entry["name"]] = target
        else:
            missing.append(entry)

    if missing:
        manager = DownloadManager(
            base_path=dest_dir,
            data_dir=dest_dir,
            download_config=DownloadConfig(
                cache_dir=dest_dir,
                force_download=False,
                resume_download=True,
                max_retries=5,
            ),
        )
        for entry in missing:
            target = dest_dir / entry["name"]
            logger.info(f"Downloading «{entry['name']}» from {entry['url']}.")
            downloaded = Path(manager.download(entry["url"]))
            shutil.copyfile(downloaded, target)
            digest = _md5(target)
            if digest != entry["md5"]:
                raise ValueError(
                    f"md5 mismatch for «{entry['name']}»: expected {entry['md5']}, got {digest}. "
                    "Refusing to use unverified bytes (T-17-01)."
                )
            logger.info(f"Verified «{entry['name']}» md5 {digest}.")
            paths[entry["name"]] = target

    return paths
