"""
(c) Inria

The SINGLE network boundary for the planktonzilla explorer (FND-06).

Every read that crosses to the network (the live public HF dataset
``project-oceania/planktonzilla-17M``) goes through THIS module and nowhere else.
The boundary is built as an injectable seam (D2):

* ``load_geo(repo_id, *, loader=None)`` — the public entry point. With ``loader``
  injected (tests pass a fake frame) the network is never reached and the real
  read code never runs. With no loader it calls the real, in-process-cached reader.
* ``_default_geo_loader`` — the real implementation. It lazy-imports
  ``huggingface_hub``/``pyarrow`` INSIDE the function body (so importing this
  module stays cheap and the pure-test path never resolves them) and does a
  COLUMN-PROJECTED parquet read of EXACTLY ``{Latitude, Longitude, dataset}`` —
  it NEVER reads/downloads the ``image`` column (17M rows; pulling images is an
  OOM/cost/timeout hazard — T-10-01). The dataset is PUBLIC, so no token is passed.

Module-scope imports are stdlib + ``polars`` + the shared
``planktonzilla_dataset.constants`` only. NO gradio/plotly anywhere (Phase 9 guard),
and NO datasets/huggingface_hub/pyarrow at module scope (lazy seam only).
"""

import functools
from pathlib import Path

import polars as pl

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# The EXACT columns the real geo read projects — and the ONLY columns it touches.
# `image` is deliberately absent: downloading the 17M-row image column is forbidden
# (T-10-01). Tests assert `"image" not in GEO_COLUMNS`.
GEO_COLUMNS = ("Latitude", "Longitude", "dataset")

# HF dataset coordinates (single config `default`, split `train` — verified 2026-06-30).
HF_SPLIT = "train"

# The committed inferred-locations CSV (local, no network) — Phase 8 output.
INFERRED_LOCATIONS_CSV = Path(__file__).parent / "data" / "inferred_dataset_locations.csv"


@functools.lru_cache(maxsize=4)
def _default_geo_loader(repo_id: str = constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID) -> pl.DataFrame:
    """Read per-sample geo from the live public HF dataset, column-projected.

    This is the ONLY function that crosses to the network. It lazy-imports its heavy
    deps inside the body and resolves the dataset's parquet shard(s) via
    ``HfFileSystem``, then reads EXACTLY ``GEO_COLUMNS`` with pyarrow's column
    projection. The ``image`` column is never listed and therefore never downloaded
    (T-10-01). The dataset is PUBLIC, so NO token is passed.

    Results are memoized per ``repo_id`` for the process lifetime (cold-boot reads
    once). Only the REAL path is cached; the injected-loader path in ``load_geo``
    short-circuits before this function, so a fake never poisons a real entry.

    Args:
        repo_id: HF dataset repo id. Defaults to the frozen public planktonzilla repo.

    Returns:
        A polars DataFrame with columns ``{Latitude, Longitude, dataset}``.
    """
    # Lazy imports: kept out of module scope so cold import stays cheap and the
    # pure/test path never resolves huggingface_hub/pyarrow.
    import pyarrow.dataset as pa_ds
    from huggingface_hub import HfFileSystem

    logger.info(f"Reading geo columns «{GEO_COLUMNS}» from «{repo_id}» (public, no token).")
    fs = HfFileSystem()  # public dataset -> no token
    base = f"datasets/{repo_id}"

    # Discover parquet shards for the `train` split. HF parquet-converted datasets
    # expose shards under the dataset root; glob both the conventional layouts.
    candidates = fs.glob(f"{base}/**/{HF_SPLIT}*.parquet") or fs.glob(f"{base}/**/*.parquet")
    if not candidates:
        raise FileNotFoundError(f"No parquet shards found for split «{HF_SPLIT}» under «{base}».")

    # pyarrow.dataset over the HF filesystem, projected to ONLY the geo columns.
    # `image` is never in the column list, so it is never read or downloaded.
    dataset = pa_ds.dataset(candidates, filesystem=fs, format="parquet")
    table = dataset.to_table(columns=list(GEO_COLUMNS))
    return pl.from_arrow(table).select(list(GEO_COLUMNS))


def load_geo(
    repo_id: str = constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID,
    *,
    loader=None,
) -> pl.DataFrame:
    """Return per-dataset geo points, through the injectable seam.

    If ``loader`` is provided (tests inject a fake), it is called as
    ``loader(repo_id)`` and the network/real-read path is NEVER reached — this is
    the primary network-free guarantee. If ``loader`` is None, the real, in-process
    cached ``_default_geo_loader`` is used.

    Args:
        repo_id: HF dataset repo id. Defaults to the frozen public planktonzilla repo.
        loader: Optional injected loader ``(repo_id) -> pl.DataFrame``. When given,
            short-circuits before any cache or real-read code.

    Returns:
        A polars DataFrame with columns ``{Latitude, Longitude, dataset}``.
    """
    if loader is not None:
        # Injected path: bypass the cache entirely so a fake never poisons a real
        # entry, and the real loader (and its lazy HF/pyarrow imports) is untouched.
        return loader(repo_id)
    return _default_geo_loader(repo_id)


def inferred_locations(csv_path: str | Path = INFERRED_LOCATIONS_CSV) -> pl.DataFrame:
    """Read the committed inferred-locations CSV (local file, NO network).

    Routes the Phase 8 inferred-locations frame through this same IO module so the
    explorer has one consistent data-access surface. The CSV uses LOWERCASE
    ``dataset,latitude,longitude,...`` columns; ``shapes.aggregate_geo`` reconciles
    the casing against the live-HF uppercase columns.

    Args:
        csv_path: Path to the inferred-locations CSV. Defaults to the committed file.

    Returns:
        A polars DataFrame of the inferred-locations CSV (all-string columns).
    """
    return pl.read_csv(Path(csv_path), infer_schema_length=0, null_values=[]).with_columns(
        pl.all().cast(pl.Utf8).fill_null("").str.strip_chars()
    )
