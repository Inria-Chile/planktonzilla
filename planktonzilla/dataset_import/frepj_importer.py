"""
(c) Inria

FREPJ-Z importer: turn the extracted two-magnification FREPJ archive
(``plankton_images/{images_40,images_100}/<taxon>/*.jpg``) into a deterministic,
collision-safe, uniformly-RGB HuggingFace imagefolder.

This module is a thin, output-preserving subclass of the frozen
:class:`~planktonzilla.dataset_import.dataset_importer.DatasetImporter`. All of the
acquisition/layout facts (archive root, magnification prefixes, RGB normalization
target, license/URL/citation constants) live in the Phase-15
:mod:`planktonzilla.dataset_import.frepj_layout` seam and are REUSED here rather than
re-implemented. The importer overrides one lifecycle hook, ``_prepare_imagefolder``, which
(1) flattens both magnification roots into one class dir per taxon via
:func:`~planktonzilla.dataset_import.frepj_layout.merge_two_magnification_roots` and
(2) normalizes every image to RGB by DECODED CONTENT (never by extension), so the five
PNG/RGBA files carrying a ``.jpg`` name become genuine RGB JPEGs.

It also implements the base class's SIDECAR protocol (``sidecar_targets`` /
``missing_sidecars`` / ``ensure_sidecars``): the redefine step joins three md5-pinned
geodata tables (Table_S1/S3/S4, figshare article 26891563) and the committed site
crosswalk on EVERY run, imagefolder reused or not. The importer — which owns the
download config, the project User-Agent and ``data_dir`` — fetches the tables into
``<data_dir>/frepj_tables`` and hands them to ``FrepjRedefiner`` through
``attach_sidecars``; the redefiner itself never downloads.
"""

from pathlib import Path

from datasets.download import DownloadConfig
from PIL import Image, UnidentifiedImageError

from planktonzilla.dataset_import import frepj_layout
from planktonzilla.dataset_import.dataset_importer import DatasetImporter
from planktonzilla.planktonzilla_dataset import frepj_tables
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# Explicit re-encode quality for the RGB-by-content normalization pass (WR-02). Pinned
# here (rather than in the frozen frepj_layout.py seam, out of Phase 16's file scope) so
# the re-encoded bytes of the handful of non-JPEG outliers don't silently drift across
# Pillow/libjpeg versions by depending on Pillow's implicit default (currently 75). 95 is
# Pillow's own commonly-recommended "visually lossless" ceiling for re-encodes.
IMPORT_JPEG_QUALITY = 95


class FREPJDatasetImporter(DatasetImporter):
    """Importer for the FREPJ-Z freshwater zooplankton dataset (two magnifications).

    The extracted archive nests two magnification roots — ``images_40`` and
    ``images_100`` — under ``plankton_images/``. This importer merges them into ONE
    class dir per taxon (magnification is per-image metadata, not a doubled class),
    resolving cross-magnification basename collisions with the ``40_``/``100_`` filename
    prefixes from the Phase-15 seam. It never silently overwrites a file.

    After merging, it normalizes every image to RGB by DECODED CONTENT: PIL sniffs the
    real format from the bytes, so a file wearing a ``.jpg`` extension but holding
    PNG/RGBA content is still converted and re-encoded to a genuine RGB JPEG. Files that
    fail to decode are left in place for the base class's ``check_image_file_integrity``
    pass (enabled in ``frepj.yaml``) to drop via ``is_valid_image_file`` — normalization
    never raises on a corrupt image.
    """

    # Class attributes, like GlobalUVP5NetDatasetImporter.OBJECTS_URL: a test points them
    # at a synthetic manifest / a fixture crosswalk without touching the instance config.
    SIDECAR_MANIFEST = frepj_tables.FREPJ_TABLE_MANIFEST
    CROSSWALK_PATH = frepj_tables.DEFAULT_CROSSWALK_PATH

    def __post_init__(self):
        super().__post_init__()
        # A sibling of raw_dir / imagefolder_dir under the RUN's data_dir (the uvp5
        # `global_uvp5_aux` idiom), named after frepj_tables.TABLES_DIRNAME so a
        # default-data_dir run and the crosswalk CLI verify and share ONE copy.
        self.sidecar_dir = self.data_dir / frepj_tables.TABLES_DIRNAME

    def sidecar_targets(self) -> list[tuple[str, str]]:
        """The three md5-pinned tables (fetched) and the committed crosswalk (bundled)."""
        return [*(("url", entry["url"]) for entry in self.SIDECAR_MANIFEST), ("bundled", str(self.CROSSWALK_PATH))]

    def missing_sidecars(self) -> list[Path]:
        """Tables absent from ``sidecar_dir`` or failing their md5 pin — what a run would fetch."""
        return [
            self.sidecar_dir / entry["name"]
            for entry in frepj_tables.missing_frepj_tables(self.sidecar_dir, self.SIDECAR_MANIFEST)
        ]

    def ensure_sidecars(self) -> dict[str, Path]:
        """Fetch the missing tables md5-verified into ``sidecar_dir``; return every sidecar path.

        The committed crosswalk is checked first: it is not downloadable, so its absence is
        a broken checkout and no fetch can repair it. A fetch failure (dead host, md5
        mismatch, unwritable directory) is re-raised with the exact remedy — every URL, its
        md5 and destination — the way a missing manual archive is reported.
        """
        crosswalk = Path(self.CROSSWALK_PATH)
        if not crosswalk.exists():
            raise FileNotFoundError(
                f"«frepj» needs the committed site crosswalk {crosswalk}, which is not on disk. It ships with the "
                f"package: restore it with `git checkout -- {crosswalk}` or regenerate it with "
                "`python -m planktonzilla.planktonzilla_dataset.frepj_crosswalk`."
            )
        try:
            tables = frepj_tables.ensure_frepj_tables(
                self.sidecar_dir, self.SIDECAR_MANIFEST, download_config=self._sidecar_download_config()
            )
        except Exception as e:
            raise RuntimeError(
                f"«frepj» could not obtain its md5-pinned sidecar tables: {type(e).__name__}: {e}\n"
                + frepj_tables.sidecar_instructions(self.sidecar_dir, self.SIDECAR_MANIFEST, source=self.hf_dataset_name)
            ) from e
        return {**tables, frepj_tables.CROSSWALK_SIDECAR_KEY: crosswalk}

    def _sidecar_download_config(self) -> DownloadConfig:
        """The importer's download config for the tables, so probe and fetch cannot disagree.

        Mirrors ``_download_and_extract`` (``num_proc=1``, the project User-Agent via
        ``storage_options``). ``force_download=True`` on purpose: this config is only used
        for a file that is absent or failed its md5, so a cached blob for that URL is by
        definition not wanted.
        """
        return DownloadConfig(
            # A scratch cache beside the pinned CSVs, not among them: the manager's blobs,
            # .json/.lock companions and any .incomplete of a failed fetch stay out of a
            # directory that is otherwise exactly the verified tables.
            cache_dir=self.sidecar_dir / ".download_cache",
            force_download=True,
            resume_download=self.resume_download,
            max_retries=self.max_download_retries,
            num_proc=1,
            disable_tqdm=not self.show_progress,
            storage_options=self.storage_options(),
        )

    def _prepare_imagefolder(self):
        """Merge both magnification roots into the imagefolder, then normalize to RGB.

        Reuses :func:`frepj_layout.merge_two_magnification_roots` for the sorted,
        collision-safe, one-class-per-taxon flatten (IMP-01 + the sorted-determinism half
        of IMP-03), then applies the RGB-by-content normalization pass (IMP-03). No
        directory is cleared first — the seam's ``if not target.exists()`` guard keeps
        rebuilds idempotent and layout-deterministic.
        """
        archive_root = Path(self.extracted_dirs) / frepj_layout.ARCHIVE_ROOT
        images_40 = archive_root / frepj_layout.MAGNIFICATION_PREFIXES[0][0]
        images_100 = archive_root / frepj_layout.MAGNIFICATION_PREFIXES[1][0]

        copied = frepj_layout.merge_two_magnification_roots(images_40, images_100, self.imagefolder_dir)
        logger.info(f"Merged {copied} FREPJ image(s) from both magnifications into «{self.imagefolder_dir}».")

        self._normalize_images_to_rgb()

    def _normalize_images_to_rgb(self):
        """Convert every merged image to RGB by decoded content (deterministic pass).

        Enumerates class dirs and files in ``sorted`` order. Each file is decoded with
        PIL (which sniffs the real format regardless of the ``.jpg`` extension); files
        already a genuine :data:`frepj_layout.IMPORT_NORMALIZATION`-mode JPEG are left
        byte-untouched to avoid needless re-encoding, while anything else (non-RGB mode,
        OR RGB-mode-but-non-JPEG content such as a PNG-container file wearing a ``.jpg``
        name) is converted and re-saved as a genuine RGB JPEG under the same filename.
        Undecodable/corrupt files are logged and skipped (never raised) so the base
        class's integrity filter can remove them.
        """
        for class_dir in sorted(p for p in self.imagefolder_dir.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                try:
                    with Image.open(image_path) as img:
                        img.load()
                        if img.mode == frepj_layout.IMPORT_NORMALIZATION and img.format == "JPEG":
                            continue
                        converted = img.convert(frepj_layout.IMPORT_NORMALIZATION)
                    converted.save(image_path, format="JPEG", quality=IMPORT_JPEG_QUALITY)
                except (OSError, SyntaxError, Image.DecompressionBombError, UnidentifiedImageError):
                    # DecompressionBombError is a direct Exception subclass (NOT OSError), raised
                    # by Image.open()/img.load() when a file's declared pixel dimensions exceed
                    # Image.MAX_IMAGE_PIXELS -- a realistic corrupt/bit-flipped-header failure mode
                    # for this externally-sourced ~88k-image archive. UnidentifiedImageError is
                    # already an OSError subclass; naming it documents intent. This except clause
                    # must never let a single bad file abort the whole import (module contract
                    # above): log and skip, leaving the file for the integrity filter.
                    logger.warning(f"Could not decode «{image_path}»; leaving it for the integrity filter.")
                    continue
