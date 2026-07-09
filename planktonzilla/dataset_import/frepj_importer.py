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
re-implemented. The importer overrides exactly one hook, ``_prepare_imagefolder``, which
(1) flattens both magnification roots into one class dir per taxon via
:func:`~planktonzilla.dataset_import.frepj_layout.merge_two_magnification_roots` and
(2) normalizes every image to RGB by DECODED CONTENT (never by extension), so the five
PNG/RGBA files carrying a ``.jpg`` name become genuine RGB JPEGs.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from planktonzilla.dataset_import import frepj_layout
from planktonzilla.dataset_import.dataset_importer import DatasetImporter
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


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
                    converted.save(image_path, format="JPEG")
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
