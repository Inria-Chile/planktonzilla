"""
(c) Inria

DAPlankton importer: turn the doubly-nested Fairdata package into a deterministic,
collision-safe HuggingFace imagefolder of 44 taxon classes.

Two facts about this source drive everything here, and both were established by
downloading the real 2.74 GB package and enumerating all 112,040 of its entries (see
:mod:`planktonzilla.dataset_import.daplankton_layout`, which records them):

1. THE ARCHIVE IS DOUBLY NESTED. The Fairdata Download API hands over a package whose
   only member is ``DAPlankton/DAPlankton.zip`` — itself the real archive. Extracting
   once yields a zip, not class folders, so this importer unwraps twice.

2. THE LABEL SPACE IS FIVE-FOLD REDUNDANT. Images are laid out as
   ``<subset>/<instrument>/<class>/``, over two subsets (cultures and Baltic Sea field
   samples) and three instruments (IFCB, CytoSense, FlowCam). Those are per-image
   provenance, not five different label spaces, so the five roots are merged into ONE
   class dir per taxon — exactly the policy FREPJ-Z applies to its two magnifications.
   Subset and instrument survive as a filename prefix and stay recoverable from
   ``original_path`` in the consolidated dataset.

The download itself is inherited unchanged from
:class:`~planktonzilla.dataset_import.dataset_importer.FairdataPackagedDatasetImporter`,
which DAPlankton shares with SYKE ZooScan 2024.
"""

from pathlib import Path

from planktonzilla.dataset_import import daplankton_layout
from planktonzilla.dataset_import.dataset_importer import (
    FairdataPackagedDatasetImporter,
    cleanup_imagefolder_empty_dirs,
    unzip,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)


class DAPlanktonDatasetImporter(FairdataPackagedDatasetImporter):
    """Importer for DAPlankton (Batrakhanov et al. 2024), published through Fairdata.

    :meth:`_prepare_imagefolder` unwraps the inner archive, locates the subset roots
    with :func:`daplankton_layout.find_archive_root` rather than a fixed path, and
    merges the five ``(subset, instrument)`` roots into one class dir per taxon via
    :func:`daplankton_layout.merge_domain_roots`.

    The merge folds 15 lab classes + 31 sea classes into **44** distinct class dirs:
    ``Aphanizomenon_flosaquae`` and ``Pseudopedinella_sp`` are the two taxa imaged in
    both subsets, and one taxon is one class. Those 44 names are frozen in
    ``tests/fixtures/daplankton/daplankton_class_dirs.tsv`` and pinned against the
    taxonomy CSV by ``tests/test_daplankton_taxonomy_coverage.py``, which is the guard
    against the silent left join that would otherwise give a mistyped class null
    taxonomy for every one of its images.

    Images are copied VERBATIM, extension included — CytoSense folders are JPEG and
    IFCB/FlowCam folders are PNG, and both are kept as they are rather than re-encoded.

    .. note::
       Disk, not time, is the binding constraint: the package is ~2.7 GB, the unwrapped
       inner archive another ~2.7 GB, and the merged imagefolder ~2.7 GB again, so a
       from-scratch import wants roughly 9 GB free under ``data_dir``. Set
       ``cleanup_after_processing=true`` to reclaim the raw copies afterwards.
    """

    def _prepare_imagefolder(self):
        """Unwrap the inner archive, then merge the five domain roots into 44 classes."""
        root = Path(self.extracted_dirs)

        # sorted() materializes the matches BEFORE the first unzip, so an archive
        # extracted into its own parent cannot make this loop rescan what it just wrote.
        # DAPlankton nests exactly one level deep, so one pass is the whole job.
        for nested in sorted(root.rglob("*.zip")):
            logger.info(f"Extracting nested archive {nested.name}.")
            unzip(nested, nested.parent, show_progress=self.show_progress)

        archive_root = daplankton_layout.find_archive_root(root)
        logger.info(f"DAPlankton subset roots located at «{archive_root}».")

        copied = daplankton_layout.merge_domain_roots(archive_root, self.imagefolder_dir)
        logger.info(
            f"Merged {copied} DAPlankton image(s) from {len(daplankton_layout.DOMAIN_PREFIXES)} subset/instrument "
            f"root(s) into «{self.imagefolder_dir}»."
        )
        if copied != daplankton_layout.N_IMAGES:
            # Not fatal — a partial archive is still importable, and the taxonomy join
            # does not depend on the count. But the recorded figure is what the authors
            # state AND what a full enumeration of the real archive produced, so any
            # other number means this run did not see the whole dataset.
            logger.warning(
                f"Expected {daplankton_layout.N_IMAGES} DAPlankton images but merged {copied}. "
                "The archive may be partial, or its layout may have changed upstream."
            )

        # Belt and braces, matching the sibling importers: a class dir whose every entry
        # was filtered out arrives empty and would otherwise become a zero-image class.
        cleanup_imagefolder_empty_dirs(self.imagefolder_dir)
