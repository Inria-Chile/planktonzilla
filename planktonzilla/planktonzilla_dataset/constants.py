"""
(c) Inria

Shared constants and helpers for the planktonzilla_dataset scripts.

Centralizes values that were previously copy-pasted across the generation
scripts. Values and ordering are preserved EXACTLY — this module only removes
the duplication, it does not change any of the constants.

Intentionally NOT centralized: the filesystem base directory. ``generate_planktonzilla``
resolves ``data/`` via pyrootutils (the repository root), while the other scripts
resolve it relative to the package (``dirname(dirname(__file__))``). Those are
different locations on disk, so each script keeps its own base-dir resolution and
only the trailing filename is shared here.
"""

import os
from pathlib import Path

DEFAULT_PLANKTONZILLA_DATASET_NAME = "planktonzilla-17M"
DEFAULT_PLANKTONZILLA_DATASET_REPO_ID = f"project-oceania/{DEFAULT_PLANKTONZILLA_DATASET_NAME}"

# Raw enriched-taxonomy CSV filename (lives under each script's own data/ dir).
DEFAULT_TAXONOMY_CSV_FILENAME = Path(__file__).parent / "planktonzilla_taxonomy.csv"

# The seven taxonomic ranks, ordered Kingdom -> Species.
TAXONOMY_RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

# Extra label/classification columns that travel alongside the taxonomy ranks.
EXTRA_COLS = ("proposed_label", "plankton", "root_class", "qualifier")

# Authoritative reference vocabulary for the ``qualifier`` column (the specimen
# condition/part a label describes). An empty ``qualifier`` cell means "unqualified"
# and is intentionally NOT a member. This is documentation/validation only: the
# generation pipeline never validates against this set (it only casts the column to
# string), so this constant does not affect generated output. Conformance of the CSV
# to this vocabulary is pinned by ``tests/test_taxonomy_known_issues.py`` (KI-11).
QUALIFIERS = (
    "full_body",
    "larvae",
    "egg",
    "like",
    "mix",
    "parasite",
    "part",
    "part_head",
    "part_tail",
    "part_tentacle",
    "part_leg",
    "part_carapace",
    "part_skin",
    "part_trunk",
)

# External-database ID columns, grouped by how the CSV stores them.
ID_STR_COLS = ("wikidata_ID", "ecotaxa_ID")  # already text in the CSV
ID_NUM_COLS = ("aphia_ID", "NCBI_ID", "BOLD_ID")  # numeric in the CSV -> text without decimals


def default_num_proc() -> int:
    """Return half the available CPUs, at least 1.

    Replaces the duplicated ``int(cpu_count() / 2)`` idiom. On any host with two
    or more CPUs this returns the same value as before; on a single-core host it
    returns 1 instead of 0 (which ``datasets.map`` / ``ThreadPoolExecutor``
    reject), and it tolerates ``os.cpu_count()`` returning ``None``.
    """
    return max(1, (os.cpu_count() or 1) // 2)
