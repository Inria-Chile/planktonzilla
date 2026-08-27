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

# Provenance columns describing the terms each image is redistributed under. Both are
# a pure function of the ``dataset`` column (see DATASET_LICENSES) and are stored as
# string, like every other text column in the consolidated dataset.
LICENSE_COLS = ("license", "license_url")

# ``dataset`` column value -> ``configs/dataset_import/<stem>.yaml``.
#
# Five of the seventeen do NOT match: the value written into the ``dataset`` column is
# the `name` field of a `cfg.datasets` entry in configs/generate_planktonzilla.yaml,
# while the importer config is named after the source. No naming rule recovers the
# difference, so it is written down here and pinned by tests/test_dataset_licenses.py.
#
# All fifteen are active entries in cfg.datasets as of 2026-08-01 — the last three
# joined once their downloads were shown not to need the manual .zip they had long been
# documented as requiring. Note the JEDI source carries three different strings:
# `jedioceans` in the data, `jedi_oceans_cpics` as a config stem, `jedi` as a redefiner
# key.
#
# `frepj` (FREPJ-Z, milestone v1.2) is the sixteenth entry of cfg.datasets since
# 2026-08-25, appended last so the fifteen above keep their concatenation index. It is
# published on its own as project-oceania/planktonzilla-frepj and is absent from the
# frozen planktonzilla-17M until the v1.2 push.
#
# `daplankton` is the seventeenth, appended after frepj for the same reason. Like frepj it
# is recorded here — and in the taxonomy CSV and the registry — ahead of its arrival in the
# published artifact, so tests/test_dataset_licenses.py lists BOTH as pending.
DATASET_IMPORT_CONFIGS = {
    "isiisnet": "isiisnet",
    "whoi": "whoi-plankton",
    "flowcamnet": "flowcamnet",
    "lensless": "lensless",
    "medplanktonset": "medplanktonset",
    "uvp6net": "uvp6net",
    "zoocamnet": "zoocamnet",
    "zooscan": "zooscannet",
    "planktonset1.0": "planktonset1",
    "syke_ifcb_2022": "syke_ifcb_2022",
    "planktoscope": "planktoscope",
    "global_uvp5": "global_uvp5net",
    # Active since 2026-08-01 (import_name jedi_oceans_cpics, redefiner key jedi).
    "jedioceans": "jedi_oceans_cpics",
    "sykezooscan2024": "sykezooscan2024",
    "zoolake": "zoolake",
    # Sixteenth entry (v1.2), appended last; not in the frozen 17M until the v1.2 push.
    "frepj": "frepj",
    # Seventeenth entry, appended after frepj for the same index-preserving reason.
    "daplankton": "daplankton",
}

# Canonical deed URL per license slug, used to fill ``license_url`` for the standard
# licenses. Slugs that are not a self-describing standard license (``mit`` here names
# a repository's code license, ``other`` names nothing at all) are NOT listed and must
# give an explicit per-dataset URL in DATASET_LICENSES below.
_LICENSE_DEEDS = {
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}

# ``dataset`` column value -> the license fields emitted for every image of that source.
#
# The slugs are inherited VERBATIM from the ``license:`` field of the matching
# configs/dataset_import/*.yaml — that file stays the upstream source of truth, and
# tests/test_dataset_licenses.py fails if the two ever disagree. This table exists
# because update_planktonzilla never composes importer configs at all, and because a
# source's TERMS must stay recorded even if it is temporarily pulled from the build.
#
# Cross-checked against the published dataset's LICENSE.md on 2026-08-01 — all fifteen
# agree. `zoolake` was corrected there and then: it had been recorded as cc-by-4.0,
# over-stating the restriction, when the originating EAWAG deposit is CC0 (no
# attribution required at all). `frepj` post-dates that LICENSE.md; its slug is pinned
# against configs/dataset_import/frepj.yaml and dataset_import.frepj_layout.LICENSE.
#
# Two entries carry a URL that is not a license deed, because their slug alone is not
# actionable (see KI-14/KI-15 in utils/KNOWN_ISSUES.md):
#   - whoi: `mit` is the license of the hsosik/WHOI-Plankton *code* repository named by
#     the config's source_url; the URL points there so a consumer can check the terms
#     that actually cover the IFCB imagery.
#   - planktonset1.0: `other` names nothing, so the URL is the NOAA NCEI DOI for
#     accession 0127422 already recorded in the config's citation. The published
#     LICENSE.md words this as "U.S. Government Work — no license stated".
#
# `daplankton` post-dates that LICENSE.md too; its slug is pinned against
# configs/dataset_import/daplankton.yaml and dataset_import.daplankton_layout.LICENSE. CC BY
# 4.0 is stated identically in three independent places — the Metax record's
# access_rights.license, the Etsin landing page, and the readme.md bundled inside the archive
# itself — all read on 2026-08-27.
DATASET_LICENSES = {
    name: {"license": slug, "license_url": url or _LICENSE_DEEDS[slug]}
    for name, slug, url in (
        ("isiisnet", "cc-by-nc-4.0", None),
        ("whoi", "mit", "https://github.com/hsosik/WHOI-Plankton"),
        ("flowcamnet", "cc-by-nc-4.0", None),
        ("lensless", "cc-by-4.0", None),
        ("medplanktonset", "cc-by-4.0", None),
        ("uvp6net", "cc-by-nc-4.0", None),
        ("zoocamnet", "cc-by-4.0", None),
        ("zooscan", "cc-by-nc-4.0", None),
        ("planktonset1.0", "other", "https://doi.org/10.7289/v5d21vjd"),
        ("syke_ifcb_2022", "cc-by-4.0", None),
        ("planktoscope", "cc-by-nc-4.0", None),
        ("global_uvp5", "cc-by-4.0", None),
        ("jedioceans", "cc-by-sa-4.0", None),
        ("sykezooscan2024", "cc-by-4.0", None),
        ("zoolake", "cc0-1.0", None),
        ("frepj", "cc-by-4.0", None),
        ("daplankton", "cc-by-4.0", None),
    )
}


def license_fields(dataset_name: str) -> dict:
    """Return the ``{license, license_url}`` pair for one ``dataset`` column value.

    A fresh dict per call, so callers can hand it straight to ``datasets.map`` without
    risking a shared-mutable-state bug across processes.

    Raises:
        KeyError: If ``dataset_name`` has no entry in DATASET_LICENSES. Missing is
            always a bug (a new source was wired up without recording its terms), and
            it must not silently degrade to a null license on published images.
    """
    try:
        return dict(DATASET_LICENSES[dataset_name])
    except KeyError:
        raise KeyError(
            f"No license recorded for dataset «{dataset_name}». Add it to "
            f"DATASET_LICENSES in {__name__} (and its importer config to "
            f"DATASET_IMPORT_CONFIGS) before building or updating the dataset."
        ) from None


def validate_license_coverage(dataset_names) -> None:
    """Fail fast if any of ``dataset_names`` has no recorded license.

    Called before the expensive work starts so a missing entry surfaces in seconds
    rather than after a multi-hour build has already written unlicensed rows.

    Raises:
        KeyError: Listing every unrecorded dataset name at once.
    """
    missing = sorted({name for name in dataset_names if name not in DATASET_LICENSES})
    if missing:
        raise KeyError(
            f"No license recorded for dataset(s) {missing}. Add them to "
            f"DATASET_LICENSES in {__name__} (and their importer configs to "
            f"DATASET_IMPORT_CONFIGS) before building or updating the dataset."
        )


# Provenance columns written by RedefineDataset._taxonomy_row: which source an example
# came from, the label that source gave it, and its path inside that source's
# imagefolder. ``dataset`` is the splice key — its values are the ``name`` field of the
# entries in the ``datasets`` table and the ``Dataset`` column of the taxonomy CSV.
IDENTITY_COLS = ("dataset", "original_label", "original_path")

# Columns flattened out of the per-source metadata JSON, in the exact order
# RedefineDataset._flatten_metadata produces them.
METADATA_COLS = (
    "Latitude",
    "Humidity",
    "Temperature",
    "Longitude",
    "ObjID",
    "Depth_max",
    "Depth_min",
    "timestamp",
)

# Source-specific metadata that has no consolidated column of its own, kept as ONE JSON
# object per row so a source can carry what only it knows (FREPJ: the magnification and
# the raw sampling-site token) without adding a sparse column to the schema for every
# source. Filled generically by RedefineDataset._flatten_metadata from whatever keys of
# the per-source metadata JSON the consolidated columns do not consume, sorted by key so
# equal content is equal text. The value is always a JSON object — the literal "{}" when
# a source has nothing to add — never null, so a consumer can json.loads() every row
# without a null check. Rows carried over from a base that predates the column get the
# same literal (make_planktonzilla.ensure_custom_metadata), so a rebuilt row and a
# carried-over row are indistinguishable.
CUSTOM_METADATA_COL = "custom_metadata"
EMPTY_CUSTOM_METADATA = "{}"

# Every column of the consolidated dataset. Used to check that a base dataset and a
# freshly built part agree before they are concatenated: datasets.concatenate_datasets
# silently NULL-FILLS a column missing from one side rather than raising, which would
# blank the column for exactly the rows just rebuilt.
CONSOLIDATED_COLUMNS = (
    "image",
    *IDENTITY_COLS,
    *TAXONOMY_RANKS,
    *EXTRA_COLS,
    *ID_STR_COLS,
    *ID_NUM_COLS,
    *METADATA_COLS,
    *LICENSE_COLS,
    CUSTOM_METADATA_COL,
)


def default_num_proc() -> int:
    """Return half the available CPUs, at least 1.

    Replaces the duplicated ``int(cpu_count() / 2)`` idiom. On any host with two
    or more CPUs this returns the same value as before; on a single-core host it
    returns 1 instead of 0 (which ``datasets.map`` / ``ThreadPoolExecutor``
    reject), and it tolerates ``os.cpu_count()`` returning ``None``.
    """
    return max(1, (os.cpu_count() or 1) // 2)
