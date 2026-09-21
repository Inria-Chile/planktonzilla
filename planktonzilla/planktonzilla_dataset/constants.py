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

# Provenance columns describing the imaging instrument each image was captured with.
# ``instrument`` is the short community name a consumer groups by; ``instrument_id`` is its
# term in the BODC/SeaVoX L22 device catalogue, which is the vocabulary the plankton-imaging
# community actually uses (JERICO-S3 best practice, Martin-Cabrera et al. 2022, carries
# exactly this crosswalk into OBIS-ENV-DATA eMoF rows). Darwin Core has no instrument term at
# all, which is why the URI and not a DwC field is the interoperable half.
#
# Unlike LICENSE_COLS these are NOT a pure function of ``dataset``: see DATASET_INSTRUMENTS.
INSTRUMENT_COLS = ("instrument", "instrument_id")

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
# `daplankton` is the seventeenth, appended after frepj for the same reason, and the four
# `tara_pacific_*` entries (v1.2, added 2026-08-26) follow it — all recorded here, in the
# taxonomy CSV and in the registry ahead of their arrival in the published artifact, so
# tests/test_dataset_licenses.py lists every one of them as pending.
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
    # Entries 18-21 (v1.2): the four Tara Pacific deposits of Mériguet et al. 2025
    # (essd-17-2761-2025), appended after daplankton so every earlier source keeps its
    # concatenation index. name == import_name for all four, on purpose — five of the
    # sixteen above differ, and there is no reason to add a sixth trap. Unlike every other
    # source these have NO archive: their SEANOE deposits publish EcoTaxa TSV exports
    # (metadata, no vignettes), so the importer walks the public EcoTaxa API instead. See
    # planktonzilla.dataset_import.tara_pacific_layout.
    "tara_pacific_bongo": "tara_pacific_bongo",
    "tara_pacific_decknet": "tara_pacific_decknet",
    "tara_pacific_hsn": "tara_pacific_hsn",
    "tara_pacific_manta": "tara_pacific_manta",
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
        # The four Tara Pacific deposits, each CC BY 4.0 as read from its SEANOE record on
        # 2026-08-26. Pinned against configs/dataset_import/tara_pacific_*.yaml and
        # dataset_import.tara_pacific_layout.LICENSE by the tests. They add nothing new to
        # the licence mix.
        ("tara_pacific_bongo", "cc-by-4.0", None),
        ("tara_pacific_decknet", "cc-by-4.0", None),
        ("tara_pacific_hsn", "cc-by-4.0", None),
        ("tara_pacific_manta", "cc-by-4.0", None),
    )
}

# Sources whose terms, taxonomy rows and importer configs are recorded AHEAD of their arrival in
# the published planktonzilla-17M. They are in DATASET_LICENSES, DATASET_IMPORT_CONFIGS, the
# default registry and planktonzilla_taxonomy.csv, and absent from the frozen artifact until the
# v1.2 push. Listed explicitly so every consumer tolerates exactly these six and nothing else;
# when a pending source lands, it comes out of this set in the same commit that regenerates
# samples.json.
#   frepj           registry since 2026-08-25, published on its own as project-oceania/planktonzilla-frepj
#   daplankton      registry since 2026-08-27
#   tara_pacific_*  registry since 2026-08-26
#
# This used to be a private set in tests/test_dataset_licenses.py, which was the wrong home: it is
# not a fact about testing. Three consumers now depend on it — that test's samples.json coverage
# check, the golden-diff harness's excluded_datasets, and the coverage-anomaly rule that makes the
# harness's "876 uncovered rows" honest rather than a licence to subtract whatever disagrees. Two
# transcriptions of six names would drift, and the drift would read as a data difference.
RECORDED_BUT_NOT_YET_PUBLISHED = frozenset(
    {
        "frepj",
        "daplankton",
        "tara_pacific_bongo",
        "tara_pacific_decknet",
        "tara_pacific_hsn",
        "tara_pacific_manta",
    }
)


def revision_kwargs(revision) -> dict:
    """``{"revision": r}`` when ``r`` is set, ``{}`` when it is not.

    The one line that keeps a default run byte-identical. A read path that spells
    ``revision=cfg.get("base_revision")`` inline forwards ``revision=None``, which is NOT the same
    call as forwarding nothing: it changes the ``datasets`` cache key, and it breaks the two
    existing test doubles that pin a one- and a two-parameter signature
    (``tests/test_sankey.py`` installs ``lambda repo_id: ...``). Every read-side pin in this
    package goes through here, so "unset means unchanged" is one testable property rather than
    seven copies of the same conditional.

    The write side already spells this idiom inline twice (``make_planktonzilla`` and
    ``update_planktonzilla``, both for ``push_revision``); those are left alone because changing
    them moves no read and this helper exists to stop the READ paths from drifting.
    """
    return {"revision": revision} if revision else {}


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


# Sentinel for a source whose instrument varies per IMAGE rather than per source. Stored in
# DATASET_INSTRUMENTS so the fact is declared rather than implied by an absence, and so
# ``instrument_fields`` can refuse to answer for it instead of quietly returning nulls.
PER_ROW_INSTRUMENT = "@per-row"

# L22 terms, resolved against vocab.nerc.ac.uk on 2026-09-18 and their prefLabels read back.
# None is deprecated. Two are deliberately the SERIES term rather than the obvious model term,
# because the configs name an instrument family and not a model:
#   - UVP5   -> TOOL2154 "...Underwater Vision Profiler 5 {UVP5} imaging sensor series",
#               NOT TOOL1577, which is the UVP5 *DEEP*.
#   - UVP6   -> TOOL2141 "...Underwater Vision Profiler 6 {UVP6} imaging sensor series",
#               NOT TOOL1578, which is the UVP6 *LP*.
_L22 = "https://vocab.nerc.ac.uk/collection/L22/current/{}/"

# ``dataset`` column value -> the instrument fields emitted for every image of that source.
#
# Transcribed VERBATIM from the ``instrument:`` / ``instrument_id:`` fields of the matching
# configs/dataset_import/*.yaml, which stay the upstream source of truth exactly as they do for
# DATASET_LICENSES; tests/test_dataset_instruments.py fails if the two disagree. Those config
# fields were ADDED for this column: before them a table here would have been a transcription
# with nothing to check it against, which is how the licence table drifted on `zoolake` once.
#
# Three kinds of null, all deliberate, none a placeholder for "look it up later":
#   - `frepj` has BOTH fields null. Nothing in the repo names an instrument for it: the config
#     says "high-resolution microscopic images" and the README "40x and 100x", which is a
#     modality, not a device. L22 does offer generic microscope terms (TOOL1034 "Inverted
#     Microscope (generic)", TOOL2302 "Unspecified polarised light microscope") and choosing one
#     would publish a guess as a fact on 229 rows. See KI-33.
#   - `zoolake`, `lensless` and `planktonset1.0` have a NAME but a null id, because L22 has no
#     term for their instrument: no Scripps plankton camera (its four "Scripps" entries are a
#     Doppler sonar, a plankton net and two titrators), no lensless imager at all, and only the
#     2008 ISIIS TOOL1561 -- not the later ISIIS-2 that planktonset1.0 names. Group those by the
#     `instrument` string; the id says only "no standard term exists", never "unknown device".
#   - `daplankton` is PER_ROW_INSTRUMENT. Its images genuinely come from three instruments and
#     that is the POINT of the source -- its own config calls it a benchmark for recognition
#     "ACROSS imaging instruments ... the domain shift a classifier has to survive is the
#     instrument rather than the label". Resolved per row by ``resolve_instrument``.
#
# One caveat that the table cannot express and KI-33 records: TOOL1583 is L22's only FlowCam
# term and names the "FlowCam VS [imaging only] (Benchtop) ... series" specifically, while the
# four FlowCam sources here say only "FlowCam". The id is therefore slightly more specific than
# the evidence; it is used because the alternative is no id at all for four sources.
DATASET_INSTRUMENTS = {
    name: {"instrument": instrument, "instrument_id": _L22.format(code) if code else None}
    for name, instrument, code in (
        ("isiisnet", "ISIIS", "TOOL1561"),
        ("whoi", "IFCB", "TOOL1588"),
        ("flowcamnet", "FlowCam", "TOOL1583"),
        ("lensless", "Lensless microscope", None),
        ("medplanktonset", "IFCB", "TOOL1588"),
        ("uvp6net", "UVP6", "TOOL2141"),
        ("zoocamnet", "ZooCAM", "TOOL1587"),
        ("zooscan", "ZooScan", "TOOL1581"),
        ("planktonset1.0", "ISIIS-2", None),
        ("syke_ifcb_2022", "IFCB", "TOOL1588"),
        ("planktoscope", "PlanktoScope", "TOOL1579"),
        ("global_uvp5", "UVP5", "TOOL2154"),
        ("jedioceans", "CPICS", "TOOL1582"),
        ("sykezooscan2024", "ZooScan", "TOOL1581"),
        ("zoolake", "Dual Scripps Plankton Camera", None),
        ("frepj", None, None),
        ("daplankton", PER_ROW_INSTRUMENT, None),
        ("tara_pacific_bongo", "FlowCam", "TOOL1583"),
        ("tara_pacific_decknet", "FlowCam", "TOOL1583"),
        ("tara_pacific_hsn", "ZooScan", "TOOL1581"),
        ("tara_pacific_manta", "ZooScan", "TOOL1581"),
    )
}

# The instrument each DAPlankton merge prefix stands for. The importer folds five
# ``<subset>/<instrument>/<class>/`` roots into one class dir per taxon and keeps the
# provenance as a filename prefix (daplankton_layout.DOMAIN_PREFIXES), so the instrument of a
# DAPlankton image is recoverable from ``original_path`` alone -- offline, with no re-fetch.
# Keyed on the prefix rather than on the (subset, code) pair so this stays a pure string
# operation on the column the dataset already carries.
DAPLANKTON_PREFIX_INSTRUMENTS = {
    "lab_cs_": ("CytoSense", _L22.format("TOOL1209")),
    "sea_cs_": ("CytoSense", _L22.format("TOOL1209")),
    "lab_fc_": ("FlowCam", _L22.format("TOOL1583")),
    "lab_ifcb_": ("IFCB", _L22.format("TOOL1588")),
    "sea_ifcb_": ("IFCB", _L22.format("TOOL1588")),
}


def instrument_fields(dataset_name: str) -> dict:
    """Return the ``{instrument, instrument_id}`` pair for one CONSTANT-instrument source.

    A fresh dict per call, for the same reason ``license_fields`` returns one.

    Raises:
        KeyError: If ``dataset_name`` has no entry in DATASET_INSTRUMENTS. Missing is always a
            bug -- a new source wired up without recording what took its pictures.
        ValueError: If the source is PER_ROW_INSTRUMENT. Returning nulls there would publish
            "instrument unknown" for a source whose whole purpose is the instrument, so the
            caller is made to go through ``resolve_instrument`` instead.
    """
    try:
        fields = DATASET_INSTRUMENTS[dataset_name]
    except KeyError:
        raise KeyError(
            f"No instrument recorded for dataset «{dataset_name}». Add `instrument:` and "
            f"`instrument_id:` to its configs/dataset_import/*.yaml and to DATASET_INSTRUMENTS "
            f"in {__name__} before building or updating the dataset."
        ) from None
    if fields["instrument"] == PER_ROW_INSTRUMENT:
        raise ValueError(
            f"«{dataset_name}» images come from more than one instrument; call "
            f"resolve_instrument(dataset_name, original_path) per row instead of "
            f"instrument_fields(dataset_name)."
        )
    return dict(fields)


def has_per_row_instrument(dataset_name: str) -> bool:
    """Whether this source's instrument varies per IMAGE rather than per source.

    Lets a caller resolve the constant pair once per split and fall back to the per-row path
    only where it is needed, without indexing DATASET_INSTRUMENTS directly — an unrecorded
    source must raise the same explanatory KeyError here as everywhere else, not a bare one
    from a dict lookup buried in a build.

    Raises:
        KeyError: If ``dataset_name`` has no entry in DATASET_INSTRUMENTS.
    """
    try:
        return DATASET_INSTRUMENTS[dataset_name]["instrument"] == PER_ROW_INSTRUMENT
    except KeyError:
        raise KeyError(
            f"No instrument recorded for dataset «{dataset_name}». Add `instrument:` and "
            f"`instrument_id:` to its configs/dataset_import/*.yaml and to DATASET_INSTRUMENTS "
            f"in {__name__} before building or updating the dataset."
        ) from None


def resolve_instrument(dataset_name: str, original_path: str | None = None) -> dict:
    """Return the ``{instrument, instrument_id}`` pair for one IMAGE.

    The single entry point the build path uses, so a per-row source cannot silently fall back
    to a source-level constant: for the 20 constant sources this is ``instrument_fields``, and
    for DAPlankton it reads the merge prefix the importer left on the filename.

    Raises:
        KeyError: As ``instrument_fields``.
        ValueError: If a per-row source's path carries no recognised prefix. That means the
            importer's merge policy changed without this table following, and silently emitting
            a null would hide it on every one of that source's rows.
    """
    fields = DATASET_INSTRUMENTS.get(dataset_name)
    if fields is None or fields["instrument"] != PER_ROW_INSTRUMENT:
        return instrument_fields(dataset_name)

    name = (original_path or "").rsplit("/", 1)[-1]
    for prefix, (instrument, identifier) in DAPLANKTON_PREFIX_INSTRUMENTS.items():
        if name.startswith(prefix):
            return {"instrument": instrument, "instrument_id": identifier}
    raise ValueError(
        f"«{dataset_name}» resolves its instrument from the merge prefix of original_path, but "
        f"{original_path!r} carries none of {sorted(DAPLANKTON_PREFIX_INSTRUMENTS)}. The "
        f"importer's merge policy and DAPLANKTON_PREFIX_INSTRUMENTS in {__name__} have diverged."
    )


def validate_instrument_coverage(dataset_names) -> None:
    """Fail fast if any of ``dataset_names`` has no recorded instrument.

    Called beside ``validate_license_coverage``, before the expensive work starts, so a source
    added without an instrument surfaces in seconds rather than after a multi-hour build has
    written rows that cannot say what took the picture.

    Raises:
        KeyError: Listing every unrecorded dataset name at once.
    """
    missing = sorted({name for name in dataset_names if name not in DATASET_INSTRUMENTS})
    if missing:
        raise KeyError(
            f"No instrument recorded for dataset(s) {missing}. Add `instrument:` and "
            f"`instrument_id:` to their configs/dataset_import/*.yaml and to DATASET_INSTRUMENTS "
            f"in {__name__} before building or updating the dataset."
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
    *INSTRUMENT_COLS,
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
