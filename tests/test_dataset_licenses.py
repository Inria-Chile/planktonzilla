"""
(c) Inria

Network-free tests for the per-image license provenance recorded in
``planktonzilla.planktonzilla_dataset.constants``.

The ``license`` / ``license_url`` columns are a pure function of the ``dataset``
column, resolved through ``DATASET_LICENSES``. That table is a transcription, and a
transcription can drift, so these tests pin it from two independent directions:

  (a) DRIFT — every slug in DATASET_LICENSES equals the ``license:`` field of the
      matching ``configs/dataset_import/*.yaml``. Those configs remain the upstream
      source of truth; the constant exists only because the three manual-download
      sources have no config reachable at update time, and because
      ``update_planktonzilla`` never composes importer configs at all.

  (b) COVERAGE — the table covers exactly the fifteen source datasets that actually
      appear in the published planktonzilla-17M, no more and no fewer. The source of
      that list is ``samples.json`` (a real scan of the frozen dataset), so a
      hand-maintained table cannot silently miss a source or invent one. The tolerated
      extras are spelled out in ``_RECORDED_BUT_NOT_YET_PUBLISHED``: sources whose
      taxonomy rows and importer config are already in the tree but whose merge into
      planktonzilla-17M is still pending.

Coverage is what makes the frozen artifact safe to relicense-annotate: a missing entry
would mean published images shipping a null license, and (b) fails before that can
happen. It also guards the naming trap documented on DATASET_IMPORT_CONFIGS: five
``dataset`` values differ from their config stem.

Reads only committed files: the importer YAMLs, ``samples.json`` and the taxonomy CSV.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import csv
import json

import pytest
import yaml

from planktonzilla.dataset_import import tara_pacific_layout
from planktonzilla.planktonzilla_dataset.constants import (
    DATASET_IMPORT_CONFIGS,
    DATASET_LICENSES,
    LICENSE_COLS,
    RECORDED_BUT_NOT_YET_PUBLISHED,
    license_fields,
    validate_license_coverage,
)

_IMPORT_CONFIG_DIR = root / "configs" / "dataset_import"
_SAMPLES_JSON = root / "samples.json"
_TAXONOMY_CSV = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"

# Canonical deed URL per standard license slug. Duplicated from constants ON PURPOSE:
# if the mapping there is edited, this independent copy makes the test fail instead of
# agreeing with the edit.
_EXPECTED_DEEDS = {
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}

# The two entries whose slug is not a self-describing license, so license_url points at
# the authoritative source record instead of a deed. See KI-14 / KI-15.
_NON_DEED_URLS = {
    "whoi": "https://github.com/hsosik/WHOI-Plankton",
    "planktonset1.0": "https://doi.org/10.7289/v5d21vjd",
}

# The six pending sources now live in constants.RECORDED_BUT_NOT_YET_PUBLISHED, imported above
# rather than restated here: the golden-diff harness subtracts the same six from its coverage,
# and two transcriptions of the list would let this test go green while the harness excused a
# source that had in fact been published. Kept as a module-level alias so the assertions below,
# and the KNOWN_ISSUES entries that name this symbol, read unchanged.
_RECORDED_BUT_NOT_YET_PUBLISHED = set(RECORDED_BUT_NOT_YET_PUBLISHED)


def _published_dataset_names():
    """The distinct ``dataset`` values present in the published planktonzilla-17M.

    Read from ``samples.json``, which ``pz_sankey --save-samples`` produced by scanning
    the frozen Hub dataset — i.e. observed data, not a restatement of the config.
    """
    counts = json.loads(_SAMPLES_JSON.read_text())["counts"]
    return {row["dataset"] for row in counts}


def _config_license(config_stem):
    """The raw ``license:`` value of one ``configs/dataset_import/<stem>.yaml``.

    Parsed as plain YAML rather than composed through Hydra: the field is a literal in
    every importer config, so this reads the upstream value without inheriting
    defaults or touching Hydra's global state.
    """
    config = yaml.safe_load((_IMPORT_CONFIG_DIR / f"{config_stem}.yaml").read_text())
    return config["license"]


@pytest.mark.parametrize("dataset_name", sorted(DATASET_LICENSES))
def test_license_matches_importer_config(dataset_name):
    """(a) DRIFT: each recorded slug still equals its importer config's ``license:``."""
    config_stem = DATASET_IMPORT_CONFIGS[dataset_name]
    assert DATASET_LICENSES[dataset_name]["license"] == _config_license(config_stem), (
        f"«{dataset_name}» disagrees with configs/dataset_import/{config_stem}.yaml. "
        "The importer config is the source of truth — update DATASET_LICENSES to match it."
    )


def test_tables_cover_the_same_datasets():
    """DATASET_LICENSES and DATASET_IMPORT_CONFIGS describe the same set of sources."""
    assert set(DATASET_LICENSES) == set(DATASET_IMPORT_CONFIGS)


def test_every_importer_config_exists():
    """Every mapped config stem resolves to a real file (catches a typo'd rename)."""
    for dataset_name, config_stem in sorted(DATASET_IMPORT_CONFIGS.items()):
        assert (_IMPORT_CONFIG_DIR / f"{config_stem}.yaml").is_file(), (
            f"«{dataset_name}» maps to a missing configs/dataset_import/{config_stem}.yaml"
        )


def test_covers_exactly_the_published_source_datasets():
    """(b) COVERAGE: the table matches the fifteen sources in the frozen dataset.

    An unrecorded source would ship published images with a null license; a stale extra
    entry would mean the table is describing something the dataset no longer contains —
    except the explicitly pending ``_RECORDED_BUT_NOT_YET_PUBLISHED`` sources.
    """
    published = _published_dataset_names()

    assert len(published) == 15, f"samples.json no longer describes 15 sources: {sorted(published)}"
    assert not published - set(DATASET_LICENSES), (
        f"Published source(s) with no recorded license: {sorted(published - set(DATASET_LICENSES))}"
    )
    extra = set(DATASET_LICENSES) - published
    assert extra == _RECORDED_BUT_NOT_YET_PUBLISHED, (
        f"Recorded license(s) for source(s) not in the dataset: {sorted(extra - _RECORDED_BUT_NOT_YET_PUBLISHED)}; "
        f"pending source(s) that are now published: {sorted(_RECORDED_BUT_NOT_YET_PUBLISHED - extra)}"
    )


def test_taxonomy_csv_datasets_are_covered():
    """The taxonomy CSV's ``Dataset`` column is covered too — a second observed list."""
    with open(_TAXONOMY_CSV, newline="") as handle:
        csv_names = {row["Dataset"] for row in csv.DictReader(handle)}

    assert not csv_names - set(DATASET_LICENSES), (
        f"Taxonomy CSV source(s) with no recorded license: {sorted(csv_names - set(DATASET_LICENSES))}"
    )


def test_license_urls_are_deeds_except_the_two_documented_cases():
    """Standard slugs point at their canonical deed; the two exceptions are explicit."""
    for dataset_name, fields in sorted(DATASET_LICENSES.items()):
        url = fields["license_url"]
        assert url.startswith("https://"), f"«{dataset_name}» license_url is not https: {url}"

        if dataset_name in _NON_DEED_URLS:
            assert url == _NON_DEED_URLS[dataset_name]
            # These are exactly the slugs a consumer cannot act on without the URL.
            assert fields["license"] in ("mit", "other")
        else:
            assert url == _EXPECTED_DEEDS[fields["license"]], (
                f"«{dataset_name}» ({fields['license']}) should point at its canonical deed"
            )


def test_license_fields_returns_both_columns_as_a_fresh_dict():
    """``license_fields`` yields exactly LICENSE_COLS, and never a shared dict.

    ``generate_planktonzilla`` binds the result into a ``functools.partial`` that
    ``datasets.map`` fans out across processes, so handing back the module-level dict
    would invite a mutation to leak into the constant.
    """
    fields = license_fields("lensless")

    assert set(fields) == set(LICENSE_COLS)
    assert fields == {"license": "cc-by-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"}
    assert fields is not DATASET_LICENSES["lensless"]

    fields["license"] = "mutated"
    assert DATASET_LICENSES["lensless"]["license"] == "cc-by-4.0"


def test_unrecorded_dataset_raises_rather_than_nulling():
    """An unknown source is an error at both seams, never a silent null license."""
    with pytest.raises(KeyError, match="no-such-source"):
        license_fields("no-such-source")

    with pytest.raises(KeyError, match="no-such-source"):
        validate_license_coverage(["lensless", "no-such-source"])

    # The happy path is silent.
    validate_license_coverage(sorted(DATASET_LICENSES))


def test_generate_config_datasets_are_all_covered():
    """Every source wired into cfg.datasets has a license, so main() never fails late."""
    generate_config = yaml.safe_load((root / "configs" / "generate_planktonzilla.yaml").read_text())
    configured = {entry["name"] for entry in generate_config["datasets"]}

    assert not configured - set(DATASET_LICENSES), (
        f"cfg.datasets source(s) with no recorded license: {sorted(configured - set(DATASET_LICENSES))}"
    )
    # The registry carries all 15 published sources plus the six joining them (frepj at
    # v1.2, daplankton, and the four Tara Pacific deposits), so a from-scratch build
    # reproduces every source of the published dataset and every newcomer. The table is
    # still declared independently of cfg.datasets: it is the record of each source's
    # TERMS, which must survive a source being temporarily removed from the build.
    assert set(DATASET_LICENSES) - configured == set()
    assert len(configured) == 21


def test_pending_source_license_agrees_with_its_layout_constants():
    """Each pending source's slug is transcribed in three places; they must not drift apart."""
    from planktonzilla.dataset_import import daplankton_layout, frepj_layout

    assert DATASET_LICENSES["frepj"] == {"license": frepj_layout.LICENSE, "license_url": frepj_layout.LICENSE_URL}
    assert DATASET_LICENSES["daplankton"] == {
        "license": daplankton_layout.LICENSE,
        "license_url": daplankton_layout.LICENSE_URL,
    }


@pytest.mark.parametrize("dataset_name", sorted(tara_pacific_layout.SOURCES))
def test_tara_pacific_license_agrees_with_its_layout_constants(dataset_name):
    """Each Tara Pacific slug is transcribed in three places; they must not drift apart.

    The importer config is the source of truth (pinned by test_license_matches_importer_config
    above); this closes the triangle against the layout module the importer itself reads.
    """
    assert DATASET_LICENSES[dataset_name] == {
        "license": tara_pacific_layout.LICENSE,
        "license_url": tara_pacific_layout.LICENSE_URL,
    }


def test_the_recorded_uncovered_datasets_match_the_pending_sources():
    """(c) CROSS-PIN: the golden diff's excuse list and this file's pending list are one fact.

    ``pz_golden_diff`` compares the published planktonzilla-17M against what this tree renders, and
    the 873 taxonomy rows it cannot cover are the rows of exactly these six sources. It records them
    in its manifest as ``excluded_datasets`` so that "subtract the rows the Hub does not have" stays
    honest — an uncovered row named by neither that list nor ``absent_pairs`` is a coverage anomaly
    and fails the gate rather than being quietly subtracted.

    Two independently maintained restatements of "not published yet" are one restatement too many:
    a source that lands in the artifact must leave BOTH lists in the same commit, and this is what
    notices when it leaves only one. Same trick this file already plays with ``samples.json``.

    Skips until the reference pair exists, because it is generated from a live Hub sweep and
    committed separately from the harness itself.
    """
    manifest_path = root / "planktonzilla" / "planktonzilla_dataset" / "utils" / "golden_reference.json"
    if not manifest_path.exists():
        pytest.skip(
            f"{manifest_path.relative_to(root)} does not exist yet — generate it with "
            "`uv run pz_golden_diff --refresh --revision <sha>` and commit the reference pair."
        )

    recorded = set(json.loads(manifest_path.read_text(encoding="utf-8"))["excluded_datasets"])
    assert recorded == _RECORDED_BUT_NOT_YET_PUBLISHED, (
        f"golden_reference.json excuses source(s) this file does not: {sorted(recorded - _RECORDED_BUT_NOT_YET_PUBLISHED)}; "
        f"pending source(s) the golden diff does not excuse: {sorted(_RECORDED_BUT_NOT_YET_PUBLISHED - recorded)}"
    )
