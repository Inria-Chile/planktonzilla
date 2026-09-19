"""
(c) Inria

Gate for the per-image instrument provenance: ``instrument`` (the short community name a
consumer groups by) and ``instrument_id`` (its term in the BODC/SeaVoX L22 device catalogue).

``DATASET_INSTRUMENTS`` is a hand-transcribed table, exactly like ``DATASET_LICENSES``, and the
licence table is on record as having drifted once — ``zoolake`` was published as ``cc-by-4.0``
when its deposit is CC0, caught by a human reading the Hub's LICENSE.md rather than by a test.
What stops the licence table drifting *now* is that every config carries a machine-readable
``license:`` field for a test to hold it against.

No config carried anything instrument-shaped before this column, so the equivalent fields were
ADDED to all 21. That is the load-bearing part of the design, and (a) below is the test it
exists for. Without it this table would be prose in twenty-one files copied into a
twenty-second, with nothing but review standing between it and a wrong device name on 17.4M
published images.

  (a) DRIFT — every value equals the ``instrument:`` / ``instrument_id:`` of the matching
      ``configs/dataset_import/*.yaml``.
  (b) COVERAGE — the table covers exactly the registry, and the build refuses to start if a
      source is missing from it.
  (c) SHAPE — every non-null id is a real L22 term URI, and the three kinds of null are the
      three documented ones rather than an unfilled cell.
  (d) RESOLUTION — DAPlankton, the one source whose instrument varies per image, resolves from
      the merge prefix its importer leaves in ``original_path``, and a path that carries no
      recognised prefix raises instead of publishing a null.

Offline: reads only committed YAML and the constants module. The L22 URIs were resolved against
vocab.nerc.ac.uk when they were written (2026-09-18) and their prefLabels read back; this suite
checks their FORM, not their liveness, so it never touches the network.
"""

import re

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

from pathlib import Path

import pytest
import yaml

from planktonzilla.dataset_import import daplankton_layout, tara_pacific_layout
from planktonzilla.planktonzilla_dataset.constants import (
    CONSOLIDATED_COLUMNS,
    DAPLANKTON_PREFIX_INSTRUMENTS,
    DATASET_IMPORT_CONFIGS,
    DATASET_INSTRUMENTS,
    INSTRUMENT_COLS,
    PER_ROW_INSTRUMENT,
    instrument_fields,
    resolve_instrument,
    validate_instrument_coverage,
)

_IMPORT_CONFIG_DIR = Path(root) / "configs" / "dataset_import"

# The L22 term URI shape. Deliberately strict about the collection: an id from any other NERC
# collection would be a different KIND of thing (L05 is instrument *categories*, P01 is
# measured parameters), and silently accepting one would publish a category where the column
# promises a device.
_L22_URI = re.compile(r"^https://vocab\.nerc\.ac\.uk/collection/L22/current/TOOL\d+/$")

# The documented nulls, each for a different reason. Listed here so that a NEW null — a source
# added without an instrument — fails rather than joining them silently.
_NO_INSTRUMENT_AT_ALL = {"frepj"}
_NAMED_BUT_NO_L22_TERM = {"zoolake", "lensless", "planktonset1.0"}
_PER_ROW = {"daplankton"}


def _config_instrument(config_stem):
    """The ``instrument`` / ``instrument_id`` pair as the importer config states it."""
    config = yaml.safe_load((_IMPORT_CONFIG_DIR / f"{config_stem}.yaml").read_text(encoding="utf-8"))
    return {column: config.get(column) for column in INSTRUMENT_COLS}


# (a) DRIFT
@pytest.mark.parametrize("dataset_name", sorted(DATASET_INSTRUMENTS))
def test_instrument_matches_importer_config(dataset_name):
    """Each recorded pair still equals its importer config's fields.

    The whole reason those config fields exist. The config is the source of truth; this table
    is a transcription for the paths that never compose a config at all.
    """
    config_stem = DATASET_IMPORT_CONFIGS[dataset_name]
    assert DATASET_INSTRUMENTS[dataset_name] == _config_instrument(config_stem), (
        f"«{dataset_name}» disagrees with configs/dataset_import/{config_stem}.yaml. "
        "The importer config is the source of truth — update DATASET_INSTRUMENTS to match it."
    )


def test_every_config_declares_both_fields_even_when_null():
    """A MISSING key and an explicit ``null`` must not be confusable.

    ``yaml.safe_load`` returns None for both, so a source whose fields were never added would
    pass (a) by accident — it would compare None against None. This is the test that catches
    the omission, by reading the raw text rather than the parsed value.
    """
    for dataset_name, config_stem in sorted(DATASET_IMPORT_CONFIGS.items()):
        text = (_IMPORT_CONFIG_DIR / f"{config_stem}.yaml").read_text(encoding="utf-8")
        for column in INSTRUMENT_COLS:
            assert re.search(rf"^{column}:", text, re.MULTILINE), (
                f"configs/dataset_import/{config_stem}.yaml ({dataset_name}) declares no "
                f"`{column}:`. Add it — an absent key reads as null and would pass the drift "
                f"test against a null table entry."
            )


# (b) COVERAGE
def test_the_table_covers_exactly_the_registry():
    """No source can be imported without recording what took its pictures."""
    assert set(DATASET_INSTRUMENTS) == set(DATASET_IMPORT_CONFIGS)


def test_both_columns_are_in_the_consolidated_schema():
    """Otherwise assert_consolidated_schema would reject every build that emits them."""
    for column in INSTRUMENT_COLS:
        assert column in CONSOLIDATED_COLUMNS


def test_an_unrecorded_source_fails_before_the_build_starts():
    """Fail-fast in seconds, not after a multi-hour build has written unattributed rows."""
    with pytest.raises(KeyError, match="No instrument recorded"):
        validate_instrument_coverage(["zooscan", "a_source_nobody_recorded"])

    validate_instrument_coverage(DATASET_INSTRUMENTS)  # the real registry passes


def test_an_unrecorded_source_raises_rather_than_nulling():
    """A null instrument must mean "no device is documented", never "nobody looked"."""
    with pytest.raises(KeyError, match="No instrument recorded"):
        instrument_fields("a_source_nobody_recorded")


# (c) SHAPE
def test_every_recorded_id_is_a_well_formed_l22_term():
    """An id from another NERC collection would be a different kind of thing entirely."""
    for dataset_name, fields in sorted(DATASET_INSTRUMENTS.items()):
        identifier = fields["instrument_id"]
        if identifier is None:
            continue
        assert _L22_URI.match(identifier), (
            f"«{dataset_name}» has instrument_id {identifier!r}, which is not an L22 term URI. "
            f"L05 holds instrument CATEGORIES and P01 measured parameters; this column promises "
            f"a device."
        )


def test_the_nulls_are_the_three_documented_kinds_and_nothing_else():
    """Pins WHY each null is null, so a fourth kind cannot appear unremarked.

    Three distinct situations, and conflating them would lose real information:
      * no instrument documented anywhere (frepj) — both fields null;
      * a documented instrument L22 has no term for (zoolake's Dual Scripps Plankton Camera,
        the lensless microscope, planktonset1.0's ISIIS-2) — a name, a null id;
      * an instrument that varies per image (daplankton) — the PER_ROW sentinel.
    """
    no_name = {name for name, fields in DATASET_INSTRUMENTS.items() if fields["instrument"] is None}
    assert no_name == _NO_INSTRUMENT_AT_ALL

    named_without_id = {
        name
        for name, fields in DATASET_INSTRUMENTS.items()
        if fields["instrument"] not in (None, PER_ROW_INSTRUMENT) and fields["instrument_id"] is None
    }
    assert named_without_id == _NAMED_BUT_NO_L22_TERM

    per_row = {name for name, fields in DATASET_INSTRUMENTS.items() if fields["instrument"] == PER_ROW_INSTRUMENT}
    assert per_row == _PER_ROW


def test_sources_imaging_with_the_same_device_share_one_id():
    """The column's whole purpose: grouping across sources.

    Four sources image with a ZooScan and three with an IFCB. If those did not resolve to one
    id each, a per-instrument evaluation would silently split them.
    """
    by_identifier = {}
    for fields in DATASET_INSTRUMENTS.values():
        if fields["instrument_id"]:
            by_identifier.setdefault(fields["instrument_id"], set()).add(fields["instrument"])

    for identifier, names in by_identifier.items():
        assert len(names) == 1, f"{identifier} is recorded under more than one name: {sorted(names)}"

    zooscan = {n for n, f in DATASET_INSTRUMENTS.items() if f["instrument"] == "ZooScan"}
    assert zooscan == {"zooscan", "sykezooscan2024", "tara_pacific_hsn", "tara_pacific_manta"}
    assert len({DATASET_INSTRUMENTS[n]["instrument_id"] for n in zooscan}) == 1


def test_instrument_fields_returns_a_fresh_dict():
    """Handed straight to datasets.map, so a shared mutable would be a cross-process bug."""
    first, second = instrument_fields("zooscan"), instrument_fields("zooscan")
    assert first == second and first is not second
    first["instrument"] = "mutated"
    assert instrument_fields("zooscan")["instrument"] == "ZooScan"


def test_the_tara_pacific_layout_literals_agree_with_the_table():
    """``tara_pacific_layout.SOURCES`` has carried an ``instrument`` per deposit all along.

    Nothing read it and no test asserted it, so it was free to drift. It is a fourth
    transcription of the same fact; pin it to the table rather than leaving it decorative.
    """
    for dataset_name, source in sorted(tara_pacific_layout.SOURCES.items()):
        assert DATASET_INSTRUMENTS[dataset_name]["instrument"] == source["instrument"], (
            f"tara_pacific_layout.SOURCES[{dataset_name!r}]['instrument'] disagrees with DATASET_INSTRUMENTS"
        )


# (d) RESOLUTION
def test_the_per_row_source_refuses_a_source_level_answer():
    """DAPlankton exists to benchmark cross-instrument shift; one value for it is a wrong value."""
    with pytest.raises(ValueError, match="more than one instrument"):
        instrument_fields("daplankton")


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [("lab_cs_", "CytoSense"), ("sea_cs_", "CytoSense"), ("lab_fc_", "FlowCam"), ("lab_ifcb_", "IFCB"), ("sea_ifcb_", "IFCB")],
)
def test_each_daplankton_prefix_resolves_to_its_instrument(prefix, expected):
    """Offline, from the merge prefix the importer already leaves on every filename."""
    resolved = resolve_instrument("daplankton", f"/DAPlankton/{prefix}Some_class/{prefix}image_0001.png")
    assert resolved["instrument"] == expected
    assert _L22_URI.match(resolved["instrument_id"])


def test_the_prefix_table_covers_the_importer_merge_policy():
    """If the importer's five merge roots change, this table must change with them.

    ``DOMAIN_PREFIXES`` is the policy; ``DAPLANKTON_PREFIX_INSTRUMENTS`` reads its output. A
    sixth root added upstream would otherwise raise on every one of that root's images at
    build time rather than here.
    """
    assert {prefix for _, prefix in daplankton_layout.DOMAIN_PREFIXES} == set(DAPLANKTON_PREFIX_INSTRUMENTS)


def test_an_unprefixed_per_row_path_raises_rather_than_nulling():
    """The failure mode this design exists to prevent: a silent null on a whole source."""
    with pytest.raises(ValueError, match="carries none of"):
        resolve_instrument("daplankton", "/DAPlankton/Some_class/no_prefix_here.png")


def test_a_constant_source_ignores_the_path():
    """resolve_instrument is the single entry point, so it must be right for both kinds."""
    assert resolve_instrument("zooscan", "/anything/at/all.jpg") == instrument_fields("zooscan")
    assert resolve_instrument("zooscan") == instrument_fields("zooscan")


def test_the_readme_table_agrees_with_the_instrument_table():
    """The README cell is the fourth transcription of this fact and the only public one.

    Config, constants and ``tara_pacific_layout`` are all pinned above. The README is what a
    reader of the repository actually sees, and it is exactly the kind of hand-maintained cell
    that drifts — the licence table's one recorded drift (`zoolake`) was caught by a human
    comparing prose, not by a test. Checked by NAME only: the cell also carries the L22 link
    and the KI-33 qualifiers, which are presentation.
    """
    table = (Path(root) / "README.md").read_text(encoding="utf-8")
    rows = {
        match.group("dataset"): match.group("instrument")
        for match in re.finditer(
            r"^\| \*\*.+?\*\*.*? \| `(?P<dataset>[^`]+)` \| [\d,]+ \| (?P<instrument>[^|]+?) \|",
            table,
            re.MULTILINE,
        )
    }
    assert set(rows) == set(DATASET_INSTRUMENTS), (
        f"README source table and DATASET_INSTRUMENTS disagree on which sources exist: {set(rows) ^ set(DATASET_INSTRUMENTS)}"
    )

    for dataset_name, cell in sorted(rows.items()):
        recorded = DATASET_INSTRUMENTS[dataset_name]["instrument"]
        if recorded is None:
            assert "not recorded" in cell, f"«{dataset_name}» has no instrument; README says {cell!r}"
        elif recorded == PER_ROW_INSTRUMENT:
            assert "per image" in cell, f"«{dataset_name}» is per-image; README says {cell!r}"
        else:
            assert recorded in cell, f"README says {cell!r} for «{dataset_name}», table says {recorded!r}"
