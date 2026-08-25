"""
(c) Inria

Tests for file-name capture-time extraction.

Every path literal in ``REAL_PATHS`` was taken from real data — the published
planktonzilla-17M parquet for the twelve sources it holds, and a local zoolake build
for the thirteenth — rather than written to match the regexes. That is the point of
them: a pattern that only parses an example invented alongside it proves nothing about
the dataset it will run on.
"""

from datetime import UTC, datetime

import pytest
from datasets import Dataset

from planktonzilla.planktonzilla_dataset.generate_planktonzilla import NoMetadataRedefiner
from planktonzilla.planktonzilla_dataset.timestamps import (
    PATTERNS,
    audit_paths,
    extract_path_timestamp,
    merge_timestamp,
)

# (source, real original_path, expected UTC capture time, expected pattern name)
REAL_PATHS = [
    (
        "zoolake",
        "/train_split/aphanizomenon/SPC-EAWAG-0P5X-1589537024858496-10282432268182-002149-027-1508-1114-64-88.jpeg",
        datetime(2020, 5, 15, 10, 3, 44, 858496, tzinfo=UTC),
        "spc-epoch-us",
    ),
    (
        "whoi",
        "/mix/IFCB1_2008_073_152345_00747.png",
        datetime(2008, 3, 13, 15, 23, 45, tzinfo=UTC),
        "ifcb-day-of-year",
    ),
    (
        "medplanktonset",
        "/Proboscia_spp._Rhizosolenia_spp/D20240405T043311_IFCB181_04070.png",
        datetime(2024, 4, 5, 4, 33, 11, tzinfo=UTC),
        "ifcb-iso-tag",
    ),
    (
        "planktonset1.0",
        "/acantharia_protist/20140602073348.076.0131_000_crop_463.jpg",
        datetime(2014, 6, 2, 7, 33, 48, 76000, tzinfo=UTC),
        "isiis-compact-ms",
    ),
    (
        "jedioceans",
        "/LClass_CC.Copepod/20151103_102539.630.0.png",
        datetime(2015, 11, 3, 10, 25, 39, 630000, tzinfo=UTC),
        "cpics-compact-ms",
    ),
    (
        "planktoscope",
        "/Eutintinnus/2024-03-12_11-55-35-346253_1.jpg",
        datetime(2024, 3, 12, 11, 55, 35, 346253, tzinfo=UTC),
        "planktoscope-dashed-us",
    ),
]

# Real paths from the sources that encode NO capture time. These are the false-positive
# guard: flowcamnet/isiisnet/uvp6net/zoocamnet/global_uvp5 name files after an EcoTaxa
# object ID, and a 9-digit ID read as an epoch is a plausible-looking 1975 date. If a
# pattern ever loosens enough to claim one of these, five sources gain a fabricated
# timestamp at once and nothing else in the pipeline would notice.
REAL_PATHS_WITHOUT_TIME = [
    ("flowcamnet", "/Dinophyceae/164743291.png"),
    ("isiisnet", "/detritus/257738613.png"),
    ("uvp6net", "/artefact/287295216.png"),
    ("zoocamnet", "/Acartiidae/227413015.png"),
    ("global_uvp5", "/2649231.jpg"),
    ("syke_ifcb_2022", "/Pennales_sp_thick/Pennales_sp._boxy_175.png"),
    # PlanktoScope's own date-less variant: a time of day with no date is not a
    # timestamp, and must not be completed with a guessed day.
    ("planktoscope", "/Eutintinnus/20_06_43_996335_3.jpg"),
]


@pytest.mark.parametrize("source,path,expected,pattern", REAL_PATHS, ids=[r[0] for r in REAL_PATHS])
def test_real_path_parses_to_the_right_instant(source, path, expected, pattern):
    stamp, matched = extract_path_timestamp(path)
    assert stamp == expected.isoformat(), f"{source} parsed to {stamp}"
    assert matched == pattern


@pytest.mark.parametrize("source,path", REAL_PATHS_WITHOUT_TIME, ids=[r[0] for r in REAL_PATHS_WITHOUT_TIME])
def test_paths_without_a_capture_time_are_not_claimed(source, path):
    assert extract_path_timestamp(path) == (None, None), f"{source} path {path} was wrongly parsed"


def test_exactly_one_pattern_claims_each_real_path():
    """No two conventions may overlap: first-match-wins would hide the ambiguity."""
    for source, path, _, _ in REAL_PATHS:
        claiming = [p.name for p in PATTERNS if p.regex.search(path)]
        assert len(claiming) == 1, f"{source} path claimed by {claiming}"


def test_implausible_instants_are_rejected():
    # A 16-digit group behind an SPC- prefix whose value lands in 1973.
    assert extract_path_timestamp("/x/SPC-EAWAG-0P5X-0000112233445566-1.jpeg") == (None, None)
    # Day-of-year 999 is structurally valid and calendrically impossible.
    assert extract_path_timestamp("/mix/IFCB1_2008_999_152345_00747.png") == (None, None)
    # Month 13.
    assert extract_path_timestamp("/x/20141302073348.076.0131_000.jpg") == (None, None)


def test_a_leap_day_of_year_resolves_correctly():
    """2008 is a leap year, so day 060 is 29 February — not 1 March."""
    stamp, _ = extract_path_timestamp("/mix/IFCB1_2008_060_000000_00001.png")
    assert stamp == datetime(2008, 2, 29, tzinfo=UTC).isoformat()
    # ... and in a non-leap year the same ordinal is 1 March.
    stamp, _ = extract_path_timestamp("/mix/IFCB1_2009_060_000000_00001.png")
    assert stamp == datetime(2009, 3, 1, tzinfo=UTC).isoformat()


@pytest.mark.parametrize("path", ["", "/", "no-digits.png", "/x/" + "9" * 200 + ".png", "SPC-"])
def test_malformed_paths_never_raise(path):
    assert extract_path_timestamp(path) == (None, None)


class TestMergeTimestamp:
    def test_no_derived_value_keeps_the_recorded_one(self):
        assert merge_timestamp("2008-03-13", None) == ("2008-03-13", "kept")

    def test_absent_recorded_value_is_filled(self):
        assert merge_timestamp(None, "2020-05-15T12:03:44+00:00") == ("2020-05-15T12:03:44+00:00", "filled")
        assert merge_timestamp("", "2020-05-15T12:03:44+00:00") == ("2020-05-15T12:03:44+00:00", "filled")

    def test_same_day_is_refined_to_the_more_precise_value(self):
        """The WHOI case: the API reports the bin's day, the file name its second."""
        merged, outcome = merge_timestamp("2008-03-13", "2008-03-13T15:23:45+00:00")
        assert (merged, outcome) == ("2008-03-13T15:23:45+00:00", "refined")

    def test_a_different_day_keeps_the_recorded_value(self):
        """A disagreement is never resolved in favour of the regex."""
        merged, outcome = merge_timestamp("2008-03-13", "2019-01-01T00:00:00+00:00")
        assert (merged, outcome) == ("2008-03-13", "conflict")


def test_audit_reports_coverage_and_range():
    paths = [p for _, p, _, _ in REAL_PATHS] + [p for _, p in REAL_PATHS_WITHOUT_TIME]
    report = audit_paths(paths, sample_name="mixed")

    assert report["total"] == len(paths)
    assert report["matched"] == len(REAL_PATHS)
    assert report["patterns"]["(no match)"] == len(REAL_PATHS_WITHOUT_TIME)
    assert report["earliest"].startswith("2008-03-13")
    assert report["latest"].startswith("2024-04-05")


class TestApplyPathTimestamps:
    """The pipeline hook, on the base class so every redefiner inherits it."""

    def _dataset(self, paths, recorded):
        return Dataset.from_dict(
            {
                "dataset": ["zoolake"] * len(paths),
                "original_path": paths,
                "timestamp": recorded,
                "Latitude": [None] * len(paths),
            }
        )

    def test_null_timestamps_are_filled_from_the_path(self):
        paths = [r[1] for r in REAL_PATHS]
        ds = self._dataset(paths, [None] * len(paths))

        out = NoMetadataRedefiner.__new__(NoMetadataRedefiner)._apply_path_timestamps(ds, "zoolake")

        assert out["timestamp"] == [r[2].isoformat() for r in REAL_PATHS]

    def test_column_order_is_preserved(self):
        """add_column appends; the frozen artifact has a fixed column order."""
        paths = [REAL_PATHS[0][1]]
        ds = self._dataset(paths, [None])
        before = list(ds.column_names)

        out = NoMetadataRedefiner.__new__(NoMetadataRedefiner)._apply_path_timestamps(ds, "zoolake")

        assert list(out.column_names) == before

    def test_a_source_with_no_pattern_is_returned_untouched(self):
        paths = [p for _, p in REAL_PATHS_WITHOUT_TIME]
        ds = self._dataset(paths, [None] * len(paths))

        out = NoMetadataRedefiner.__new__(NoMetadataRedefiner)._apply_path_timestamps(ds, "flowcamnet")

        assert out["timestamp"] == [None] * len(paths)

    def test_a_conflicting_path_never_overwrites_a_recorded_date(self):
        path = REAL_PATHS[0][1]  # parses to 2020-05-15
        ds = self._dataset([path], ["1999-01-01"])

        out = NoMetadataRedefiner.__new__(NoMetadataRedefiner)._apply_path_timestamps(ds, "zoolake")

        assert out["timestamp"] == ["1999-01-01"]

    def test_a_dataset_without_the_columns_is_a_no_op(self):
        ds = Dataset.from_dict({"dataset": ["x"], "original_path": ["/a/b.png"]})

        out = NoMetadataRedefiner.__new__(NoMetadataRedefiner)._apply_path_timestamps(ds, "x")

        assert out.column_names == ["dataset", "original_path"]
