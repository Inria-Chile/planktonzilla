"""
(c) Inria

Network-free tests for the FREPJ sidecar-table fetch + pure CSV parsers.

Offline BY CONSTRUCTION: every assertion reads only in-repo synthetic fixtures
under ``tests/fixtures/frepj/`` and the importable helpers in
``planktonzilla.planktonzilla_dataset.frepj_tables``. The one download path
(``ensure_frepj_tables``) is exercised with a SYNTHETIC manifest whose md5s are
computed from the tiny fixtures — the real 8.5 MB tables are never referenced,
and ``DownloadManager`` is patched to blow up if the skip branch ever tries to
reach the network.

Zero behavioral drift: these tests PIN the documented parser behavior (the
``East latitude``-holds-longitude source typo, trailing-space headers, S4's
stray trailing commas, integer-stem canonicalization), they do not "fix" it.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import hashlib
import json
from pathlib import Path

import pytest

from planktonzilla.planktonzilla_dataset import frepj_tables

FIXTURES = Path(__file__).parent / "fixtures" / "frepj"
S1 = FIXTURES / "table_s1_sample.csv"
S3 = FIXTURES / "table_s3_sample.csv"
S4 = FIXTURES / "table_s4_sample.csv"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- parse_frepj_filename -------------------------------------------------------------


def test_parse_frepj_filename_valid():
    """40_/100_ prefixes with an integer stem parse to (magnification, id)."""
    assert frepj_tables.parse_frepj_filename("40_123.jpg") == (40, "123")
    assert frepj_tables.parse_frepj_filename("100_9.jpg") == (100, "9")


def test_parse_frepj_filename_rejects_bad_prefix_or_stem():
    """Unknown prefix or non-numeric stem returns None and never raises."""
    assert frepj_tables.parse_frepj_filename("50_1.jpg") is None
    assert frepj_tables.parse_frepj_filename("40_abc.jpg") is None
    assert frepj_tables.parse_frepj_filename("no_underscore.jpg") is None
    assert frepj_tables.parse_frepj_filename("40_.jpg") is None
    assert frepj_tables.parse_frepj_filename("") is None


# --- read_site_coordinates ------------------------------------------------------------


def test_read_site_coordinates_east_latitude_is_longitude():
    """``North latitude`` -> lat and ``East latitude`` -> LONGITUDE (source typo)."""
    coords = frepj_tables.read_site_coordinates(S1)
    # Lake Biwa: lat 35.2, lon 136.1 (the "East latitude" column carries the longitude).
    assert coords["Lake Biwa"] == (pytest.approx(35.2), pytest.approx(136.1))
    assert coords["Akigawa Dam"] == (pytest.approx(35.72), pytest.approx(139.19))
    assert coords["Haji Dam"] == (pytest.approx(34.7), pytest.approx(132.5))


def test_read_site_coordinates_out_of_range_is_none():
    """A row whose lat/lon fall outside valid ranges resolves to (None, None)."""
    coords = frepj_tables.read_site_coordinates(S1)
    assert coords["Bad Site"] == (None, None)


def test_read_site_coordinates_conflicting_duplicate_site_resolves_to_null(tmp_path):
    """Two Table_S1 rows for the SAME site name that disagree by more than the
    conflict tolerance (real-world Tsurugajo case, ~138 km apart) resolve to
    (None, None) -- never a silently-picked, potentially wrong coordinate (CR-01)."""
    csv_path = tmp_path / "table_s1_conflict.csv"
    csv_path.write_text(
        "site,North latitude,East latitude,date\n"
        "Tsurugajo,37.48819753,139.9283394,2022.09.11\n"
        "Tsurugajo,38.72911302,139.8242578,2022.10.16\n"
    )
    coords = frepj_tables.read_site_coordinates(csv_path)
    assert coords["Tsurugajo"] == (None, None)


def test_read_site_coordinates_agreeing_duplicate_rows_resolve_to_centroid(tmp_path):
    """Duplicate rows for the same site that agree to within the conflict tolerance
    (up to 10 km -- ordinary same-site measurement imprecision, e.g. a large reservoir
    surveyed from different shore points; the real Lake Ashino case, ~5.5 km apart)
    resolve to their CENTROID -- the arithmetic mean of every valid row -- rather than
    last-write-wins or either individual row."""
    csv_path = tmp_path / "table_s1_noise.csv"
    csv_path.write_text(
        "site,North latitude,East latitude,date\n"
        "Lake Ashino,35.190234,139.024848,2023.04.28\n"
        "Lake Ashino,35.234145,138.99671,2023.04.28\n"
    )
    coords = frepj_tables.read_site_coordinates(csv_path)
    expected = (pytest.approx((35.190234 + 35.234145) / 2), pytest.approx((139.024848 + 138.99671) / 2))
    assert coords["Lake Ashino"] == expected
    # The centroid must differ from BOTH individual rows -- proof this is a mean, not a
    # last-write-wins pick or either row selected outright.
    assert coords["Lake Ashino"] != (pytest.approx(35.190234), pytest.approx(139.024848))
    assert coords["Lake Ashino"] != (pytest.approx(35.234145), pytest.approx(138.99671))


# --- read_per_image_site_index --------------------------------------------------------


def test_read_per_image_site_index_maps_magnification_and_token():
    """S3 rows key on magnification 40, S4 rows on 100; token+date preserved."""
    index = frepj_tables.read_per_image_site_index(S3, S4)
    assert index[(40, "101")] == ("akigawadam", "2018.03.15")
    assert index[(100, "201")] == ("biwako", "2020.05")
    assert index[(100, "202")] == ("(baiyou)", "2018")


def test_read_per_image_site_index_canonicalizes_integer_stem():
    """A zero-padded ID (``0104``) canonicalizes to ``str(int(...))`` == ``104``."""
    index = frepj_tables.read_per_image_site_index(S3, S4)
    assert (40, "104") in index
    assert (40, "0104") not in index


def test_read_per_image_site_index_skips_non_integer_id():
    """A non-integer ID row is skipped, not raised — no ``abc`` key appears."""
    index = frepj_tables.read_per_image_site_index(S3, S4)
    assert not any(id_ == "abc" for _, id_ in index)


def test_read_per_image_site_index_tolerates_trailing_columns_and_headers():
    """Trailing-space headers (``Order ``/``Others ``) and S4's stray trailing
    commas do not break selection of ID/site/date."""
    index = frepj_tables.read_per_image_site_index(S3, S4)
    # S4 rows carry 3 stray trailing-comma columns yet still parse cleanly.
    assert index[(100, "201")][0] == "biwako"
    # Both magnifications contributed rows.
    mags = {mag for mag, _ in index}
    assert mags == {40, 100}


# --- distinct_site_tokens -------------------------------------------------------------


def test_distinct_site_tokens():
    """The set of tokens is exactly the resolvable-or-not tokens in the index."""
    index = frepj_tables.read_per_image_site_index(S3, S4)
    assert frepj_tables.distinct_site_tokens(index) == {
        "akigawadam",
        "hajidamu",
        "biwako",
        "(baiyou)",
    }


# --- manifest + path constants --------------------------------------------------------


def test_manifest_matches_committed_figshare_manifest():
    """FREPJ_TABLE_MANIFEST pins the same file_id/url/md5 as the committed manifest."""
    manifest = json.loads((FIXTURES / "frepj_figshare_manifest.json").read_text())
    by_name = {f["name"]: f for f in manifest["files"]}

    assert {e["name"] for e in frepj_tables.FREPJ_TABLE_MANIFEST} == {
        "Table_S1.csv",
        "Table_S3.csv",
        "Table_S4.csv",
    }
    for entry in frepj_tables.FREPJ_TABLE_MANIFEST:
        src = by_name[entry["name"]]
        assert entry["md5"] == src["md5"]
        assert entry["file_id"] == src["id"]
        assert str(src["id"]) in entry["url"]


def test_path_constants_are_defined():
    """The module exposes the path constants Plan 17-02 + the generator consume."""
    assert frepj_tables.PACKAGE_DIR.name == "planktonzilla_dataset"
    assert frepj_tables.DEFAULT_CROSSWALK_PATH.name == "frepj_site_crosswalk.csv"
    assert frepj_tables.DEFAULT_OVERRIDES_PATH.name == "frepj_site_overrides.csv"
    assert frepj_tables.DEFAULT_TABLES_DIR.parts[-2:] == ("data", "frepj_tables")


# --- ensure_frepj_tables --------------------------------------------------------------


def _synthetic_manifest(dest: Path) -> list[dict]:
    """A manifest that points at copies of the tiny fixtures with their real md5s,
    so the skip branch can be exercised without any network access."""
    mapping = {"Table_S1.csv": S1, "Table_S3.csv": S3, "Table_S4.csv": S4}
    manifest = []
    for name, fixture in mapping.items():
        target = dest / name
        target.write_bytes(fixture.read_bytes())
        manifest.append(
            {
                "name": name,
                "file_id": 0,
                "url": "https://example.invalid/never-fetched",
                "md5": _md5(target),
                "size": target.stat().st_size,
            }
        )
    return manifest


def test_ensure_frepj_tables_skips_download_when_md5_matches(tmp_path, monkeypatch):
    """When all files already exist with matching md5s, NO download is attempted."""

    def _boom(*args, **kwargs):
        raise AssertionError("DownloadManager must not be constructed on the skip path")

    monkeypatch.setattr(frepj_tables, "DownloadManager", _boom)

    manifest = _synthetic_manifest(tmp_path)
    paths = frepj_tables.ensure_frepj_tables(tmp_path, manifest=manifest)

    assert set(paths) == {"Table_S1.csv", "Table_S3.csv", "Table_S4.csv"}
    for name, path in paths.items():
        assert Path(path).exists()
        assert _md5(Path(path)) == next(e["md5"] for e in manifest if e["name"] == name)


def test_ensure_frepj_tables_raises_on_post_download_md5_mismatch(tmp_path, monkeypatch):
    """A downloaded file whose md5 does not match the manifest raises ValueError."""

    class _FakeDownloadManager:
        def __init__(self, *args, **kwargs):
            pass

        def download(self, url):
            bad = tmp_path / "_downloaded_bad.bin"
            bad.write_bytes(b"corrupted-bytes-that-will-not-match")
            return str(bad)

    monkeypatch.setattr(frepj_tables, "DownloadManager", _FakeDownloadManager)

    manifest = [
        {
            "name": "Table_S1.csv",
            "file_id": 49092727,
            "url": "https://example.invalid/files/49092727",
            "md5": "0" * 32,  # deliberately wrong
            "size": 14088,
        }
    ]

    with pytest.raises(ValueError, match="md5"):
        frepj_tables.ensure_frepj_tables(tmp_path, manifest=manifest)


# --- sampling-date normalization (KI-26) ----------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2022.08.28", "2022-08-28"),
        ("2019.11.6", "2019-11-06"),
        (" 2018.03.15 ", "2018-03-15"),
        ("2022.06,10", "2022-06-10"),
        ("20200917", "2020-09-17"),
        ("230815inba_funato", "2023-08-15"),
        ("230427ashinoko4", "2023-04-27"),
        ("biwako_20211122(462)", "2021-11-22"),
        ("biwa230213_100", "2023-02-13"),
        # Never guessed: three-digit day, month-only, bare tokens, impossible dates.
        ("2021.11.011", None),
        ("2020.08.dd", None),
        ("2016.11.", None),
        ("2311asahiyama_dai", None),
        ("akanko1", None),
        ("tsuruoka_100", None),
        ("akanko_557,558", None),
        ("2020.02.30", None),
        ("19990101", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_sampling_date(raw, expected):
    """Each KI-26 family maps by exactly one rule to ISO, or to None — never to a guess."""
    assert frepj_tables.normalize_sampling_date(raw) == expected


def test_three_digit_day_candidates():
    """A three-digit day means the day with any ONE digit dropped (real calendar dates only)."""
    assert frepj_tables.three_digit_day_candidates("2021.11.015") == ["2021-11-01", "2021-11-05", "2021-11-15"]
    assert frepj_tables.three_digit_day_candidates("2021.11.011") == ["2021-11-01", "2021-11-11"]
    assert frepj_tables.three_digit_day_candidates("2021.11.15") == []
    assert frepj_tables.three_digit_day_candidates(None) == []


def test_resolve_three_digit_day_requires_exactly_one_table_s1_match():
    """Resolves only when Table_S1 confirms exactly one candidate for the site."""
    assert frepj_tables.resolve_three_digit_day("2021.11.011", {"2021-11-01"}) == "2021-11-01"
    assert frepj_tables.resolve_three_digit_day("2021.11.015", {"2021-11-01", "2021-11-15"}) is None
    assert frepj_tables.resolve_three_digit_day("2021.11.015", {"2021-11-22"}) is None
    assert frepj_tables.resolve_three_digit_day("2021.11.015", None) is None
    assert frepj_tables.resolve_three_digit_day("2021.11.015", set()) is None


def test_parse_sampling_date_applies_rules_first_then_table_s1():
    """The fixed rules decide when they can; Table_S1 is consulted only for a three-digit day."""
    assert frepj_tables.parse_sampling_date("20200917", {"2020-09-01"}) == "2020-09-17"
    assert frepj_tables.parse_sampling_date("2021.11.011", {"2021-11-01"}) == "2021-11-01"
    assert frepj_tables.parse_sampling_date("2021.11.011") is None
    assert frepj_tables.parse_sampling_date("akanko1", {"2021-11-01"}) is None


def test_read_site_sampling_dates_normalizes_and_skips_unreadable(tmp_path):
    """Table_S1 (site, date) rows become {site: {ISO dates}}; month-only/blank rows add nothing."""
    csv_path = tmp_path / "table_s1_dates.csv"
    csv_path.write_text(
        "site,North latitude,East latitude,date\n"
        "Lake Biwa,35.25,136.05,2021.11.01\n"
        "Lake Biwa,35.25,136.05,2021.11.15\n"
        "Lake Biwa,35.25,136.05,2021.07\n"
        "Lake Akan,43.455,144.110,20100714\n"
        "Empty Site,1,1,\n"
    )
    dates = frepj_tables.read_site_sampling_dates(csv_path)
    assert dates == {"Lake Biwa": {"2021-11-01", "2021-11-15"}, "Lake Akan": {"2010-07-14"}}
