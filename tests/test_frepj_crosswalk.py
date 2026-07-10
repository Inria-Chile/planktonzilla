"""
(c) Inria

Network-free tests for the FREPJ romanized-site -> lat/lon crosswalk builder.

Offline BY CONSTRUCTION: the build/load logic is exercised only over the tiny
synthetic fixtures from ``test_frepj_tables`` (Table_S1/S3/S4 samples) — the real
8.5 MB tables are never referenced here. The precedence under test is strict:
trivial-normalize -> difflib auto-fuzzy (high-confidence only) -> hand-curated
override -> null. An ambiguous token NEVER receives a guessed coordinate.

Zero behavioral drift: these tests PIN the resolution semantics, they do not
"fix" the source data.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from collections import Counter
from pathlib import Path

import pytest

from planktonzilla.planktonzilla_dataset import frepj_crosswalk, frepj_tables

FIXTURES = Path(__file__).parent / "fixtures" / "frepj"
S1 = FIXTURES / "table_s1_sample.csv"
S3 = FIXTURES / "table_s3_sample.csv"
S4 = FIXTURES / "table_s4_sample.csv"


@pytest.fixture()
def site_coords():
    return frepj_tables.read_site_coordinates(S1)


@pytest.fixture()
def token_counts():
    index = frepj_tables.read_per_image_site_index(S3, S4)
    return dict(Counter(token for token, _date in index.values()))


def _by_token(rows):
    return {r["site_token"]: r for r in rows}


# --- normalize_token ------------------------------------------------------------------


def test_normalize_token_collapses_separators_and_case():
    """Case, whitespace, and non-alphanumeric separators are dropped for comparison."""
    assert frepj_crosswalk.normalize_token("Akigawa Dam") == frepj_crosswalk.normalize_token("akigawadam")
    assert frepj_crosswalk.normalize_token("  (Baiyou) ") == "baiyou"
    # "Lake Biwa" and "biwako" are romanization variants that do NOT share a trivial key.
    assert frepj_crosswalk.normalize_token("Lake Biwa") != frepj_crosswalk.normalize_token("biwako")


# --- build_crosswalk precedence -------------------------------------------------------


def test_build_crosswalk_trivial_match(site_coords, token_counts):
    """A token whose normalized form equals a site name resolves via method='trivial'."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, token_counts, {})
    row = _by_token(rows)["akigawadam"]
    assert row["method"] == "trivial"
    assert row["resolved_site"] == "Akigawa Dam"
    assert row["Latitude"] == pytest.approx(35.72)
    assert row["Longitude"] == pytest.approx(139.19)
    assert row["n_images"] == 2


def test_build_crosswalk_fuzzy_match(site_coords, token_counts):
    """A high-confidence romanization variant resolves via method='fuzzy'."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, token_counts, {})
    row = _by_token(rows)["hajidamu"]
    assert row["method"] == "fuzzy"
    assert row["resolved_site"] == "Haji Dam"
    assert (row["Latitude"], row["Longitude"]) == (pytest.approx(34.7), pytest.approx(132.5))


def test_build_crosswalk_fuzzy_tie_with_distant_coords_is_null():
    """Two Table_S1 candidates that TIE on fuzzy score but map to materially distant
    real-world coordinates are never guessed between -- resolves to null (WR-01)."""
    tied_site_coords = {
        "Test Site 1": (35.0, 140.0),
        "Test Site 2": (36.0, 141.0),  # ~140 km from "Test Site 1" -- a real conflict
    }
    rows = frepj_crosswalk.build_crosswalk(tied_site_coords, {"testsite3": 1}, {})
    row = _by_token(rows)["testsite3"]
    assert row["method"] == "null"
    assert row["resolved_site"] == ""
    assert row["Latitude"] is None
    assert row["Longitude"] is None


def test_build_crosswalk_fuzzy_tie_with_near_identical_coords_still_resolves():
    """A fuzzy score tie between two candidates whose coordinates are near-identical
    (the real ``sakurajo`` case, ~60 m apart) still resolves via method='fuzzy' --
    the tie guard only nulls MATERIALLY distant ties, not every tie (WR-01)."""
    tied_site_coords = {
        "Test Site 1": (35.00000, 140.00000),
        "Test Site 2": (35.00050, 140.00000),  # ~55 m from "Test Site 1" -- noise, not a conflict
    }
    rows = frepj_crosswalk.build_crosswalk(tied_site_coords, {"testsite3": 1}, {})
    row = _by_token(rows)["testsite3"]
    assert row["method"] == "fuzzy"
    assert row["resolved_site"] in tied_site_coords
    assert row["Latitude"] is not None and row["Longitude"] is not None


def test_build_crosswalk_resolved_site_with_null_coords_downgrades_to_null_row(token_counts):
    """A token that resolves (trivial) to a Table_S1 site whose OWN coordinate is
    (None, None) -- e.g. a same-name collision nulled by CR-01 -- downgrades to a full
    null row rather than emitting a non-null method with blank coordinates."""
    conflicted_site_coords = {"Akigawa Dam": (None, None)}
    rows = frepj_crosswalk.build_crosswalk(conflicted_site_coords, token_counts, {})
    row = _by_token(rows)["akigawadam"]
    assert row["method"] == "null"
    assert row["resolved_site"] == ""
    assert row["Latitude"] is None
    assert row["Longitude"] is None


def test_build_crosswalk_override_wins(site_coords):
    """An override entry resolves a token that neither trivially nor fuzzily matches."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, {"biwako": 3}, {"biwako": "Lake Biwa"})
    row = _by_token(rows)["biwako"]
    assert row["method"] == "override"
    assert row["resolved_site"] == "Lake Biwa"
    assert row["Latitude"] == pytest.approx(35.2)
    assert row["Longitude"] == pytest.approx(136.1)


def test_build_crosswalk_override_naming_unknown_site_raises(site_coords):
    """An override whose target is NOT a Table_S1 site name (e.g. a typo in
    ``frepj_site_overrides.csv``) fails loud at build time instead of silently
    emitting method='override' with blank coordinates (WR-02)."""
    with pytest.raises(ValueError, match="unknown Table_S1 site"):
        frepj_crosswalk.build_crosswalk(site_coords, {"biwako": 3}, {"biwako": "Lake Biwaaa"})


def test_build_crosswalk_unoverridden_ambiguous_is_null(site_coords):
    """Without an override the same low-confidence token stays null (never guessed)."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, {"biwako": 3}, {})
    row = _by_token(rows)["biwako"]
    assert row["method"] == "null"
    assert row["Latitude"] is None
    assert row["Longitude"] is None


def test_build_crosswalk_unknown_token_is_null_and_does_not_raise(site_coords, token_counts):
    """An unresolvable token (e.g. ``(baiyou)``) yields method='null' with empty coords."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, token_counts, {})
    row = _by_token(rows)["(baiyou)"]
    assert row["method"] == "null"
    assert row["resolved_site"] == ""
    assert row["Latitude"] is None
    assert row["Longitude"] is None


def test_build_crosswalk_covers_every_token(site_coords, token_counts):
    """Exactly one row per distinct token — none dropped."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, token_counts, {})
    assert set(_by_token(rows)) == set(token_counts)


# --- load_crosswalk round-trip --------------------------------------------------------


def test_load_crosswalk_round_trip(tmp_path, site_coords):
    """A written crosswalk round-trips to token -> (lat, lon), None for null rows."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, {"akigawadam": 2, "(baiyou)": 1}, {})
    out = tmp_path / "cw.csv"
    frepj_crosswalk.write_crosswalk(rows, out)

    mapping = frepj_crosswalk.load_crosswalk(out)
    lat, lon = mapping["akigawadam"]
    assert (lat, lon) == (pytest.approx(35.72), pytest.approx(139.19))
    assert mapping["(baiyou)"] == (None, None)


def test_load_overrides_skips_comments_and_blanks(tmp_path):
    """load_overrides tolerates a comment-only override table (returns {})."""
    p = tmp_path / "overrides.csv"
    p.write_text("site_token,resolved_site\n# a comment line, not a row\n")
    assert frepj_crosswalk.load_overrides(p) == {}

    p.write_text("site_token,resolved_site\niwaodam,Iwao Dam\n# trailing comment\n")
    assert frepj_crosswalk.load_overrides(p) == {"iwaodam": "Iwao Dam"}


# --- committed crosswalk coverage (reads ONLY the committed CSV — network-free) --------


def _committed_crosswalk_rows():
    """Load the committed crosswalk. Reads ONLY the committed CSV, never the raw tables."""
    import polars as pl

    df = pl.read_csv(frepj_tables.DEFAULT_CROSSWALK_PATH, infer_schema_length=0)
    return df.to_dicts()


def _is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def test_committed_crosswalk_coverage():
    """The committed crosswalk meets the >=55-token / >=72%-image baseline with valid
    coordinate ranges and ambiguous tokens left null — asserted from the CSV alone."""
    rows = _committed_crosswalk_rows()
    assert rows, "committed crosswalk is empty"

    resolved = [r for r in rows if not _is_blank(r["Latitude"])]

    # (a) resolved-token count must not regress below the trivial baseline.
    assert len(resolved) >= 55, f"only {len(resolved)} resolved tokens (need >= 55)"

    # (b) resolved images cover at least 72% of all images.
    total_images = sum(int(r["n_images"]) for r in rows)
    resolved_images = sum(int(r["n_images"]) for r in resolved)
    assert total_images > 0
    assert resolved_images / total_images >= 0.72, (
        f"image coverage {resolved_images}/{total_images} = {resolved_images / total_images:.3f} < 0.72"
    )

    for r in rows:
        method = r["method"]
        if method == "null":
            # (d) null rows carry NO coordinate.
            assert _is_blank(r["Latitude"]) and _is_blank(r["Longitude"]), f"null row has coords: {r['site_token']}"
        else:
            # (c) every non-null coordinate is within valid geographic ranges.
            latitude, longitude = float(r["Latitude"]), float(r["Longitude"])
            assert -90.0 <= latitude <= 90.0, f"latitude out of range for {r['site_token']}: {latitude}"
            assert -180.0 <= longitude <= 180.0, f"longitude out of range for {r['site_token']}: {longitude}"
            # (e) a resolved row always names its Table_S1 site.
            assert not _is_blank(r["resolved_site"]), f"resolved row missing resolved_site: {r['site_token']}"
