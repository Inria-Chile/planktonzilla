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


def test_build_crosswalk_override_wins(site_coords):
    """An override entry resolves a token that neither trivially nor fuzzily matches."""
    rows = frepj_crosswalk.build_crosswalk(site_coords, {"biwako": 3}, {"biwako": "Lake Biwa"})
    row = _by_token(rows)["biwako"]
    assert row["method"] == "override"
    assert row["resolved_site"] == "Lake Biwa"
    assert row["Latitude"] == pytest.approx(35.2)
    assert row["Longitude"] == pytest.approx(136.1)


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
