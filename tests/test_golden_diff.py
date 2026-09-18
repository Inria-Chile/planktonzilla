"""
(c) Inria

The golden-diff harness's test suite, written FROM THE SPEC rather than from the implementation.

`planktonzilla/planktonzilla_dataset/utils/golden_diff.py` compares the taxonomy columns of the
PUBLISHED `project-oceania/planktonzilla-17M` against what this tree renders. It has two stages
that must never be confused: `--refresh` reads the Hub and writes a committed reference pair
(`utils/golden_reference.csv` + `utils/golden_reference.json`), and `--report` compares that
committed reference against a fresh render with no network at all. Only the second runs in CI, and
only the second is what makes a green PR mean anything.

Every test here was written against PLAN.md steps 14-20 BEFORE the module existed. That is
deliberate and it is the whole value of this file: a test written from the implementation pins the
implementation's bugs, while a test written from the spec pins the contract the implementation owes.
A disagreement between this file and the module is therefore a finding to adjudicate, not a test to
relax. The names this suite assumes of the module are listed in ONE place below, `# ── assumed API`,
so a rename is one edit rather than thirty.

Three properties are worth more than the rest:

  * THE KEYSTONE — `test_the_reference_writer_reproduces_render_bytes_for_the_rows_it_covers` pins
    the entire inversion table (PLAN.md step 14) in a single byte comparison. Every trap in that
    table — `plankton` as Arrow `bool` rather than `true`/`false`, the `.0` suffix on three id
    columns and NOT on `ecotaxa_ID`, `living` synthesised from `root_class`, keys renamed but never
    case-folded — fails this one assertion if it is got wrong.
  * EXIT CODES ARE NOT INTERCHANGEABLE (step 17). 1 means the DATA differs; 2 means the harness
    could not read or will not vouch; 3 means published rows disagree with each other and a human
    must adjudicate. A flaky network can only ever produce 2. A reviewer who cannot rely on that
    separation cannot rely on a green run either.
  * THE HARNESS NEVER PICKS A VALUE (step 16). Four places in this tree silently collapse a
    duplicate key; this one emits a `«DISAGREE:…»` sentinel and a finding instead.

THE FIXTURE TRICK: the fake Hub rows every test here uses are built from the committed CSV itself —
drop the six sources that are not published yet, push the rest through `render.published_projection`
(already the published contract: `plankton` as `bool`, `living` excluded). There are no fixture files
to maintain and they cannot drift from the build, because they ARE the build.

Network-free by construction. The autouse `_offline` fixture is the four-layer guard copied from
`tests/test_gen_planktonzilla_lensless_e2e.py`, and
`test_the_report_stage_imports_nothing_that_speaks_http` PROVES the report stage stays offline from
the import graph rather than asserting it. There is no pytest marker: `pyproject.toml` registers
none, and one added here would be the first in the tree.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import ast
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import datasets
import huggingface_hub
import pyarrow as pa
import pytest
import requests

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, render
from planktonzilla.planktonzilla_dataset.taxonomy.model import LEGACY_HEADER

# ── assumed API ────────────────────────────────────────────────────────────────────────────────
# The module does not exist when this file is written, so these names ARE the contract, taken from
# PLAN.md steps 14-20. Anything the module spells differently is a finding for the next stage; fix
# it here, in one place, rather than scattering getattr fallbacks through the tests.
#
#   hub_cell_to_csv(column, value) -> str          step 14, named verbatim in the plan
#   hub_row_to_csv_row(row) -> dict                step 14, the row-level counterpart
#   reference_bytes(hub_rows) -> bytes             step 15.8-9, sorted then render.write_wide_csv
#   CellDisagreement(dataset, original_label, column, values) with a .finding_id property   step 16
#   refresh(repo_id, revision, *, workers, block_size, shards, reference_path, manifest_path,
#           api, fs) -> Reference                  step 15, with api/fs injected the way
#                                                  check_base_on_hub already takes `api=None`
#   list_shards(api, repo_id, sha) / read_shard(fs, path, columns) / read_shard_schema(fs, path)
#                                                  step 21, "the only networked functions"
#   report(reference_csv, manifest, package_dir, waivers) -> int          step 19, paths
#   main(argv) -> int                              step 20
#   GoldenDiffError with an .exit_code             step 17's table needs a carrier
#
# Two shapes are assumed of the manifest as well: `excluded_datasets` is a container of dataset
# NAMES (a list or a name->count mapping both satisfy `set(...)`), and a serialised
# CellDisagreement's `values` is a list of `[value, n_rows, [shard, ...]]` triples.
_SENTINEL_PREFIX = "«DISAGREE:"

_REPO_ID = "project-oceania/planktonzilla-17M"
_FROZEN_SHA = "7fd542ef0c1b4a9d8e3f6021b5c7d84a9e0f1b23"

_TAXONOMY_CSV = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"
_SAMPLES_JSON = root / "samples.json"
_UTILS_DIR = root / "planktonzilla" / "planktonzilla_dataset" / "utils"
_COMMITTED_REFERENCE = _UTILS_DIR / "golden_reference.csv"
_COMMITTED_MANIFEST = _UTILS_DIR / "golden_reference.json"

# The Hub's names for the two key columns. `Dataset`/`Raw_Labels` in the CSV, `dataset`/
# `original_label` on the Hub — a rename and nothing else, which is why they are spelled out here
# rather than lower-cased from the CSV header.
_HUB_KEY_COLUMNS = ("dataset", "original_label")
_HUB_COLUMNS = (*_HUB_KEY_COLUMNS, *render.PUBLISHED_TAXONOMY_COLUMNS)

# Measured on the committed table; each number is also an independent assertion somewhere below.
_CSV_ROWS = 2358
_COVERED_ROWS = 1485
_EXCLUDED_ROWS = 873
_NUMERIC_ID_CELLS = 5733
_ECOTAXA_CELLS = 1523
_NON_LOWERCASE_LABELS = 1842


# ── offline guard ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Four layers of network guard, so an accidental Hub read in the report path fails loudly.

    The env vars alone are not enough: `datasets` and `huggingface_hub` read them at IMPORT time,
    which already happened, so the captured flags are flipped directly too. The `requests` patches
    are the backstop for a code path that reaches HTTP without going through either library — the
    exact failure this guard exists to make impossible is a `--report` run that quietly phones the
    Hub on a developer's machine, passes, and then fails in CI where there is no token.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True, raising=False)

    def _no_network(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests.Session, "get", _no_network)


@pytest.fixture
def gd():
    """The harness module, imported lazily.

    A module-level import would make collection of this whole file fail while `golden_diff.py` is
    still being written, taking the offline-proof test and the committed-artefact tests down with
    it. Importing inside a fixture keeps the failure scoped to the tests that genuinely need the
    module, and keeps it an ERROR rather than a skip — a silently skipped contract test is worse
    than no test.
    """
    from planktonzilla.planktonzilla_dataset.utils import golden_diff

    return golden_diff


# ── the fixture trick ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def csv_rows():
    """All 2,358 committed wide rows, derived from the model exactly as the build derives them."""
    return loader.load_taxonomy(loader.PACKAGE_DIR).rows()


@pytest.fixture(scope="module")
def published_datasets():
    """The fifteen sources that actually appear in the published artifact.

    Read from `samples.json`, a real scan of the frozen dataset, rather than restated here. That is
    the same independent-derivation trick `tests/test_dataset_licenses.py` plays: a hand-written
    list of fifteen names would agree with whatever it was copied from and prove nothing.
    """
    return {row["dataset"] for row in json.loads(_SAMPLES_JSON.read_text())["counts"]}


@pytest.fixture(scope="module")
def covered_rows(csv_rows, published_datasets):
    """The committed rows a Hub reference can cover: everything outside the six pending sources."""
    return [row for row in csv_rows if row["Dataset"] in published_datasets]


@pytest.fixture(scope="module")
def hub_rows(covered_rows):
    """Fake Hub rows built from the committed CSV — the fixture trick, in three lines."""
    return _fake_hub_rows(covered_rows)


@pytest.fixture(scope="module")
def covered_reference(covered_rows):
    """The bytes a correct refresh must produce for those rows: the render's own writer, sorted."""
    return render.write_wide_csv(sorted(covered_rows, key=_key))


def _fake_hub_rows(rows):
    """Wide CSV rows as the Hub publishes them, by pushing them through the build's own projection.

    `published_projection` IS the published contract — `plankton` as a Python `bool`, `living`
    excluded, ids decimal-free, blanks as `None` — so these rows differ from real Hub rows only in
    that they carry no images. Hand-written fixtures would encode somebody's belief about the
    published schema; these encode the code that produces it, and go stale the moment it changes.
    """
    return [
        {"dataset": row["Dataset"], "original_label": row["Raw_Labels"], **values}
        for row, values in zip(rows, render.published_projection(rows), strict=True)
    ]


def _key(row):
    """The sort key a committed reference is written in, so its diffs read like a table."""
    return (row["Dataset"], row["Raw_Labels"])


def _keyed(data: bytes) -> dict:
    """`(Dataset, Raw_Labels) -> row`, parsed the same way `write.diff_published` parses it."""
    return {(row["Dataset"], row["Raw_Labels"]): row for row in csv.DictReader(io.StringIO(data.decode("utf-8")))}


def _rewrite(data: bytes, header, mutate) -> bytes:
    """Re-render reference bytes under a possibly different header, applying `mutate` per row.

    Used to build the deliberately-damaged references the refusal tests need — a dropped column, a
    twentieth column, a changed cell — without hand-writing 1,485 rows of CSV in a fixture.
    """
    rows = [dict(row) for row in csv.DictReader(io.StringIO(data.decode("utf-8")))]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for row in rows:
        mutate(row)
        writer.writerow([row.get(column, "") for column in header])
    return buffer.getvalue().encode("utf-8")


# ── manifest and reference plumbing for the report tests ───────────────────────────────────────
def _manifest_for(reference: bytes, published_datasets, **overrides) -> dict:
    """A step-18 manifest describing `reference`, with every key the report stage pre-flights.

    Built rather than committed as a fixture for the same reason the rows are: a stale fixture
    manifest would keep passing after the schema it describes had moved.
    """
    csv_datasets = {row["Dataset"] for row in loader.load_taxonomy(loader.PACKAGE_DIR).rows()}
    now = datetime.now(UTC).isoformat()
    manifest = {
        "repo_id": _REPO_ID,
        "revision": _FROZEN_SHA,
        "revision_requested": _FROZEN_SHA,
        "hub_last_modified": now,
        "refreshed_at": now,
        "complete": True,
        "shards": {"expected": 189, "read": 189, "failed": []},
        "schema_fingerprint": hashlib.sha256(b"schema").hexdigest(),
        "rows_scanned": 17_402_900,
        "pairs": len(_keyed(reference)),
        "csv_sha256": hashlib.sha256(reference).hexdigest(),
        "taxonomy_csv_sha256": hashlib.sha256(_TAXONOMY_CSV.read_bytes()).hexdigest(),
        "covered_columns": list(render.PUBLISHED_TAXONOMY_COLUMNS),
        "synthesised_columns": ["living"],
        "excluded_datasets": sorted(csv_datasets - published_datasets),
        "absent_pairs": [],
        "disagreements": [],
    }
    manifest.update(overrides)
    return manifest


def _write_pair(tmp_path, reference: bytes, published_datasets, **overrides):
    """Write a reference/manifest pair into `tmp_path` and return both paths."""
    reference_path = tmp_path / "golden_reference.csv"
    manifest_path = tmp_path / "golden_reference.json"
    reference_path.write_bytes(reference)
    manifest = _manifest_for(reference, published_datasets, **overrides)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return reference_path, manifest_path


def _waivers(tmp_path, *finding_ids):
    """A waiver file in `verify_label_consistency`'s shape, which the harness reuses verbatim."""
    path = tmp_path / "GOLDEN_DIFF_WAIVERS.json"
    payload = {"waivers": [{"finding_id": finding_id, "reason": "adjudicated in a test"} for finding_id in finding_ids]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _unexpected(offenders, allowed):
    """The offending `function -> calls` pairs, sorted, for a failure message worth reading."""
    return sorted((name, sorted(calls)) for name, calls in offenders.items() if name not in allowed)


def _require_committed_artefacts():
    """Skip, loudly and by name, until `--refresh` has been run once against the live Hub.

    Not `xfail` and not a silent pass: these two tests are the ones CI will actually run, so the
    reason a green suite is not yet checking them has to be legible in the test output.
    """
    for path in (_COMMITTED_REFERENCE, _COMMITTED_MANIFEST):
        if not path.exists():
            pytest.skip(
                f"{path.relative_to(root)} does not exist yet — generate it with "
                f"`uv run pz_golden_diff --refresh --revision <sha>` and commit the pair."
            )


# ── fake Hub shards for the refresh tests ──────────────────────────────────────────────────────
def _hub_schema(*, plankton_type=None, drop=()):
    """The published Hub schema over the 18 covered columns: `plankton` bool, the other 17 string."""
    plankton_type = plankton_type or pa.bool_()
    fields = [
        pa.field(column, plankton_type if column == "plankton" else pa.string())
        for column in _HUB_COLUMNS
        if column not in drop
    ]
    return pa.schema(fields)


def _hub_table(rows, schema=None):
    """An Arrow table of fake Hub rows under `schema`, as `read_shard` is expected to return one."""
    schema = schema or _hub_schema()
    return pa.table({field.name: [row.get(field.name) for row in rows] for field in schema}, schema=schema)


def _hub_row(dataset, original_label, **overrides):
    """One synthetic Hub row: every covered column present, nulls where the CSV would be blank."""
    row = dict.fromkeys(render.PUBLISHED_TAXONOMY_COLUMNS)
    row.update({"dataset": dataset, "original_label": original_label, "plankton": True, "root_class": "living"})
    row.update(overrides)
    return row


class _FakeHub:
    """A scripted Hub: a fixed sha, a schema per shard path, and a table or exception per read.

    Installed over the module's three networked seams rather than over `HfApi`/`HfFileSystem`, so
    these tests exercise the harness's own reduce, resolve, invert and write steps and only stub
    the bytes arriving from the network. Counting reads per shard is the point of half of them:
    "retried twice then succeeded" and "counted exactly once" are the same claim seen from two
    sides.
    """

    def __init__(self, tables, schemas=None, sha=_FROZEN_SHA):
        self.tables = dict(tables)
        self.schemas = dict(schemas) if schemas is not None else {path: _hub_schema() for path in self.tables}
        self.sha = sha
        self.reads = Counter()
        self.schema_reads = Counter()

    def dataset_info(self, repo_id, revision=None, timeout=None, **kwargs):
        return SimpleNamespace(id=repo_id, sha=self.sha, last_modified=datetime.now(UTC))

    def list_shards(self, api, repo_id, sha):
        assert sha == self.sha, f"the refresh listed {sha!r}, not the sha it resolved"
        return sorted(self.tables)

    def read_shard_schema(self, fs, path):
        self.schema_reads[path] += 1
        return self.schemas[path]

    def read_shard(self, fs, path, columns=None):
        self.reads[path] += 1
        entry = self.tables[path]
        if callable(entry):
            entry = entry(self.reads[path])
        if isinstance(entry, Exception):
            raise entry
        return entry

    def install(self, monkeypatch, module):
        monkeypatch.setattr(module, "list_shards", self.list_shards)
        monkeypatch.setattr(module, "read_shard_schema", self.read_shard_schema)
        monkeypatch.setattr(module, "read_shard", self.read_shard)


def _refresh(gd, hub, tmp_path, **kwargs):
    """Drive one refresh against a `_FakeHub`, serially, into `tmp_path`.

    `workers=1` is not incidental: the real refresh fans out over a `ProcessPoolExecutor`, and a
    monkeypatched `read_shard` cannot cross a process boundary. An in-process serial path at
    `workers=1` is therefore part of the contract, not a convenience.
    """
    return gd.refresh(
        _REPO_ID,
        _FROZEN_SHA,
        workers=1,
        reference_path=tmp_path / "golden_reference.csv",
        manifest_path=tmp_path / "golden_reference.json",
        api=hub,
        fs=object(),
        **kwargs,
    )


# ── step 14: the inversion ─────────────────────────────────────────────────────────────────────
def test_the_dot_zero_inversion_is_exact_for_every_committed_id(gd, csv_rows):
    """`f"{hub_value}.0"` reproduces all 5,733 committed numeric id cells, exactly.

    The forward direction IS a float round trip — `render._as_decimal_free_string` is
    `str(int(float(value)))` — so appending `.0` inverts it only while every committed id is a
    canonical integer string. That is a property of today's data, not of the format, which is why
    it is measured here over the whole table rather than assumed, and refused at refresh time by
    the test below.
    """
    seen = 0
    for row in csv_rows:
        for column in constants.ID_NUM_COLS:
            cell = row[column]
            if not cell:
                continue
            seen += 1
            hub_value = render._as_decimal_free_string(cell)
            assert gd.hub_cell_to_csv(column, hub_value) == cell, f"{column} {hub_value!r} does not invert to {cell!r}"
    assert seen == _NUMERIC_ID_CELLS, f"the numeric id population moved: {seen} cells, not {_NUMERIC_ID_CELLS}"


def test_a_non_canonical_hub_id_is_refused(gd):
    """A Hub id that is not a canonical integer string stops the refresh instead of round-tripping.

    `"0123"`, `"12.5"` and `" 12"` all survive `str(int(float(v)))` as something OTHER than
    themselves, so appending `.0` would publish an id the Hub does not carry. The harness cannot
    know which side is right, so it refuses with code 2 — "will not vouch" — rather than guessing.
    """
    for bad in ("0123", "12.5", " 12", "1e3"):
        with pytest.raises(gd.GoldenDiffError) as caught:
            gd.hub_cell_to_csv("aphia_ID", bad)
        assert caught.value.exit_code == 2, "a non-canonical id is unreadable data, not a data difference"
        assert bad.strip() in str(caught.value), f"the refusal must name the offending value {bad!r}"

    assert gd.hub_cell_to_csv("aphia_ID", "104464") == "104464.0"
    assert gd.hub_cell_to_csv("aphia_ID", None) == ""


def test_plankton_is_python_str_bool_not_arrow_lowercase(gd, hub_rows):
    """`plankton` renders as `True`/`False`, never Arrow's `true`/`false`.

    The column is a real Arrow `bool` on the Hub and a `True`/`False` string in the CSV. A
    `cast(Utf8)` — the obvious way to get a string out of an Arrow boolean — yields lowercase and
    would report all 4,716 published plankton cells as changed, which is the single loudest false
    positive this harness could produce on its first run.
    """
    assert gd.hub_cell_to_csv("plankton", True) == "True"
    assert gd.hub_cell_to_csv("plankton", False) == "False"

    emitted = {gd.hub_row_to_csv_row(row)["plankton"] for row in hub_rows}
    assert emitted <= {"True", "False"}, f"lowercase leaked into the plankton column: {sorted(emitted)}"


def test_ecotaxa_id_never_gains_a_decimal_suffix(gd, csv_rows):
    """`ecotaxa_ID` inverts verbatim, because it is multi-valued and not one of the numeric ids.

    `MULTI_VALUED_COLUMNS` routes it through the `";".join` branch BEFORE the suffix branch, so it
    is already a string of possibly several ids. Treating it as numeric would append `.0` to 1,523
    cells and make nonsense of the 598 that carry more than one id.
    """
    assert "ecotaxa_ID" not in constants.ID_NUM_COLS, "the routing this test pins has moved"

    assert gd.hub_cell_to_csv("ecotaxa_ID", "84963") == "84963"
    assert gd.hub_cell_to_csv("ecotaxa_ID", "84963;84964") == "84963;84964"
    assert gd.hub_cell_to_csv("ecotaxa_ID", None) == ""

    seen = 0
    for row in csv_rows:
        cell = row["ecotaxa_ID"]
        if not cell:
            continue
        seen += 1
        assert gd.hub_cell_to_csv("ecotaxa_ID", cell) == cell
    assert seen == _ECOTAXA_CELLS, f"the ecotaxa population moved: {seen} cells, not {_ECOTAXA_CELLS}"


def test_keys_are_renamed_but_never_stripped_or_case_folded(gd, csv_rows, hub_rows, covered_rows):
    """`dataset`/`original_label` become `Dataset`/`Raw_Labels` and are otherwise untouched.

    Nothing strips or case-folds on either side of this boundary — not the writer, not the reader.
    1,842 of the committed `Raw_Labels` are not lowercase, so a `.lower()` borrowed from
    `sankey.scan_dataset` would report 3,684 spurious row changes: every one of those pairs would
    read as a removal plus an addition rather than as a match.
    """
    odd = _hub_row("zooscan", "  Calanus Finmarchicus  ")
    inverted = gd.hub_row_to_csv_row(odd)
    assert inverted["Dataset"] == "zooscan"
    assert inverted["Raw_Labels"] == "  Calanus Finmarchicus  "

    keys = {(row["Dataset"], row["Raw_Labels"]) for row in map(gd.hub_row_to_csv_row, hub_rows)}
    assert keys == {_key(row) for row in covered_rows}

    mixed = sum(1 for row in csv_rows if row["Raw_Labels"] != row["Raw_Labels"].lower())
    assert mixed == _NON_LOWERCASE_LABELS, f"the mixed-case population moved: {mixed}, not {_NON_LOWERCASE_LABELS}"
    covered_mixed = sum(1 for row in covered_rows if row["Raw_Labels"] != row["Raw_Labels"].lower())
    assert sum(1 for key in keys if key[1] != key[1].lower()) == covered_mixed


def test_living_is_synthesised_from_root_class_exactly_as_the_renderer_derives_it(gd, csv_rows, hub_rows, covered_rows):
    """`living` is `root_class == "living"`, reproduced from the renderer and not from the vocab file.

    `living` is never published, so it has to be synthesised, and there are two candidate sources:
    `render.py`'s one-line derivation and `data/vocab/root_class.tsv`. The renderer is the right
    one — the vocab table is loaded and read by nothing — and reproducing the wrong one would
    silently disagree with the build for any root class the two spell differently.
    """
    for root_class, expected in (("living", "True"), ("detritus", "False"), ("artefact", "False"), ("inert", "False")):
        row = _hub_row("zooscan", f"probe_{root_class}", root_class=root_class)
        assert gd.hub_row_to_csv_row(row)["living"] == expected

    observed = Counter(row["root_class"] for row in csv_rows)
    assert observed == {"living": 2013, "detritus": 165, "artefact": 158, "inert": 22}
    assert "living" not in render.PUBLISHED_TAXONOMY_COLUMNS, "living is published after all; the synthesis is wrong"

    committed = {_key(row): row["living"] for row in covered_rows}
    for inverted in map(gd.hub_row_to_csv_row, hub_rows):
        assert inverted["living"] == committed[_key(inverted)]


# ── step 15: the refresh stage ─────────────────────────────────────────────────────────────────
def test_a_shard_that_never_reads_fails_the_refresh_and_writes_no_reference(gd, monkeypatch, tmp_path):
    """A shard that fails all four attempts aborts the whole refresh with code 2 and writes nothing.

    The alternative — write what was read and mark it partial — is how a reference silently becomes
    a subset of the Hub, and every pair the failed shard carried then reads as "the CSV has a row
    the Hub does not". That is a data difference the data does not have. Code 2, not 1: a flaky
    network must never be able to produce a data-difference verdict.
    """
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    good = "data/train-00000-of-00002.parquet"
    bad = "data/train-00001-of-00002.parquet"
    hub = _FakeHub({good: _hub_table([_hub_row("zooscan", "calanus")]), bad: RuntimeError("503 from the CDN")})
    hub.install(monkeypatch, gd)

    with pytest.raises(gd.GoldenDiffError) as caught:
        _refresh(gd, hub, tmp_path)

    assert caught.value.exit_code == 2
    assert bad in str(caught.value), "the refusal must name the shard PATH; a rank is not actionable"
    assert hub.reads[bad] == 4, "four attempts, then loud failure"
    assert not (tmp_path / "golden_reference.csv").exists(), "a partial reference was written anyway"
    assert not (tmp_path / "golden_reference.json").exists()


def test_a_shard_that_fails_twice_then_succeeds_is_counted_exactly_once(gd, monkeypatch, tmp_path):
    """A retried shard contributes its rows once, not once per attempt.

    The retry `sankey.scan_dataset` has never had a test for: a per-shard retry that accumulates
    into a shared counter rather than returning a fresh one double-counts every row of every shard
    that ever blipped, and the resulting reference is wrong in a way no schema check can see.
    """
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    path = "data/train-00000-of-00001.parquet"
    table = _hub_table([_hub_row("zooscan", "calanus"), _hub_row("zooscan", "oithona")])

    def flaky(attempt):
        return table if attempt > 2 else RuntimeError(f"transient {attempt}")

    hub = _FakeHub({path: flaky})
    hub.install(monkeypatch, gd)
    _refresh(gd, hub, tmp_path)

    assert hub.reads[path] == 3
    assert sleeps == [2, 4], "the backoff is sleep(2 * (k + 1)), copied from sankey.py"
    manifest = json.loads((tmp_path / "golden_reference.json").read_text(encoding="utf-8"))
    assert manifest["rows_scanned"] == 2, "the two failed attempts were counted as rows"
    assert manifest["pairs"] == 2
    assert len(_keyed((tmp_path / "golden_reference.csv").read_bytes())) == 2


def test_a_missing_published_column_is_a_refusal_naming_the_column(gd, monkeypatch, tmp_path):
    """A shard whose footer lacks a covered column stops the refresh BEFORE any data is read.

    Never a null-fill: a missing `qualifier` filled with nulls inverts to 1,482 empty cells and
    reports every non-blank qualifier in the table as deleted. The gate runs over all footers first
    precisely so this costs one metadata read rather than a full sweep.
    """
    path = "data/train-00000-of-00001.parquet"
    hub = _FakeHub({path: _hub_table([_hub_row("zooscan", "calanus")])}, schemas={path: _hub_schema(drop=("qualifier",))})
    hub.install(monkeypatch, gd)

    with pytest.raises(gd.GoldenDiffError) as caught:
        _refresh(gd, hub, tmp_path)

    message = str(caught.value)
    assert caught.value.exit_code == 2
    assert "qualifier" in message and path in message, f"the refusal names neither the column nor the shard: {message}"
    assert hub.reads == Counter(), "the schema gate must refuse before the first data read"
    assert not (tmp_path / "golden_reference.csv").exists()


def test_a_plankton_column_that_is_not_bool_is_a_refusal(gd, monkeypatch, tmp_path):
    """A `plankton` that arrives as string is a refusal, never a coercion.

    The inversion for this column depends entirely on it being an Arrow boolean: given strings, the
    harness cannot tell a published `"True"` from a published `"true"`, and either guess writes a
    reference that disagrees with the Hub it claims to describe.
    """
    path = "data/train-00000-of-00001.parquet"
    schema = _hub_schema(plankton_type=pa.string())
    hub = _FakeHub({path: _hub_table([_hub_row("zooscan", "calanus", plankton="True")], schema)}, schemas={path: schema})
    hub.install(monkeypatch, gd)

    with pytest.raises(gd.GoldenDiffError) as caught:
        _refresh(gd, hub, tmp_path)

    message = str(caught.value)
    assert caught.value.exit_code == 2
    assert "plankton" in message and "bool" in message.lower()
    assert not (tmp_path / "golden_reference.csv").exists()


def test_schema_drift_between_shards_is_caught_not_just_shard_zero(gd, monkeypatch, tmp_path):
    """The gate reads EVERY footer, so a column that vanishes at shard 3 is still caught.

    A build that appends shards written by a newer pipeline drifts at the tail, not at the head.
    Checking shard zero and trusting the rest is the cheap version of this gate and it would pass
    on exactly the corpus it exists to catch.
    """
    paths = [f"data/train-{index:05d}-of-00004.parquet" for index in range(4)]
    tables = {path: _hub_table([_hub_row("zooscan", f"label_{index}")]) for index, path in enumerate(paths)}
    schemas = dict.fromkeys(paths, _hub_schema())
    schemas[paths[3]] = _hub_schema(drop=("BOLD_ID",))
    hub = _FakeHub(tables, schemas=schemas)
    hub.install(monkeypatch, gd)

    with pytest.raises(gd.GoldenDiffError) as caught:
        _refresh(gd, hub, tmp_path)

    message = str(caught.value)
    assert caught.value.exit_code == 2
    assert "BOLD_ID" in message and paths[3] in message
    assert hub.schema_reads[paths[3]] == 1, "the last shard's footer was never read"


# ── step 16: pairs whose published rows disagree ───────────────────────────────────────────────
def test_a_pair_whose_rows_disagree_is_a_finding_and_a_sentinel_not_a_guess(gd, monkeypatch, tmp_path):
    """Two published values for one cell produce `«DISAGREE:…»` and a finding — never a winner.

    Four places in this tree collapse a duplicate key silently: first-wins in `sankey`, last-wins in
    `render` and in `write.diff_published`'s own `keyed()`, and one-representative-row in
    `frepj_validate`. The committed CSV has zero duplicate keys, so a split key here is a defect in
    the PUBLISHED rows — build history no in-tree test can see — and picking a side would hide it.
    The disagreement is per CELL, so the other fifteen columns' evidence survives.
    """
    first = "data/train-00000-of-00002.parquet"
    second = "data/train-00001-of-00002.parquet"
    agreeing = {"Genus": "Calanus", "qualifier": "full_body"}
    hub = _FakeHub(
        {
            first: _hub_table([_hub_row("zooscan", "Calanus", proposed_label="calanus", **agreeing)] * 3),
            second: _hub_table([_hub_row("zooscan", "Calanus", proposed_label="Calanus", **agreeing)]),
        }
    )
    hub.install(monkeypatch, gd)
    _refresh(gd, hub, tmp_path)

    row = _keyed((tmp_path / "golden_reference.csv").read_bytes())[("zooscan", "Calanus")]
    assert row["proposed_label"].startswith(_SENTINEL_PREFIX), f"the harness picked a value: {row['proposed_label']!r}"
    assert row["proposed_label"].endswith("»")
    assert row["Genus"] == "Calanus" and row["qualifier"] == "full_body", "agreeing cells were withheld too"

    manifest = json.loads((tmp_path / "golden_reference.json").read_text(encoding="utf-8"))
    (finding,) = manifest["disagreements"]
    assert finding["finding_id"] in row["proposed_label"]
    assert (finding["dataset"], finding["original_label"], finding["column"]) == ("zooscan", "Calanus", "proposed_label")
    assert {(value, n_rows) for value, n_rows, _shards in finding["values"]} == {("calanus", 3), ("Calanus", 1)}
    assert {shard for _value, _n, shards in finding["values"] for shard in shards} == {first, second}


def test_a_disagreement_finding_id_survives_a_new_shard_but_not_a_new_value(gd):
    """The id hashes the CLAIM — pair, column, values — and not the evidence behind it.

    A waiver has to survive a re-refresh. Row counts and shard paths move whenever the corpus is
    repartitioned or grows, so hashing them would invalidate every adjudication on the next sweep
    and train reviewers to re-waive without reading. A genuinely NEW value is a new claim and must
    arrive unwaived.
    """
    claim = ("zooscan", "Calanus", "proposed_label")
    original = gd.CellDisagreement(*claim, (("calanus", 17_402_900, ("data/a.parquet",)), ("Calanus", 3, ("data/b.parquet",))))
    repartitioned = gd.CellDisagreement(
        *claim, (("calanus", 18_000_000, ("data/a.parquet", "data/c.parquet")), ("Calanus", 4, ("data/b.parquet",)))
    )
    new_value = gd.CellDisagreement(
        *claim, (("calanus", 17_402_900, ("data/a.parquet",)), ("Calanoida", 3, ("data/b.parquet",)))
    )

    assert original.finding_id == repartitioned.finding_id
    assert original.finding_id != new_value.finding_id
    assert len(original.finding_id) == 12, "the house finding_id is sha256 over the claim, truncated to 12"


def test_a_waived_disagreement_exits_zero_and_a_stale_waiver_exits_nonzero(
    gd, tmp_path, covered_rows, covered_reference, published_datasets
):
    """An adjudicated disagreement stops being a failure; a waiver matching nothing becomes one.

    Both halves matter. Without the first, the gate cannot be green while a real published
    disagreement is being worked on and somebody waives the whole check. Without the second, a
    waiver outlives the defect it excused and quietly covers the next one.
    """
    finding_id = "a1b2c3d4e5f6"
    sentinel = f"{_SENTINEL_PREFIX}{finding_id}»"
    target = _key(min(covered_rows, key=_key))

    def mutate(row):
        if (row["Dataset"], row["Raw_Labels"]) == target:
            row["proposed_label"] = sentinel

    damaged = _rewrite(covered_reference, LEGACY_HEADER, mutate)
    disagreement = {
        "dataset": target[0],
        "original_label": target[1],
        "column": "proposed_label",
        "finding_id": finding_id,
        "values": [["calanus", 3, ["data/a.parquet"]], ["Calanus", 1, ["data/b.parquet"]]],
    }
    reference_path, manifest_path = _write_pair(tmp_path, damaged, published_datasets, disagreements=[disagreement])

    unwaived = gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json")
    assert unwaived == 3, "an unwaived disagreement is unresolvable (3), not a data difference (1)"

    waived = gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, _waivers(tmp_path, finding_id))
    assert waived == 0

    stale = gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, _waivers(tmp_path, finding_id, "deadbeef0000"))
    assert stale != 0, "a waiver matching no finding must be deleted with the change that fixed it"


# ── step 18: the manifest's coverage rule ──────────────────────────────────────────────────────
def test_every_uncovered_csv_row_is_accounted_for_by_the_manifest(csv_rows):
    """Every committed row the reference does not cover is named by one of the two excuse lists.

    This is what keeps "subtract the rows the Hub does not have" from degenerating into "subtract
    whatever disagrees". The two lists are different kinds of gap and must not be merged: 873 rows
    in six sources that have not been published yet, and 3 planktoscope rows whose source IS
    published but which carry no image — the second is a real per-row finding, the first is not.
    """
    _require_committed_artefacts()
    manifest = json.loads(_COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    reference = _keyed(_COMMITTED_REFERENCE.read_bytes())

    excluded = set(manifest["excluded_datasets"])
    absent = {tuple(pair) for pair in manifest["absent_pairs"]}
    uncovered = [_key(row) for row in csv_rows if _key(row) not in reference]

    orphans = [pair for pair in uncovered if pair[0] not in excluded and pair not in absent]
    assert not orphans, f"uncovered row(s) in neither excuse list — the reference is wrong, not the data: {orphans[:5]}"

    by_dataset = sum(1 for pair in uncovered if pair[0] in excluded)
    assert by_dataset + len(absent) == len(uncovered) == _CSV_ROWS - len(reference)
    assert len(reference) == len(csv_rows) - _EXCLUDED_ROWS - len(absent)


# ── step 19 + 21: the report stage ─────────────────────────────────────────────────────────────
def test_the_reference_writer_reproduces_render_bytes_for_the_rows_it_covers(gd, hub_rows, covered_rows, covered_reference):
    """THE KEYSTONE. Inverting the published rows reproduces the committed CSV, byte for byte.

    One assertion pins the entire step-14 inversion table at once: the key rename without
    case-folding, `None` to empty string, `plankton` as a Python `str(bool)`, the `.0` suffix on
    exactly three id columns and not on `ecotaxa_ID`, `living` synthesised from `root_class`, the
    column ORDER (three different id orders exist in this tree and conflating two swaps id values
    between columns), and the writer itself. Anything wrong in any of them changes these bytes.

    The input is shuffled on purpose: a reference is sorted by `(Dataset, Raw_Labels)` so a
    committed-file diff is legible, and shard order on the Hub is not that order.
    """
    shuffled = list(hub_rows)
    random.Random(20260918).shuffle(shuffled)

    produced = gd.reference_bytes(shuffled)

    assert produced == covered_reference
    assert len(_keyed(produced)) == _COVERED_ROWS == len(covered_rows)
    assert produced.split(b"\n", 1)[0] == ",".join(LEGACY_HEADER).encode("utf-8")

    # Not merely "equals a re-render": every emitted data line is a line of the frozen table.
    committed = set(_TAXONOMY_CSV.read_bytes().split(b"\n"))
    assert set(produced.split(b"\n")) <= committed


def test_a_reference_missing_a_column_raises_rather_than_reporting(gd, tmp_path, covered_reference, published_datasets):
    """An 18-column reference raises `KeyError`, which is why the harness emits all nineteen.

    `write.diff_published` subscripts the reference row with every column of the RENDERED row, so a
    reference missing one cannot be diffed at all. The register's "then compare the other
    seventeen" fallback is therefore unreachable, and the harness's job is to emit `living` rather
    than to hope the differ tolerates its absence.
    """
    header = tuple(column for column in LEGACY_HEADER if column != "living")
    reference_path, manifest_path = _write_pair(
        tmp_path, _rewrite(covered_reference, header, lambda row: None), published_datasets
    )

    with pytest.raises(KeyError) as caught:
        gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json")
    assert "living" in str(caught.value)


def test_extra_columns_in_the_reference_are_free(gd, tmp_path, covered_reference, published_datasets):
    """A twentieth column in the reference changes nothing, because the differ iterates the render.

    Worth pinning because it is the asymmetry that makes the harness forward-compatible: the day
    `instrument` joins the published columns, an older reference carrying it does not go red, while
    a reference MISSING a rendered column does. Symmetric tolerance would have hidden the case
    above.
    """
    header = (*LEGACY_HEADER, "instrument")
    extra = _rewrite(covered_reference, header, lambda row: row.update({"instrument": "zooscan"}))
    reference_path, manifest_path = _write_pair(tmp_path, extra, published_datasets)

    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 0


def test_a_reconstructed_reference_reports_green_offline(gd, tmp_path, covered_reference, published_datasets):
    """The whole report stage, end to end and network-free, over a reference that should be green.

    The offline twin of the committed-artefact test below: it exercises the same pre-flights,
    the same `diff_published` call and the same coverage subtraction, but on a reference this file
    builds, so it goes red the day the pipeline breaks rather than the day somebody regenerates the
    committed pair.
    """
    reference_path, manifest_path = _write_pair(tmp_path, covered_reference, published_datasets)
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 0


def test_a_missing_reference_or_manifest_is_a_refusal_naming_the_file(gd, tmp_path, covered_reference, published_datasets):
    """A missing half of the pair is code 2 and says which file to generate.

    "Could not read" is not "the data differs", and a gate that returns 1 when its own reference is
    absent teaches reviewers that 1 sometimes means "not set up yet".
    """
    reference_path, manifest_path = _write_pair(tmp_path, covered_reference, published_datasets)

    manifest_path.unlink()
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2

    manifest_path.write_text(json.dumps(_manifest_for(covered_reference, published_datasets)), encoding="utf-8")
    reference_path.unlink()
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2


def test_a_reference_that_disagrees_with_its_manifest_sha_is_a_refusal(gd, tmp_path, covered_reference, published_datasets):
    """The pair was edited apart, so neither file describes the other: code 2.

    The manifest's `csv_sha256` is what makes the two one artifact. Without this check, hand-editing
    a cell out of the reference is indistinguishable from the Hub having published that cell, and
    the gate becomes a way to launder a change past review.
    """
    reference_path, manifest_path = _write_pair(tmp_path, covered_reference, published_datasets)
    reference_path.write_bytes(covered_reference + b"zooscan,edited-in-by-hand" + b"," * 17 + b"\n")

    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2


def test_an_incomplete_refresh_cannot_be_reported_green(gd, tmp_path, covered_reference, published_datasets):
    """`complete: false`, or any failed shard, refuses rather than reporting on a partial sweep.

    A `--shards` run is a legitimate debugging tool and its output is a legitimate file; what it is
    not is evidence about the corpus. The flag in the manifest is how the report stage tells the two
    apart, and both spellings of "partial" have to refuse.
    """
    reference_path, manifest_path = _write_pair(tmp_path, covered_reference, published_datasets, complete=False)
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2

    shards = {"expected": 189, "read": 188, "failed": ["data/train-00042-of-00189.parquet"]}
    reference_path, manifest_path = _write_pair(tmp_path, covered_reference, published_datasets, shards=shards)
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2


def test_a_reference_predating_the_schema_gate_is_a_refusal(gd, tmp_path, covered_reference, published_datasets):
    """A manifest with no `schema_fingerprint` was written before the gate existed: code 2.

    Such a reference may be perfectly correct, and that is exactly the problem — nothing recorded
    which schema it was read under, so a green report over it vouches for something nobody checked.
    """
    manifest = _manifest_for(covered_reference, published_datasets)
    del manifest["schema_fingerprint"]
    reference_path = tmp_path / "golden_reference.csv"
    manifest_path = tmp_path / "golden_reference.json"
    reference_path.write_bytes(covered_reference)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2


def test_a_changed_published_cell_is_a_data_difference(gd, tmp_path, covered_rows, covered_reference, published_datasets):
    """A published cell that disagrees with the render is code 1 — the gate's whole reason to exist."""
    target = _key(min(covered_rows, key=_key))

    def mutate(row):
        if (row["Dataset"], row["Raw_Labels"]) == target:
            row["Genus"] = "NotTheGenusTheBuildRenders"

    reference_path, manifest_path = _write_pair(
        tmp_path, _rewrite(covered_reference, LEGACY_HEADER, mutate), published_datasets
    )
    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 1


def test_a_hub_pair_the_csv_does_not_map_is_a_data_difference(gd, tmp_path, covered_reference, published_datasets):
    """A pair the Hub publishes and the CSV does not map is code 1, under its own heading.

    Measured at zero today, which is why it needs a test: a bucket that is empty on the day it ships
    is the one nobody notices has stopped working. It is a data difference, not a coverage gap —
    the published corpus carries a label this tree cannot explain.
    """
    orphan = {column: "" for column in LEGACY_HEADER}
    orphan.update({"Dataset": "zooscan", "Raw_Labels": "a_label_no_mapping_file_carries", "plankton": "False"})
    orphan["living"] = "False"
    reference = covered_reference + ",".join(orphan[column] for column in LEGACY_HEADER).encode("utf-8") + b"\n"
    reference_path, manifest_path = _write_pair(tmp_path, reference, published_datasets)

    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 1


def test_an_uncovered_row_in_neither_excuse_list_is_a_coverage_anomaly(
    gd, tmp_path, covered_rows, covered_reference, published_datasets
):
    """A committed row the reference skips, with no excuse recorded, is code 2 and not a difference.

    This is step 18's rule enforced from the report side. Dropping a whole published source from the
    reference makes 1,485 rows uncovered; if the harness were allowed to subtract them because they
    happen to be missing, the gate would excuse precisely the failure mode it exists to catch.
    """
    dropped = "zoolake"
    kept = [row for row in sorted(covered_rows, key=_key) if row["Dataset"] != dropped]
    assert len(kept) < len(covered_rows), f"{dropped} is no longer a published source; pick another"
    reference_path, manifest_path = _write_pair(tmp_path, render.write_wide_csv(kept), published_datasets)

    assert gd.report(reference_path, manifest_path, loader.PACKAGE_DIR, tmp_path / "absent.json") == 2


def test_the_committed_reference_agrees_with_the_committed_taxonomy_csv(gd):
    """THE TEST CI RUNS: `pz_golden_diff --report` over the committed pair, green, offline."""
    _require_committed_artefacts()
    assert gd.main(["--report"]) == 0


def test_the_report_stage_imports_nothing_that_speaks_http():
    """Offline by construction, proven from the import graph rather than asserted.

    Importing the module must not pull in `requests`, `urllib.request`, `http.client` or `aiohttp`,
    which means `huggingface_hub`, `HfFileSystem` and `datasets` are imported inside the refresh
    functions and nowhere else. This is the house pattern (`verify_label_consistency` carries the
    same probe) and it is what lets `--report` run on every PR with no token and no egress: an
    assertion that the report "does not use the network" is a comment, while this fails.
    """
    probe = (
        "import sys;"
        "import planktonzilla.planktonzilla_dataset.utils.golden_diff;"
        "print([m for m in ('requests', 'urllib.request', 'http.client', 'aiohttp') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=root)
    assert result.stdout.strip() == "[]", f"HTTP machinery reached the import graph: {result.stdout}"


def test_refresh_is_the_only_entry_point_that_lists_or_opens_hub_files(gd):
    """Every Hub call sits inside one of the four functions allowed to make one.

    The import probe above proves nothing is imported; this proves nothing is CALLED outside the
    refresh path. Together they are why `report` can be trusted offline. A new networked helper is
    not forbidden — it has to be named here, which is the review step.
    """
    allowed = {"refresh", "list_shards", "read_shard", "read_shard_schema", "check_revision"}
    hub_calls = {"HfApi", "HfFileSystem", "list_repo_files", "dataset_info", "ParquetFile", "read_schema", "load_dataset"}

    tree = ast.parse(Path(gd.__file__).read_text(encoding="utf-8"))
    owner = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                owner[id(child)] = node.name

    offenders = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in hub_calls:
            offenders.setdefault(owner.get(id(node), "<module level>"), set()).add(name)

    assert "report" not in offenders, f"the report stage reaches the Hub: {sorted(offenders.get('report', ()))}"
    assert set(offenders) <= allowed, f"Hub call(s) outside the refresh path: {_unexpected(offenders, allowed)}"
