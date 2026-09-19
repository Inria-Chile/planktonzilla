"""
(c) Inria

golden_diff.py
==============
The GOLDEN gate: does ``planktonzilla_taxonomy.csv`` still describe the rows that are actually
published in ``project-oceania/planktonzilla-17M``?

Why this exists
---------------
``verify_taxonomy_ids.py`` checks the table against external authorities. ``verify_label_consistency.py``
checks the table against itself. Neither of them ever looks at the artifact the table is supposed
to describe. The published dataset is frozen, the table is not, and nothing in the tree compares
the two — so an edit to the CSV that contradicts 17.4 million already-published cells lands green.
That is the gap ``docs/CODE_REVIEW.md:53`` ("does the change move a byte already published on the
Hub?") asks every reviewer to close by hand, on every diff, from memory.

This harness closes it mechanically, in two stages that never run together:

    refresh   the NETWORKED stage. Reads every published parquet shard's taxonomy columns,
              reduces them to one row per ``(dataset, original_label)`` pair, inverts the
              published values back into the CSV's spelling, and writes a reference CSV plus a
              manifest recording exactly what was read and what it could not see.

    report    the OFFLINE stage, and the one CI runs. Diffs the committed reference against a
              fresh render of the taxonomy package. Imports nothing that speaks HTTP, which
              ``tests/test_golden_diff.py`` proves with a ``sys.modules`` probe rather than
              asserting.

The harness only ever READS the Hub. Nothing here can move a published byte.

What it cannot see, and says so
-------------------------------
The Hub carries 1,482 distinct pairs across 15 datasets; the CSV carries 2,358 rows across 21.
The 876-row difference is not slack to be absorbed — it is two named, counted gaps, recorded in
the manifest and re-proved on every report:

    excluded_datasets   873 rows in the six sources recorded ahead of their publication
                        (``constants.RECORDED_BUT_NOT_YET_PUBLISHED``). "Not published yet."
    absent_pairs        3 planktoscope rows in a PUBLISHED source that carry no published image.
                        A genuine per-row finding, adjudicated rather than absorbed.

**Any uncovered CSV row in neither list is a coverage anomaly and refuses the run.** Without that
rule, "subtract the rows the Hub does not have" degenerates into "subtract whatever disagrees",
and the gate silently stops being a gate.

The harness never picks a value
-------------------------------
Four places in this tree collapse a duplicate key silently — first-wins in ``sankey``, last-wins in
``render`` and in ``write.diff_published``'s own ``keyed()``, one-representative-row in
``frepj_validate``. The committed CSV has zero duplicate keys, so published rows that disagree
about a cell are a defect in the BUILD HISTORY, which is exactly what no in-tree test can see.
Such a cell is emitted as the sentinel ``«DISAGREE:<finding_id>»`` and is a FAILURE (exit 3), never
a guess and never an excused removal. Adjudicate it in ``GOLDEN_DIFF_WAIVERS.json``.

Exit codes
----------
    0   every cell this reference can see agrees
    1   THE DATA DIFFERS — a published cell disagrees with the render, or a Hub pair the CSV
        does not map
    2   COULD NOT READ / WILL NOT VOUCH — shard failure, missing column, type change,
        non-canonical id, ``csv_sha256`` mismatch, ``complete: false``, stale revision, a broken
        ``living`` derivation, or a coverage anomaly
    3   UNRESOLVABLE — an unwaived cell disagreement among published rows, or a stale waiver

**1 and 2 are never conflated, and a flaky network can only ever produce 2.** That is the single
property a reviewer needs before trusting a green run: code 1 means the table moved, and nothing
else does. Code 3 extends the house scheme deliberately (see DECISIONS item 1) — adjudicating a
waiver and fixing a cell are different work, and folding them together would hide which is needed.

Usage:
    # networked stage: re-harvest the published rows (~58 s, ~190 MB, 16 processes)
    pz_golden_diff --refresh --revision 7fd542ef9fc99e6e850bed06bbd8884f43192347

    # offline stage: the gate. Zero network, and what CI runs.
    pz_golden_diff --report

    # the same gate, plus one dataset_info call asking whether the Hub has moved under it
    pz_golden_diff --report --check-revision
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import argparse
import hashlib
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.taxonomy import loader, render, write
from planktonzilla.planktonzilla_dataset.taxonomy.model import LEGACY_HEADER
from planktonzilla.planktonzilla_dataset.utils.verify_label_consistency import apply_waivers, load_waivers
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# ── What is compared, and under which names ────────────────────────────────────────────────────

# The two Hub columns that carry the CSV's key, and the CSV columns they become. The rename is the
# WHOLE transformation: nothing strips, nothing case-folds, on either side. 1,842 of the 2,358
# `Raw_Labels` values are not lowercase, so a stray `.lower()` here would report 3,684 spurious
# row changes — which is precisely what `sankey.py:296-298` does to its own scan, and one of the
# three reasons that scan could not be reused.
HUB_KEY_COLUMNS = ("dataset", "original_label")
CSV_KEY_COLUMNS = ("Dataset", "Raw_Labels")
KEY_COLUMN_OF_HUB = dict(zip(HUB_KEY_COLUMNS, CSV_KEY_COLUMNS))

# The 18 Hub columns the schema gate insists on: the two key columns plus the 16 taxonomy columns
# the build actually writes. `render.PUBLISHED_TAXONOMY_COLUMNS` is the build's own WRITE list, not
# a transcription of it, so this READ list cannot drift from what is published. The 16 keep their
# CSV names on the Hub; only the key columns are renamed.
COVERED_HUB_COLUMNS = (*HUB_KEY_COLUMNS, *render.PUBLISHED_TAXONOMY_COLUMNS)

# `living` is emitted but never read: it is not published at all, and it is not independent — it IS
# `root_class == "living"` (render.py:148). It is synthesised so the reference carries all 19
# columns, because `write.diff_published` subscripts every column of the render and a 18-column
# reference raises `KeyError: 'living'` rather than reporting. It therefore cannot fail on its own,
# which the coverage report says out loud rather than letting a reader count it as checked.
SYNTHESISED_COLUMNS = ("living",)

# The taxonomy columns actually compared: the 19 emitted, minus the 2 key columns, minus `living`.
COMPARED_COLUMNS = tuple(c for c in LEGACY_HEADER if c not in CSV_KEY_COLUMNS and c not in SYNTHESISED_COLUMNS)

# ── Sentinels, paths and tunables ───────────────────────────────────────────────────────────────

# Guillemets match the house error style, and the closing one makes the sentinel greppable as a
# whole token. A cell holding this is NOT a value — it is the harness refusing to invent one.
DISAGREE_PREFIX = "«DISAGREE:"
DISAGREE_SUFFIX = "»"

DEFAULT_REPO_ID = constants.DEFAULT_PLANKTONZILLA_DATASET_REPO_ID
DEFAULT_REFERENCE_PATH = Path(__file__).parent / "golden_reference.csv"
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "golden_reference.json"
DEFAULT_WAIVERS_PATH = Path(__file__).parent / "GOLDEN_DIFF_WAIVERS.json"

# 53.9 MB per shard transfers with the naive fsspec block; 1.13 MB at 64 KiB, at no time penalty.
# Over a 189-shard sweep that is 10.2 GB against 214 MB, so this default is not a micro-optimisation
# — it is the difference between a gate someone runs and a gate someone stops running.
DEFAULT_BLOCK_SIZE = 65536

# PROCESSES, not threads. `HfFileSystem` funnels every read through one fsspec asyncio loop thread,
# so a thread pool tops out around 1.42x no matter how wide it is. `workers=1` runs in-process
# instead, which is how the offline tests drive `refresh` without a subprocess boundary standing
# between them and their fakes.
#
# Derived from the host rather than fixed at 16. Each worker materialises one shard's projected
# columns, so the width is bounded by MEMORY, not by how many reads the Hub will serve: a first
# live sweep at 16 on a 4-CPU host lost 72 of 189 shards to `BrokenProcessPool`, because one
# OOM-killed worker takes the whole pool — and the Hub answered a run that wide with 429s besides.
# Twice the CPU count keeps the pool fed while a worker waits on the network, and the cap keeps a
# large host from rediscovering the same wall.
DEFAULT_WORKERS = max(2, min(16, (os.cpu_count() or 2) * 2))

SHARD_READ_ATTEMPTS = 4
SHARD_RETRY_BACKOFF_SECONDS = 2.0

# Shard paths are evidence for a disagreement, not part of its identity, so recording every one of
# up to 189 per distinct value would cost tens of megabytes to say what eight already say. The
# finding_id does not depend on them, so this bound cannot destabilise a waiver.
MAX_RECORDED_SHARDS = 8

# An old reference of a FROZEN artifact is still correct, so age warns and never fails: making it
# fail would train people to pass --allow-stale-revision reflexively, which is how a real staleness
# refusal comes to be ignored.
STALE_AFTER_DAYS = 90

EXIT_OK = 0
EXIT_DIFFERS = 1
EXIT_REFUSED = 2
EXIT_UNRESOLVABLE = 3


class GoldenDiffError(RuntimeError):
    """A refusal: something could not be read, or would have to be guessed at.

    Every raise of this becomes exit 2 and never exit 1. That separation is the harness's one
    load-bearing promise — a reviewer reading a green CI run needs to know that a flaky S3 range
    request cannot have been rendered as "the taxonomy is fine", and that a red run at code 1 is
    about the data and nothing else.

    The code travels ON the exception rather than being re-derived at each catch site: two
    conversion sites is how one of these eventually comes back as a 1.
    """

    exit_code = EXIT_REFUSED


class ShardReadError(GoldenDiffError):
    """One shard that would not read after every attempt, carrying the path that failed.

    The path is kept as an attribute rather than only interpolated into the message because the
    caller reports every failed shard together: a sweep that loses four shards to one flaky mirror
    should say so once, not surface whichever one happened to be collected first.
    """

    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path


# ── Step 14: the inversion, from a published cell back to the CSV's spelling ───────────────────


def _is_sentinel(value) -> bool:
    """Whether a resolved cell is a disagreement marker rather than a published value."""
    return isinstance(value, str) and value.startswith(DISAGREE_PREFIX)


def _canonical_integer(column: str, value) -> str:
    """The value, unchanged, when it is a canonical integer string; a refusal when it is not.

    This is the guard that keeps the ``.0`` inversion honest. The forward direction IS a float
    round trip — ``render._as_decimal_free_string`` is literally ``str(int(float(value)))``
    (``render.py:182-195``) — so appending ``".0"`` reconstructs the CSV cell only while every id
    that reached the Hub was a canonical integer on the way out. ``"007"``, ``"1.5"`` and
    ``"1e3"`` all survive the forward trip and none of them comes back.

    Measured: 0 counterexamples across the 5,733 non-empty numeric id cells published today. That
    is a property of today's data, not of the code, which is why it is re-checked on every refresh
    instead of being assumed once and written down.
    """
    try:
        canonical = str(int(value))
    except (TypeError, ValueError):
        canonical = None
    if canonical != value:
        raise GoldenDiffError(
            f"published {column}={value!r} is not a canonical integer string, so the CSV's "
            f"'{value}.0' spelling cannot be reconstructed from it exactly. The build renders ids "
            f"through str(int(float(v))) (render.py:182-195), which this cell would not survive. "
            f"Adjudicate the published id; the harness will not guess at it."
        )
    return canonical


def hub_cell_to_csv(column: str, value) -> str:
    """One published cell, in the spelling the 19-column CSV would have written for it.

    The inversion table, and every trap in it, in one place. ``column`` is the CSV's name for the
    cell; the 16 taxonomy columns keep their names on the Hub, so only the caller's key columns are
    renamed. A disagreement sentinel passes through untouched, because it is not a value.

    The traps, each of which costs a specific number of false differences if got wrong:

    * ``plankton`` arrives as an Arrow ``bool``. ``cast(Utf8)`` would spell it ``true``/``false``
      and report 4,716 false diffs, so it goes through Python ``str()`` and a non-bool is refused
      rather than coerced.
    * ``aphia_ID``/``NCBI_ID``/``BOLD_ID`` are published decimal-free and the CSV stores them with
      a ``.0`` suffix, but only after the canonical-integer guard above has proved the suffix is
      the whole difference.
    * ``ecotaxa_ID`` is ALSO numeric-looking and must NOT gain a suffix: it is multi-valued, so
      ``render.py:81-82`` routes it through the ``";".join`` branch before the suffix branch ever
      runs. Treating it as numeric appends ``.0`` to 1,523 cells.
    * a null becomes ``""``, never ``"None"`` and never a dropped column.
    """
    if column not in COMPARED_COLUMNS:
        raise GoldenDiffError(
            f"{column!r} is not one of the {len(COMPARED_COLUMNS)} published taxonomy columns "
            f"{COMPARED_COLUMNS}; the key columns are renamed by the caller and 'living' is "
            f"synthesised from root_class, neither of which is invertible from a cell alone."
        )
    if _is_sentinel(value):
        return value
    if column == "plankton":
        if value is None:
            return ""
        if not isinstance(value, bool):
            raise GoldenDiffError(
                f"published plankton={value!r} is a {type(value).__name__}, not the Arrow bool the "
                f"schema gate requires. Rendering it as text here is how 'true' reaches a CSV that "
                f"spells it 'True'."
            )
        return str(value)
    if value is None:
        return ""
    if column in constants.ID_NUM_COLS:
        return f"{_canonical_integer(column, value)}.0"
    return value


def _key_cell(value) -> str:
    """A key cell, verbatim: a null becomes ``""`` and nothing else is touched.

    Deliberately not ``.strip()`` and deliberately not ``.lower()``. ``Raw_Labels`` legitimately
    preserves each source's own casing (KI-9), and ``write.diff_published`` keys on the exact
    bytes, so normalising here would not tidy the diff — it would split every pair whose label is
    not already lowercase into a removal plus an addition.
    """
    return "" if value is None else value


def _living_from(root_class: str) -> str:
    """``living``, reproduced exactly as the renderer derives it at ``render.py:148``.

    From ``root_class``, and NOT from ``data/vocab/root_class.tsv``: that vocabulary is loaded at
    ``loader.py:124-126`` and read by nothing, so deriving from it would pin a table the build does
    not consult. A ``root_class`` the published rows disagree about yields the SAME sentinel rather
    than ``"False"`` — "not the string 'living'" is true of a sentinel too, and answering it would
    be the harness quietly picking a value for a cell it just refused to pick one for.
    """
    if _is_sentinel(root_class):
        return root_class
    return "True" if root_class == "living" else "False"


def hub_row_to_csv_row(values) -> dict:
    """One published pair's cells as a full 19-column CSV row, in ``LEGACY_HEADER`` order.

    All 19 are emitted, not the 18 the Hub carries. ``write.diff_published`` subscripts
    ``new[key][column]`` for every column of the render (``write.py:891``), so a reference short of
    one raises ``KeyError: 'living'`` instead of reporting — which is why the KNOWN_ISSUES register's
    "compare the other seventeen" fallback is not reachable, and why the count here is 19.

    Columns are read out of ``values`` BY NAME. Three different id orders exist in this repository
    (``render.py:15-17``) and the CSV header's ``wikidata, aphia, NCBI, BOLD, ecotaxa`` is not
    ``CONSOLIDATED_COLUMNS``'s ``wikidata, ecotaxa, aphia, NCBI, BOLD``; a positional read would
    swap id values between columns and every row would still look well formed.
    """
    row = {csv_column: _key_cell(values.get(hub_column)) for hub_column, csv_column in KEY_COLUMN_OF_HUB.items()}
    for column in render.PUBLISHED_TAXONOMY_COLUMNS:
        row[column] = hub_cell_to_csv(column, values.get(column))
    row["living"] = _living_from(row["root_class"])
    return {column: row[column] for column in LEGACY_HEADER}


# ── Step 16: a pair whose published rows disagree ──────────────────────────────────────────────


def _claim_text(value) -> str:
    """One published value as the text that identifies it inside a claim."""
    return "" if value is None else str(value)


@dataclass(frozen=True)
class CellDisagreement:
    """One ``(dataset, original_label, column)`` cell that the published rows do not agree on.

    Per CELL and not per row: withholding the whole key would throw away the other 15 columns'
    evidence, and those columns are usually the ones that say which of the two variants is the
    build-history accident.

    Row counts travel with each value because remediation turns on them. "17,402,900 rows say
    ``Calanus`` and 3 say ``calanus``" is a typo with an obvious fix; a 50/50 split is a question
    about which build wrote which shard, and the two want different people.
    """

    dataset: str
    original_label: str
    column: str
    values: tuple[tuple[object, int, tuple[str, ...]], ...]
    recorded_id: str = ""

    @property
    def finding_id(self) -> str:
        """Stable 12-hex-char identity for waiver matching, keyed on the CLAIM.

        The claim is "these distinct values are published for this cell" — so the id survives a new
        shard appearing, a shard disappearing, and any change in row counts, and does NOT survive a
        new value joining the disagreement. That is the split a waiver needs: re-sharding the
        dataset must not invalidate an adjudication, and a third spelling arriving must.

        A finding read back from a manifest keeps the id the manifest recorded. The reference's
        sentinel cells cite that id, and the two files are pinned to each other by ``csv_sha256``,
        so recomputing it here would only let a future change to this hash silently orphan every
        sentinel in a reference that is otherwise provably intact.
        """
        if self.recorded_id:
            return self.recorded_id
        distinct = "\x1f".join(sorted(_claim_text(value) for value, _rows, _shards in self.values))
        payload = "|".join((self.dataset, self.original_label, self.column, distinct))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def sentinel(self) -> str:
        """The cell the reference carries in place of a value the harness will not pick."""
        return f"{DISAGREE_PREFIX}{self.finding_id}{DISAGREE_SUFFIX}"

    def describe(self) -> str:
        """One human block, row counts first, as the CLI prints it."""
        head = f"{self.finding_id}  {self.dataset}/{self.original_label}.{self.column}"
        lines = [head]
        for value, rows, shards in self.values:
            sample = ", ".join(Path(shard).name for shard in shards)
            lines.append(f"        {rows:>12,} rows  {value!r:<40} [{sample}]")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """The manifest's JSON shape, with the id materialised so a reader need not recompute it.

        ``values`` stays a list of ``[value, n_rows, [shard, ...]]`` TRIPLES rather than becoming a
        list of objects: the triple is the dataclass's own shape, so the JSON and the Python cannot
        drift into disagreeing about which field is which, and a reviewer reading the manifest sees
        the row counts lined up in a column.
        """
        return {
            "finding_id": self.finding_id,
            "dataset": self.dataset,
            "original_label": self.original_label,
            "column": self.column,
            "values": [[value, rows, list(shards)] for value, rows, shards in self.values],
        }

    @classmethod
    def from_manifest(cls, entry) -> "CellDisagreement":
        """Rebuild a finding from its manifest entry, so the offline stage can waive it.

        An entry with no recorded id recomputes one, which is what makes a hand-written manifest
        usable without a reviewer having to hash the claim themselves.
        """
        values = tuple((value, rows, tuple(shards)) for value, rows, shards in entry["values"])
        return cls(
            dataset=entry["dataset"],
            original_label=entry["original_label"],
            column=entry["column"],
            values=values,
            recorded_id=entry.get("finding_id", ""),
        )


# ── The networked seam: the only two functions in this module that speak to the Hub ────────────


@dataclass(frozen=True)
class ShardRead:
    """What one shard yielded: its schema always, its reduced cells when any columns were asked for.

    The REDUCTION travels back from a worker, never the table. One shard's 18 projected columns are
    17.8 MB in memory, so shipping 189 of those to the parent would move 3.4 GB through a pickle
    that the parent immediately collapses to a few thousand distinct rows. The worker collapses it
    first, and what crosses the boundary is the answer rather than the evidence for it.
    """

    path: str
    schema: object
    cells: dict = field(default_factory=dict)
    rows_scanned: int = 0


def list_shards(api, repo_id: str, sha: str) -> list:
    """Every published parquet shard of ONE commit, as ``HfFileSystem`` paths, sorted.

    By filename, and not by ``num_shards``: a finding has to name the shard a reviewer can open,
    and ``split_dataset_by_node`` — the shape ``sankey.scan_dataset`` uses — hands back a rank
    instead, which identifies nothing once the dataset is re-sharded.

    The commit is carried IN THE PATH, as ``datasets/<repo>@<sha>/...``, rather than on the
    filesystem object. Two reasons, and the second is a bug avoided: a path that names its own
    commit is a path a reviewer can paste, and ``HfFileSystem(revision=sha)`` silently LOSES its
    revision when pickled to a worker process — every shard of the sweep would then have been read
    from the default branch while the manifest recorded the sha.
    """
    names = api.list_repo_files(repo_id, repo_type="dataset", revision=sha)
    return sorted(f"datasets/{repo_id}@{sha}/{name}" for name in names if name.endswith(".parquet"))


class TunedFilesystem:
    """An ``HfFileSystem`` whose ``open`` carries this harness's read block size.

    The block size has to reach ``fs.open`` to have any effect — ``HfFileSystem`` ignores the
    instance's ``blocksize`` attribute and takes only the explicit keyword — and it must NOT reach
    the read functions, whose signatures are the seam a test replaces. Binding it to the filesystem
    is the factoring that satisfies both, and it is also where it belongs: it is a transport
    tunable, not something a parquet read has an opinion about.

    It is worth this much care because the number is large. The naive fsspec block transfers
    53.9 MB per shard — 10.2 GB over a 189-shard sweep — against 1.13 MB tuned, at no time penalty.
    """

    def __init__(self, filesystem, block_size: int = DEFAULT_BLOCK_SIZE):
        self.filesystem = filesystem
        self.block_size = block_size

    def open(self, path, mode="rb", **kwargs):
        """Open one published file, defaulting the block size rather than overriding a given one."""
        kwargs.setdefault("block_size", self.block_size)
        return self.filesystem.open(path, mode, **kwargs)


def tuned_filesystem(filesystem, block_size: int = DEFAULT_BLOCK_SIZE) -> TunedFilesystem:
    """Wrap a filesystem so every read it serves uses ``block_size``, wrapping an already-tuned one once."""
    if isinstance(filesystem, TunedFilesystem):
        return TunedFilesystem(filesystem.filesystem, block_size)
    return TunedFilesystem(filesystem, block_size)


def read_shard_schema(fs, path: str):
    """One shard's Arrow schema, read from its footer and nothing else.

    Separate from :func:`read_shard` because the gate has to run over all 189 footers BEFORE a
    single data byte is fetched. A coercion or a null-fill applied to a shard whose schema had
    drifted would turn a structural break into a plausible-looking data difference, and the whole
    value of code 2 is that it is never mistaken for code 1.
    """
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as handle:
        return pq.read_schema(handle)


def read_shard(fs, path: str, columns):
    """One shard's projected columns, as an Arrow table.

    The projection is pushed into the parquet reader rather than applied after it. ``sankey.py``'s
    scan calls ``select_columns`` on an already-streaming dataset, which is a post-hoc
    ``pa_table.select``: every shard's image bytes are fetched on 16 threads and thrown away. Here
    ``columns`` reaches the reader, so the 10 fields this harness does not want are never read.
    """
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as handle:
        return pq.ParquetFile(handle).read(columns=list(columns))


def reduce_table(table) -> tuple:
    """One shard's table as ``({pair: {column: Counter[value]}}, rows_scanned)``.

    Pure, and deliberately not part of :func:`read_shard`: the seam that speaks to the network has
    to stay small enough that a test can replace it with a literal table, and everything downstream
    of it has to be drivable from one.

    An Arrow ``group_by`` over all 18 columns does the collapsing — a ~92,000-row shard holds only
    a few thousand distinct rows, and grouping them in Arrow before any Python touches them is what
    makes 17.4 million rows tractable at all. Per-row Python here is the version that does not run.

    A Counter PER CELL, not per row. A row-keyed counter — the shape ``sankey.scan_dataset``
    aggregates into — structurally cannot represent "these published rows disagree about this one
    column": it can only report two rows, leaving a reader to work out which of sixteen cells
    differs and whether the other fifteen agreed.
    """
    names = [name for name in table.schema.names if name in COVERED_HUB_COLUMNS]
    value_columns = [name for name in names if name not in HUB_KEY_COLUMNS]
    grouped = table.group_by(names).aggregate([([], "count_all")])

    cells: dict = {}
    for row in grouped.to_pylist():
        count = row["count_all"]
        pair = (_key_cell(row["dataset"]), _key_cell(row["original_label"]))
        bucket = cells.setdefault(pair, {})
        for column in value_columns:
            bucket.setdefault(column, Counter())[row[column]] += count
    return cells, table.num_rows


def with_retry(call, args, path: str, *, attempts: int = SHARD_READ_ATTEMPTS, attempt: int = 1, sleep=None):
    """Run one shard's read, retrying a transient failure, then failing loudly with its path.

    Written as a self-call rather than a ``for`` loop around a ``try`` on purpose: ``PERF203``
    forbids a ``try``/``except`` inside a loop, so the retry is a helper the shard loop CALLS and
    that contains no loop of its own. Depth is bounded by ``attempts``.

    ``datasets``/``huggingface_hub`` already retry transient HTTP inside themselves; this outer
    layer only re-runs a shard whose read still failed after those. The backoff is
    ``sleep(2 * (k + 1))``, copied from ``sankey.py:289-304``.

    ``sleep`` defaults to ``None`` and resolves to ``time.sleep`` at CALL time rather than being
    bound as a default at def time. That is not a style choice: a default of ``time.sleep`` captures
    the function object when the module is imported, so a test that patches ``time.sleep`` to record
    the backoff records nothing and then waits the real 12 seconds to find out.
    """
    try:
        return call(*args)
    except Exception as exc:
        if attempt >= attempts:
            raise ShardReadError(path, f"shard {path} would not read after {attempts} attempt(s): {exc}") from exc
        (sleep or time.sleep)(SHARD_RETRY_BACKOFF_SECONDS * attempt)
        return with_retry(call, args, path, attempts=attempts, attempt=attempt + 1, sleep=sleep)


def _shard_job(job) -> ShardRead:
    """One unit of work as a process pool can see it: a top-level function over a picklable tuple.

    An empty ``columns`` is the gate's pass and reads the footer alone; anything else reads the
    projection and reduces it here, in the worker, so only the reduction crosses back.
    """
    fs, path, columns, attempts = job
    if not columns:
        schema = with_retry(read_shard_schema, (fs, path), path, attempts=attempts)
        return ShardRead(path=path, schema=schema)
    table = with_retry(read_shard, (fs, path, columns), path, attempts=attempts)
    cells, rows_scanned = reduce_table(table)
    return ShardRead(path=path, schema=table.schema, cells=cells, rows_scanned=rows_scanned)


def _settle(call, argument):
    """Run one job and return its outcome — result or exception — as a VALUE.

    A failed shard must not abort the other 188: the sweep has to finish so the refusal can name
    every path that failed at once. Returning the exception rather than raising it is also what
    keeps the dispatch loops below free of the ``try`` that ``PERF203`` forbids there.
    """
    try:
        return call(argument)
    except Exception as exc:
        return exc


def _dispatch(jobs, workers: int, label: str) -> list:
    """Run every job, in processes when ``workers > 1`` and in-process when it is not.

    ``workers=1`` is not only a debugging affordance: it is how the offline tests drive a whole
    refresh against fakes. A process pool re-imports this module in each child, so a monkeypatched
    ``read_shard`` in the parent would be invisible to it, and the tests would end up proving
    something about the real Hub instead.
    """
    if workers <= 1:
        return [_settle(_shard_job, job) for job in jobs]

    outcomes = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(_shard_job, job) for job in jobs]
        for index, future in enumerate(futures, start=1):
            outcomes.append(_settle(lambda done: done.result(), future))
            if index % 20 == 0 or index == len(futures):
                logger.info("  %s: %d/%d shards", label, index, len(futures))
    return outcomes


def _outcome_path(outcome, job) -> str:
    """The shard an outcome belongs to, whether it succeeded, failed, or died with the pool.

    A `ShardReadError` carries its own path, but a `BrokenProcessPool` carries nothing — and that
    is exactly the failure a reviewer most needs named, because one OOM-killed worker fails every
    outstanding future at once and a refusal listing 72 question marks says nothing about which
    read was too big. The job tuple always knows, so the path comes from there when the exception
    cannot supply it.
    """
    return getattr(outcome, "path", None) or job[1]


def _with_paths(outcomes, jobs) -> list:
    """Pair each outcome with its shard's path, so a refusal can name what failed."""
    return [(_outcome_path(outcome, job), outcome) for outcome, job in zip(outcomes, jobs, strict=True)]


# ── Step 15, part 1: the schema gate ───────────────────────────────────────────────────────────


def schema_pairs(schema) -> tuple:
    """One Arrow schema as sorted ``(name, str(type))`` pairs — the text every refusal quotes."""
    return tuple(sorted((field_.name, str(field_.type)) for field_ in schema))


def schema_fingerprint(schema) -> str:
    """sha256 over the sorted ``name:type`` pairs of one shard's schema.

    Over the WHOLE schema, not only the 18 covered columns. Restricting it to the covered ones
    would make the fingerprint a constant — the gate has just asserted those types — and a constant
    records nothing about the artifact a reference was taken from.
    """
    payload = "\n".join(f"{name}:{type_name}" for name, type_name in schema_pairs(schema))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_shard_schema(shard: ShardRead) -> None:
    """Refuse a shard that does not carry all 18 covered columns at the expected types.

    Never a null-fill and never a coercion. ``datasets.concatenate_datasets`` null-fills a missing
    column rather than raising, which is the behaviour ``constants.CONSOLIDATED_COLUMNS`` exists to
    guard the build against; doing the same here would publish a reference full of empty cells and
    report them as 1,482 rows' worth of data differences.
    """
    types = dict(schema_pairs(shard.schema))
    for column in COVERED_HUB_COLUMNS:
        expected = "bool" if column == "plankton" else "string"
        actual = types.get(column)
        if actual is None:
            raise GoldenDiffError(
                f"{shard.path} carries no {column!r} column. The harness reads the published "
                f"taxonomy by name and will not substitute nulls for a column that is not there."
            )
        if actual != expected:
            raise GoldenDiffError(
                f"{shard.path} publishes {column!r} as {actual}, not {expected}. A cast here would "
                f"turn a schema change into a data difference; adjudicate the published schema first."
            )


def gate_schemas(reads) -> str:
    """Check every shard's footer and return the one fingerprint they all share.

    Over ALL shards, not shard zero. A drift that starts at shard 137 is exactly the drift a
    sampled gate misses, and it is also the likeliest kind: a partial re-push writes new shards
    beside old ones.
    """
    by_fingerprint: dict = {}
    for shard in reads:
        check_shard_schema(shard)
        by_fingerprint.setdefault(schema_fingerprint(shard.schema), []).append(shard.path)
    if len(by_fingerprint) > 1:
        groups = ", ".join(
            f"{fingerprint[:12]} ({len(paths)} shards, e.g. {paths[0]})" for fingerprint, paths in by_fingerprint.items()
        )
        raise GoldenDiffError(f"the published shards do not share one schema: {groups}")
    return next(iter(by_fingerprint))


# ── Step 15, part 2: reduce, resolve, invert ───────────────────────────────────────────────────


def merge_shard(cells, shard: ShardRead) -> None:
    """Fold one shard's reduction into the sweep's, keeping row counts and a shard sample per value.

    The parent keeps ``{pair: {column: {value: [n_rows, [shard paths]]}}}``: the worker's Counter
    says how many rows, and the merge is the only place that knows WHICH shard they came from. Both
    halves are needed to describe a disagreement, and neither is worth carrying when there is not
    one — hence the bound on the recorded paths.
    """
    for pair, columns in shard.cells.items():
        target = cells.setdefault(pair, {})
        for column, counter in columns.items():
            bucket = target.setdefault(column, {})
            for value, count in counter.items():
                record = bucket.get(value)
                if record is None:
                    bucket[value] = [count, [shard.path]]
                    continue
                record[0] += count
                if len(record[1]) < MAX_RECORDED_SHARDS and shard.path not in record[1]:
                    record[1].append(shard.path)


def resolve_cells(cells) -> tuple:
    """Turn the reduction into one value per cell, or a finding and a sentinel where there is not one.

    THE RULE, and the reason this function exists at all: where the published rows disagree, the
    harness emits a marker and a finding, and never a value. Omitting the column instead raises
    ``KeyError`` at ``write.py:891``; omitting the value compares as ``""`` and reads as a real
    difference, which would file a build-history defect under "the taxonomy changed".
    """
    resolved: dict = {}
    findings = []
    for pair in sorted(cells):
        dataset, original_label = pair
        values: dict = {}
        for column in render.PUBLISHED_TAXONOMY_COLUMNS:
            bucket = cells[pair].get(column) or {}
            if len(bucket) == 1:
                values[column] = next(iter(bucket))
                continue
            if not bucket:
                raise GoldenDiffError(
                    f"({dataset}, {original_label}) was reduced with no value at all for {column!r}; "
                    f"the reduction is wrong, and a reference built from it would be too."
                )
            finding = CellDisagreement(
                dataset=dataset,
                original_label=original_label,
                column=column,
                # Most rows first: it is what tells a "3 rows out of 17 million" typo apart from a
                # 50/50 split, and the two want different remediation.
                values=tuple(
                    sorted(
                        ((value, record[0], tuple(record[1])) for value, record in bucket.items()),
                        key=lambda item: (-item[1], _claim_text(item[0])),
                    )
                ),
            )
            findings.append(finding)
            values[column] = finding.sentinel
        resolved[pair] = values
    return resolved, findings


def hub_rows(resolved) -> list:
    """The resolved pairs as plain Hub-shaped row dicts, one per pair.

    The bridge between the reduce, which is keyed by pair, and the inversion, which takes a row.
    Keeping it a named step rather than a comprehension inside :func:`refresh` is what lets a test
    hand :func:`reference_bytes` rows it built itself.
    """
    return [
        {HUB_KEY_COLUMNS[0]: dataset, HUB_KEY_COLUMNS[1]: original_label, **values}
        for (dataset, original_label), values in resolved.items()
    ]


def reference_rows(rows) -> list:
    """Every published row as a 19-column CSV row, sorted by ``(Dataset, Raw_Labels)``.

    The order is diff-irrelevant — ``write.diff_published`` keys its comparison by pair
    (``write.py:876-877``) — and is chosen only so that a reviewer reading the committed reference's
    own git diff sees a change where the data changed, rather than a re-sort.
    """
    rows = list(rows)
    inverted = [hub_row_to_csv_row(row) for row in rows]
    if len(inverted) != len(rows):
        raise GoldenDiffError(f"the inversion emitted {len(inverted)} rows for {len(rows)} pairs")
    inverted.sort(key=lambda row: (row["Dataset"], row["Raw_Labels"]))
    return inverted


def reference_bytes(rows) -> bytes:
    """The reference CSV's bytes for a set of published rows.

    Through ``render.write_wide_csv`` and never through a local ``csv.writer``: the reference has to
    be a file ``write.diff_published`` can read back, and a divergence in quoting or line ending
    would reach a reviewer as a data difference rather than as the formatting accident it is.
    """
    return render.write_wide_csv(reference_rows(rows))


# ── Step 18: the manifest ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reference:
    """One refresh's whole output: the rows, the bytes they render to, and what was read to get them.

    Returned rather than written, so a caller — a test, or a CLI run whose sweep then fails — holds
    a complete reference before anything reaches the disk. :func:`write_reference` is the only
    function that persists it, which is how "a failed shard writes no reference file at all" is a
    property of the control flow rather than a cleanup step that could be skipped.
    """

    rows: list
    csv_bytes: bytes
    manifest: dict
    disagreements: list


def sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of a byte string, used for both the reference and the frozen taxonomy CSV."""
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    repo_id: str,
    sha: str,
    revision_requested,
    hub_last_modified,
    complete: bool,
    expected: int,
    read: int,
    failed,
    fingerprint: str,
    rows_scanned: int,
    resolved,
    csv_bytes: bytes,
    taxonomy_csv: Path,
    csv_pairs,
    disagreements,
) -> dict:
    """The committed record of what one refresh could see, and of what it could not.

    ``excluded_datasets`` and ``absent_pairs`` are kept apart deliberately. They are both "rows the
    reference does not cover" and they are not the same kind of gap: the first is a source that has
    not been published yet, the second is a row in a source that HAS been published for which no
    image carries the label — a finding, adjudicated in KI-34, not an exemption. Merging them into
    one number would let the second kind grow silently inside the first.

    ``taxonomy_csv_sha256`` records the table this reference was taken ALONGSIDE, not the table it
    must be compared against. A mismatch is the normal state of a PR that edits the taxonomy, so the
    report warns about it and does not refuse — refusing there would disable the gate in exactly the
    situation it exists for.
    """
    excluded = sorted(constants.RECORDED_BUT_NOT_YET_PUBLISHED)
    absent = sorted(pair for pair in csv_pairs - set(resolved) if pair[0] not in constants.RECORDED_BUT_NOT_YET_PUBLISHED)
    if absent:
        # DERIVED, and therefore said out loud. The report stage refuses an uncovered row that this
        # list does not name — but this list is computed from whatever was uncovered, so a refresh
        # run after the taxonomy grew would quietly absolve the new rows instead of failing on them,
        # and the coverage-anomaly rule would never fire again. It cannot be a refusal (the three
        # planktoscope rows are real and have to ship accounted-for, or the gate is red on day one
        # and someone waives the whole check), so it is a warning that names every pair. A reviewer
        # reads this list in the committed manifest's diff; growing it is a decision, not a default.
        logger.warning(
            "%d pair(s) are mapped by the taxonomy in a PUBLISHED source but carry no published "
            "image, and are being recorded in absent_pairs: %s. Each is a per-row finding to "
            "adjudicate (KI-34), not an exemption — check this list in the manifest's diff.",
            len(absent),
            [f"{dataset}/{label}" for dataset, label in absent],
        )
    return {
        "repo_id": repo_id,
        "revision": sha,
        "revision_requested": revision_requested,
        "hub_last_modified": hub_last_modified,
        "refreshed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "complete": complete,
        "shards": {"expected": expected, "read": read, "failed": list(failed)},
        "schema_fingerprint": fingerprint,
        "rows_scanned": rows_scanned,
        "pairs": len(resolved),
        "csv_sha256": sha256_bytes(csv_bytes),
        "taxonomy_csv_sha256": sha256_bytes(Path(taxonomy_csv).read_bytes()),
        "covered_columns": list(COVERED_HUB_COLUMNS),
        "synthesised_columns": list(SYNTHESISED_COLUMNS),
        "excluded_datasets": excluded,
        "absent_pairs": [list(pair) for pair in absent],
        "disagreements": [finding.as_dict() for finding in disagreements],
    }


def write_reference(reference: Reference, reference_path, manifest_path) -> None:
    """Persist the reference CSV and its manifest, in that order.

    The CSV first, because the manifest's ``csv_sha256`` is the only thing that proves the pair was
    written together; a manifest on disk naming bytes that were never written is the one state the
    report stage cannot tell apart from a hand edit.
    """
    Path(reference_path).write_bytes(reference.csv_bytes)
    Path(manifest_path).write_text(json.dumps(reference.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("reference: %s (%d bytes)", reference_path, len(reference.csv_bytes))
    logger.info("manifest:  %s", manifest_path)


# ── Step 15: the refresh stage ─────────────────────────────────────────────────────────────────


def refresh(
    repo_id,
    revision,
    *,
    workers: int = DEFAULT_WORKERS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    shards=None,
    taxonomy_csv=constants.DEFAULT_TAXONOMY_CSV_FILENAME,
    api=None,
    fs=None,
    reference_path=None,
    manifest_path=None,
) -> Reference:
    """Read the published taxonomy columns at one commit and build the reference from them.

    The whole stage turns on resolving ``revision`` to a sha ONCE, up front, and using only that
    sha afterwards. A symbolic name re-resolved per call is a branch that can move between the
    listing and the last shard, and the reference would then be stitched from two commits with
    nothing in it saying so.

    Raises:
        GoldenDiffError: If any shard will not read, if any shard's schema has drifted, or if a
            published id cannot be inverted exactly. Raising rather than returning a partial
            reference is what makes "the refresh wrote no file" the guaranteed consequence of a
            flaky network, so a half-read sweep can never be mistaken for a green gate.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if fs is None:
        from huggingface_hub import HfFileSystem

        # No ``revision=`` on the filesystem: it is in every path (see :func:`list_shards`), which
        # is the only form that survives the pickle into a worker process.
        fs = tuned_filesystem(HfFileSystem(), block_size)

    info = api.dataset_info(repo_id, **constants.revision_kwargs(revision))
    sha = info.sha
    logger.info("golden diff refresh: %s @ %s (requested %s)", repo_id, sha, revision or "the default branch")

    paths = list_shards(api, repo_id, sha)
    if not paths:
        raise GoldenDiffError(f"{repo_id}@{sha} lists no parquet shard; there is nothing to build a reference from")

    # The gate runs over every listed shard even when --shards narrows the data read: a reference
    # that vouches for a schema it only sampled is a reference that vouches for nothing.
    started = time.time()
    gate_jobs = [(fs, path, (), SHARD_READ_ATTEMPTS) for path in paths]
    gate_outcomes = _dispatch(gate_jobs, workers, "schema gate")
    fingerprint = gate_schemas(_settled_reads(_with_paths(gate_outcomes, gate_jobs)))
    logger.info("schema gate: %d footers, %.1f s, fingerprint %s", len(paths), time.time() - started, fingerprint[:12])

    indices = None if shards is None else resolve_shard_selection(shards, len(paths))
    selected = paths if indices is None else [paths[index] for index in indices]
    complete = indices is None
    if not complete:
        logger.warning(
            "--shards selected %d of %d shards: the reference will be marked complete=false and the "
            "report stage will refuse it",
            len(selected),
            len(paths),
        )

    started = time.time()
    jobs = [(fs, path, COVERED_HUB_COLUMNS, SHARD_READ_ATTEMPTS) for path in selected]
    reads = _settled_reads(_with_paths(_dispatch(jobs, workers, "shard read"), jobs))

    cells: dict = {}
    rows_scanned = 0
    for shard in reads:
        merge_shard(cells, shard)
        rows_scanned += shard.rows_scanned
    logger.info("read %d shards, %d rows, %.1f s", len(reads), rows_scanned, time.time() - started)

    resolved, disagreements = resolve_cells(cells)
    rows = reference_rows(hub_rows(resolved))
    csv_bytes = render.write_wide_csv(rows)

    csv_pairs = {(row["Dataset"], row["Raw_Labels"]) for row in loader.load_taxonomy(taxonomy_csv).rows()}
    manifest = build_manifest(
        repo_id=repo_id,
        sha=sha,
        revision_requested=revision,
        hub_last_modified=_isoformat(getattr(info, "last_modified", None) or getattr(info, "lastModified", None)),
        complete=complete,
        expected=len(paths),
        read=len(reads),
        failed=[],
        fingerprint=fingerprint,
        rows_scanned=rows_scanned,
        resolved=resolved,
        csv_bytes=csv_bytes,
        taxonomy_csv=taxonomy_csv,
        csv_pairs=csv_pairs,
        disagreements=disagreements,
    )
    reference = Reference(rows=rows, csv_bytes=csv_bytes, manifest=manifest, disagreements=disagreements)
    if reference_path is not None and manifest_path is not None:
        write_reference(reference, reference_path, manifest_path)
    return reference


def _settled_reads(outcomes) -> list:
    """Every successful read, or one refusal naming every shard that failed.

    All of them, in one message. A sweep that loses four shards to one flaky mirror should say so
    once — reporting whichever failure was collected first sends a reviewer chasing a shard that is
    no more broken than the other three.
    """
    failures = [(path, outcome) for path, outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        paths = [path for path, _ in failures]
        shown = paths if len(paths) <= 8 else [*paths[:8], f"... and {len(paths) - 8} more"]
        raise GoldenDiffError(
            f"{len(failures)} of {len(outcomes)} shard(s) would not read, so NO reference was "
            f"written: {shown}. The first failure was: {failures[0][1]}"
        )
    return [outcome for _path, outcome in outcomes]


def _isoformat(value) -> str | None:
    """A Hub timestamp as ISO text, tolerating the ``None`` an unset ``last_modified`` gives."""
    return None if value is None else value.isoformat()


def resolve_shard_selection(shards, count: int) -> list:
    """``--shards`` as indices, from either a spec string or an iterable of them.

    Resolved HERE and not in the CLI, because the count it has to be checked against only exists
    once the repo has been listed — and listing it twice would mean two ``dataset_info`` calls, one
    of them outside the networked seam this module keeps deliberately small.
    """
    indices = parse_shard_spec(shards, count) if isinstance(shards, str) else sorted({int(index) for index in shards})
    out_of_range = [index for index in indices if not 0 <= index < count]
    if out_of_range:
        raise GoldenDiffError(f"--shards names {out_of_range}, but the repo has {count} shard(s) (0-{count - 1})")
    return indices


# ── Step 19: the report stage ──────────────────────────────────────────────────────────────────


def recorded_revision(manifest_path) -> str | None:
    """The sha the committed manifest was taken at, or ``None`` when there is no manifest yet.

    This is what ``--revision`` defaults to, and deliberately not ``main``. The artifact is frozen,
    so "the Hub moved" is news and has to be loud; defaulting to the default branch would silently
    re-baseline the gate against whatever was pushed.
    """
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("revision")


def check_living_derivation(rendered_rows) -> list:
    """Every rendered row where ``living`` is not ``root_class == "living"``.

    The one column the reference synthesises, so this is the only check standing between a
    synthesised value and a reference the harness cannot justify. If the renderer ever stops
    deriving ``living`` at ``render.py:148`` — because a curator pinned it, or because the column
    grew a third state — then synthesising it here is inventing data, and the harness must refuse
    rather than emit a column it is quietly guessing.
    """
    return [row for row in rendered_rows if row["living"] != ("True" if row["root_class"] == "living" else "False")]


def _sentinel_id(value: str) -> str:
    """The ``finding_id`` inside a sentinel cell."""
    return value[len(DISAGREE_PREFIX) :].removesuffix(DISAGREE_SUFFIX)


def _split_key(key: str) -> tuple:
    """A ``write.Change`` key back into its ``(Dataset, Raw_Labels)`` pair.

    ``diff_published`` joins the pair with ``/`` (``write.py:882``). Splitting on the FIRST ``/``
    recovers it exactly, because no ``dataset`` value contains one while ``Raw_Labels`` values do.
    """
    dataset, _, raw_label = key.partition("/")
    return dataset, raw_label


def partition_changes(changes) -> dict:
    """The four kinds of change a reference-versus-render diff can produce, kept apart.

    They are four different verdicts and merging any two of them loses the verdict. In particular a
    pair the Hub has and the CSV does not (``differs``, exit 1) and a CSV row the Hub does not carry
    (``uncovered``, subtracted only once the manifest accounts for it) arrive from
    ``diff_published`` as the same ``column == "*"`` shape, distinguishable only by direction.

    Note that ``write.py:873``'s byte short-circuit can never fire on this diff: a 1,482-row
    reference is never byte-equal to a 2,358-row render. Nobody should optimise for it.
    """
    buckets: dict = {"differs": [], "unresolvable": [], "hub_only": [], "uncovered": []}
    for change in changes:
        if change.column == "*":
            buckets["hub_only" if change.after == "" else "uncovered"].append(change)
        elif _is_sentinel(change.before):
            buckets["unresolvable"].append(change)
        else:
            buckets["differs"].append(change)
    return buckets


def coverage_anomalies(uncovered, manifest) -> list:
    """Uncovered CSV rows that neither recorded gap accounts for.

    THE RULE that makes the 876 honest. Every row the reference does not cover has to be either a
    source that is not published yet or a pair recorded as carrying no published image. Anything
    else means the REFERENCE is wrong — a shard silently skipped, a pair lost in the reduction —
    and that is a refusal, not a data difference. Without this, the subtraction that makes the gate
    readable is also the hole that makes it useless.
    """
    excluded = set(manifest.get("excluded_datasets", ()))
    absent = {tuple(pair) for pair in manifest.get("absent_pairs", ())}
    return [
        change for change in uncovered if _split_key(change.key)[0] not in excluded and _split_key(change.key) not in absent
    ]


def _coverage_lines(manifest, rendered_rows, buckets, waived_ids) -> list:
    """The coverage block, printed before the verdict on EVERY run, green included.

    A gate that only explains itself when it is red teaches its readers that green means "checked",
    when green here means "checked for 62.9% of the rows, and here is precisely which 37.1% it did
    not look at". The percentage belongs next to the verdict or it will not be read at all.
    """
    pairs = manifest["pairs"]
    total = len(rendered_rows)
    excluded = set(manifest.get("excluded_datasets", ()))
    absent = [tuple(pair) for pair in manifest.get("absent_pairs", ())]

    by_dataset = Counter(row["Dataset"] for row in rendered_rows if row["Dataset"] in excluded)
    pending = ", ".join(f"{name} {count}" for name, count in by_dataset.most_common())
    absent_sources = ", ".join(sorted({dataset for dataset, _label in absent}))
    # `compared_cells` is the DENOMINATOR — what the gate claims to have checked — and it excludes
    # `living` deliberately (23,712 = 1,482 x 16, not x 17). The numerator beside it is whatever
    # `diff_published` reported, and that DOES include `living`, because the reference carries all
    # 19 columns and the differ iterates every one of them.
    #
    # So a changed `root_class` prints TWO differing cells, not one: the real change and its
    # synthesised shadow. That asymmetry is deliberate and is left as it is, because it can only
    # ever over-report. Making the numerator match the denominator would mean filtering `living`
    # out of the differ's output, and a filter that hides a column is exactly the kind of thing
    # that later hides a column it should not have.
    compared_cells = pairs * len(COMPARED_COLUMNS)
    synthesised = ", ".join(SYNTHESISED_COLUMNS)

    lines = [
        f"golden diff — {manifest['repo_id']} @ {manifest['revision'][:7]}  (refreshed {manifest['refreshed_at']})",
        f"  columns  {len(LEGACY_HEADER)} of {len(LEGACY_HEADER)} emitted: {len(COMPARED_COLUMNS)} compared, "
        f"{len(SYNTHESISED_COLUMNS)} synthesised ({synthesised}, from root_class —",
        "           render.py:148, so it cannot independently fail), 2 are the key",
        f"  rows     {pairs:,} of {total:,} compared ({100.0 * pairs / total:.1f}%)",
        f"             {sum(by_dataset.values()):,} in {len(by_dataset)} datasets not yet published: {pending}",
        f"             {len(absent):,} mapped but carrying no published image ({absent_sources or 'none'})",
        f"  cells    {compared_cells:,} compared, {len(buckets['differs']):,} differ",
        f"  gaps     {len(manifest.get('disagreements', ())):,} pairs whose rows disagree, {len(waived_ids):,} waived",
    ]
    if buckets["hub_only"]:
        lines.append(f"  hub-only {len(buckets['hub_only']):,} published pair(s) the taxonomy does not map")
    return lines


def report(reference_csv, manifest, package_dir, waivers, *, summary: bool = False, as_json: bool = False) -> int:
    """Diff the committed reference against a fresh render, offline, and return the exit code.

    Nothing in this function or anything it imports speaks HTTP; ``tests/test_golden_diff.py``
    proves that with a subprocess ``sys.modules`` probe rather than asserting it, because the
    assertion is only worth anything if it keeps being true after someone adds an import.

    The pre-flight refusals below run in order and each names its fix. They exist because every one
    of them describes a state in which the diff would still PRODUCE a report — a plausible, precise,
    wrong report — and a wrong green run is worse here than no run at all.
    """
    reference_path, manifest_path = Path(reference_csv), Path(manifest)
    missing = [str(path) for path in (reference_path, manifest_path) if not path.exists()]
    if missing:
        return _refuse(f"no reference to report against: {missing} — run `pz_golden_diff --refresh` first")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_bytes = reference_path.read_bytes()

    if sha256_bytes(reference_bytes) != manifest_data.get("csv_sha256"):
        return _refuse(
            f"{reference_path.name} and {manifest_path.name} were edited apart: the manifest records "
            f"csv_sha256={manifest_data.get('csv_sha256')} and the file hashes to "
            f"{sha256_bytes(reference_bytes)}. Re-run --refresh; do not patch either by hand."
        )
    if not manifest_data.get("complete", False) or manifest_data.get("shards", {}).get("failed"):
        return _refuse(
            f"the reference is partial (complete={manifest_data.get('complete')}, "
            f"failed={manifest_data.get('shards', {}).get('failed')}). A gate cannot vouch for rows "
            f"it never read — re-run --refresh without --shards."
        )
    if not manifest_data.get("schema_fingerprint"):
        return _refuse(
            "the manifest records no schema_fingerprint, so it predates the schema gate and nothing "
            "proves the columns it read were the columns it names. Re-run --refresh."
        )

    store = loader.load_taxonomy(Path(package_dir))
    rendered_rows = store.rows()
    broken = check_living_derivation(rendered_rows)
    if broken:
        example = broken[0]
        return _refuse(
            f"{len(broken)} rendered row(s) no longer satisfy living == (root_class == 'living'), e.g. "
            f"{example['Dataset']}/{example['Raw_Labels']}: living={example['living']!r} "
            f"root_class={example['root_class']!r}. The reference SYNTHESISES living from root_class "
            f"exactly as render.py:148 derives it, so that synthesis is now invention and the harness "
            f"refuses rather than emit a column it cannot justify."
        )

    # Hashed from the rows THIS report is diffing, not from the default CSV path. Under a
    # non-default --package those are different tables, and warning about a file the run never
    # looked at is worse than not warning: it is a specific claim about the wrong artifact. The
    # bytes are already in hand, since `rendered_rows` is what the diff below compares.
    committed_sha = sha256_bytes(render.write_wide_csv(rendered_rows))
    if committed_sha != manifest_data.get("taxonomy_csv_sha256"):
        # A WARNING and not a refusal: comparing a CHANGED table against the frozen Hub is the whole
        # point, so refusing here would disable the gate exactly when it is needed.
        logger.warning(
            "the taxonomy CSV has moved since this reference was taken (%s -> %s); that is the "
            "normal state of a PR that edits the table, and is what the diff below is for.",
            (manifest_data.get("taxonomy_csv_sha256") or "?")[:12],
            committed_sha[:12],
        )

    changes = write.diff_published(Path(package_dir), reference_bytes)
    buckets = partition_changes(changes.changes)

    findings = [CellDisagreement.from_manifest(entry) for entry in manifest_data.get("disagreements", ())]
    unwaived, waived, stale = apply_waivers(findings, load_waivers(Path(waivers)))
    waived_ids = {finding.finding_id for finding in waived}
    unwaived_ids = {finding.finding_id for finding in unwaived}

    anomalies = coverage_anomalies(buckets["uncovered"], manifest_data)
    orphans = sorted({_sentinel_id(change.before) for change in buckets["unresolvable"]} - {f.finding_id for f in findings})

    lines = _coverage_lines(manifest_data, rendered_rows, buckets, waived_ids)
    # Every sentinel in the reference is cross-checked against the manifest that describes it, so a
    # manifest entry quietly deleted cannot turn an unwaived disagreement into a silent green.
    cited = {_sentinel_id(change.before) for change in buckets["unresolvable"]}
    logger.debug("%d sentinel(s) cited by the reference, %d unwaived", len(cited), len(unwaived_ids))

    if as_json:
        print(json.dumps(_json_payload(manifest_data, buckets, unwaived, waived, stale, anomalies, orphans), indent=2))
    else:
        print("\n".join(lines))
        _print_verdict(buckets, unwaived, stale, anomalies, orphans, summary=summary)

    # Refusals outrank everything: a 2 says "this run cannot be trusted", and letting a 1 or a 3
    # mask it would hand a reviewer a specific-sounding answer from a reference known to be wrong.
    if anomalies or orphans:
        return EXIT_REFUSED
    if unwaived or stale:
        return EXIT_UNRESOLVABLE
    if buckets["differs"] or buckets["hub_only"]:
        return EXIT_DIFFERS
    return EXIT_OK


def _refuse(message: str) -> int:
    """Print one refusal and hand back exit 2."""
    print(f"REFUSED: {message}")
    return EXIT_REFUSED


def _print_verdict(buckets, unwaived, stale, anomalies, orphans, *, summary: bool) -> None:
    """The verdict, and — unless ``--summary`` — the changes behind it."""
    limit = 0 if summary else 40
    for change in buckets["differs"][:limit]:
        print(f"  DIFFERS  {change.describe()}")
    for change in buckets["hub_only"][:limit]:
        print(f"  HUB-ONLY {change.key}: published, but the taxonomy maps no such pair")
    for change in anomalies[:limit]:
        print(f"  ANOMALY  {change.key}: uncovered by the reference and named by neither recorded gap")
    for finding in unwaived[:limit]:
        print(f"  UNWAIVED {finding.describe()}")
    for waiver_id in stale:
        print(f"  STALE WAIVER {waiver_id}: matches no current finding — delete it")
    for orphan in orphans:
        print(f"  ORPHAN SENTINEL {orphan}: the reference cites a finding the manifest does not record")

    if anomalies or orphans:
        print(f"REFUSED: {len(anomalies)} coverage anomaly/-ies, {len(orphans)} orphan sentinel(s)")
    elif unwaived or stale:
        print(f"UNRESOLVABLE: {len(unwaived)} unwaived disagreement(s), {len(stale)} stale waiver(s)")
    elif buckets["differs"] or buckets["hub_only"]:
        print(f"DIFFERS: {len(buckets['differs'])} cell(s), {len(buckets['hub_only'])} hub-only pair(s)")
    else:
        print("OK: every cell this reference can see agrees with the render")


def _json_payload(manifest, buckets, unwaived, waived, stale, anomalies, orphans) -> dict:
    """The same verdict as a machine-readable object, for a reviewer diffing two runs."""
    return {
        "coverage": {
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
            "refreshed_at": manifest["refreshed_at"],
            "pairs": manifest["pairs"],
            "compared_columns": list(COMPARED_COLUMNS),
            "synthesised_columns": list(SYNTHESISED_COLUMNS),
            "excluded_datasets": manifest.get("excluded_datasets", []),
            "absent_pairs": manifest.get("absent_pairs", []),
        },
        "differs": [change.describe() for change in buckets["differs"]],
        "hub_only": [change.key for change in buckets["hub_only"]],
        "coverage_anomalies": [change.key for change in anomalies],
        "unwaived": [finding.as_dict() for finding in unwaived],
        "waived": [finding.as_dict() for finding in waived],
        "stale_waivers": stale,
        "orphan_sentinels": orphans,
    }


# ── Staleness: the one networked call outside refresh, and it is opt-in ────────────────────────


def check_revision(manifest, *, allow_stale: bool = False, api=None) -> int:
    """Ask the Hub whether it has moved under the committed reference. Opt-in, one call.

    Imported lazily and called only from ``main`` under ``--check-revision``, so ``report`` keeps
    its offline-by-construction property. This fires the day after a ``push_revision`` push, which
    is why the read-revision pins and this harness are one change set: a reference of a repo the
    gate cannot pin to a revision is a reference that goes quietly wrong.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    recorded = manifest["revision"]
    # ONE call, and it asks the only question that matters: has the repo's default branch moved off
    # the commit this reference was taken at? Re-reading the recorded sha would always succeed and
    # would answer nothing, because a sha is exactly the thing that cannot move.
    current = api.dataset_info(manifest["repo_id"]).sha
    if current == recorded:
        return EXIT_OK

    message = (
        f"{manifest['repo_id']} now resolves to {current} and the reference was taken at {recorded}. "
        f"The gate is checking a commit the repo has moved past — re-run --refresh, or pass "
        f"--allow-stale-revision if that is deliberate."
    )
    if allow_stale:
        logger.warning("%s", message)
        return EXIT_OK
    print(f"REFUSED: {message}")
    return EXIT_REFUSED


def warn_if_aged(manifest) -> None:
    """Warn when a reference is older than :data:`STALE_AFTER_DAYS`, and change no verdict.

    Age is not a failure. The artifact is frozen, so an old reference of it is still a correct
    reference of it; making age red would produce a gate that is red for reasons unrelated to the
    diff, and a gate that is routinely red is a gate nobody reads.
    """
    refreshed = manifest.get("refreshed_at")
    if not refreshed:
        return
    age = (datetime.now(UTC) - datetime.fromisoformat(refreshed)).days
    if age >= STALE_AFTER_DAYS:
        logger.warning("this reference is %d days old (> %d); consider --refresh", age, STALE_AFTER_DAYS)


# ── Step 20: the CLI ───────────────────────────────────────────────────────────────────────────


def parse_shard_spec(spec: str, count: int) -> list:
    """``"0-1"`` / ``"0,5,7"`` / ``"3"`` into shard indices.

    Indices and not a count, so ``--shards 4-7`` names the same four shards on every run and a
    finding in one of them can be reproduced by re-reading exactly those. ``count`` is accepted so
    an open-ended form could be spelled here later; the range check itself lives in
    :func:`resolve_shard_selection`, which is the one place both spellings pass through.
    """
    logger.debug("resolving shard spec %r against %d shard(s)", spec, count)
    indices: set = set()
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue
        low, _, high = piece.partition("-")
        indices.update(range(int(low), int(high or low) + 1))
    return sorted(indices)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as ``pz_golden_diff``."""
    parser = argparse.ArgumentParser(
        description="Compare planktonzilla_taxonomy.csv against the rows actually published on the Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 agrees | 1 the data differs | 2 could not read / will not vouch | "
            "3 unresolvable (an unwaived disagreement among published rows, or a stale waiver). "
            "1 and 2 are never conflated: a flaky network can only ever produce 2."
        ),
    )
    parser.add_argument("--refresh", action="store_true", help="network stage: re-read the published shards")
    parser.add_argument("--report", action="store_true", help="network-free stage: diff the reference against the render")
    parser.add_argument(
        "--check-revision",
        action="store_true",
        help="with --report, make ONE dataset_info call asking whether the Hub has moved past the reference",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="published dataset to read")
    parser.add_argument(
        "--revision",
        default=None,
        help="revision to READ. Default: the sha the committed manifest records, NOT the default branch — "
        "the artifact is frozen, so the Hub having moved must be loud rather than absorbed.",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH, help="reference CSV path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="reference manifest JSON path")
    parser.add_argument("--waivers", type=Path, default=DEFAULT_WAIVERS_PATH, help="adjudicated-disagreements JSON")
    parser.add_argument("--package", type=Path, default=loader.PACKAGE_DIR, help="taxonomy package directory to render")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="refresh PROCESSES (1 reads in-process)")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="fsspec read block size, in bytes")
    parser.add_argument("--shards", default=None, help="restrict the refresh to shard indices, e.g. 0-1 or 0,5,7")
    parser.add_argument("--summary", action="store_true", help="print the coverage block and the verdict, no change list")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit the verdict as JSON")
    parser.add_argument("--allow-stale-revision", action="store_true", help="with --check-revision, warn instead of refusing")
    return parser


def main(argv=None) -> int:
    """CLI entry point.

    Returns:
        The process exit status — 0, 1, 2 or 3, as the module docstring defines them.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.refresh or args.report):
        parser.error("choose --refresh and/or --report")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.refresh:
        status = _guarded(_refresh_stage, args)
        if status != EXIT_OK:
            return status

    if not args.report:
        return EXIT_OK
    return _guarded(_report_stage, args)


def _guarded(stage, args) -> int:
    """Run one CLI stage and turn every way it can fail into the code that failure actually IS.

    This is the function that makes "1 and 2 are never conflated" true of the PROCESS and not only
    of the docstring. Without it, a stage that raises leaves ``main`` through the interpreter's own
    error path, and Python exits 1 — so an 18-column reference nobody can diff, a manifest with a
    truncated JSON tail, or one flaky ``dataset_info`` call all report themselves to CI as "the
    taxonomy moved". That is the single worst answer this harness can give, because it is specific,
    plausible and wrong, and it sends a reviewer looking for a change to the table instead of at the
    reference that could not be read.

    One conversion site for both stages, and it reads the code OFF the exception rather than
    re-deciding it: two places that each decide what a refusal is worth is how one of them
    eventually returns a 1. An UNEXPECTED exception is a 2 for the same reason — "could not read"
    is the honest description of a traceback, and "the data differs" is a claim about the data that
    nothing here is in a position to make. The traceback survives at debug level, because a refusal
    a maintainer cannot reproduce is its own problem.
    """
    try:
        return stage(args)
    except GoldenDiffError as error:
        print(f"REFUSED: {error}")
        return error.exit_code
    except Exception as error:
        logger.debug("the %s stage raised", stage.__name__, exc_info=True)
        print(
            f"REFUSED: {type(error).__name__}: {error}. The harness could not read its own inputs, "
            f"which is exit 2 and never exit 1 — a traceback is not evidence that the taxonomy has "
            f"moved. Re-run --refresh rather than hand-editing the reference or its manifest."
        )
        return EXIT_REFUSED


def _report_stage(args) -> int:
    """The offline stages in order: the age warning, the opt-in revision check, then the diff.

    ``warn_if_aged`` runs on EVERY report and not only under ``--check-revision``. Age is offline
    information — it is a date the manifest already carries — so gating the warning on the networked
    flag meant the one state it describes, a reference nobody has refreshed in a year, was silent in
    exactly the CI run that reads the manifest on every PR.
    """
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None:
        warn_if_aged(manifest)

    if args.check_revision:
        if manifest is None:
            return _refuse(f"--check-revision needs a manifest and {manifest_path} does not exist")
        status = check_revision(manifest, allow_stale=args.allow_stale_revision)
        if status != EXIT_OK:
            return status

    return report(args.reference, args.manifest, args.package, args.waivers, summary=args.summary, as_json=args.as_json)


def _refresh_stage(args) -> int:
    """The refresh half of the CLI. Every refusal it can raise is converted by :func:`_guarded`.

    It raises rather than catching, so the conversion from an exception to an exit code happens in
    exactly one place for both stages.
    """
    revision = args.revision or recorded_revision(args.manifest)
    if revision is None:
        logger.warning(
            "no --revision and no manifest to take one from, so this refresh reads %s's default "
            "branch and the reference will be baselined against whatever is there.",
            args.repo_id,
        )
    reference = refresh(
        args.repo_id,
        revision,
        workers=args.workers,
        block_size=args.block_size,
        shards=args.shards,
        reference_path=args.reference,
        manifest_path=args.manifest,
    )
    logger.info("refreshed %d pairs, %d disagreement(s)", reference.manifest["pairs"], len(reference.disagreements))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
