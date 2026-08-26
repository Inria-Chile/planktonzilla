"""
(c) Inria

End-to-end VAL-02 validation for a built FREPJ-only dataset — a hard gate.

Fully network-free (no HF Hub reads, no push, no API): it reads a locally-saved HF
dataset (``load_from_disk``) OR an already-in-memory ``datasets.Dataset`` plus the
committed ``planktonzilla_taxonomy.csv`` and asserts the four VAL-02 properties of
the built FREPJ-only composite:

  1. COUNT reconciliation — 229 distinct classes and ~88,686 images (optionally
     reconciled per-class against the frozen class-dir contract and / or the
     extracted imagefolder file counts).
  2. NON-NULL TAXONOMY — every built class carries a non-null ``proposed_label``
     (closes the silent left-join miss).
  3. METADATA COVERAGE — ``magnification`` / ``site`` present in every row's
     ``custom_metadata`` JSON object, ``Latitude`` / ``Longitude`` for at least the
     ~86.9% floor (rest null by design), and ``timestamp`` ISO ``YYYY-MM-DD`` wherever
     set and set for at least the ~98.1% floor (the upstream sampling dates are
     hand-typed free text — KI-26 — and the build nulls what it cannot read without
     guessing).
  4. OVERLAP + FIDELITY — every built row's ``proposed_label`` + five external-ID
     columns match ``planktonzilla_taxonomy.csv`` for its ``(frepj, Raw_Labels)``
     key, and shared taxa reuse the canonical non-frepj external IDs.

``main()`` renders a markdown report and EXITS NON-ZERO if ANY check fails — the
enforcement that a failing check blocks publish rather than deferring the issue.
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import polars as pl

from planktonzilla.planktonzilla_dataset import constants

# The lookup columns pulled from the taxonomy CSV, ordered exactly as
# ``RedefineDataset._build_lookup`` builds them so external-ID comparison is
# apples-to-apples with the built dataset (which was produced by that same lookup).
_TAXONOMY_COLS = constants.TAXONOMY_RANKS
_EXTRA_COLS = constants.EXTRA_COLS
_ID_STR_COLS = constants.ID_STR_COLS  # already text in the CSV
_ID_NUM_COLS = constants.ID_NUM_COLS  # numeric in the CSV -> text without decimals
_LOOKUP_COLS = (*_TAXONOMY_COLS, *_EXTRA_COLS, *_ID_STR_COLS, *_ID_NUM_COLS)

# ``timestamp`` values must be ISO dates: the build normalizes the hand-typed upstream
# "Sampling date" (KI-26) and never lets a raw value through.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The five external-ID columns compared for overlap fidelity.
_ID_COLS = (*_ID_STR_COLS, *_ID_NUM_COLS)


@dataclass
class CheckResult:
    """A single named VAL-02 check with its pass/fail verdict + observed numbers."""

    name: str
    passed: bool
    observed: str
    expected: str

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class ValidationReport:
    """Structured collection of check results; ``passed`` is the AND of them all."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, observed: str, expected: str) -> None:
        self.checks.append(CheckResult(name=name, passed=bool(passed), observed=str(observed), expected=str(expected)))


def _norm(value):
    """Empty or blank strings become ``None``; everything else is left as is.

    Mirrors ``RedefineDataset._norm`` so blank-vs-None comparisons match the build.
    """
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _cmp_norm(value):
    """Normalise a value to a comparable string (or ``None``).

    The built dataset stores every ID / label as an arrow string, whereas the
    polars-read CSV may infer un-cast str-ID columns as integers. Stringifying both
    sides after ``_norm`` makes the fidelity comparison representation-agnostic
    (apples-to-apples) without loosening a genuine value mismatch.
    """
    value = _norm(value)
    if value is None:
        return None
    return str(value).strip()


def _build_taxonomy_lookup(csv_path) -> dict:
    """Build the ``(Dataset, Raw_Labels) -> {column: value}`` lookup from the CSV.

    Mirrors ``RedefineDataset._build_lookup``: numeric ID columns are normalised to
    decimal-free strings and blank values to ``None``.
    """
    df = pl.read_csv(csv_path)

    for col in _ID_NUM_COLS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False).cast(pl.Utf8).alias(col))

    present = [col for col in _LOOKUP_COLS if col in df.columns]
    keys = zip(df["Dataset"].to_list(), df["Raw_Labels"].to_list())
    rows = df.select(present).to_dicts()

    lookup = {}
    for key, row in zip(keys, rows):
        lookup[key] = {col: _norm(row.get(col)) for col in _LOOKUP_COLS}
    return lookup


def _as_dataset(dataset_or_path) -> datasets.Dataset:
    """Return a single ``datasets.Dataset`` from a path, Dataset or DatasetDict.

    A path is read with ``load_from_disk`` (network-free); a ``DatasetDict`` is
    flattened by concatenating its splits so the checks see every built row.
    """
    if isinstance(dataset_or_path, (str, Path)):
        loaded = datasets.load_from_disk(str(dataset_or_path))
    else:
        loaded = dataset_or_path

    if isinstance(loaded, datasets.DatasetDict):
        parts = [loaded[split] for split in loaded.keys()]
        return datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return loaded


def _col(ds: datasets.Dataset, name: str, n: int) -> list:
    """Return the column as a list, or ``[None] * n`` when it is absent.

    Column access never decodes the image column, so this stays cheap even on the
    real ~88,686-row build.
    """
    if name in ds.column_names:
        return list(ds[name])
    return [None] * n


def _load_class_dirs(class_dirs_tsv) -> dict | None:
    """Parse the frozen class-dir contract TSV into ``{class_dir: n_total}``."""
    if class_dirs_tsv is None:
        return None

    lines = Path(class_dirs_tsv).read_text().splitlines()
    if not lines:
        return {}

    header = lines[0].split("\t")
    idx_total = header.index("n_total") if "n_total" in header else None

    result = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        class_dir = fields[0]
        total = int(fields[idx_total]) if idx_total is not None and idx_total < len(fields) else None
        result[class_dir] = total
    return result


def _count_imagefolder(imagefolder_dir) -> dict | None:
    """Count non-hidden files per class subdir of the extracted imagefolder."""
    if imagefolder_dir is None:
        return None

    base = Path(imagefolder_dir)
    if not base.exists():
        return None

    counts = {}
    for entry in base.iterdir():
        if entry.is_dir():
            counts[entry.name] = sum(1 for f in entry.iterdir() if f.is_file() and not f.name.startswith("."))
    return counts


def _nn_fraction(seq, denom: int) -> float:
    """Fraction of ``seq`` entries that are neither ``None`` nor a blank string."""
    good = sum(1 for value in seq if not (value is None or (isinstance(value, str) and value.strip() == "")))
    return good / denom


def _check_count(report, labels, class_dirs, imagefolder_counts, expected_classes, expected_images, count_tolerance):
    """CHECK 1 — distinct classes + total images reconcile against the contract."""
    built_counts = Counter(labels)
    n_classes = len(built_counts)
    total = len(labels)

    low = expected_images * (1 - count_tolerance)
    high = expected_images * (1 + count_tolerance)
    ok = (n_classes == expected_classes) and (low <= total <= high)

    # Per-class reconciliation against the frozen class-dir contract (n_total).
    if class_dirs is not None:
        if set(built_counts) != set(class_dirs):
            ok = False
        for class_dir, n_total in class_dirs.items():
            if n_total is None:
                continue
            built = built_counts.get(class_dir, 0)
            if not (n_total * (1 - count_tolerance) <= built <= n_total * (1 + count_tolerance)):
                ok = False

    # Reconciliation against the extracted imagefolder (built <= imagefolder; the
    # global corrupt-drop delta must stay within tolerance).
    if imagefolder_counts is not None:
        drop = 0
        img_total = 0
        for class_dir, img_n in imagefolder_counts.items():
            built = built_counts.get(class_dir, 0)
            img_total += img_n
            drop += max(0, img_n - built)
            if built > img_n:
                ok = False
        if img_total and (drop / img_total) > count_tolerance:
            ok = False

    report.add(
        "Count Reconciliation",
        ok,
        f"{n_classes} classes / {total} images",
        f"{expected_classes} classes / {int(low)}-{int(high)} images",
    )


def _check_taxonomy(report, labels, proposed_labels):
    """CHECK 2 — every distinct class has a non-null, non-empty proposed_label."""
    seen = set()
    missing = set()
    for label, proposed in zip(labels, proposed_labels):
        seen.add(label)
        if proposed is None or (isinstance(proposed, str) and proposed.strip() == ""):
            missing.add(label)

    ok = len(missing) == 0
    report.add(
        "Non-null Taxonomy",
        ok,
        f"{len(seen) - len(missing)}/{len(seen)} classes resolved",
        f"{len(seen)}/{len(seen)} classes resolved",
    )


def _custom_metadata_field(custom_metadata, key) -> list:
    """Pull one key out of every row's ``custom_metadata`` JSON object (None when absent)."""
    values = []
    for raw in custom_metadata:
        try:
            obj = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            obj = {}
        values.append(obj.get(key) if isinstance(obj, dict) else None)
    return values


def _check_metadata(report, n, magnification, site):
    """CHECK 3a — magnification / site present in ``custom_metadata`` for 100% of rows."""
    denom = n or 1
    mag_frac = _nn_fraction(magnification, denom)
    site_frac = _nn_fraction(site, denom)

    ok = (n > 0) and mag_frac == 1.0 and site_frac == 1.0
    report.add("Metadata Coverage", ok, f"mag {mag_frac:.1%} / site {site_frac:.1%}", "mag+site 100%")


def _check_timestamp(report, n, timestamps, timestamp_floor, timestamp_tolerance):
    """CHECK 3c — ``timestamp`` is ISO ``YYYY-MM-DD`` wherever set, and set for >= the floor.

    The upstream sampling dates are hand-typed (KI-26); the build normalizes them and
    nulls what it cannot read without guessing. Shape is a hard 100% — one malformed
    value fails — while coverage is a floor (98.1% on the real tables).
    """
    denom = n or 1
    present = [t for t in timestamps if t is not None and str(t).strip() != ""]
    malformed = [t for t in present if not _ISO_DATE_RE.match(str(t))]
    report.add("Timestamp Shape", (n > 0) and not malformed, f"{len(malformed)} malformed of {len(present)} set", "0 malformed")

    coverage = len(present) / denom
    floor = timestamp_floor - timestamp_tolerance
    report.add("Timestamp Coverage", (n > 0) and coverage >= floor, f"{coverage:.1%}", f">= {floor:.1%}")


def _check_latlon(report, n, latitude, longitude, latlon_floor, latlon_tolerance):
    """CHECK 3b — Latitude / Longitude present for at least the ~86.9% floor."""
    denom = n or 1
    latlon_nn = sum(1 for lat, lon in zip(latitude, longitude) if lat is not None and lon is not None)
    coverage = latlon_nn / denom

    floor = latlon_floor - latlon_tolerance
    ok = (n > 0) and coverage >= floor
    report.add("Lat/Lon Coverage", ok, f"{coverage:.1%}", f">= {floor:.1%}")


def _check_overlap(report, labels, id_data, taxonomy_lookup):
    """CHECK 4 — proposed_label + external IDs match the taxonomy CSV.

    Part A pins the ``(frepj, Raw_Labels)`` join (the build did not drift). Part B
    re-asserts TAX-04 reuse: where a proposed_label is shared with a canonical
    non-frepj CSV row, the frepj external IDs equal the non-frepj values.
    """
    compare_cols = ("proposed_label", *_ID_COLS)

    # One representative value set per built class (all rows in a class share them).
    rep = {}
    for i, label in enumerate(labels):
        if label not in rep:
            rep[label] = {col: id_data[col][i] for col in compare_cols if col in id_data}

    mismatches = set()

    # Part A — fidelity of the (frepj, Raw_Labels) join.
    for label, values in rep.items():
        expected = taxonomy_lookup.get(("frepj", label))
        if expected is None:
            continue  # a missing join is surfaced by the non-null-taxonomy check.
        for col in compare_cols:
            if _cmp_norm(values.get(col)) != _cmp_norm(expected.get(col)):
                mismatches.add(label)
                break

    # Part B — shared proposed_label reuses the canonical non-frepj external IDs.
    canonical = {}
    for (dataset_name, _raw), values in taxonomy_lookup.items():
        if dataset_name == "frepj":
            continue
        proposed = _cmp_norm(values.get("proposed_label"))
        if not proposed:
            continue
        id_tuple = tuple(_cmp_norm(values.get(col)) for col in _ID_COLS)
        canonical.setdefault(proposed.lower(), set()).add(id_tuple)

    for label, values in rep.items():
        proposed = _cmp_norm(values.get("proposed_label"))
        if not proposed:
            continue
        options = canonical.get(proposed.lower())
        if not options or len(options) != 1:
            continue  # unknown / ambiguous canonical -> nothing unambiguous to enforce.
        canon_tuple = next(iter(options))
        frepj_tuple = tuple(_cmp_norm(values.get(col)) for col in _ID_COLS)
        for canon_value, frepj_value in zip(canon_tuple, frepj_tuple):
            if canon_value is not None and frepj_value is not None and canon_value != frepj_value:
                mismatches.add(label)
                break

    ok = len(mismatches) == 0
    report.add(
        "Overlap & Fidelity",
        ok,
        f"{len(rep) - len(mismatches)}/{len(rep)} classes consistent",
        f"{len(rep)}/{len(rep)} classes consistent",
    )


def validate_frepj_dataset(
    dataset_or_path,
    taxonomy_csv,
    class_dirs_tsv=None,
    imagefolder_dir=None,
    expected_classes=229,
    expected_images=88686,
    count_tolerance=0.02,
    latlon_floor=0.869,
    latlon_tolerance=0.01,
    timestamp_floor=0.98,
    timestamp_tolerance=0.005,
) -> ValidationReport:
    """Validate a built FREPJ-only dataset end-to-end (VAL-02) and return a report.

    ``dataset_or_path`` is a saved HF dataset directory (``load_from_disk``) or an
    in-memory ``datasets.Dataset``. ``taxonomy_csv`` is the committed
    ``planktonzilla_taxonomy.csv``. ``class_dirs_tsv`` (the frozen 229-class
    contract) and ``imagefolder_dir`` (the extracted archive) are optional
    per-class count-reconciliation sources.
    """
    ds = _as_dataset(dataset_or_path)
    n = len(ds)

    labels = _col(ds, "original_label", n)
    proposed = _col(ds, "proposed_label", n)
    id_data = {col: _col(ds, col, n) for col in ("proposed_label", *_ID_COLS)}
    custom_metadata = _col(ds, constants.CUSTOM_METADATA_COL, n)
    magnification = _custom_metadata_field(custom_metadata, "magnification")
    site = _custom_metadata_field(custom_metadata, "site")
    timestamps = _col(ds, "timestamp", n)
    latitude = _col(ds, "Latitude", n)
    longitude = _col(ds, "Longitude", n)

    taxonomy_lookup = _build_taxonomy_lookup(taxonomy_csv)
    class_dirs = _load_class_dirs(class_dirs_tsv)
    imagefolder_counts = _count_imagefolder(imagefolder_dir)

    report = ValidationReport()
    _check_count(report, labels, class_dirs, imagefolder_counts, expected_classes, expected_images, count_tolerance)
    _check_taxonomy(report, labels, proposed)
    _check_metadata(report, n, magnification, site)
    _check_latlon(report, n, latitude, longitude, latlon_floor, latlon_tolerance)
    _check_timestamp(report, n, timestamps, timestamp_floor, timestamp_tolerance)
    _check_overlap(report, labels, id_data, taxonomy_lookup)
    return report


def render_report(report: ValidationReport) -> str:
    """Render the report as a markdown table (check, status, observed, expected).

    A fully-passing report contains no ``FAIL`` token anywhere in its body, so the
    gate's ``grep`` count of failing lines is zero.
    """
    lines = [
        "# FREPJ-only Validation Report (VAL-02)",
        "",
        f"**Overall status:** {'PASS' if report.passed else 'FAIL'}",
        "",
        "| Check | Status | Observed | Expected |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {check.name} | {check.status} | {check.observed} | {check.expected} |" for check in report.checks)
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    """CLI hard gate: write the markdown report and exit non-zero on any failure."""
    parser = argparse.ArgumentParser(description="Validate a built FREPJ-only dataset (VAL-02 hard gate).")
    parser.add_argument("--dataset-path", required=True, help="Saved HF dataset dir (load_from_disk).")
    parser.add_argument("--taxonomy", default=str(constants.DEFAULT_TAXONOMY_CSV_FILENAME), help="Taxonomy CSV path.")
    parser.add_argument("--class-dirs", default=None, help="Optional frozen class-dir contract TSV.")
    parser.add_argument("--imagefolder", default=None, help="Optional extracted imagefolder dir.")
    parser.add_argument("--report-out", required=True, help="Markdown report output path.")
    parser.add_argument("--expected-classes", type=int, default=229)
    parser.add_argument("--expected-images", type=int, default=88686)
    parser.add_argument("--count-tolerance", type=float, default=0.02)
    parser.add_argument("--latlon-floor", type=float, default=0.869)
    parser.add_argument("--latlon-tolerance", type=float, default=0.01)
    parser.add_argument("--timestamp-floor", type=float, default=0.98)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.005)
    args = parser.parse_args(argv)

    report = validate_frepj_dataset(
        args.dataset_path,
        args.taxonomy,
        class_dirs_tsv=args.class_dirs,
        imagefolder_dir=args.imagefolder,
        expected_classes=args.expected_classes,
        expected_images=args.expected_images,
        count_tolerance=args.count_tolerance,
        latlon_floor=args.latlon_floor,
        latlon_tolerance=args.latlon_tolerance,
        timestamp_floor=args.timestamp_floor,
        timestamp_tolerance=args.timestamp_tolerance,
    )

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(report))

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
