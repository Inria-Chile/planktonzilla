"""
(c) Inria

Network-free tests for the FREPJ-only validation module (Plan 19-01, VAL-02).

They drive ``validate_frepj_dataset`` over tiny in-memory ``datasets.Dataset``
fixtures (scalar columns only — no images, no HF Hub, no API) plus a synthetic
taxonomy CSV, and assert the PASS path plus each of the behaviour-block failure
modes:

  - all checks PASS on a well-formed built dataset,
  - a null proposed_label -> Non-null Taxonomy FAIL (the silent-join guard),
  - a row missing magnification -> Metadata Coverage FAIL (100% required),
  - lat/lon coverage below the ~85.9% floor -> Lat/Lon Coverage FAIL,
  - an external ID drifting from the CSV -> Overlap & Fidelity FAIL,
  - a class-count / total-image mismatch -> Count Reconciliation FAIL,
  - main() writes a markdown report and exits non-zero on any failure (the gate).

Small ``expected_classes`` / ``expected_images`` overrides exercise the real check
logic without needing 88,686 rows.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import copy

import datasets
import pytest

from planktonzilla.planktonzilla_dataset import frepj_validate as fv

_TAXO_HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,root_class,qualifier,"
    "wikidata_ID,ecotaxa_ID,aphia_ID,NCBI_ID,BOLD_ID"
)

# Per-class taxonomy + external IDs shared by the built fixture and the CSV.
_CLASS_INFO = {
    "classA": {
        "proposed_label": "copepoda",
        "wikidata_ID": "Q100",
        "ecotaxa_ID": "10",
        "aphia_ID": "111",
        "NCBI_ID": "222",
        "BOLD_ID": "333",
    },
    "classB": {
        "proposed_label": "cladocera",
        "wikidata_ID": "Q200",
        "ecotaxa_ID": "20",
        "aphia_ID": None,
        "NCBI_ID": None,
        "BOLD_ID": None,
    },
    "classC": {
        "proposed_label": "rotifera",
        "wikidata_ID": None,
        "ecotaxa_ID": None,
        "aphia_ID": "444",
        "NCBI_ID": None,
        "BOLD_ID": None,
    },
}
_CLASSES = ("classA", "classB", "classC")


def _write_taxo(path):
    """Write a synthetic taxonomy CSV whose frepj rows match ``_CLASS_INFO``."""
    lines = [_TAXO_HEADER]
    for cls in _CLASSES:
        info = _CLASS_INFO[cls]
        lines.append(
            ",".join(
                [
                    "frepj",
                    cls,
                    "Animalia",
                    "Arthropoda",
                    "",
                    "",
                    "",
                    "",
                    "",
                    info["proposed_label"] or "",
                    "True",
                    "zoo",
                    "",
                    info["wikidata_ID"] or "",
                    info["ecotaxa_ID"] or "",
                    info["aphia_ID"] or "",
                    info["NCBI_ID"] or "",
                    info["BOLD_ID"] or "",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _good_dict(n_per_class=40, latlon_cov=1.0):
    """Build the column dict for a well-formed built FREPJ-only dataset."""
    columns = {
        key: []
        for key in (
            "original_label",
            "proposed_label",
            "wikidata_ID",
            "ecotaxa_ID",
            "aphia_ID",
            "NCBI_ID",
            "BOLD_ID",
            "custom_metadata",
            "timestamp",
            "Latitude",
            "Longitude",
        )
    }
    for cls in _CLASSES:
        info = _CLASS_INFO[cls]
        for _ in range(n_per_class):
            columns["original_label"].append(cls)
            columns["proposed_label"].append(info["proposed_label"])
            for id_col in ("wikidata_ID", "ecotaxa_ID", "aphia_ID", "NCBI_ID", "BOLD_ID"):
                columns[id_col].append(info[id_col])
            columns["custom_metadata"].append('{"magnification": "40", "site": "akigawadam"}')
            columns["timestamp"].append("2018-03-15")

    n = len(columns["original_label"])
    n_latlon = round(latlon_cov * n)
    for i in range(n):
        if i < n_latlon:
            columns["Latitude"].append(35.4)
            columns["Longitude"].append(137.4)
        else:
            columns["Latitude"].append(None)
            columns["Longitude"].append(None)
    return columns


def _to_ds(columns):
    return datasets.Dataset.from_dict(columns)


def _validate(columns, taxo_path, **kwargs):
    defaults = dict(expected_classes=3, expected_images=120, count_tolerance=0.02, latlon_floor=0.869, latlon_tolerance=0.01)
    defaults.update(kwargs)
    return fv.validate_frepj_dataset(_to_ds(columns), str(taxo_path), **defaults)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_all_checks_pass(tmp_path):
    """A well-formed built dataset passes every VAL-02 check."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    report = _validate(_good_dict(), taxo)
    assert report.passed, [(c.name, c.observed) for c in report.checks if not c.passed]
    assert {c.name for c in report.checks} == {
        "Count Reconciliation",
        "Non-null Taxonomy",
        "Metadata Coverage",
        "Lat/Lon Coverage",
        "Timestamp Shape",
        "Timestamp Coverage",
        "Overlap & Fidelity",
    }


def test_null_taxonomy_fails(tmp_path):
    """A class with a null proposed_label fails the non-null-taxonomy gate."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    columns = _good_dict()
    for i, label in enumerate(columns["original_label"]):
        if label == "classC":
            columns["proposed_label"][i] = None
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Non-null Taxonomy").passed


def test_missing_metadata_fails(tmp_path):
    """A single row whose custom_metadata lacks magnification fails the 100% metadata-coverage gate."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    columns = _good_dict()
    columns["custom_metadata"][0] = '{"site": "akigawadam"}'
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Metadata Coverage").passed
    # mag/site/date and lat/lon are distinct checks — lat/lon is unaffected.
    assert _check(report, "Lat/Lon Coverage").passed


def test_low_latlon_coverage_fails(tmp_path):
    """Lat/lon coverage below the ~85.9% floor fails the lat/lon-coverage gate."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    # 83.3% coverage (100/120) is below latlon_floor - latlon_tolerance = 85.9%.
    columns = _good_dict(latlon_cov=100 / 120)
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Lat/Lon Coverage").passed
    # The 100%-required metadata check is unaffected.
    assert _check(report, "Metadata Coverage").passed


def test_overlap_id_drift_fails(tmp_path):
    """A built external ID drifting from the CSV fails the overlap/fidelity gate."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    columns = _good_dict()
    for i, label in enumerate(columns["original_label"]):
        if label == "classA":
            columns["aphia_ID"][i] = "999999"  # CSV says 111 for classA.
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Overlap & Fidelity").passed


def test_class_count_mismatch_fails(tmp_path):
    """A distinct-class count != expected fails the count-reconciliation gate."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    report = _validate(_good_dict(), taxo, expected_classes=5)
    assert not report.passed
    assert not _check(report, "Count Reconciliation").passed


def test_total_image_count_out_of_band_fails(tmp_path):
    """A total-image count outside the tolerance band fails count reconciliation."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    report = _validate(_good_dict(), taxo, expected_images=100000)
    assert not report.passed
    assert not _check(report, "Count Reconciliation").passed


def test_class_dirs_contract_reconciliation_passes(tmp_path):
    """Per-class counts reconcile against a frozen class-dir contract TSV."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    tsv = tmp_path / "class_dirs.tsv"
    tsv.write_text("class_dir\tn_total\nclassA\t40\nclassB\t40\nclassC\t40\n")
    report = _validate(_good_dict(), taxo, class_dirs_tsv=str(tsv))
    assert report.passed, [(c.name, c.observed) for c in report.checks if not c.passed]


def test_imagefolder_reconciliation_flags_excess_drop(tmp_path):
    """A too-large corrupt-drop delta vs the imagefolder fails count reconciliation."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    # Imagefolder holds far more files than the built rows -> drop fraction > tolerance.
    imagefolder = tmp_path / "imagefolder"
    for cls in _CLASSES:
        class_dir = imagefolder / cls
        class_dir.mkdir(parents=True)
        for i in range(80):  # built has 40/class -> 50% dropped, well over 2%.
            (class_dir / f"{i}.jpg").write_bytes(b"x")
    report = _validate(_good_dict(), taxo, imagefolder_dir=str(imagefolder))
    assert not report.passed
    assert not _check(report, "Count Reconciliation").passed


def test_load_from_disk_path_input(tmp_path):
    """validate_frepj_dataset accepts a saved HF dataset path (load_from_disk)."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    ds_path = tmp_path / "built"
    _to_ds(_good_dict()).save_to_disk(str(ds_path))
    report = fv.validate_frepj_dataset(str(ds_path), str(taxo), expected_classes=3, expected_images=120)
    assert report.passed, [(c.name, c.observed) for c in report.checks if not c.passed]


def test_main_writes_report_and_exits_zero_on_pass(tmp_path):
    """main() writes a markdown report with no FAIL token and exits 0 when all pass."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    ds_path = tmp_path / "built"
    _to_ds(_good_dict()).save_to_disk(str(ds_path))
    report_out = tmp_path / "report.md"

    argv = [
        "--dataset-path",
        str(ds_path),
        "--taxonomy",
        str(taxo),
        "--report-out",
        str(report_out),
        "--expected-classes",
        "3",
        "--expected-images",
        "120",
    ]
    with pytest.raises(SystemExit) as exc:
        fv.main(argv)
    assert exc.value.code == 0

    text = report_out.read_text()
    assert "PASS" in text
    # Mirror the gate's grep: a passing report has no "fail" in any non-# line.
    body = [line for line in text.splitlines() if not line.startswith("#")]
    assert not any("fail" in line.lower() for line in body)


def test_main_exits_nonzero_on_failure(tmp_path):
    """main() exits 1 and renders a FAIL when any check fails (the hard gate)."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    ds_path = tmp_path / "built"
    _to_ds(_good_dict()).save_to_disk(str(ds_path))
    report_out = tmp_path / "report.md"

    argv = [
        "--dataset-path",
        str(ds_path),
        "--taxonomy",
        str(taxo),
        "--report-out",
        str(report_out),
        "--expected-classes",
        "3",
        "--expected-images",
        "100000",  # total 120 is far outside the band.
    ]
    with pytest.raises(SystemExit) as exc:
        fv.main(argv)
    assert exc.value.code == 1
    assert "FAIL" in report_out.read_text()


def test_report_is_structured(tmp_path):
    """The returned report exposes named checks with a boolean pass verdict."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    report = _validate(_good_dict(), taxo)
    # Deep-copy proves the checks are plain data, not tied to the live dataset.
    snapshot = copy.deepcopy(report.checks)
    assert all(isinstance(c.passed, bool) and c.name and c.status in ("PASS", "FAIL") for c in snapshot)


# --- timestamp: ISO shape is a hard 100%, coverage is a floor (KI-26) ------------------


def test_malformed_timestamp_fails(tmp_path):
    """One non-ISO timestamp — a raw upstream value leaking through — fails Timestamp Shape."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    columns = _good_dict()
    columns["timestamp"][0] = "2018.03.15"
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Timestamp Shape").passed
    # Shape and coverage are distinct checks — the value is present, so coverage is still 100%.
    assert _check(report, "Timestamp Coverage").passed


def test_timestamp_coverage_below_floor_fails(tmp_path):
    """Coverage under the floor fails; the floor is a parameter, so a lower one accepts the same build."""
    taxo = tmp_path / "taxo.csv"
    _write_taxo(taxo)
    columns = _good_dict()
    n = len(columns["timestamp"])
    for i in range(int(n * 0.05)):
        columns["timestamp"][i] = None  # 95% coverage, under the 98% (-0.5%) default floor
    report = _validate(columns, taxo)
    assert not report.passed
    assert not _check(report, "Timestamp Coverage").passed
    assert _check(report, "Timestamp Shape").passed
    assert _check(_validate(columns, taxo, timestamp_floor=0.9), "Timestamp Coverage").passed
