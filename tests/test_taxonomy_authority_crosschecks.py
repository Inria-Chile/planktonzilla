"""
(c) Inria

Automated EXTERNAL-AUTHORITY verification of `planktonzilla_taxonomy.csv`, run as a test.

`planktonzilla/planktonzilla_dataset/utils/verify_taxonomy_ids.py` splits the verification in
two: a network stage that harvests WoRMS, NCBI Taxonomy and Wikidata into a committed snapshot,
and a pure comparison stage. Only the second runs here, so this suite is network-free like
every other committed test — it reads the CSV, the snapshot and the waiver file, and nothing
else. `test_report_path_makes_no_network_calls` proves that rather than asserting it.

Two kinds of test live here, and the distinction matters:

  * ENGINE tests build tiny synthetic CSV+snapshot pairs and assert the checker fires (and does
    not fire) where it should. They are what makes a green data-state test meaningful.
  * DATA-STATE tests run the checker over the real shipped table. The gate is "no UNWAIVED
    ERROR": the published table is frozen on HuggingFace Hub under the zero-behavioural-drift
    rule, so a genuine defect found here is recorded in `utils/AUTHORITY_WAIVERS.json` with its
    justification rather than silently corrected. Adding a waiver is a reviewed act; a NEW
    discrepancy — from a CSV edit or from an authority revising a name — arrives unwaived and
    turns this suite red.

Refresh the snapshot with:
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxonomy_ids --refresh-snapshot
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=False,
)

import hashlib
import json

import pytest
import requests

from planktonzilla.planktonzilla_dataset.utils import verify_taxonomy_ids as vti

_CSV_PATH = root / "planktonzilla" / "planktonzilla_dataset" / "planktonzilla_taxonomy.csv"

_HEADER = (
    "Dataset,Raw_Labels,Kingdom,Phylum,Class,Order,Family,Genus,Species,"
    "proposed_label,plankton,living,root_class,qualifier,"
    "wikidata_ID,aphia_ID,NCBI_ID,BOLD_ID,ecotaxa_ID"
)


# ── Fixtures over the real shipped data ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rows():
    return vti.read_taxonomy(_CSV_PATH)


@pytest.fixture(scope="module")
def snapshot():
    return vti.load_snapshot()


@pytest.fixture(scope="module")
def waivers():
    return vti.load_waivers()


@pytest.fixture(scope="module")
def findings(rows, snapshot):
    return vti.crosscheck(rows, snapshot)


# ── Engine tests: synthetic CSV + snapshot, one check at a time ─────────────────────────────
def _write_case(tmp_path, csv_row: str, snap: dict):
    """Write a one-row CSV and a matching snapshot, and return the cross-check findings."""
    csv_path = tmp_path / "taxo.csv"
    csv_path.write_text(_HEADER + "\n" + csv_row + "\n", encoding="utf-8")
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(snap), encoding="utf-8")
    return vti.crosscheck(vti.read_taxonomy(csv_path), vti.load_snapshot(snap_path))


def _worms(aphia, name, rank, chain, status="accepted", valid_name=None):
    return {
        aphia: {
            "aphia_id": aphia,
            "scientific_name": name,
            "rank": rank,
            "status": status,
            "unacceptreason": None,
            "valid_aphia_id": aphia,
            "valid_name": valid_name or name,
            "classification": chain,
        }
    }


CLEAN_CHAIN = [["Kingdom", "Animalia"], ["Phylum", "Cnidaria"], ["Class", "Hydrozoa"], ["Family", "Abylidae"]]
CLEAN_ROW = "ds,raw_lbl,animalia,cnidaria,hydrozoa,,abylidae,,,abylidae,True,True,living,full_body,,135336.0,,,"


def test_engine_clean_row_produces_no_findings(tmp_path):
    """A row whose lineage matches the register exactly is silent."""
    out = _write_case(tmp_path, CLEAN_ROW, {"worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN)})
    assert out == [], f"expected silence, got {[f.check for f in out]}"


def test_engine_detects_lineage_contradiction(tmp_path):
    """A rank name absent from the register's chain is an ERROR — the id denotes another organism."""
    row = "ds,raw_lbl,animalia,mollusca,hydrozoa,,abylidae,,,abylidae,True,True,living,full_body,,135336.0,,,"
    out = _write_case(tmp_path, row, {"worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN)})
    hits = [f for f in out if f.check == "worms_lineage_contradiction"]
    assert len(hits) == 1 and hits[0].severity == "ERROR"
    assert hits[0].csv_value == "Phylum=mollusca"


def test_engine_detects_rank_slot_drift(tmp_path):
    """A name the register knows, sitting in the wrong CSV rank column, is a WARN not an ERROR."""
    row = "ds,raw_lbl,animalia,cnidaria,,hydrozoa,abylidae,,,abylidae,True,True,living,full_body,,135336.0,,,"
    out = _write_case(tmp_path, row, {"worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN)})
    hits = [f for f in out if f.check == "worms_rank_slot_drift"]
    assert len(hits) == 1 and hits[0].severity == "WARN"
    assert hits[0].csv_value == "Order=hydrozoa" and "class" in hits[0].authority_value


def test_engine_detects_unresolved_and_unaccepted_ids(tmp_path):
    """A dead AphiaID is an ERROR; a live-but-superseded one is a WARN carrying the current name."""
    missing = _write_case(tmp_path, CLEAN_ROW, {"worms": {}})
    assert [f.check for f in missing] == ["worms_id_unresolved"] and missing[0].severity == "ERROR"

    superseded = _write_case(
        tmp_path,
        CLEAN_ROW,
        {"worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN, status="unaccepted", valid_name="Abylidae")},
    )
    hits = [f for f in superseded if f.check == "worms_status_not_accepted"]
    assert len(hits) == 1 and hits[0].severity == "WARN"


def test_engine_detects_name_mismatch(tmp_path):
    """A label the register does not know under that id is an ERROR."""
    row = "ds,raw_lbl,animalia,cnidaria,hydrozoa,,diphyidae,,,diphyidae,True,True,living,full_body,,135336.0,,,"
    out = _write_case(tmp_path, row, {"worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN)})
    assert "worms_name_mismatch" in {f.check for f in out}


def test_engine_ncbi_higher_ranks_are_info_not_error(tmp_path):
    """NCBI lacks Chromista and Heterokontophyta by design, so their absence must not be an ERROR.

    Genus is different: the two registers share genus names, so a genus missing from an NCBI
    lineage is real evidence the taxid is the wrong organism.
    """
    row = (
        "ds,raw_lbl,chromista,heterokontophyta,bacillariophyceae,chaetocerotanae,chaetocerotaceae,chaetoceros,,"
        "chaetoceros,True,True,living,full_body,,,49237.0,,"
    )
    snap = {
        "ncbi": {
            "49237": {
                "tax_id": "49237",
                "scientific_name": "Chaetoceros",
                "rank": "genus",
                "lineage": [["superkingdom", "Eukaryota"], ["clade", "Sar"], ["class", "Bacillariophyceae"]],
                "other_names": [],
            }
        }
    }
    out = _write_case(tmp_path, row, snap)
    assert not [f for f in out if f.severity == "ERROR"], [f.check for f in out if f.severity == "ERROR"]
    assert {f.severity for f in out if f.csv_value.startswith("Kingdom=")} == {"INFO"}

    wrong_genus = row.replace(",chaetoceros,,chaetoceros,", ",thalassiosira,,thalassiosira,")
    out2 = _write_case(tmp_path, wrong_genus, snap)
    assert "ncbi_lineage_contradiction" in {f.check for f in out2 if f.severity == "ERROR"}


def test_engine_replays_the_wikidata_id_harvest(tmp_path):
    """The CSV's aphia/NCBI/BOLD ids came from these Wikidata claims; divergence must surface."""
    row = "ds,raw_lbl,animalia,cnidaria,hydrozoa,,abylidae,,,abylidae,True,True,living,full_body,Q3386609,999999.0,,,"
    snap = {
        "wikidata": {
            "Q3386609": {
                "qid": "Q3386609",
                "label_en": "Abylidae",
                "taxon_name": "Abylidae",
                "rank_qid": "Q35409",
                "rank": "family",
                "p850_worms": ["135336"],
                "p685_ncbi": [],
                "p3606_bold": [],
            }
        },
        "worms": _worms("999999", "Abylidae", "Family", CLEAN_CHAIN),
    }
    out = _write_case(tmp_path, row, snap)
    hits = [f for f in out if f.check == "wikidata_worms_claim_divergence"]
    assert len(hits) == 1 and hits[0].severity == "WARN"
    assert hits[0].csv_value == "999999" and hits[0].authority_value == "135336"


def test_engine_detects_cross_authority_disagreement(tmp_path):
    """Two registers naming different organisms for one row is the strongest wrong-id evidence."""
    row = "ds,raw_lbl,animalia,cnidaria,hydrozoa,,abylidae,,,abylidae,True,True,living,full_body,,135336.0,6656.0,,"
    snap = {
        "worms": _worms("135336", "Abylidae", "Family", CLEAN_CHAIN),
        "ncbi": {
            "6656": {
                "tax_id": "6656",
                "scientific_name": "Arthropoda",
                "rank": "phylum",
                "lineage": [["kingdom", "Metazoa"]],
                "other_names": [],
            }
        },
    }
    out = _write_case(tmp_path, row, snap)
    assert "authority_name_disagreement" in {f.check for f in out if f.severity == "ERROR"}


def test_engine_detects_identifier_reuse(tmp_path):
    """One authority identifier stamped on two distinct labels is a mapping defect (KI-13 shape)."""
    csv_path = tmp_path / "reuse.csv"
    rows = [
        "ds,a,animalia,cnidaria,hydrozoa,,abylidae,,,abylidae,True,True,living,full_body,,,418941.0,,",
        "ds,b,animalia,cnidaria,hydrozoa,,diphyidae,,,diphyidae,True,True,living,full_body,,,418941.0,,",
    ]
    csv_path.write_text(_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    out = vti.crosscheck(vti.read_taxonomy(csv_path), {"ncbi": {}})
    hits = [f for f in out if f.check == "id_reused_across_taxa"]
    assert len(hits) == 1 and hits[0].csv_value == "abylidae;diphyidae"


def test_engine_normalizers():
    """The CSV/authority casing, `.0` id serialization and WoRMS parentheticals all normalize."""
    assert vti.norm_id("135336.0") == "135336" and vti.norm_id("") == "" and vti.norm_id("Q123") == "Q123"
    assert vti.norm_name("Abylopsis  tetragona") == "abylopsis tetragona"
    assert vti.norm_name("Calanus (Calanus) finmarchicus") == "calanus finmarchicus"
    assert vti.expected_label({"Genus": "Abylopsis", "Species": "tetragona"}) == "abylopsis tetragona"


# ── Snapshot integrity ─────────────────────────────────────────────────────────────────────
def test_snapshot_provenance_is_recorded(snapshot):
    """The snapshot must say what it was built from and when, or it cannot be audited."""
    provenance = snapshot["provenance"]
    for key in ("generated_utc", "tool", "taxonomy_csv_sha256", "taxonomy_csv_rows", "sources"):
        assert provenance.get(key), f"provenance is missing {key}"
    assert set(provenance["sources"]) == {"worms", "ncbi", "wikidata"}
    for source in provenance["sources"].values():
        assert source["endpoint"] and source["requested"] >= 0


def test_snapshot_matches_the_committed_csv(snapshot):
    """A snapshot harvested against a different CSV revision would verify the wrong table."""
    digest = hashlib.sha256(_CSV_PATH.read_bytes()).hexdigest()
    assert snapshot["provenance"]["taxonomy_csv_sha256"] == digest, (
        "authority_snapshot.json was harvested against a different planktonzilla_taxonomy.csv; re-run --refresh-snapshot"
    )


def test_snapshot_covers_every_populated_identifier(rows, snapshot):
    """Every id in the CSV is either resolved in the snapshot or recorded as unresolvable.

    Without this, dropping a record from the snapshot would silently skip its checks.
    """
    ids = vti.distinct_ids(rows)
    for column, authority in vti.ID_COL_AUTHORITY.items():
        resolved = set(snapshot[authority])
        unresolved = set(snapshot["provenance"]["sources"][authority]["unresolved"])
        uncovered = set(ids[column]) - resolved - unresolved
        assert not uncovered, f"{column}: {len(uncovered)} identifier(s) absent from the snapshot, e.g. {sorted(uncovered)[:5]}"


def test_report_path_makes_no_network_calls(rows, snapshot, monkeypatch):
    """The verification stage must be pure, so CI never depends on three external services."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the report path attempted a network call")

    monkeypatch.setattr(requests.Session, "get", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    assert vti.crosscheck(rows, snapshot) is not None


# ── Waiver hygiene ─────────────────────────────────────────────────────────────────────────
def test_every_waiver_is_justified(waivers):
    """A waiver without a reason is an undocumented defect; that is what this file exists against."""
    assert waivers, "AUTHORITY_WAIVERS.json is empty — expected the reviewed findings to be recorded there"
    for finding_id, entry in waivers.items():
        assert len(finding_id) == 12, f"{finding_id} is not a 12-char finding id"
        for key in ("check", "proposed_label", "reason"):
            assert entry.get(key), f"waiver {finding_id} is missing {key}"
        assert len(entry["reason"]) >= 25, f"waiver {finding_id} has a reason too short to be a justification"


def test_no_stale_waivers(findings, waivers):
    """A waiver matching nothing means the finding is gone and the waiver should be deleted."""
    _, _, stale = vti.apply_waivers(findings, waivers)
    assert not stale, f"{len(stale)} waiver(s) no longer match any finding: {stale}"


# ── The gate ───────────────────────────────────────────────────────────────────────────────
def test_no_unwaived_errors(findings, waivers):
    """The CI gate: every ERROR-severity finding is either fixed or explicitly waived."""
    unwaived, _, _ = vti.apply_waivers(findings, waivers)
    errors = [f for f in unwaived if f.severity == "ERROR"]
    rendered = [f"{f.finding_id} {f.check} {f.proposed_label} csv={f.csv_value} authority={f.authority_value}" for f in errors]
    assert not errors, "unwaived ERROR findings:\n  " + "\n  ".join(rendered)
