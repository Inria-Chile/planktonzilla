"""
(c) Inria

verify_taxonomy_ids.py
======================
Automated EXTERNAL-AUTHORITY verification of ``planktonzilla_taxonomy.csv`` — the table
that translates each source dataset's raw labels into the planktonzilla taxonomy.

Why this exists
---------------
``extract_taxon_ids.py`` populated ``aphia_ID`` / ``NCBI_ID`` / ``BOLD_ID`` by reading
Wikidata claims (P850 / P685 / P3606) off the Qcode it had matched for each taxon. Nothing
downstream ever asked WoRMS or NCBI whether the ID they were handed denotes the organism the
row claims. That is what this module does: it goes to each authority directly and checks the
identifier against the row's ``proposed_label`` and its seven rank columns.

Two-stage design, so the committed tests stay network-free
----------------------------------------------------------
Stage 1 — ``--refresh-snapshot`` (network, occasional):
    Resolves every distinct populated identifier in the CSV against its authority and writes
    a distilled snapshot JSON next to this module:

      * WoRMS    ``AphiaRecordByAphiaID`` (name, rank, status, valid_AphiaID/valid_name)
                 + ``AphiaClassificationByAphiaID`` (the full ranked ancestor chain)
      * NCBI     ``efetch db=taxonomy`` over E-utilities HTTP, batched (name, rank, LineageEx)
      * Wikidata ``wbgetentities``, batched (label, P225 taxon name, P105 rank, P850/P685/P3606)

Stage 2 — ``--report`` (network-free, what CI runs):
    Cross-checks the CSV against the committed snapshot and emits findings. Pure function of
    two files on disk, so it is deterministic and reviewable in a PR diff.

The lineage test
----------------
Comparing rank slot against rank slot would drown in false positives, because WoRMS, NCBI and
the CSV do not use the same intermediate ranks. Instead, for each populated CSV rank cell the
check asks where that name sits in the authority's ancestor chain:

    * present at the same rank                  -> agreement
    * present in the chain at a DIFFERENT rank  -> ``rank_slot_drift`` (the KI-8 shape)
    * absent from the chain entirely            -> ``lineage_contradiction``

Only the third case is evidence that the identifier denotes a different organism.

Usage:
    # refresh the committed snapshot (needs network + NCBI_EMAIL)
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxonomy_ids --refresh-snapshot

    # verify the CSV against the committed snapshot (no network)
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxonomy_ids --report

    # write the findings table somewhere else / restrict to one authority
    python -m ...verify_taxonomy_ids --report --findings-csv /tmp/findings.csv --authority worms
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
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from planktonzilla.planktonzilla_dataset.constants import (
    DEFAULT_TAXONOMY_CSV_FILENAME,
    ID_NUM_COLS,
    ID_STR_COLS,
    TAXONOMY_RANKS,
)
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DEFAULT_SNAPSHOT_PATH = HERE / "authority_snapshot.json"
DEFAULT_WAIVERS_PATH = HERE / "AUTHORITY_WAIVERS.json"
DEFAULT_FINDINGS_CSV = HERE / "authority_findings.csv"

# ── Endpoints ──────────────────────────────────────────────────────────────────────────────
WORMS_REST = "https://www.marinespecies.org/rest"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Wikimedia's robot policy requires a descriptive, non-browser user agent that identifies the
# client and gives a contact point. https://w.wiki/4wJS
USER_AGENT = "planktonzilla-taxonomy-verifier/1.0 (+https://github.com/Inria-Chile/planktonzilla)"

# Politeness. WoRMS is a single-record REST service with no batch endpoint, so the 746 distinct
# AphiaIDs cost two requests each; a small thread pool keeps that to a few minutes without
# hammering the host. Wikidata and NCBI both batch, so they need no concurrency at all.
WORMS_WORKERS = 4
WORMS_MAX_RETRIES = 4
WIKIDATA_BATCH = 50
NCBI_BATCH = 200

# ── Column groups ──────────────────────────────────────────────────────────────────────────
RANKS = tuple(TAXONOMY_RANKS)
KEY_COLS = ("Dataset", "Raw_Labels")
ID_COLS = (*ID_STR_COLS, *ID_NUM_COLS)
# The authority behind each ID column. ``BOLD_ID`` and ``ecotaxa_ID`` have no verification
# authority wired up here: BOLD's taxonomy browser exposes no stable by-id JSON record, and
# EcoTaxa category ids are a per-project UI vocabulary, not a nomenclatural authority. Both are
# still checked INDIRECTLY, through Wikidata's P3606 round-trip for BOLD.
ID_COL_AUTHORITY = {"aphia_ID": "worms", "NCBI_ID": "ncbi", "wikidata_ID": "wikidata"}

# Severity ladder. ERROR = evidence the identifier denotes a different organism than the row
# claims. WARN = the identifier is right but something about it has moved (nomenclature
# superseded, rank slot drifted, a Wikidata claim was edited away). INFO = the authority simply
# has nothing to say, which for a freshwater taxon in a marine register is expected.
SEVERITIES = ("ERROR", "WARN", "INFO")

# Column order of the exported findings table.
FINDINGS_COLUMNS = (
    "finding_id",
    "severity",
    "authority",
    "check",
    "proposed_label",
    "id_column",
    "id_value",
    "csv_value",
    "authority_value",
    "n_rows",
    "datasets",
    "detail",
)

_WS = re.compile(r"\s+")
_FLOAT_INT = re.compile(r"^(\d+)\.0+$")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


@dataclass
class Finding:
    """One verification finding, attributable to a taxon and a check.

    Attributes:
        check: Stable check name, e.g. ``worms_lineage_contradiction``.
        severity: One of ``SEVERITIES``.
        authority: ``worms`` / ``ncbi`` / ``wikidata``.
        proposed_label: The planktonzilla label the finding is about.
        id_column: CSV column holding the identifier under test.
        id_value: The identifier as normalized for lookup (no ``.0`` suffix).
        csv_value: What the CSV asserts.
        authority_value: What the authority says instead.
        detail: Free-text amplification, safe to change without breaking waivers.
        datasets: Source datasets whose rows carry this taxon.
        n_rows: How many CSV rows the finding touches.
    """

    check: str
    severity: str
    authority: str
    proposed_label: str
    id_column: str
    id_value: str
    csv_value: str
    authority_value: str
    detail: str = ""
    datasets: tuple[str, ...] = ()
    n_rows: int = 0

    @property
    def finding_id(self) -> str:
        """Stable 12-hex-char identity for waiver matching.

        Deliberately excludes ``detail``, ``datasets`` and ``n_rows``: a waiver should survive
        prose edits and the arrival of a new source dataset carrying an already-known taxon,
        but must NOT survive a change in what the authority actually says.
        """
        payload = "|".join((self.check, self.authority, self.proposed_label, self.id_column, self.id_value, self.csv_value))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ── Normalization helpers ──────────────────────────────────────────────────────────────────
def norm_name(value: str | None) -> str:
    """Normalize a taxon name for comparison across the CSV and the authorities.

    The CSV lowercases every taxonomic column while the authorities capitalize; WoRMS also
    appends the authority citation to some names and parenthesizes subgenera. Strip all of
    that so only the nomenclatural content is compared.

    Args:
        value: Raw name from either side, possibly ``None``.

    Returns:
        Lowercased, whitespace-collapsed name with parentheticals removed; ``""`` for ``None``.
    """
    if not value:
        return ""
    return _WS.sub(" ", _PARENTHETICAL.sub("", str(value))).strip().lower()


def norm_id(value: str | None) -> str:
    """Normalize an identifier cell to its canonical string form.

    ``aphia_ID`` / ``NCBI_ID`` / ``BOLD_ID`` are stored as ``"135336.0"`` (KI-12); authorities
    speak in plain integers.

    Args:
        value: Raw identifier cell.

    Returns:
        Decimal-free identifier string, or ``""`` when the cell is blank.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    match = _FLOAT_INT.match(text)
    return match.group(1) if match else text


def deepest_rank(row: dict) -> tuple[str, str]:
    """Return the deepest populated rank of a CSV row and the name in it.

    Args:
        row: A CSV row as a mapping of column name to string.

    Returns:
        ``(rank_name, taxon_name)``; ``("", "")`` when no rank column is populated, which is
        the normal state of ``artefact`` / ``detritus`` rows.
    """
    for rank in reversed(RANKS):
        value = norm_name(row.get(rank))
        if value:
            return rank, value
    return "", ""


def expected_label(row: dict) -> str:
    """The ``proposed_label`` implied by a row's rank columns.

    Species rows carry the epithet alone in ``Species``, so the label is the binomial.

    Args:
        row: A CSV row as a mapping of column name to string.

    Returns:
        Normalized implied label, or ``""`` for rows with no populated rank.
    """
    rank, value = deepest_rank(row)
    if rank == "Species":
        return norm_name(f"{row.get('Genus', '')} {row.get('Species', '')}")
    return value


# ── Stage 1: authority harvest ─────────────────────────────────────────────────────────────
def _session() -> requests.Session:
    """Build a requests session carrying the policy-compliant user agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _get_with_retry(session: requests.Session, url: str, **kwargs) -> requests.Response | None:
    """GET with bounded exponential backoff on rate limits and server errors.

    Args:
        session: Session to issue the request on.
        url: Absolute URL.
        **kwargs: Passed through to ``session.get``.

    Returns:
        The response, or ``None`` when every attempt failed at the transport level.
    """
    delay = 1.0
    for attempt in range(WORMS_MAX_RETRIES):
        try:
            response = session.get(url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            logger.warning("transport error on %s (attempt %d): %s", url, attempt + 1, exc)
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue
        return response
    return None


def fetch_worms(aphia_ids: list[str]) -> tuple[dict, list[str]]:
    """Resolve AphiaIDs against WoRMS.

    Two calls per id: the record (name, rank, nomenclatural status, valid-name redirect) and
    the classification (the full ranked ancestor chain used by the lineage test).

    Args:
        aphia_ids: Distinct AphiaIDs as decimal-free strings.

    Returns:
        ``(records, unresolved)`` where ``records`` maps AphiaID to the distilled record and
        ``unresolved`` lists the ids the register returned no record for.
    """
    session = _session()
    records: dict[str, dict] = {}
    unresolved: list[str] = []

    def flatten(node: dict | None) -> list[list[str]]:
        chain: list[list[str]] = []
        while isinstance(node, dict) and node.get("scientificname"):
            chain.append([node.get("rank") or "", node["scientificname"]])
            node = node.get("child")
        return chain

    def one(aphia_id: str) -> tuple[str, dict | None]:
        record = _get_with_retry(session, f"{WORMS_REST}/AphiaRecordByAphiaID/{aphia_id}")
        if record is None or record.status_code != 200:
            return aphia_id, None
        try:
            payload = record.json()
        except ValueError:
            return aphia_id, None
        if not isinstance(payload, dict):
            return aphia_id, None
        classification = _get_with_retry(session, f"{WORMS_REST}/AphiaClassificationByAphiaID/{aphia_id}")
        chain: list[list[str]] = []
        if classification is not None and classification.status_code == 200:
            try:
                chain = flatten(classification.json())
            except ValueError:
                chain = []
        return aphia_id, {
            "aphia_id": str(payload.get("AphiaID", aphia_id)),
            "scientific_name": payload.get("scientificname"),
            "rank": payload.get("rank"),
            "status": payload.get("status"),
            "unacceptreason": payload.get("unacceptreason"),
            "valid_aphia_id": None if payload.get("valid_AphiaID") is None else str(payload["valid_AphiaID"]),
            "valid_name": payload.get("valid_name"),
            "classification": chain,
        }

    with ThreadPoolExecutor(max_workers=WORMS_WORKERS) as pool:
        for done, (aphia_id, distilled) in enumerate(pool.map(one, aphia_ids), start=1):
            if distilled is None:
                unresolved.append(aphia_id)
            else:
                records[aphia_id] = distilled
            if done % 100 == 0:
                logger.info("WoRMS %d/%d resolved", done, len(aphia_ids))
    return records, unresolved


# NCBI ``OtherNames`` child tags that carry a SCIENTIFIC alternative for the taxon. Vernacular
# tags (``CommonName``, ``GenbankCommonName``) are deliberately excluded: accepting them would
# let a common name mask a genuine scientific-name mismatch.
NCBI_SYNONYM_TAGS = (
    "Synonym",
    "GenbankSynonym",
    "EquivalentName",
    "Anamorph",
    "GenbankAnamorph",
    "Teleomorph",
    "Includes",
    "Misspelling",
    "Misnomer",
)


def fetch_ncbi(tax_ids: list[str], email: str | None = None) -> tuple[dict, list[str]]:
    """Resolve NCBI taxonomy ids via batched E-utilities ``efetch``.

    Talks to E-utilities over plain HTTP rather than through ``Bio.Entrez`` so the harvest does
    not hard-require a contact address: NCBI only *recommends* one, and it is sent when
    available. Batched at ``NCBI_BATCH`` ids per request, within the unauthenticated rate limit.

    Args:
        tax_ids: Distinct NCBI taxids as decimal-free strings.
        email: Contact address to identify the requests; falls back to ``NCBI_EMAIL``. Omitted
            entirely when neither is set.

    Returns:
        ``(records, unresolved)`` mapping taxid to distilled record, plus the ids NCBI had no
        taxon for.
    """
    import xml.etree.ElementTree as ET

    session = _session()
    base_params = {"db": "taxonomy", "retmode": "xml", "tool": "planktonzilla-taxonomy-verifier"}
    resolved_email = email or os.environ.get("NCBI_EMAIL")
    if resolved_email:
        base_params["email"] = resolved_email
    if os.environ.get("NCBI_API_KEY"):
        base_params["api_key"] = os.environ["NCBI_API_KEY"]

    records: dict[str, dict] = {}
    for start in range(0, len(tax_ids), NCBI_BATCH):
        batch = tax_ids[start : start + NCBI_BATCH]
        response = _get_with_retry(session, NCBI_EFETCH, params={**base_params, "id": ",".join(batch)})
        if response is None or response.status_code != 200:
            logger.warning("NCBI batch starting at %s failed", batch[0])
            continue
        try:
            tree = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning("NCBI batch starting at %s returned unparseable XML: %s", batch[0], exc)
            continue
        for taxon in tree.findall("Taxon"):
            tax_id = (taxon.findtext("TaxId") or "").strip()
            if not tax_id:
                continue
            lineage_node = taxon.find("LineageEx")
            lineage = []
            if lineage_node is not None:
                lineage.extend(
                    [(node.findtext("Rank") or "").strip(), (node.findtext("ScientificName") or "").strip()]
                    for node in lineage_node.findall("Taxon")
                )
            other = taxon.find("OtherNames")
            synonyms: set[str] = set()
            if other is not None:
                for tag in NCBI_SYNONYM_TAGS:
                    synonyms.update((node.text or "").strip() for node in other.findall(tag) if (node.text or "").strip())
                for node in other.findall("Name"):
                    display = (node.findtext("DispName") or "").strip()
                    if display:
                        synonyms.add(display)
            records[tax_id] = {
                "tax_id": tax_id,
                "scientific_name": (taxon.findtext("ScientificName") or "").strip(),
                "rank": (taxon.findtext("Rank") or "").strip(),
                "lineage": lineage,
                "other_names": sorted(synonyms),
            }
        logger.info("NCBI %d/%d resolved", min(start + NCBI_BATCH, len(tax_ids)), len(tax_ids))
        time.sleep(0.4)
    return records, [t for t in tax_ids if t not in records]


def fetch_wikidata(qids: list[str]) -> tuple[dict, list[str]]:
    """Resolve Wikidata Qcodes via batched ``wbgetentities``.

    Captures the English label, the P225 taxon name, the P105 taxon rank and the three
    cross-reference claims the CSV's ID columns were originally harvested from, so the harvest
    can be replayed against today's Wikidata.

    Args:
        qids: Distinct Qcodes.

    Returns:
        ``(records, unresolved)`` mapping Qcode to distilled record, plus missing Qcodes.
    """
    session = _session()
    records: dict[str, dict] = {}

    def claim_values(claims: dict, prop: str) -> list[str]:
        out = []
        for claim in claims.get(prop, []):
            snak = claim.get("mainsnak", {})
            value = snak.get("datavalue", {}).get("value")
            if isinstance(value, dict):
                value = value.get("id")
            if value is not None:
                out.append(str(value))
        return out

    def harvest(ids: list[str]) -> dict:
        out: dict[str, dict] = {}
        for start in range(0, len(ids), WIKIDATA_BATCH):
            batch = ids[start : start + WIKIDATA_BATCH]
            response = _get_with_retry(
                session,
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "format": "json",
                    "props": "labels|claims",
                    "languages": "en",
                },
            )
            if response is None or response.status_code != 200:
                logger.warning("Wikidata batch starting at %s failed", batch[0])
                continue
            for qid, entity in response.json().get("entities", {}).items():
                if "missing" in entity:
                    continue
                claims = entity.get("claims", {})
                out[qid] = {
                    "qid": qid,
                    "label_en": entity.get("labels", {}).get("en", {}).get("value"),
                    "taxon_name": (claim_values(claims, "P225") or [None])[0],
                    "rank_qid": (claim_values(claims, "P105") or [None])[0],
                    "p850_worms": claim_values(claims, "P850"),
                    "p685_ncbi": claim_values(claims, "P685"),
                    "p3606_bold": claim_values(claims, "P3606"),
                }
            time.sleep(0.2)
        return out

    records = harvest(qids)
    # Second pass: resolve the P105 rank Qcodes to English labels rather than hardcoding the
    # species/genus/family/... Qcode table, so a rank we have not seen before still reads.
    rank_qids = sorted({r["rank_qid"] for r in records.values() if r.get("rank_qid")})
    rank_labels = {qid: rec.get("label_en") for qid, rec in harvest(rank_qids).items()}
    for record in records.values():
        record["rank"] = rank_labels.get(record.get("rank_qid") or "")
    return records, [q for q in qids if q not in records]


def read_taxonomy(csv_path: Path) -> list[dict]:
    """Read the taxonomy CSV as a list of all-string row dicts.

    Uses the same polars reader family as the generation path, with every column forced to
    string so ID formatting is compared exactly as committed.

    Args:
        csv_path: Path to ``planktonzilla_taxonomy.csv``.

    Returns:
        One dict per CSV row.
    """
    frame = pl.read_csv(csv_path, infer_schema_length=0)
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in frame.to_dicts()]


def distinct_ids(rows: list[dict]) -> dict[str, list[str]]:
    """Collect the distinct populated identifiers per ID column.

    Args:
        rows: CSV rows from :func:`read_taxonomy`.

    Returns:
        Mapping of ID column name to a sorted list of decimal-free identifier strings.
    """
    buckets: dict[str, set[str]] = {col: set() for col in ID_COL_AUTHORITY}
    for row in rows:
        for col in ID_COL_AUTHORITY:
            value = norm_id(row.get(col))
            if value:
                buckets[col].add(value)
    return {col: sorted(values, key=lambda v: (len(v), v)) for col, values in buckets.items()}


def build_snapshot(csv_path: Path, email: str | None = None) -> dict:
    """Harvest all three authorities for the identifiers present in the CSV.

    Args:
        csv_path: Path to the taxonomy CSV.
        email: Contact address for NCBI Entrez.

    Returns:
        The snapshot dict, ready for :func:`write_snapshot`.
    """
    rows = read_taxonomy(csv_path)
    ids = distinct_ids(rows)
    logger.info("distinct ids to resolve: %s", {k: len(v) for k, v in ids.items()})

    worms, worms_missing = fetch_worms(ids["aphia_ID"])
    ncbi, ncbi_missing = fetch_ncbi(ids["NCBI_ID"], email=email)
    wikidata, wikidata_missing = fetch_wikidata(ids["wikidata_ID"])

    return {
        "provenance": {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": "planktonzilla.planktonzilla_dataset.utils.verify_taxonomy_ids",
            "taxonomy_csv": csv_path.name,
            "taxonomy_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "taxonomy_csv_rows": len(rows),
            "sources": {
                "worms": {
                    "endpoint": f"{WORMS_REST}/AphiaRecordByAphiaID + AphiaClassificationByAphiaID",
                    "requested": len(ids["aphia_ID"]),
                    "resolved": len(worms),
                    "unresolved": worms_missing,
                },
                "ncbi": {
                    "endpoint": NCBI_EFETCH,
                    "requested": len(ids["NCBI_ID"]),
                    "resolved": len(ncbi),
                    "unresolved": ncbi_missing,
                },
                "wikidata": {
                    "endpoint": WIKIDATA_API,
                    "requested": len(ids["wikidata_ID"]),
                    "resolved": len(wikidata),
                    "unresolved": wikidata_missing,
                },
            },
        },
        "worms": worms,
        "ncbi": ncbi,
        "wikidata": wikidata,
    }


def write_snapshot(snapshot: dict, path: Path = DEFAULT_SNAPSHOT_PATH) -> Path:
    """Write the snapshot as sorted, indented JSON so PR diffs stay readable.

    Args:
        snapshot: Snapshot dict from :func:`build_snapshot`.
        path: Destination file.

    Returns:
        The path written.
    """
    path.write_text(json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("snapshot written: %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_snapshot(path: Path = DEFAULT_SNAPSHOT_PATH) -> dict:
    """Load a committed authority snapshot.

    Args:
        path: Snapshot file.

    Returns:
        The snapshot dict.

    Raises:
        FileNotFoundError: If the snapshot has not been harvested yet.
    """
    if not path.exists():
        raise FileNotFoundError(f"No authority snapshot at {path}. Run with --refresh-snapshot first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ── Stage 2: the cross-check (network-free) ────────────────────────────────────────────────
# Ranks whose NAME is shared between WoRMS and NCBI, so an absence from an NCBI lineage is
# evidence about the organism rather than about the classification system. Above family the two
# registers genuinely disagree by design — WoRMS keeps the Chromista kingdom and the
# Heterokontophyta phylum, NCBI has neither (it routes the same organisms through Sar >
# Stramenopiles and calls the metazoans Metazoa, not Animalia). Flagging those as defects would
# be flagging the CSV for following the register it is aligned to.
NCBI_STRICT_RANKS = frozenset({"Genus", "Species"})
NCBI_SOFT_RANKS = frozenset({"Family"})

# WoRMS status strings that mean "this name is the current one".
WORMS_ACCEPTED_STATUS = frozenset({"accepted"})


@dataclass
class Taxon:
    """One verification unit: a distinct taxon assertion, with the rows that carry it.

    Rows from different source datasets that make the same taxonomic claim are verified once
    and the finding is attributed back to all of them.
    """

    proposed_label: str
    ranks: dict[str, str]
    ids: dict[str, str]
    datasets: tuple[str, ...] = ()
    n_rows: int = 0
    root_class: str = ""
    csv_ranks: dict[str, str] = field(default_factory=dict)


def group_taxa(rows: list[dict]) -> list[Taxon]:
    """Collapse CSV rows into distinct taxon assertions.

    Args:
        rows: CSV rows from :func:`read_taxonomy`.

    Returns:
        One :class:`Taxon` per distinct (label, ranks, identifiers) combination, carrying the
        set of source datasets whose rows assert it.
    """
    buckets: dict[tuple, dict] = {}
    for row in rows:
        ranks = {rank: norm_name(row.get(rank)) for rank in RANKS}
        ids = {col: norm_id(row.get(col)) for col in ID_COL_AUTHORITY}
        label = norm_name(row.get("proposed_label"))
        key = (label, tuple(ranks.items()), tuple(ids.items()))
        entry = buckets.setdefault(
            key,
            {
                "taxon": Taxon(
                    proposed_label=label,
                    ranks=ranks,
                    ids=ids,
                    root_class=norm_name(row.get("root_class")),
                    csv_ranks={rank: row.get(rank, "") for rank in RANKS},
                ),
                "datasets": set(),
                "n": 0,
            },
        )
        entry["datasets"].add(row.get("Dataset", ""))
        entry["n"] += 1
    out = []
    for entry in buckets.values():
        taxon = entry["taxon"]
        taxon.datasets = tuple(sorted(entry["datasets"]))
        taxon.n_rows = entry["n"]
        out.append(taxon)
    return sorted(out, key=lambda t: (t.proposed_label, t.ids["aphia_ID"]))


def _chain_index(chain: list[list[str]]) -> dict[str, set[str]]:
    """Index an authority ancestor chain by normalized name.

    Args:
        chain: ``[[rank, name], ...]`` as stored in the snapshot.

    Returns:
        Mapping of normalized name to the set of ranks the authority places it at.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for rank, name in chain:
        key = norm_name(name)
        if key:
            index[key].add(norm_name(rank))
    return index


def rank_cells(taxon: Taxon) -> list[tuple[str, frozenset[str]]]:
    """The populated rank cells of a taxon, each with the name forms an authority may use.

    The ``Species`` column holds the bare epithet (``tetragona``), while every authority
    records the binomial (``Abylopsis tetragona``). Both forms are therefore acceptable at that
    rank; comparing the epithet alone would report a contradiction on every species row.

    Args:
        taxon: The taxon under test.

    Returns:
        ``[(rank, {acceptable names}), ...]`` in Kingdom-to-Species order.
    """
    cells: list[tuple[str, frozenset[str]]] = []
    for rank in RANKS:
        value = taxon.ranks[rank]
        if not value:
            continue
        if rank == "Species":
            binomial = norm_name(f"{taxon.ranks['Genus']} {value}")
            cells.append((rank, frozenset({value, binomial} - {""})))
        else:
            cells.append((rank, frozenset({value})))
    return cells


def _chain_rank_bound(index: dict[str, set[str]]) -> int:
    """Deepest of the seven schema ranks that an authority chain actually reaches.

    An identifier's ancestor chain cannot contain ranks BELOW the identifier's own rank, so
    testing a species row's ``Species`` cell against a genus-level identifier's chain would
    report a contradiction that is really a statement about the identifier's rank. Cells deeper
    than this bound are therefore left to the coarseness check instead.

    Args:
        index: Chain index from :func:`_chain_index`.

    Returns:
        Index into ``RANKS`` of the deepest schema rank present, or ``len(RANKS) - 1`` when the
        chain carries none of them (nothing is then skipped).
    """
    chain_ranks = {rank for ranks in index.values() for rank in ranks}
    reached = [i for i, rank in enumerate(RANKS) if rank.lower() in chain_ranks]
    return max(reached) if reached else len(RANKS) - 1


def _coarser_than_label(taxon: Taxon, authority_names: set[str]) -> str:
    """Whether an authority's name for the row's identifier is one of the row's own ancestors.

    A genus identifier stamped on a species row is not a wrong-organism error — the identifier
    denotes a real ancestor of the labelled taxon, and the row's own ``Genus`` cell agrees with
    it. It is the "too coarse" defect class (KI-6), reported once at WARN rather than as a name
    mismatch plus a cascade of lineage contradictions below it.

    Args:
        taxon: The taxon under test.
        authority_names: Normalized names the authority gives for the identifier.

    Returns:
        The rank whose cell the authority name matches, or ``""`` when it matches none.
    """
    label_rank, _ = deepest_rank(taxon.csv_ranks)
    label_depth = RANKS.index(label_rank) if label_rank in RANKS else len(RANKS)
    for rank, names in rank_cells(taxon):
        if RANKS.index(rank) < label_depth and names & authority_names:
            return rank
    return ""


def _lineage_findings(
    taxon: Taxon,
    authority: str,
    id_column: str,
    id_value: str,
    chain: list[list[str]],
    strict_ranks: frozenset[str],
    soft_ranks: frozenset[str],
) -> list[Finding]:
    """Test every populated CSV rank cell against one authority's ancestor chain.

    Three outcomes per cell — same rank (silent), present at a different rank
    (``rank_slot_drift``), absent from the chain (``lineage_contradiction`` at a rank the two
    registers share, ``higher_rank_divergence`` above it). Cells deeper than the chain's own
    deepest schema rank are skipped; see :func:`_chain_rank_bound`.

    Args:
        taxon: The taxon under test.
        authority: Authority name, used in the check names.
        id_column: CSV column holding the identifier.
        id_value: The identifier being tested.
        chain: The authority's ranked ancestor chain, deepest node included.
        strict_ranks: CSV ranks at which an absence is treated as ERROR.
        soft_ranks: CSV ranks at which an absence is treated as WARN.

    Returns:
        The findings for this taxon/authority pair.
    """
    index = _chain_index(chain)
    bound = _chain_rank_bound(index)
    findings: list[Finding] = []
    for rank, csv_names in rank_cells(taxon):
        if RANKS.index(rank) > bound:
            continue
        csv_name = max(csv_names, key=len)
        hit = next((name for name in sorted(csv_names, key=len, reverse=True) if name in index), "")
        if hit:
            authority_ranks = index[hit]
            if rank.lower() not in authority_ranks:
                findings.append(
                    Finding(
                        check=f"{authority}_rank_slot_drift",
                        severity="WARN",
                        authority=authority,
                        proposed_label=taxon.proposed_label,
                        id_column=id_column,
                        id_value=id_value,
                        csv_value=f"{rank}={csv_name}",
                        authority_value=f"{'/'.join(sorted(authority_ranks))}={hit}",
                        detail=f"the authority places {hit!r} at a different rank than the CSV column it occupies",
                    )
                )
            continue
        if rank in strict_ranks:
            severity, check = "ERROR", f"{authority}_lineage_contradiction"
            detail = f"{csv_name!r} is absent from the authority's classification of {id_value}"
        elif rank in soft_ranks:
            severity, check = "WARN", f"{authority}_lineage_contradiction"
            detail = f"{csv_name!r} is absent from the authority's classification of {id_value}"
        else:
            severity, check = "INFO", f"{authority}_higher_rank_divergence"
            detail = "expected above family: the two registers use different higher classifications"
        findings.append(
            Finding(
                check=check,
                severity=severity,
                authority=authority,
                proposed_label=taxon.proposed_label,
                id_column=id_column,
                id_value=id_value,
                csv_value=f"{rank}={csv_name}",
                authority_value="(absent from chain)",
                detail=detail,
            )
        )
    return findings


def _name_findings(
    taxon: Taxon,
    authority: str,
    id_column: str,
    id_value: str,
    authority_names: set[str],
    display_name: str,
    mismatch_detail: str,
) -> tuple[list[Finding], str]:
    """Compare a row's label to the names one authority gives its identifier.

    Three outcomes: the label is one of them (silent), the label is a descendant of one of them
    (``id_coarser_than_label``, WARN — the identifier is a real ancestor, just not at the
    label's rank), or the name is unrelated to the row's whole lineage (``name_mismatch``,
    ERROR — the identifier denotes a different organism).

    Args:
        taxon: The taxon under test.
        authority: Authority name, used in the check names.
        id_column: CSV column holding the identifier.
        id_value: The identifier being tested.
        authority_names: Normalized names the authority accepts for it.
        display_name: The authority's primary name, for the report.
        mismatch_detail: Detail text for the ERROR case.

    Returns:
        ``(findings, coarser_rank)``; ``coarser_rank`` is non-empty when the identifier sits
        above the label's rank, in which case rank-level checks are redundant.
    """
    label = taxon.proposed_label
    names = {name for name in authority_names if name}
    if not label or not names or label in names:
        return [], ""
    label_rank, _ = deepest_rank(taxon.csv_ranks)
    coarser = _coarser_than_label(taxon, names)
    if coarser:
        return [
            Finding(
                check=f"{authority}_id_coarser_than_label",
                severity="WARN",
                authority=authority,
                proposed_label=label,
                id_column=id_column,
                id_value=id_value,
                csv_value=f"{label_rank or 'label'}={label}",
                authority_value=f"{coarser}={display_name}",
                detail=f"the identifier resolves to the row's {coarser}, not to its {label_rank or 'label'}",
            )
        ], coarser
    return [
        Finding(
            check=f"{authority}_name_mismatch",
            severity="ERROR",
            authority=authority,
            proposed_label=label,
            id_column=id_column,
            id_value=id_value,
            csv_value=label,
            authority_value=display_name,
            detail=mismatch_detail,
        )
    ], ""


def check_worms(taxon: Taxon, worms: dict) -> list[Finding]:
    """Verify a taxon's ``aphia_ID`` against the WoRMS snapshot.

    Args:
        taxon: The taxon under test.
        worms: The snapshot's ``worms`` section.

    Returns:
        Findings for this taxon; empty when it carries no AphiaID or everything agrees.
    """
    aphia = taxon.ids["aphia_ID"]
    if not aphia:
        return []
    record = worms.get(aphia)
    if record is None:
        return [
            Finding(
                check="worms_id_unresolved",
                severity="ERROR",
                authority="worms",
                proposed_label=taxon.proposed_label,
                id_column="aphia_ID",
                id_value=aphia,
                csv_value=aphia,
                authority_value="(no record)",
                detail="WoRMS returns no record for this AphiaID",
            )
        ]

    findings: list[Finding] = []
    scientific, valid = norm_name(record.get("scientific_name")), norm_name(record.get("valid_name"))
    status = norm_name(record.get("status"))

    if status not in WORMS_ACCEPTED_STATUS:
        findings.append(
            Finding(
                check="worms_status_not_accepted",
                severity="WARN",
                authority="worms",
                proposed_label=taxon.proposed_label,
                id_column="aphia_ID",
                id_value=aphia,
                csv_value=taxon.proposed_label,
                authority_value=f"{status}; current name {record.get('valid_name')} ({record.get('valid_aphia_id')})",
                detail=str(record.get("unacceptreason") or "nomenclatural status is not 'accepted'"),
            )
        )

    name_findings, coarser = _name_findings(
        taxon,
        "worms",
        "aphia_ID",
        aphia,
        {scientific, valid},
        record.get("scientific_name") or "",
        "the AphiaID names neither the CSV label nor its accepted synonym",
    )
    findings.extend(name_findings)

    csv_rank, _ = deepest_rank(taxon.csv_ranks)
    authority_rank = norm_name(record.get("rank"))
    if csv_rank and authority_rank and not coarser:
        seven = {r.lower() for r in RANKS}
        if authority_rank not in seven:
            findings.append(
                Finding(
                    check="worms_rank_outside_seven",
                    severity="INFO",
                    authority="worms",
                    proposed_label=taxon.proposed_label,
                    id_column="aphia_ID",
                    id_value=aphia,
                    csv_value=csv_rank,
                    authority_value=record.get("rank") or "",
                    detail="WoRMS rank has no column in the seven-rank schema, so it was folded into the nearest one",
                )
            )
        elif authority_rank != csv_rank.lower():
            findings.append(
                Finding(
                    check="worms_rank_mismatch",
                    severity="WARN",
                    authority="worms",
                    proposed_label=taxon.proposed_label,
                    id_column="aphia_ID",
                    id_value=aphia,
                    csv_value=csv_rank,
                    authority_value=record.get("rank") or "",
                    detail="the deepest populated CSV rank disagrees with the AphiaID's own rank",
                )
            )

    chain = record.get("classification") or []
    if not chain:
        findings.append(
            Finding(
                check="worms_classification_unavailable",
                severity="INFO",
                authority="worms",
                proposed_label=taxon.proposed_label,
                id_column="aphia_ID",
                id_value=aphia,
                csv_value="(7 ranks)",
                authority_value="(no classification returned)",
                detail="lineage could not be tested for this AphiaID",
            )
        )
    else:
        # WoRMS is the register the CSV's higher classification follows, so every rank is
        # strict here — an absence is a real contradiction, not a system difference.
        findings.extend(_lineage_findings(taxon, "worms", "aphia_ID", aphia, chain, frozenset(RANKS), frozenset()))
    return findings


def check_ncbi(taxon: Taxon, ncbi: dict) -> list[Finding]:
    """Verify a taxon's ``NCBI_ID`` against the NCBI Taxonomy snapshot.

    Args:
        taxon: The taxon under test.
        ncbi: The snapshot's ``ncbi`` section.

    Returns:
        Findings for this taxon.
    """
    tax_id = taxon.ids["NCBI_ID"]
    if not tax_id:
        return []
    record = ncbi.get(tax_id)
    if record is None:
        return [
            Finding(
                check="ncbi_id_unresolved",
                severity="ERROR",
                authority="ncbi",
                proposed_label=taxon.proposed_label,
                id_column="NCBI_ID",
                id_value=tax_id,
                csv_value=tax_id,
                authority_value="(no taxon)",
                detail="NCBI Taxonomy returns no taxon for this taxid",
            )
        ]

    findings: list[Finding] = []
    names = {norm_name(record.get("scientific_name"))} | {norm_name(n) for n in record.get("other_names", [])}
    name_findings, _ = _name_findings(
        taxon,
        "ncbi",
        "NCBI_ID",
        tax_id,
        names,
        record.get("scientific_name") or "",
        "the taxid names neither the CSV label nor any NCBI synonym of it",
    )
    findings.extend(name_findings)

    chain = list(record.get("lineage") or [])
    chain.append([record.get("rank") or "", record.get("scientific_name") or ""])
    findings.extend(_lineage_findings(taxon, "ncbi", "NCBI_ID", tax_id, chain, NCBI_STRICT_RANKS, NCBI_SOFT_RANKS))
    return findings


def check_wikidata(taxon: Taxon, wikidata: dict) -> list[Finding]:
    """Verify a taxon's ``wikidata_ID`` and replay the original ID harvest against Wikidata.

    The CSV's ``aphia_ID`` / ``NCBI_ID`` / ``BOLD_ID`` were read off this Qcode's P850 / P685 /
    P3606 claims. Re-reading them says whether the harvest is still what Wikidata asserts.

    Args:
        taxon: The taxon under test.
        wikidata: The snapshot's ``wikidata`` section.

    Returns:
        Findings for this taxon.
    """
    qid = taxon.ids["wikidata_ID"]
    if not qid:
        return []
    record = wikidata.get(qid)
    if record is None:
        return [
            Finding(
                check="wikidata_id_unresolved",
                severity="ERROR",
                authority="wikidata",
                proposed_label=taxon.proposed_label,
                id_column="wikidata_ID",
                id_value=qid,
                csv_value=qid,
                authority_value="(missing entity)",
                detail="Wikidata has no entity for this Qcode (deleted or merged away)",
            )
        ]

    findings: list[Finding] = []
    names = {norm_name(record.get("taxon_name")), norm_name(record.get("label_en"))}
    name_findings, coarser = _name_findings(
        taxon,
        "wikidata",
        "wikidata_ID",
        qid,
        names,
        record.get("taxon_name") or record.get("label_en") or "",
        "neither the P225 taxon name nor the English label matches the CSV label",
    )
    findings.extend(name_findings)

    for claim_key, csv_col, label in (
        ("p850_worms", "aphia_ID", "WoRMS"),
        ("p685_ncbi", "NCBI_ID", "NCBI"),
        ("p3606_bold", "BOLD_ID", "BOLD"),
    ):
        csv_id = taxon.ids.get(csv_col) or ""
        claimed = [str(v) for v in record.get(claim_key, [])]
        if not csv_id:
            continue
        if not claimed:
            findings.append(
                Finding(
                    check=f"wikidata_{label.lower()}_claim_absent",
                    severity="INFO",
                    authority="wikidata",
                    proposed_label=taxon.proposed_label,
                    id_column=csv_col,
                    id_value=csv_id,
                    csv_value=csv_id,
                    authority_value="(no claim)",
                    detail=f"the Qcode carries no {label} identifier claim, so the harvested value cannot be replayed",
                )
            )
        elif csv_id not in claimed:
            findings.append(
                Finding(
                    check=f"wikidata_{label.lower()}_claim_divergence",
                    severity="WARN",
                    authority="wikidata",
                    proposed_label=taxon.proposed_label,
                    id_column=csv_col,
                    id_value=csv_id,
                    csv_value=csv_id,
                    authority_value=";".join(claimed),
                    detail=f"Wikidata now asserts a different {label} identifier than the CSV holds",
                )
            )

    csv_rank, _ = deepest_rank(taxon.csv_ranks)
    authority_rank = norm_name(record.get("rank"))
    seven = {r.lower() for r in RANKS}
    if csv_rank and not coarser and authority_rank in seven and authority_rank != csv_rank.lower():
        findings.append(
            Finding(
                check="wikidata_rank_mismatch",
                severity="WARN",
                authority="wikidata",
                proposed_label=taxon.proposed_label,
                id_column="wikidata_ID",
                id_value=qid,
                csv_value=csv_rank,
                authority_value=record.get("rank") or "",
                detail="the P105 taxon rank disagrees with the deepest populated CSV rank",
            )
        )
    return findings


def check_cross_authority(taxon: Taxon, worms: dict, ncbi: dict) -> list[Finding]:
    """Compare WoRMS and NCBI against each other for the same taxon.

    Two independent registers naming different organisms for the same row is the strongest
    available evidence that one of the two identifiers is wrong. A pure depth difference (one
    register resolved to the genus, the other to a species inside it) is reported separately.

    Args:
        taxon: The taxon under test.
        worms: The snapshot's ``worms`` section.
        ncbi: The snapshot's ``ncbi`` section.

    Returns:
        At most one finding.
    """
    worms_record, ncbi_record = worms.get(taxon.ids["aphia_ID"]), ncbi.get(taxon.ids["NCBI_ID"])
    if not (worms_record and ncbi_record):
        return []
    worms_name = norm_name(worms_record.get("valid_name") or worms_record.get("scientific_name"))
    ncbi_name = norm_name(ncbi_record.get("scientific_name"))
    if not (worms_name and ncbi_name) or worms_name == ncbi_name:
        return []

    # NCBI records cross-register synonyms in OtherNames — 'Animalia' is listed there under
    # Metazoa, 'Acantharia' under Acantharea. Two registers using different accepted spellings
    # for the same taxon is not a disagreement about which organism the row denotes.
    worms_names = {worms_name, norm_name(worms_record.get("scientific_name"))} - {""}
    ncbi_names = {ncbi_name} | {norm_name(n) for n in ncbi_record.get("other_names", [])} - {""}
    if worms_names & ncbi_names:
        return []

    # Otherwise: is one of them simply deeper in the other's lineage? That is a rank-depth
    # difference between the two identifiers, not a claim about different organisms.
    worms_chain = _chain_index(worms_record.get("classification") or [])
    ncbi_chain = _chain_index([*(ncbi_record.get("lineage") or []), [ncbi_record.get("rank") or "", ncbi_name]])
    nested = bool(worms_names & set(ncbi_chain)) or bool(ncbi_names & set(worms_chain))
    return [
        Finding(
            check="authority_depth_disagreement" if nested else "authority_name_disagreement",
            severity="WARN" if nested else "ERROR",
            authority="worms+ncbi",
            proposed_label=taxon.proposed_label,
            id_column="aphia_ID+NCBI_ID",
            id_value=f"{taxon.ids['aphia_ID']}+{taxon.ids['NCBI_ID']}",
            csv_value=worms_name,
            authority_value=ncbi_name,
            detail=(
                "the two registers resolve the row's identifiers to different depths of the same lineage"
                if nested
                else "the two registers resolve the row's identifiers to different organisms"
            ),
        )
    ]


def check_id_reuse(taxa: list[Taxon]) -> list[Finding]:
    """Flag an authority identifier stamped on more than one distinct planktonzilla label.

    An identifier denotes exactly one taxon at the authority, so two different labels sharing
    one is a mapping defect on at least one of the two rows. This generalizes the single
    hand-found instance recorded as KI-13.

    Args:
        taxa: All taxon assertions.

    Returns:
        One finding per over-used identifier.
    """
    findings: list[Finding] = []
    for col in ID_COL_AUTHORITY:
        by_id: dict[str, set[str]] = defaultdict(set)
        for taxon in taxa:
            value = taxon.ids[col]
            if value and taxon.proposed_label:
                by_id[value].add(taxon.proposed_label)
        for value, labels in sorted(by_id.items()):
            if len(labels) > 1:
                findings.append(
                    Finding(
                        check="id_reused_across_taxa",
                        severity="WARN",
                        authority=ID_COL_AUTHORITY[col],
                        proposed_label=min(labels),
                        id_column=col,
                        id_value=value,
                        csv_value=";".join(sorted(labels)),
                        authority_value="(one identifier denotes one taxon)",
                        detail=f"{len(labels)} distinct planktonzilla labels share this identifier",
                    )
                )
    return findings


def crosscheck(rows: list[dict], snapshot: dict, authorities: tuple[str, ...] = ("worms", "ncbi", "wikidata")) -> list[Finding]:
    """Run the full cross-check of the CSV against an authority snapshot.

    Args:
        rows: CSV rows from :func:`read_taxonomy`.
        snapshot: Snapshot from :func:`load_snapshot`.
        authorities: Subset of authorities to test.

    Returns:
        All findings, with source datasets and row counts attributed.
    """
    worms, ncbi, wikidata = snapshot.get("worms", {}), snapshot.get("ncbi", {}), snapshot.get("wikidata", {})
    taxa = group_taxa(rows)
    findings: list[Finding] = []
    for taxon in taxa:
        produced: list[Finding] = []
        if "worms" in authorities:
            produced += check_worms(taxon, worms)
        if "ncbi" in authorities:
            produced += check_ncbi(taxon, ncbi)
        if "wikidata" in authorities:
            produced += check_wikidata(taxon, wikidata)
        if {"worms", "ncbi"} <= set(authorities):
            produced += check_cross_authority(taxon, worms, ncbi)
        for finding in produced:
            finding.datasets = taxon.datasets
            finding.n_rows = taxon.n_rows
        findings += produced
    findings += [f for f in check_id_reuse(taxa) if f.authority in authorities]
    return sorted(findings, key=lambda f: (SEVERITIES.index(f.severity), f.check, f.proposed_label, f.id_value))


# ── Waivers ────────────────────────────────────────────────────────────────────────────────
def load_waivers(path: Path = DEFAULT_WAIVERS_PATH) -> dict[str, dict]:
    """Load the waiver file keyed by ``finding_id``.

    A waiver records a finding that has been reviewed and accepted — most of them because the
    published table is frozen under the zero-behavioural-drift rule, so a real defect is
    documented rather than corrected.

    Args:
        path: Waiver JSON file. A missing file means "no waivers".

    Returns:
        Mapping of ``finding_id`` to the waiver entry.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {entry["finding_id"]: entry for entry in payload.get("waivers", [])}


def apply_waivers(findings: list[Finding], waivers: dict[str, dict]) -> tuple[list[Finding], list[Finding], list[str]]:
    """Split findings by waiver status and report waivers that no longer match anything.

    Args:
        findings: All findings from :func:`crosscheck`.
        waivers: Mapping from :func:`load_waivers`.

    Returns:
        ``(unwaived, waived, stale_waiver_ids)``. A stale waiver means the CSV or the authority
        changed such that the waived finding is gone — the waiver should be deleted.
    """
    seen = {f.finding_id for f in findings}
    unwaived = [f for f in findings if f.finding_id not in waivers]
    waived = [f for f in findings if f.finding_id in waivers]
    return unwaived, waived, sorted(set(waivers) - seen)


# ── Reporting ──────────────────────────────────────────────────────────────────────────────
def findings_frame(findings: list[Finding]) -> pl.DataFrame:
    """Render findings as a polars frame for CSV export.

    Args:
        findings: Findings to tabulate.

    Returns:
        A frame with one row per finding.
    """
    if not findings:
        return pl.DataFrame({c: [] for c in FINDINGS_COLUMNS}, schema={c: pl.String for c in FINDINGS_COLUMNS})
    records = []
    for finding in findings:
        record = asdict(finding)
        record["datasets"] = ";".join(finding.datasets)
        record["finding_id"] = finding.finding_id
        records.append(record)
    return pl.DataFrame(records).select(FINDINGS_COLUMNS)


def summarize(findings: list[Finding]) -> dict:
    """Aggregate findings by severity, authority and check.

    Args:
        findings: Findings to aggregate.

    Returns:
        Nested counts plus the number of CSV rows touched per severity.
    """
    by_severity: dict[str, int] = defaultdict(int)
    by_authority: dict[str, int] = defaultdict(int)
    by_check: dict[str, int] = defaultdict(int)
    rows_touched: dict[str, set] = defaultdict(set)
    for finding in findings:
        by_severity[finding.severity] += 1
        by_authority[finding.authority] += 1
        by_check[f"{finding.severity}:{finding.check}"] += 1
        rows_touched[finding.severity].add((finding.proposed_label, finding.id_value))
    return {
        "total": len(findings),
        "by_severity": dict(sorted(by_severity.items(), key=lambda kv: SEVERITIES.index(kv[0]))),
        "by_authority": dict(sorted(by_authority.items())),
        "by_check": dict(sorted(by_check.items())),
        "distinct_taxa_by_severity": {k: len(v) for k, v in sorted(rows_touched.items())},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status — non-zero when unwaived findings at or above ``--fail-on`` exist.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-snapshot", action="store_true", help="network stage: re-harvest all three authorities")
    parser.add_argument("--report", action="store_true", help="network-free stage: cross-check the CSV against the snapshot")
    parser.add_argument("--csv", type=Path, default=Path(DEFAULT_TAXONOMY_CSV_FILENAME), help="taxonomy CSV to verify")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH, help="authority snapshot JSON")
    parser.add_argument("--waivers", type=Path, default=DEFAULT_WAIVERS_PATH, help="waiver JSON file")
    parser.add_argument("--findings-csv", type=Path, default=None, help="write the findings table here")
    parser.add_argument("--authority", action="append", choices=["worms", "ncbi", "wikidata"], help="restrict to one authority")
    parser.add_argument("--fail-on", default="ERROR", choices=[*SEVERITIES, "NONE"], help="lowest severity that fails the run")
    parser.add_argument("--email", default=None, help="contact address for NCBI Entrez (else NCBI_EMAIL)")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress and summary log")
    args = parser.parse_args(argv)

    if not (args.refresh_snapshot or args.report):
        parser.error("choose --refresh-snapshot and/or --report")

    # Run as a script there is no handler on the root logger, so the summary would go nowhere.
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")

    if args.refresh_snapshot:
        write_snapshot(build_snapshot(args.csv, email=args.email), args.snapshot)

    if not args.report:
        return 0

    rows = read_taxonomy(args.csv)
    snapshot = load_snapshot(args.snapshot)
    authorities = tuple(args.authority) if args.authority else ("worms", "ncbi", "wikidata")
    findings = crosscheck(rows, snapshot, authorities=authorities)
    unwaived, waived, stale = apply_waivers(findings, load_waivers(args.waivers))

    logger.info("findings: %d total, %d waived, %d unwaived", len(findings), len(waived), len(unwaived))
    for severity, count in summarize(unwaived)["by_severity"].items():
        logger.info("  unwaived %s: %d", severity, count)
    if stale and len(authorities) == 3:
        # Only meaningful on a full run: restricting --authority makes the other authorities'
        # waivers unmatchable, which is not staleness.
        logger.warning("stale waivers (no longer match any finding): %s", stale)

    destination = args.findings_csv or DEFAULT_FINDINGS_CSV
    findings_frame(findings).write_csv(destination)
    logger.info("findings table: %s", destination)

    if args.fail_on == "NONE":
        return 0
    threshold = SEVERITIES.index(args.fail_on)
    blocking = [f for f in unwaived if SEVERITIES.index(f.severity) <= threshold]
    if blocking:
        logger.error("%d unwaived finding(s) at or above %s", len(blocking), args.fail_on)
        for finding in blocking[:20]:
            logger.error(
                "  [%s] %s %s %s -> %s",
                finding.severity,
                finding.check,
                finding.proposed_label,
                finding.csv_value,
                finding.authority_value,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
