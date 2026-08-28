"""
(c) Inria

verify_taxon_ids.py
===================
Answers, for every row of ``planktonzilla_taxonomy.csv``: **does this external ID
actually name this taxon?**

The committed table's ID columns are well-formed by construction (KI-12 pins their
serialization) and internally consistent (one value per label). Neither property says the
id points at the RIGHT organism — the failure `resolve_frepj_ids.verify_ncbi_lineage`
caught on 9 frepj draft rows, where a bare-epithet search resolved a copepod to the
hydrozoan *Sarsia*. That guard ran once, over 229 draft rows, at fill time. This module
generalizes it to the whole table and every registry, as a check that can be re-run.

Two layers, because the test suite is network-free by construction:

1. **Fetch** (this module's ``main``; needs network) — resolve every distinct
   ``aphia_ID`` / ``NCBI_ID`` / ``wikidata_ID`` against WoRMS / NCBI Taxonomy /
   Wikidata, and every distinct ``proposed_label`` against the GBIF backbone. Write
   what each registry says — accepted name, rank, lineage — to a snapshot TSV.
2. **Score** (the pure functions below; no network) — compare each row against the
   snapshot and return a verdict. ``tests/test_taxonomy_external_ids.py`` imports these
   and fails on ``contradiction``.

Scoring is deliberately generous about NAMES and strict about ORGANISMS. Registries
disagree on orthography (`globorotalidae` vs *Globorotaliidae*), on rank depth (a species
row legitimately carrying its genus's taxid — KI-13's documented coarse-propagation
bucket), on accepted-vs-synonym naming, and on which genus a species belongs to after a
recombination. None of those is a defect. What IS a defect is an id whose organism has
nothing to do with the row: its name matches nothing the row asserts, and its lineage
agrees with the row's nowhere at Phylum or deeper. Kingdom agreement alone does not
count — most of this table is `animalia` or `chromista`, so it certifies nothing. Only
that returns ``contradiction``.

A wrong id that is also NEW has no snapshot record and would score as ``unchecked``, so
coverage is the second half of the guard: the test suite fails when the table carries an
identifier the snapshot has never resolved, which forces the refresh that then judges it.

Usage:
    # refresh the committed snapshot (network; ~10 min, polite concurrency)
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids --only worms
    python -m planktonzilla.planktonzilla_dataset.utils.verify_taxon_ids --report

Requirements: requests (already a project dependency).
"""

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree

from planktonzilla.planktonzilla_dataset.constants import DEFAULT_TAXONOMY_CSV_FILENAME, TAXONOMY_RANKS
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_TSV = _REPO_ROOT / "tests" / "fixtures" / "taxonomy" / "external_id_snapshot.tsv"

SNAPSHOT_COLUMNS = ("source", "key", "status", "name", "rank", "lineage")

# `source` values: which registry answered, keyed by the CSV column it verifies.
SOURCE_FOR_COLUMN = {
    "aphia_ID": "worms",
    "NCBI_ID": "ncbi",
    "wikidata_ID": "wikidata",
}
# GBIF is keyed by `proposed_label` rather than by an ID column: the CSV has no GBIF
# column, so GBIF answers a different question — is this NAME a taxon at all, and does
# its backbone lineage agree with the row's rank columns?
GBIF_SOURCE = "gbif"

# A registry answered, and knows the key / does not / could not be reached. `unknown` is
# a finding (we ship an id nobody recognizes); `error` is never read as a defect.
STATUS_OK = "ok"
STATUS_UNKNOWN = "unknown"
STATUS_ERROR = "error"

HEADERS = {"User-Agent": "planktonzilla-taxonomy-verifier (https://github.com/Inria-Chile/planktonzilla)"}
TIMEOUT = 30

# Two names this close, after normalization, are the same name spelled differently
# (`globorotalidae` / `globorotaliidae` = 0.97). Below it they are different names, and
# the lineage checks decide.
NEAR_MATCH_RATIO = 0.90


# --- Normalization ---------------------------------------------------------------------


def normalize_name(value: str) -> str:
    """A scientific name reduced to what two registries can be expected to agree on.

    Registries return a name with its authority attached (`Tripos Bory, 1823`); the table
    stores the name alone. Binomial nomenclature makes the split reliable without a
    parser: the name is the leading token plus any following all-lowercase tokens (the
    epithets), and an authority always begins with a capital or a digit. So the authority
    is cut BEFORE lowercasing, then the usual qualifiers (`sp.`, `cf.`, `incertae sedis`)
    and punctuation are dropped.
    """
    text = re.sub(r"\([^)]*\)", " ", (value or "").strip())  # subgenus / authority parens
    tokens = text.split()
    kept = tokens[:1]
    for token in tokens[1:]:
        if not token.strip(".,;").isalpha() or not token.strip(".,;").islower():
            break  # an authority surname, an initial, or a year — the name ended
        kept.append(token)
    text = " ".join(kept).lower()
    text = re.sub(r"\b(sp|spp|cf|aff|var|subsp|f|indet|incertae|sedis)\b\.?", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_similarity(left: str, right: str) -> float:
    """Similarity of two normalized names in [0, 1] (stdlib only, deterministic).

    The max of whole-string ratio and token-set ratio, so `chaetoceros dadayi` vs
    `dadayi chaetoceros` and `neoceratium` vs `neoceratium gibberum` both score high.
    """
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    whole = SequenceMatcher(None, a, b).ratio()
    tokens_a, tokens_b = set(a.split()), set(b.split())
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    return max(whole, overlap)


# --- Scoring (pure; the test imports these) --------------------------------------------

VERDICT_EXACT = "exact"  # the registry name IS the row's taxon name
VERDICT_NEAR = "near"  # same name, different orthography
VERDICT_ANCESTOR = "ancestor"  # a coarser rank of the row's own lineage (KI-13 bucket)
VERDICT_LINEAGE = "lineage"  # different name, but demonstrably the same branch
VERDICT_RECOMBINATION = "recombination"  # same epithet, different genus — one organism, two combinations
VERDICT_NAME_MISMATCH = "name_mismatch"  # names differ and no lineage exists to judge by
VERDICT_CONTRADICTION = "contradiction"  # names a different organism — the defect
VERDICT_UNKNOWN = "unknown"  # the registry does not know this id
VERDICT_UNCHECKED = "unchecked"  # no snapshot entry / lookup failed — never a defect


@dataclass(frozen=True)
class Verdict:
    """One row-vs-registry judgement, with the evidence that produced it."""

    verdict: str
    detail: str

    @property
    def is_defect(self) -> bool:
        return self.verdict == VERDICT_CONTRADICTION


def row_names(row: dict) -> list[str]:
    """Every name the row itself asserts, deepest first: label, binomial, then ranks."""
    names = [row["proposed_label"]]
    genus, species = row.get("Genus", ""), row.get("Species", "")
    if genus and species:
        names.append(f"{genus} {species}")
    names.extend(row[rank] for rank in reversed(TAXONOMY_RANKS))
    return [name for name in (n.strip().lower() for n in names) if name]


def score_row_against_record(row: dict, record: dict) -> Verdict:
    """Judge one registry record against one taxonomy row.

    Order matters — each rule is strictly weaker than the one above it, so a verdict
    names the STRONGEST relationship found.
    """
    if record is None:
        return Verdict(VERDICT_UNCHECKED, "no snapshot entry")
    if record["status"] == STATUS_ERROR:
        return Verdict(VERDICT_UNCHECKED, "registry lookup did not complete")
    if record["status"] == STATUS_UNKNOWN:
        return Verdict(VERDICT_UNKNOWN, "the registry does not know this identifier")

    registry_name = record["name"]
    normalized_registry = normalize_name(registry_name)
    names = row_names(row)
    deepest = names[0] if names else ""

    # 1. The registry names the row's own taxon.
    if any(normalize_name(name) == normalized_registry for name in names[:2]):
        return Verdict(VERDICT_EXACT, f"{registry_name!r} is the row's taxon")

    # 2. Same name, different orthography (registry spelling vs the table's).
    best = max((name_similarity(name, registry_name), name) for name in names[:2])
    if best[0] >= NEAR_MATCH_RATIO:
        return Verdict(VERDICT_NEAR, f"{registry_name!r} ~ {best[1]!r} ({best[0]:.2f})")

    # 3. A coarser rank of this row's own lineage — the documented KI-13 bucket: an
    #    id-precision limitation, not a wrong organism.
    for rank in reversed(TAXONOMY_RANKS):
        value = row[rank].strip().lower()
        if value and name_similarity(value, registry_name) >= NEAR_MATCH_RATIO:
            return Verdict(VERDICT_ANCESTOR, f"{registry_name!r} is the row's {rank}")

    # 4. Different name, same branch: the registry's lineage contains the row's taxon,
    #    or the row's own genus+phylum sit in it (accepted-name/synonym pairs land here).
    lineage = record["lineage"]
    genus, phylum = row["Genus"].strip().lower(), row["Phylum"].strip().lower()
    if lineage:
        if any(normalize_name(name) in lineage for name in names[:2]):
            return Verdict(VERDICT_LINEAGE, f"the row's taxon is in {registry_name!r}'s lineage")
        # Kingdom agreement is nearly free here — most of the table is `animalia` or
        # `chromista` — so it certifies nothing on its own: WoRMS's *Bivalvia* record
        # "shares the kingdom" of every copepod row. Agreement has to reach Phylum or
        # deeper before it means the same organism.
        shared = {
            value
            for rank, value in ((rank, row[rank].strip().lower()) for rank in TAXONOMY_RANKS[1:])
            if value and value in lineage
        }
        if shared:
            return Verdict(VERDICT_LINEAGE, f"{registry_name!r} shares the row's {', '.join(sorted(shared))}")

    # 5. A recombination: same species epithet, different genus. Epithets are stable
    #    when a species moves between genera, so `odontella mobiliensis` vs the
    #    registry's `trieres mobiliensis` is one organism under two combinations, not
    #    two organisms. Reported rather than failed — a shared epithet across unrelated
    #    genera is possible (the table already documents `socialis`, `caudatum`), so
    #    this wants a human, not a verdict.
    species = row["Species"].strip().lower()
    if species and normalized_registry.split(" ")[-1] == species and " " in normalized_registry:
        return Verdict(
            VERDICT_RECOMBINATION,
            f"{registry_name!r} shares the epithet {species!r} under a different genus (the row says {row['Genus'] or '∅'})",
        )

    # A rank-less bucket row (`artefact`, `other`, `egg`) asserts no organism, so there
    # is nothing an identifier can contradict.
    if not genus and not phylum:
        return Verdict(VERDICT_UNCHECKED, "the row asserts no lineage to check against")

    # A contradiction needs POSITIVE evidence of a different organism. Without a lineage
    # from the registry there is none — a bare name mismatch is reported for a human to
    # read (Wikidata entities with no P171 parent land here) but never fails the suite.
    if not lineage:
        return Verdict(
            VERDICT_NAME_MISMATCH,
            f"{registry_name!r} matches nothing the row asserts, and the registry gave no lineage to judge it by",
        )

    return Verdict(
        VERDICT_CONTRADICTION,
        f"{registry_name!r} (rank {record['rank'] or '?'}) matches neither {deepest!r} nor any rank of the row, "
        f"and its lineage ({', '.join(sorted(lineage)) or '∅'}) shares nothing with the row's lineage "
        f"(genus {genus or '∅'}, phylum {phylum or '∅'})",
    )


def snapshot_key(column: str, value: str) -> str:
    """The snapshot key for one CSV id cell (`X.0` floats normalized to the integer)."""
    text = value.strip()
    if column in ("aphia_ID", "NCBI_ID", "BOLD_ID") and text.endswith(".0"):
        return text[:-2]
    return text


def load_snapshot(path=None) -> dict:
    """The committed registry snapshot as ``{(source, key): record}``."""
    path = Path(path or DEFAULT_SNAPSHOT_TSV)
    records = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            records[(row["source"], row["key"])] = {
                "status": row["status"],
                "name": row["name"],
                "rank": row["rank"],
                "lineage": {part for part in row["lineage"].split("|") if part},
            }
    return records


def score_table(rows, snapshot) -> list[dict]:
    """Every (row, id-column) judgement over the whole table, snapshot-driven."""
    findings = []
    for index, row in enumerate(rows):
        for column, source in SOURCE_FOR_COLUMN.items():
            value = row[column].strip()
            if not value:
                continue
            record = snapshot.get((source, snapshot_key(column, value)))
            verdict = score_row_against_record(row, record)
            findings.append(
                {
                    "row": index,
                    "dataset": row["Dataset"],
                    "raw_label": row["Raw_Labels"],
                    "label": row["proposed_label"],
                    "column": column,
                    "value": value,
                    "verdict": verdict.verdict,
                    "detail": verdict.detail,
                }
            )
    return findings


def score_labels_against_gbif(rows, snapshot) -> list[dict]:
    """Judge each distinct `proposed_label` against the GBIF backbone.

    A different question from the ID checks: not "is this id right" but "is this name a
    taxon at all, at the rank the row files it under". A label GBIF cannot match, or
    matches into a different kingdom, is worth a human look.
    """
    findings, seen = [], set()
    for index, row in enumerate(rows):
        label = row["proposed_label"].strip().lower()
        if not label or label in seen or not any(row[rank].strip() for rank in TAXONOMY_RANKS):
            continue
        seen.add(label)
        record = snapshot.get((GBIF_SOURCE, label))
        verdict = score_row_against_record(row, record)
        findings.append(
            {
                "row": index,
                "dataset": row["Dataset"],
                "raw_label": row["Raw_Labels"],
                "label": label,
                "column": "proposed_label",
                "value": label,
                "verdict": verdict.verdict,
                "detail": verdict.detail,
            }
        )
    return findings


# --- Fetching (network) ----------------------------------------------------------------


def _session():
    import requests

    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _record(status, name="", rank="", lineage=()):
    return {"status": status, "name": name.strip().lower(), "rank": rank.strip().lower(), "lineage": set(lineage)}


def fetch_worms(session, aphia_id: str) -> dict:
    """One WoRMS AphiaRecord: accepted name, rank, and its Kingdom..Genus lineage."""
    url = f"https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{aphia_id}"
    try:
        response = session.get(url, timeout=TIMEOUT)
        if response.status_code in (204, 400, 404):
            return _record(STATUS_UNKNOWN)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        logger.warning(f"[worms] {aphia_id}: {error}")
        return _record(STATUS_ERROR)
    lineage = {str(data.get(rank) or "").strip().lower() for rank in ("kingdom", "phylum", "class", "order", "family", "genus")}
    # `valid_name` is the ACCEPTED name; an unaccepted id keeps its own name too, so a
    # synonym in the table still scores as a name match rather than a contradiction.
    for key in ("scientificname", "valid_name"):
        value = str(data.get(key) or "").strip().lower()
        if value:
            lineage.add(value)
    name = str(data.get("valid_name") or data.get("scientificname") or "").strip()
    return _record(STATUS_OK, name, str(data.get("rank") or ""), {part for part in lineage if part})


def fetch_ncbi_batch(session, taxids: list[str]) -> dict:
    """A batch of NCBI taxids via esummary (name + rank), then efetch for lineages.

    esummary carries no lineage, so a second efetch pass fills it; a failure there
    degrades to a name-only record rather than an error — the name checks still apply.
    """
    results = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "taxonomy", "id": ",".join(taxids), "retmode": "json"}
    try:
        response = session.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json().get("result", {})
    except Exception as error:
        logger.warning(f"[ncbi] esummary batch failed: {error}")
        return {taxid: _record(STATUS_ERROR) for taxid in taxids}

    for taxid in taxids:
        entry = payload.get(taxid)
        if not isinstance(entry, dict) or entry.get("error"):
            results[taxid] = _record(STATUS_UNKNOWN)
            continue
        results[taxid] = _record(STATUS_OK, str(entry.get("scientificname") or ""), str(entry.get("rank") or ""))

    known = [taxid for taxid in taxids if results[taxid]["status"] == STATUS_OK]
    if known:
        try:
            response = session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "taxonomy", "id": ",".join(known), "retmode": "xml"},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            # Parsed, not regexed: an NCBI record nests <Taxon> elements inside
            # <LineageEx>, so a non-greedy <Taxon>...</Taxon> match ends at the first
            # nested close tag and silently drops most lineages.
            root = ElementTree.fromstring(response.text)
            for node in root.findall("Taxon"):
                taxid = (node.findtext("TaxId") or "").strip()
                if taxid not in results:
                    continue
                names = {part.strip().lower() for part in (node.findtext("Lineage") or "").split(";")}
                names |= {
                    (entry.findtext("ScientificName") or "").strip().lower() for entry in node.findall("./LineageEx/Taxon")
                }
                results[taxid]["lineage"] = {name for name in names if name}
        except Exception as error:
            logger.warning(f"[ncbi] efetch lineage batch failed: {error}")
    return results


def fetch_wikidata_batch(session, qids: list[str]) -> dict:
    """A batch of Wikidata entities: taxon name (P225), rank label, parent-taxon chain.

    Wikidata rate-limits aggressively, so batches are capped at 50 and retried with
    backoff; a batch that never lands is `error` for its whole slice.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|claims",
        "languages": "en",
        "format": "json",
    }
    payload = None
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json().get("entities", {})
            break
        except Exception as error:
            logger.warning(f"[wikidata] batch attempt {attempt + 1} failed: {error}")
            time.sleep(2**attempt)
    if payload is None:
        return {qid: _record(STATUS_ERROR) for qid in qids}

    results, parents = {}, {}
    for qid in qids:
        entity = payload.get(qid)
        if not isinstance(entity, dict) or "missing" in entity:
            results[qid] = _record(STATUS_UNKNOWN)
            continue
        claims = entity.get("claims", {})
        # P225 is the scientific name; the English label is the fallback for entities
        # that are a taxon by lineage but carry a vernacular label.
        name = ""
        for claim in claims.get("P225", []):
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(value, str):
                name = value
                break
        if not name:
            name = str(entity.get("labels", {}).get("en", {}).get("value") or "")
        results[qid] = _record(STATUS_OK, name, "")
        # P171 (parent taxon) is the only lineage Wikidata gives cheaply. One level is
        # enough to tell a plausible parent from a cross-phylum one, and it costs a
        # single extra batched call for the distinct parents.
        for claim in claims.get("P171", []):
            parent = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(parent, dict) and parent.get("id"):
                parents.setdefault(qid, []).append(parent["id"])

    parent_names = _fetch_wikidata_names(session, sorted({p for ids in parents.values() for p in ids}))
    for qid, parent_ids in parents.items():
        results[qid]["lineage"] = {parent_names[p] for p in parent_ids if parent_names.get(p)}
    return results


def _fetch_wikidata_names(session, qids: list[str]) -> dict:
    """``{QID: taxon name}`` for parent-taxon entities (names only, batched)."""
    names = {}
    for batch in _batched(qids, 50):
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|claims",
            "languages": "en",
            "format": "json",
        }
        for attempt in range(6):
            try:
                response = session.get("https://www.wikidata.org/w/api.php", params=params, timeout=TIMEOUT)
                if response.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                response.raise_for_status()
                for qid, entity in response.json().get("entities", {}).items():
                    if not isinstance(entity, dict) or "missing" in entity:
                        continue
                    value = ""
                    for claim in entity.get("claims", {}).get("P225", []):
                        candidate = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
                        if isinstance(candidate, str):
                            value = candidate
                            break
                    names[qid] = (value or str(entity.get("labels", {}).get("en", {}).get("value") or "")).strip().lower()
                break
            except Exception as error:
                logger.warning(f"[wikidata] parent-name batch attempt {attempt + 1} failed: {error}")
                time.sleep(2**attempt)
        time.sleep(1.0)
    return names


def fetch_gbif(session, name: str, hints: dict | None = None) -> dict:
    """GBIF backbone match for one name: is it a taxon, at what rank, in what lineage.

    Three GBIF behaviours to respect. Its matcher is CASE-SENSITIVE — `abylidae` returns
    no match where `Abylidae` is exact — and this table stores names lowercased, so the
    query is capitalized. A name it cannot place is not rejected but silently promoted:
    `Aegina` alone comes back as `matchType=HIGHERRANK` naming kingdom *Animalia*, so
    only EXACT and FUZZY count as a match. And it refuses outright on a homonym
    ("Multiple equal matches for Aegina") unless told which branch to look in — so the
    row's own Kingdom..Family are passed as hints. That turns the query into the
    stronger question anyway: is this name a taxon IN THIS LINEAGE?
    """
    url = "https://api.gbif.org/v1/species/match"
    query = name[:1].upper() + name[1:] if name else name
    params = {"name": query, "strict": "false"}
    for rank in ("Kingdom", "Phylum", "Class", "Order", "Family"):
        value = (hints or {}).get(rank, "")
        if value:
            params[rank.lower()] = value[:1].upper() + value[1:]
    try:
        response = session.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        logger.warning(f"[gbif] {name!r}: {error}")
        return _record(STATUS_ERROR)
    if str(data.get("matchType", "NONE")).upper() not in ("EXACT", "FUZZY"):
        return _record(STATUS_UNKNOWN)
    lineage = {
        str(data.get(rank) or "").strip().lower()
        for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    }
    return _record(
        STATUS_OK,
        str(data.get("canonicalName") or data.get("scientificName") or ""),
        str(data.get("rank") or ""),
        {part for part in lineage if part},
    )


def _batched(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def collect_keys(rows) -> dict:
    """Every distinct lookup the table needs, grouped by source.

    The GBIF entry carries each label's rank columns as disambiguation hints. Keying by
    label alone is sound because one label has exactly one lineage — an invariant
    ``tests/test_taxonomy_validation.py`` enforces.
    """
    keys = {source: set() for source in SOURCE_FOR_COLUMN.values()}
    gbif_hints = {}
    for row in rows:
        for column, source in SOURCE_FOR_COLUMN.items():
            if row[column].strip():
                keys[source].add(snapshot_key(column, row[column]))
        label = row["proposed_label"].strip().lower()
        if label and any(row[rank].strip() for rank in TAXONOMY_RANKS):
            gbif_hints.setdefault(label, {rank: row[rank].strip() for rank in TAXONOMY_RANKS})
    collected = {source: sorted(values) for source, values in keys.items()}
    collected[GBIF_SOURCE] = gbif_hints
    return collected


def fetch_all(keys: dict, only=None) -> dict:
    """Resolve every key against its registry. Returns ``{(source, key): record}``."""
    session = _session()
    snapshot = {}

    if only in (None, "aphia", "worms"):
        aphia = keys["worms"]
        logger.info(f"[worms] resolving {len(aphia)} AphiaIDs")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for key, record in zip(aphia, pool.map(lambda value: fetch_worms(session, value), aphia)):
                snapshot[("worms", key)] = record

    if only in (None, "ncbi"):
        taxids = keys["ncbi"]
        logger.info(f"[ncbi] resolving {len(taxids)} taxids")
        for batch in _batched(taxids, 150):
            for key, record in fetch_ncbi_batch(session, batch).items():
                snapshot[("ncbi", key)] = record
            time.sleep(0.5)  # NCBI asks for <=3 requests/second unauthenticated

    if only in (None, "wikidata"):
        qids = keys["wikidata"]
        logger.info(f"[wikidata] resolving {len(qids)} entities")
        # Wikidata rate-limits by request rate, not by payload: fewer, fuller batches
        # with a real pause between them get through where 50-id bursts do not.
        for batch in _batched(qids, 25):
            for key, record in fetch_wikidata_batch(session, batch).items():
                snapshot[("wikidata", key)] = record
            time.sleep(2.5)

    if only in (None, "gbif"):
        hints = keys[GBIF_SOURCE]
        names = sorted(hints)
        logger.info(f"[gbif] matching {len(names)} labels")
        with ThreadPoolExecutor(max_workers=8) as pool:
            matched = pool.map(lambda value: fetch_gbif(session, value, hints[value]), names)
            for key, record in zip(names, matched):
                snapshot[(GBIF_SOURCE, key)] = record

    return snapshot


def write_snapshot(snapshot: dict, path=None) -> Path:
    """Write the snapshot TSV, sorted, so a refresh diffs row-by-row."""
    path = Path(path or DEFAULT_SNAPSHOT_TSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SNAPSHOT_COLUMNS)
        for (source, key), record in sorted(snapshot.items()):
            writer.writerow(
                [source, key, record["status"], record["name"], record["rank"], "|".join(sorted(record["lineage"]))]
            )
    return path


def read_rows(path=None) -> list[dict]:
    with Path(path or DEFAULT_TAXONOMY_CSV_FILENAME).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, type=Path, help="Master taxonomy CSV (default: the committed one).")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_TSV, type=Path)
    parser.add_argument("--only", choices=["worms", "ncbi", "wikidata", "gbif"], help="Refresh one registry only.")
    parser.add_argument("--report", action="store_true", help="Score against the committed snapshot; fetch nothing.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)

    if not args.report:
        keys = collect_keys(rows)
        snapshot = fetch_all(keys, only=args.only)
        if args.snapshot.exists():
            # Refreshes are quality-monotonic: a transient failure must never DOWNGRADE a
            # record the committed snapshot already resolved, or one flaky run would
            # quietly stop verifying those identifiers while the suite stayed green.
            merged = dict(load_snapshot(args.snapshot))
            for key, record in snapshot.items():
                if record["status"] == STATUS_ERROR and merged.get(key, {}).get("status") == STATUS_OK:
                    continue
                merged[key] = record
            snapshot = merged
        # Prune to what the table still uses: a `--only` refresh merges over the old
        # snapshot, so an identifier removed from the CSV would otherwise linger here
        # forever and the staleness test would fail on a file the fetcher itself wrote.
        live = {(source, key) for source, values in keys.items() for key in values}
        snapshot = {key: value for key, value in snapshot.items() if key in live}
        write_snapshot(snapshot, args.snapshot)
        logger.info(f"Wrote {len(snapshot)} snapshot record(s) to {args.snapshot}.")

    snapshot = load_snapshot(args.snapshot)
    findings = score_table(rows, snapshot) + score_labels_against_gbif(rows, snapshot)
    flagged = [f for f in findings if f["verdict"] in (VERDICT_CONTRADICTION, VERDICT_UNKNOWN)]

    if args.json:
        json.dump(findings, sys.stdout, indent=2)
        return 0

    counts = {}
    for finding in findings:
        counts[finding["verdict"]] = counts.get(finding["verdict"], 0) + 1
    logger.info(f"Scored {len(findings)} row-identifier pair(s): {counts}")
    for finding in flagged:
        logger.warning(
            f"[{finding['verdict']}] row {finding['row']} {finding['dataset']}/{finding['raw_label']!r} "
            f"{finding['column']}={finding['value']} ({finding['label']}): {finding['detail']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
