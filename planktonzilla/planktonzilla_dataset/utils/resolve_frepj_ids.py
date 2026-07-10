"""
(c) Inria

resolve_frepj_ids.py
====================
One-time CURATION that fills the four external-ID columns
(``wikidata_ID`` / ``aphia_ID`` / ``NCBI_ID`` / ``BOLD_ID``) for the 229 ``frepj``
rows appended to ``planktonzilla_taxonomy.csv`` by Plan 18-01 (TAX-02).

Two resolution paths:

Overlap (network-free):
    A ``frepj`` taxon whose species-binomial or Genus already exists in a
    non-``frepj`` row REUSES that row's verified IDs VERBATIM (byte-identical,
    keeping the ``.0`` float format). These cross the no trust boundary — they are
    trusted in-repo values already published on HuggingFace Hub.

Draft (network, one-time):
    The remaining new-to-FREPJ taxa are DRAFT-resolved by re-running the existing
    ``extract_taxon_ids`` resolver (Wikidata Qcode -> WoRMS/NCBI/BOLD from the
    deepest known rank). Species rows are searched on the FULL binomial
    (``"Genus species"``), not the bare CSV epithet, so a bare epithet can no longer
    collide with an unrelated genus (CR-01). A Wikidata-label-vs-Genus guard BLANKS
    any Species/Genus hit whose label disagrees with the row's genus, and a
    cross-row guard BLANKS any ``wikidata_ID`` shared across differing genera. Every
    surviving network-derived ID is still DRAFT (marine-bias / rank-drift caveats),
    gated by the BLOCKING human-verify checkpoint in Plan 18-03. Freshwater taxa with
    no marine WoRMS hit keep a blank ``aphia_ID`` (expected).

Finalization (``--finalize``, network — needs NCBI_EMAIL):
    (1) BLANKS the draft rows that resolved ABOVE genus (Order/Class/Phylum) — a
    whole higher-taxon stamped onto a species/genus row is a KI-6 error, so no such
    ID ships. (2) LINEAGE-GUARDS every draft ``NCBI_ID``: it fetches the taxid's
    scientific name + lineage and BLANKS the row if the taxid's genus/phylum
    contradicts the row (a wrong-taxon substitution — the corroboration now confirms
    the taxid is the RIGHT one, not merely that a taxid exists — WR-01). (3)
    CORROBORATES every distinct surviving ``NCBI_ID`` against NCBI via
    ``extract_cox.get_cox_sequences`` (COX1 presence) + a taxonomy-id validity check;
    an ``NCBI_ID`` that does not resolve at NCBI is FLAGGED. Corroboration is advisory
    only (a taxon legitimately lacking COX1 is fine) and is resilient to rate limits.

This module is a CURATION script, exactly like the Phase-15 download and Phase-17
crosswalk — the committed tests stay network-free. Three entry modes keep the
network steps and the CSV write separate and auditable:

    # Resolve + write the drafted-ID summary (Wikidata network; no CSV write):
    python -m planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids

    # Blank too-coarse KI-6 matches + NCBI COX1 corroboration, rewrite summary (NCBI network):
    python -m planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids --finalize

    # Backfill the frepj rows from the committed summary (network-free, idempotent):
    python -m planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids --backfill

Requirements:
    polars, requests (Wikidata); biopython (NCBI COX1 corroboration). NCBI_EMAIL
    (+ optional NCBI_API_KEY) required for --finalize.
"""

import argparse
import csv
import io
import logging
import time
from pathlib import Path

import polars as pl

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.planktonzilla_dataset.utils import extract_taxon_ids
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

DATASET = "frepj"

# The four external-ID columns filled here; ecotaxa_ID stays blank for frepj.
WIKIDATA_ID = "wikidata_ID"
NUMERIC_ID_FIELDS = ("aphia_ID", "NCBI_ID", "BOLD_ID")
ID_FIELDS = (WIKIDATA_ID, *NUMERIC_ID_FIELDS)

# Kingdom -> Species, the rank columns the Wikidata resolver consumes.
RANK_COLS = list(constants.TAXONOMY_RANKS)

CSV_PATH = constants.DEFAULT_TAXONOMY_CSV_FILENAME
SUMMARY_PATH = Path(__file__).parent / "FREPJ_DRAFTED_IDS.md"

# Fences delimiting the machine-readable mapping embedded in the summary. The
# --backfill step reads this block back as the single source of truth for the
# CSV write (so backfilling never needs the network).
TSV_FENCE_OPEN = "```tsv"
TSV_FENCE_CLOSE = "```"
TSV_HEADER = ("raw_label", *ID_FIELDS, "provenance", "matched_rank", "cox")

PROV_UNRESOLVED = "unresolved"
PROV_DRAFT = "draft:wikidata"
# A draft match at these ranks is too coarse to ship (KI-6 rank drift): a whole
# order/class/phylum ID stamped onto a species/genus row. Finalization blanks them.
PROV_BLANKED = "blanked:too-coarse-KI6"
BLANK_COARSE_RANKS = frozenset({"Order", "Class", "Phylum", "Kingdom"})
# A draft match whose resolved Wikidata/NCBI entity contradicts the row's parsed
# lineage — a wrong-taxon substitution (CR-01 / WR-01). The canonical example is the
# bare epithet ``"sarsi"`` colliding with the unrelated hydrozoan genus ``Sarsia``.
# All four IDs are BLANKED rather than ship a known-wrong id: this milestone never
# ships wrong data, so a conservative blank beats a confident wrong answer.
PROV_GUARD_BLANKED = "blanked:lineage-guard"

# Positions of Genus/Species in RANK_COLS — used to build the full-binomial query.
GENUS_IDX = RANK_COLS.index("Genus")
SPECIES_IDX = RANK_COLS.index("Species")


# ── CSV helpers ──────────────────────────────────────────────────────────────────


def read_csv_rows(csv_path: Path) -> list[dict]:
    """Read the master taxonomy CSV, preserving row order and cell values."""
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def format_numeric_id(value) -> str:
    """Normalize a WoRMS/NCBI/BOLD id to the CSV's ``<int>.0`` float format.

    The Wikidata external-id claim value is an integer-valued string (e.g.
    ``"106265"``); the CSV stores it as ``"106265.0"``. Blank/None stays ``""``.
    A non-integer value is passed through verbatim with a warning (never crashes).
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return f"{int(float(text))}.0"
    except (TypeError, ValueError):
        logger.warning(f"Non-numeric external id {text!r}; keeping verbatim.")
        return text


# ── Overlap path (network-free reuse of existing verified IDs) ────────────────────


def _register_existing(index: dict, key, ids: tuple[str, ...], source: str) -> None:
    """Record an existing (verified) id set for reuse, failing LOUD on a conflict.

    WR-02: two sources that disagree on the IDs for the same species/genus key must
    not be silently tie-broken by insertion order. The first-seen set is kept (it is
    a trusted, already-published value), but any DIFFERENT later set is logged as a
    loud warning so the conflict surfaces at the 18-03 checkpoint instead of being
    swallowed. No current overlap key triggers this; it guards the next import that
    shares a genus with FREPJ.
    """
    prev = index.get(key)
    if prev is None:
        index[key] = (ids, source)
        return
    prev_ids, prev_source = prev
    if prev_ids != ids:
        logger.warning(
            f"[overlap] conflicting existing IDs for {key!r}: «{prev_source}» has {prev_ids}, "
            f"«{source}» has {ids}. Keeping «{prev_source}» (first seen); verify at 18-03."
        )


def build_existing_lookups(rows: list[dict]) -> tuple[dict, dict]:
    """Index the non-frepj rows by species-binomial and by genus-level id.

    Returns:
        ``(ex_species, ex_genus)`` where
        ``ex_species[(genus, species)] = (ids, source)`` for rows carrying both a
        Genus and a Species, and ``ex_genus[genus] = (ids, source)`` for
        genus-level rows (Species blank). ``ids`` is the ``ID_FIELDS`` tuple copied
        verbatim (``.0`` floats untouched). First occurrence wins; the overlap
        sources are known to agree byte-for-byte (18-01 reconciliation Section A),
        and any later source that DISAGREES is logged loudly rather than silently
        tie-broken (WR-02, via ``_register_existing``).
    """
    ex_species: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
    ex_genus: dict[str, tuple[tuple[str, ...], str]] = {}
    for r in rows:
        if r["Dataset"] == DATASET:
            continue
        genus = (r["Genus"] or "").strip().lower()
        species = (r["Species"] or "").strip().lower()
        ids = tuple((r[c] or "") for c in ID_FIELDS)
        if not any(ids):
            continue
        if genus and species:
            _register_existing(ex_species, (genus, species), ids, r["Dataset"])
        elif genus:
            _register_existing(ex_genus, genus, ids, r["Dataset"])
    return ex_species, ex_genus


def resolve_overlaps(frepj_rows: list[dict], ex_species: dict, ex_genus: dict) -> tuple[dict, list[dict]]:
    """Map overlap frepj rows to verbatim existing IDs; collect the rest for drafting.

    A frepj row reuses IDs when its (Genus, Species) binomial matches an existing
    species row, else when its Genus matches an existing genus-level row. Anything
    unmatched (including a genus-level frepj row whose genus only exists at the
    species level, e.g. ``keratella``) is returned for the draft path.

    Returns:
        ``(mapping, draft_rows)`` — ``mapping[raw_label]`` holds the reused record;
        ``draft_rows`` are the frepj rows still needing a network draft.
    """
    mapping: dict[str, dict] = {}
    draft_rows: list[dict] = []
    for r in frepj_rows:
        genus = (r["Genus"] or "").strip().lower()
        species = (r["Species"] or "").strip().lower()
        reuse = None
        matched_rank = ""
        if genus and species and (genus, species) in ex_species:
            reuse, matched_rank = ex_species[(genus, species)], "Species"
        elif genus and genus in ex_genus:
            reuse, matched_rank = ex_genus[genus], "Genus"

        if reuse is None:
            draft_rows.append(r)
            continue

        ids, source = reuse
        mapping[r["Raw_Labels"]] = {
            "raw_label": r["Raw_Labels"],
            "proposed_label": r["proposed_label"],
            "matched_rank": matched_rank,
            **dict(zip(ID_FIELDS, ids, strict=True)),
            "provenance": f"reused:{source}",
            "cox": "",
        }
    return mapping, draft_rows


# ── Draft path (one-time Wikidata resolution via extract_taxon_ids) ───────────────


def _blank_record(rec: dict, reason: str) -> None:
    """Blank a draft record's four external-ID cells and mark it lineage-guard-blanked.

    Used by both the Wikidata-label guard (``resolve_drafts``) and the NCBI
    lineage/name guard (``verify_ncbi_lineage``). Records the removed IDs and the
    reason for the 18-03 audit table. Idempotent: re-blanking keeps the first reason.
    """
    if rec["provenance"] != PROV_GUARD_BLANKED:
        rec["_former_ids"] = f"wd={rec[WIKIDATA_ID]} aphia={rec['aphia_ID']} NCBI={rec['NCBI_ID']} BOLD={rec['BOLD_ID']}"
        rec["_guard_reason"] = reason
    for col in ID_FIELDS:
        rec[col] = ""
    rec["provenance"] = PROV_GUARD_BLANKED
    rec["cox"] = ""


def _query_species(genus: str, species: str) -> str:
    """Return the Wikidata *search string* for a Species cell — the full binomial.

    ROOT-CAUSE FIX for CR-01. In this schema the Species column stores the BARE
    epithet only (``"sarsi"``, ``"affinis"``), which textually collides with
    unrelated genus names (``"sarsi"`` -> the hydrozoan genus ``Sarsia``) and with
    each other (three unrelated genera all searching ``"affinis"`` share one cached
    hit). Searching the ``"Genus species"`` binomial makes the query unambiguous, so
    the resolver can no longer pick a homonymous taxon. Genus-only rows (no species)
    are unchanged.
    """
    genus = (genus or "").strip()
    species = (species or "").strip()
    return f"{genus} {species}" if genus and species else species


def _rank_key(row: dict) -> tuple[str, ...]:
    """Unique-taxon key: lowercased rank columns, Species as the full binomial.

    The identical transform is applied when building the resolver's input frame and
    when mapping the resolved IDs back onto each draft row, so the two always agree.
    """
    vals = [(row[c] or "").strip().lower() for c in RANK_COLS]
    vals[SPECIES_IDX] = _query_species(vals[GENUS_IDX], vals[SPECIES_IDX])
    return tuple(vals)


def _label_matches_genus(matched_label: str, matched_rank: str, genus: str) -> bool:
    """True if a Species/Genus-rank Wikidata hit's label is consistent with the genus.

    LINEAGE GUARD (name layer). A Species/Genus-rank match whose Wikidata label does
    not begin with the row's Genus is a wrong-taxon hit (e.g. row genus
    ``sinodiaptomus`` resolving to a label ``Sarsia``). Coarser matches
    (Family/Order/…) are legitimately labelled with the higher taxon and are tracked
    by ``matched_rank`` (KI-6), not by this guard, so they pass. An absent label or
    genus also passes here (the NCBI lineage guard is the second line of defence).
    """
    if matched_rank not in ("Species", "Genus"):
        return True
    genus = (genus or "").strip().lower()
    label = (matched_label or "").strip().lower()
    if not genus or not label:
        return True
    return label.split()[0] == genus


def _unique_rank_frame(rows: list[dict]) -> pl.DataFrame:
    """Build the unique (Kingdom..Species) frame the Wikidata resolver consumes.

    The Species cell carries the FULL binomial (``"Genus species"``) rather than the
    bare CSV epithet, so the Wikidata search cannot collide a bare epithet with an
    unrelated genus (CR-01). See ``_query_species``.
    """
    seen: dict[tuple[str, ...], None] = {}
    for r in rows:
        seen.setdefault(_rank_key(r), None)
    data = {c: [key[i] for key in seen] for i, c in enumerate(RANK_COLS)}
    return pl.DataFrame(data, schema={c: pl.String for c in RANK_COLS})


def resolve_drafts(draft_rows: list[dict]) -> dict:
    """DRAFT-resolve the non-overlap frepj rows via Wikidata (one-time network).

    Resolves each unique deepest-rank taxon once (``fetch_wikidata_ids`` +
    ``fetch_external_ids``), normalizes numeric IDs to the CSV format, and maps the
    result back onto every frepj row sharing that rank tuple. Rows whose Wikidata
    lookup returns nothing keep blank IDs and provenance ``unresolved``.

    Returns:
        ``mapping[raw_label]`` for every draft row (draft IDs or expected blanks).
    """
    mapping: dict[str, dict] = {}
    if not draft_rows:
        return mapping

    unique = _unique_rank_frame(draft_rows)
    logger.info(f"[draft] resolving {unique.height} unique taxa via Wikidata.")

    resolved = extract_taxon_ids.fetch_wikidata_ids(unique)
    has_qcode = resolved.filter(pl.col(WIKIDATA_ID).is_not_null()).height
    if has_qcode:
        resolved = extract_taxon_ids.fetch_external_ids(resolved)
    else:
        logger.warning("[draft] no Wikidata Qcodes resolved; leaving all draft IDs blank.")
        for col in NUMERIC_ID_FIELDS:
            resolved = resolved.with_columns(pl.lit(None).alias(col))

    # Index the resolved unique taxa by their rank tuple (Species = full binomial).
    by_rank: dict[tuple[str, ...], dict] = {}
    for row in resolved.iter_rows(named=True):
        key = tuple((row[c] or "") for c in RANK_COLS)
        by_rank[key] = {
            WIKIDATA_ID: row.get(WIKIDATA_ID) or "",
            "aphia_ID": format_numeric_id(row.get("aphia_ID")),
            "NCBI_ID": format_numeric_id(row.get("NCBI_ID")),
            "BOLD_ID": format_numeric_id(row.get("BOLD_ID")),
            "matched_rank": row.get("Matched Rank") or "",
            "matched_label": row.get("Matched Label") or "",
        }

    guard_blanked = 0
    for r in draft_rows:
        res = by_rank.get(_rank_key(r), {})
        wikidata = res.get(WIKIDATA_ID, "")
        matched_rank = res.get("matched_rank", "")
        matched_label = res.get("matched_label", "")
        genus = (r["Genus"] or "").strip().lower()

        rec = {
            "raw_label": r["Raw_Labels"],
            "proposed_label": r["proposed_label"],
            "matched_rank": matched_rank,
            WIKIDATA_ID: wikidata,
            "aphia_ID": res.get("aphia_ID", ""),
            "NCBI_ID": res.get("NCBI_ID", ""),
            "BOLD_ID": res.get("BOLD_ID", ""),
            "provenance": PROV_DRAFT if wikidata else PROV_UNRESOLVED,
            "cox": "",
        }

        # LINEAGE GUARD (name layer): a Species/Genus hit whose Wikidata label
        # disagrees with the row's Genus is a wrong-taxon substitution — blank it
        # rather than ship a confident wrong id (CR-01).
        if wikidata and not _label_matches_genus(matched_label, matched_rank, genus):
            logger.warning(
                f"[guard] {r['Raw_Labels']!r}: Wikidata label {matched_label!r} "
                f"({matched_rank or 'no-rank'}) is not consistent with genus {genus!r}; blanking IDs."
            )
            _blank_record(rec, reason=f"wikidata label {matched_label!r} != genus {genus!r}")
            guard_blanked += 1

        mapping[r["Raw_Labels"]] = rec

    if guard_blanked:
        logger.warning(f"[guard] blanked {guard_blanked} draft rows on Wikidata-label/genus mismatch.")
    return mapping


def blank_cross_genus_shared_ids(mapping: dict, frepj_rows: list[dict]) -> list[str]:
    """Blank draft rows that share one ``wikidata_ID`` across DIFFERENT genera (CR-01).

    Two distinct genera cannot legitimately be the same Wikidata taxon, so a
    ``wikidata_ID`` shared by draft rows whose Genus differs is a bare-epithet
    collision — the original ``affinis`` bug, where three unrelated genera inherited
    one cached hit. Every draft row in such a group is BLANKED (conservative). Reused
    overlap rows legitimately share a genus-level id WITHIN a single genus and are
    never considered here. Network-free. Returns the blanked raw_labels.
    """
    genus_of = {r["Raw_Labels"]: (r["Genus"] or "").strip().lower() for r in frepj_rows}
    by_qid: dict[str, set[str]] = {}
    for rec in mapping.values():
        if rec["provenance"] == PROV_DRAFT and rec[WIKIDATA_ID]:
            by_qid.setdefault(rec[WIKIDATA_ID], set()).add(genus_of.get(rec["raw_label"], ""))

    conflicting = {qid for qid, genera in by_qid.items() if len({g for g in genera if g}) > 1}
    blanked: list[str] = []
    for rec in mapping.values():
        if rec["provenance"] == PROV_DRAFT and rec[WIKIDATA_ID] in conflicting:
            shared_genera = sorted(g for g in by_qid[rec[WIKIDATA_ID]] if g)
            logger.warning(
                f"[guard] {rec['raw_label']!r}: wikidata_ID {rec[WIKIDATA_ID]} shared across "
                f"differing genera {shared_genera}; blanking IDs."
            )
            _blank_record(rec, reason=f"wikidata_ID {rec[WIKIDATA_ID]} shared across genera {shared_genera}")
            blanked.append(rec["raw_label"])
    return blanked


def blank_coarse_matches(mapping: dict) -> list[str]:
    """Blank the external IDs of draft rows that resolved above genus (KI-6).

    A ``draft:wikidata`` row whose ``matched_rank`` is Order/Class/Phylum/Kingdom
    carries a misleadingly-coarse ID (a whole higher-taxon stamped onto a
    species/genus row). Blank all four ID cells so no such ID ships; the taxonomy
    rank columns + proposed_label are left untouched by this (backfill edits only
    the ID cells). Returns the blanked ``raw_label`` list (CSV row order preserved
    by the caller). Idempotent: an already-blanked row stays blanked.
    """
    blanked: list[str] = []
    for rec in mapping.values():
        if rec["provenance"] in (PROV_DRAFT, PROV_BLANKED) and rec["matched_rank"] in BLANK_COARSE_RANKS:
            if rec["provenance"] != PROV_BLANKED:
                rec["_former_ids"] = (
                    f"wd={rec[WIKIDATA_ID]} aphia={rec['aphia_ID']} NCBI={rec['NCBI_ID']} BOLD={rec['BOLD_ID']}"
                )
            for col in ID_FIELDS:
                rec[col] = ""
            rec["provenance"] = PROV_BLANKED
            rec["cox"] = ""
            blanked.append(rec["raw_label"])
    return blanked


def _taxid_resolves(tax_id: str) -> bool | None:
    """Return True if NCBI Taxonomy knows ``tax_id``, False if unknown, None if unsure.

    ``efetch(db=taxonomy)`` returns an empty record for an unknown taxid (it does
    not raise), so validity is ``len(record) > 0``. Retries on rate limit; returns
    None only when the lookup could not be completed (so a transient error is never
    mislabelled as a wrong ID).
    """
    from Bio import Entrez

    for attempt in range(3):
        try:
            handle = Entrez.efetch(db="taxonomy", id=str(tax_id), retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            return len(record) > 0
        except Exception as e:  # network / rate limit — retry then give up as "unsure"
            if "429" in str(e) or "Too Many" in str(e):
                time.sleep(2**attempt)
                continue
            logger.warning(f"[cox] taxonomy lookup failed for {tax_id}: {e}")
            return None
    return None


def _lineage_names(node: dict) -> set[str]:
    """Lowercased set of every scientific name in an NCBI taxonomy record's lineage."""
    names: set[str] = set()
    for entry in node.get("LineageEx", []) or []:
        name = str(entry.get("ScientificName", "")).strip().lower()
        if name:
            names.add(name)
    for name in str(node.get("Lineage", "")).split(";"):
        name = name.strip().lower()
        if name:
            names.add(name)
    return names


_TAXID_INFO_CACHE: dict[str, dict | None] = {}


def fetch_taxid_info(tax_id: str) -> dict | None:
    """Fetch an NCBI taxid's scientific name + rank + lineage names (cached).

    Returns ``{"name", "rank", "lineage"}`` (all lowercased; ``lineage`` a set) or
    ``None`` when the taxid is unknown or the lookup could not complete. Retries on
    rate limit; a transient failure returns ``None`` so it is never read as a wrong
    id. Feeds the WR-01 lineage guard.
    """
    if tax_id in _TAXID_INFO_CACHE:
        return _TAXID_INFO_CACHE[tax_id]

    from Bio import Entrez

    info = None
    for attempt in range(3):
        try:
            handle = Entrez.efetch(db="taxonomy", id=str(tax_id), retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            if record:
                node = record[0]
                info = {
                    "name": str(node.get("ScientificName", "")).strip().lower(),
                    "rank": str(node.get("Rank", "")).strip().lower(),
                    "lineage": _lineage_names(node),
                }
            break
        except Exception as e:  # network / rate limit — retry then give up as "unsure"
            if "429" in str(e) or "Too Many" in str(e):
                time.sleep(2**attempt)
                continue
            logger.warning(f"[guard] taxid info lookup failed for {tax_id}: {e}")
            break

    _TAXID_INFO_CACHE[tax_id] = info
    return info


def _ncbi_lineage_consistent(info: dict, genus: str, phylum: str) -> bool:
    """True if an NCBI taxid's name/lineage is consistent with the row's Genus+Phylum.

    The taxid must agree on BOTH the genus (its scientific-name first word equals the
    row genus, or the genus appears anywhere in its lineage) AND the phylum (the row
    phylum appears in its lineage, or equals its name). Empty row values are not
    checked. This is what turns "a taxid exists" into "it is the RIGHT taxid" (WR-01):
    the wrong copepod→hydrozoan taxid (Sinodiaptomus→Sarsia) fails the phylum test.
    """
    name = info["name"]
    lineage = info["lineage"]
    genus_ok = True
    if genus:
        first_word = name.split()[0] if name else ""
        genus_ok = (genus == first_word) or (genus in lineage)
    phylum_ok = True
    if phylum:
        phylum_ok = (phylum in lineage) or (phylum == name)
    return genus_ok and phylum_ok


def verify_ncbi_lineage(mapping: dict) -> list[str]:
    """Blank draft rows whose NCBI taxid contradicts the row's Genus/Phylum (WR-01).

    For every DRAFT row carrying an NCBI_ID, fetch the taxid's scientific name +
    lineage and check it against the row's parsed Genus and Phylum. A contradiction
    (wrong-taxon substitution) BLANKS all four IDs — the taxid, and the Wikidata
    entity it was derived from, are the wrong organism. Rows whose lookup is
    inconclusive are left untouched (a transient error must never be read as a wrong
    id). Only ``draft:wikidata`` rows are checked; reused-overlap IDs are trusted and
    never blanked. Network step (NCBI Entrez). Returns the blanked raw_labels.
    """
    draft = [rec for rec in mapping.values() if rec["provenance"] == PROV_DRAFT and rec["NCBI_ID"]]
    logger.info(f"[guard] verifying NCBI lineage for {len(draft)} draft rows with an NCBI_ID.")
    blanked: list[str] = []
    for rec in draft:
        tax_id = str(int(float(rec["NCBI_ID"])))
        genus = rec.get("_genus", "")
        phylum = rec.get("_phylum", "")
        info = fetch_taxid_info(tax_id)
        if info is None:
            logger.warning(f"[guard] {rec['raw_label']!r}: taxid {tax_id} lineage inconclusive; leaving as-is.")
            continue
        if not _ncbi_lineage_consistent(info, genus, phylum):
            logger.warning(
                f"[guard] {rec['raw_label']!r}: NCBI taxid {tax_id} ({info['name']!r}, rank {info['rank']!r}) "
                f"contradicts row genus={genus!r}/phylum={phylum!r}; blanking IDs."
            )
            _blank_record(rec, reason=f"NCBI taxid {tax_id} ({info['name']}) contradicts genus {genus}/phylum {phylum}")
            blanked.append(rec["raw_label"])
    return blanked


def corroborate_ncbi_ids(ncbi_ids: list[str]) -> dict[str, str]:
    """Cross-check NCBI_IDs against NCBI: COX1 presence + taxid validity.

    Corroboration only. Per distinct NCBI_ID the status is one of:
        ``cox:<n>``         n>=1 COX1 records found -> corroborated
        ``cox:0``           taxid valid at NCBI, no COX1 record (fine, not a defect)
        ``cox:invalid-id``  taxid does NOT resolve at NCBI -> FLAG (potential wrong id)
        ``cox:inconclusive`` lookup could not complete (rate limit / error)
    Resilient to rate limits (extract_cox sleeps + retries; failures log + continue).
    Requires ``NCBI_EMAIL`` (raises if unset — the finalize caller has verified it).
    """
    from planktonzilla.planktonzilla_dataset.utils import extract_cox

    extract_cox.configure_entrez()
    out: dict[str, str] = {}
    total = len(ncbi_ids)
    for idx, raw in enumerate(ncbi_ids, start=1):
        tax_id = str(int(float(raw)))
        logger.info(f"[cox] {idx}/{total} NCBI_ID {tax_id}")
        try:
            records = extract_cox.get_cox_sequences(tax_id, expand_to_children=True, max_results=5)
            n = len(records)
        except Exception as e:  # corroboration only, never fatal
            logger.warning(f"[cox] COX1 lookup failed for {tax_id}: {e}")
            n = 0
        if n > 0:
            out[raw] = f"cox:{n}"
            continue
        valid = _taxid_resolves(tax_id)
        out[raw] = {True: "cox:0", False: "cox:invalid-id", None: "cox:inconclusive"}[valid]
    return out


def apply_corroboration(mapping: dict, statuses: dict[str, str]) -> None:
    """Stamp each taxon's ``cox`` field from its NCBI_ID's corroboration status."""
    for rec in mapping.values():
        rec["cox"] = statuses.get(rec["NCBI_ID"], "") if rec["NCBI_ID"] else ""


# ── Summary emission + machine-readable mapping ──────────────────────────────────


def _is_corroborated(cox: str) -> bool:
    """True when the cox status records at least one COX1 record (``cox:<n>``, n>=1)."""
    return cox.startswith("cox:") and cox[4:].isdigit() and int(cox[4:]) > 0


def _counts(mapping: dict) -> dict:
    """Compute per-source, per-fill and COX1-corroboration tallies for the summary."""
    reused = sum(1 for r in mapping.values() if r["provenance"].startswith("reused:"))
    draft = sum(1 for r in mapping.values() if r["provenance"] == PROV_DRAFT)
    unresolved = sum(1 for r in mapping.values() if r["provenance"] == PROV_UNRESOLVED)
    blanked = sum(1 for r in mapping.values() if r["provenance"] == PROV_BLANKED)
    guard_blanked = sum(1 for r in mapping.values() if r["provenance"] == PROV_GUARD_BLANKED)
    filled = {c: sum(1 for r in mapping.values() if r[c]) for c in ID_FIELDS}
    cox_vals = [r["cox"] for r in mapping.values() if r["NCBI_ID"]]
    cox = {
        "checked": sum(1 for v in cox_vals if v),
        "corroborated": sum(1 for v in cox_vals if _is_corroborated(v)),
        "no_cox": sum(1 for v in cox_vals if v == "cox:0"),
        "flagged": sum(1 for v in cox_vals if v == "cox:invalid-id"),
        "inconclusive": sum(1 for v in cox_vals if v == "cox:inconclusive"),
    }
    return {
        "reused": reused,
        "draft": draft,
        "unresolved": unresolved,
        "blanked": blanked,
        "guard_blanked": guard_blanked,
        "filled": filled,
        "cox": cox,
    }


def _machine_block(mapping: dict, order: list[str]) -> str:
    """Render the TSV mapping block (backfill source of truth), one row per taxon."""
    buf = io.StringIO()
    buf.write("\t".join(TSV_HEADER) + "\n")
    for raw in order:
        rec = mapping[raw]
        buf.write("\t".join(str(rec[c]) for c in TSV_HEADER) + "\n")
    return buf.getvalue()


def write_summary(mapping: dict, order: list[str]) -> None:
    """Write FREPJ_DRAFTED_IDS.md: counts, human tables, KI caveats + TSV mapping."""
    c = _counts(mapping)
    lines: list[str] = []
    lines.append("<!--")
    lines.append("(c) Inria")
    lines.append("-->")
    lines.append("")
    lines.append("# FREPJ Drafted External-ID Summary (TAX-02)")
    lines.append("")
    lines.append(
        "Generated by `resolve_frepj_ids.py`. Records the external IDs "
        "(`wikidata_ID` / `aphia_ID` / `NCBI_ID` / `BOLD_ID`) filled into the 229 "
        "`frepj` rows of `planktonzilla_taxonomy.csv`, with per-taxon **provenance** so "
        "the Plan 18-03 human-verify checkpoint knows exactly what to spot-check. "
        "`ecotaxa_ID` stays blank for all `frepj` rows."
    )
    lines.append("")
    lines.append("## Provenance legend")
    lines.append("")
    lines.append("- `reused:<source>` — copied VERBATIM from an existing verified non-frepj row (no network, trusted).")
    lines.append("- `draft:wikidata` — DRAFT, resolved live from Wikidata this run (UNTRUSTED; needs sign-off).")
    lines.append(
        "- `blanked:too-coarse-KI6` — a draft that resolved ABOVE genus (Order/Class/Phylum); IDs BLANKED "
        "at finalization so no misleadingly-coarse whole-higher-taxon ID ships."
    )
    lines.append(
        "- `blanked:lineage-guard` — a draft whose resolved Wikidata/NCBI entity CONTRADICTS the row's "
        "genus/phylum (a wrong-taxon substitution, e.g. `sarsi`→hydrozoan `Sarsia`); all four IDs BLANKED "
        "rather than ship a confident wrong id (CR-01 / WR-01)."
    )
    lines.append("- `unresolved` — Wikidata returned no biological match; all four IDs left blank.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Taxa total: **{len(mapping)}** (must equal 229).")
    lines.append(f"- Reused verbatim (overlap): **{c['reused']}**.")
    lines.append(f"- Draft-resolved via Wikidata (shipping): **{c['draft']}**.")
    lines.append(f"- Blanked too-coarse (KI-6, removed at finalization): **{c['blanked']}**.")
    lines.append(f"- Blanked lineage-guard (wrong-taxon, CR-01 / WR-01): **{c['guard_blanked']}**.")
    lines.append(f"- Unresolved (all IDs blank): **{c['unresolved']}**.")
    lines.append(
        f"- Filled cells — wikidata_ID: **{c['filled'][WIKIDATA_ID]}**, "
        f"aphia_ID: **{c['filled']['aphia_ID']}**, "
        f"NCBI_ID: **{c['filled']['NCBI_ID']}**, "
        f"BOLD_ID: **{c['filled']['BOLD_ID']}**."
    )
    lines.append("")
    lines.append("## NCBI COX1 corroboration (finalization cross-check)")
    lines.append("")
    lines.append(
        "Each distinct shipping `NCBI_ID` was cross-checked against NCBI (COX1 presence + taxid validity). "
        "This is CORROBORATION ONLY — a taxon legitimately lacking a COX1 record is fine. Status codes: "
        "`cox:<n>` = n COX1 records found (corroborated); `cox:0` = valid taxid, no COX1 (fine); "
        "`cox:invalid-id` = taxid does NOT resolve at NCBI (**FLAG — potential wrong id**); "
        "`cox:inconclusive` = lookup could not complete."
    )
    lines.append("")
    lines.append(f"- NCBI_IDs checked (per-row): **{c['cox']['checked']}**.")
    lines.append(f"- Corroborated (>=1 COX1 record): **{c['cox']['corroborated']}**.")
    lines.append(f"- Valid taxid, no COX1 (fine): **{c['cox']['no_cox']}**.")
    lines.append(f"- **FLAGGED invalid taxid (verify): {c['cox']['flagged']}**.")
    lines.append(f"- Inconclusive (retry at review): **{c['cox']['inconclusive']}**.")
    lines.append("")
    flagged = [mapping[r] for r in order if mapping[r]["NCBI_ID"] and mapping[r]["cox"] == "cox:invalid-id"]
    if flagged:
        lines.append("**Flagged NCBI_IDs (taxid did not resolve at NCBI — likely wrong id):**")
        lines.append("")
        lines.append("| proposed_label | NCBI_ID | provenance | raw_label |")
        lines.append("| --- | --- | --- | --- |")
        lines.extend(
            f"| {rec['proposed_label']} | {rec['NCBI_ID']} | {rec['provenance']} | {rec['raw_label']} |" for rec in flagged
        )
    else:
        lines.append("No NCBI_ID failed to resolve at NCBI — every shipping NCBI_ID is a valid NCBI taxid.")
    lines.append("")
    lines.append("## Caveats — KI-3 / KI-5 / KI-6 (what to spot-check at 18-03)")
    lines.append("")
    lines.append(
        "- **KI-3 (bare-epithet / homonym collision):** species rows are now searched on the FULL "
        "binomial (`Genus species`), not the bare epithet, so a bare epithet (`sarsi`) can no longer "
        "collide with an unrelated genus (`Sarsia`). Two guards back this up: a Wikidata-label vs Genus "
        "check and an NCBI taxid genus+phylum lineage check; a resolved entity that contradicts the row's "
        "lineage is BLANKED (`blanked:lineage-guard`), never shipped. At 18-03 still spot-check every "
        "`draft:wikidata` row's Wikidata label against its Genus (not only cross-kingdom homonyms) — the "
        "demonstrated failure was a same-kingdom bare-epithet collision."
    )
    lines.append(
        "- **KI-5 (marine bias):** `aphia_ID` (WoRMS) is a marine register. Freshwater "
        "FREPJ taxa legitimately have **no** `aphia_ID` — a blank aphia is EXPECTED, not a defect."
    )
    lines.append(
        "- **KI-6 (rank drift):** when a species has no Wikidata entry the resolver falls back "
        "up the ranks; check the `matched_rank` column — a genus/family-level match assigned to a "
        "species row is a DRAFT approximation."
    )
    lines.append("")

    reused = [mapping[r] for r in order if mapping[r]["provenance"].startswith("reused:")]
    drafts = [mapping[r] for r in order if mapping[r]["provenance"] == PROV_DRAFT]
    blanked = [mapping[r] for r in order if mapping[r]["provenance"] == PROV_BLANKED]
    guard_blanked = [mapping[r] for r in order if mapping[r]["provenance"] == PROV_GUARD_BLANKED]

    lines.append("## Blanked lineage-guard matches (CR-01 / WR-01 — wrong-taxon, removed)")
    lines.append("")
    lines.append(
        f"{len(guard_blanked)} draft taxa resolved to an entity whose genus/phylum CONTRADICTS the row "
        "(a bare-epithet homonym collision or a wrong NCBI taxid), so all four ID cells were BLANKED — no "
        "known-wrong id ships. Taxonomy rank columns + proposed_label are unchanged. Listed for the 18-03 record:"
    )
    lines.append("")
    lines.append("| raw_label | proposed_label | former IDs (removed) | reason |")
    lines.append("| --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['raw_label']} | {rec['proposed_label']} | {rec.get('_former_ids', '(blanked)')} "
        f"| {rec.get('_guard_reason', '')} |"
        for rec in guard_blanked
    )
    lines.append("")

    lines.append("## Blanked too-coarse matches (KI-6 — removed at finalization)")
    lines.append("")
    lines.append(
        f"{len(blanked)} draft taxa resolved ABOVE genus (Order/Class/Phylum) — a whole higher-taxon "
        "stamped onto a species/genus row — so all four ID cells were BLANKED (they do NOT ship). "
        "Taxonomy rank columns + proposed_label are unchanged. Listed for the 18-03 record:"
    )
    lines.append("")
    lines.append("| raw_label | former matched_rank | former IDs (removed) |")
    lines.append("| --- | --- | --- |")
    lines.extend(f"| {rec['raw_label']} | {rec['matched_rank']} | {rec.get('_former_ids', '(blanked)')} |" for rec in blanked)
    lines.append("")

    # Any remaining draft matches still above genus (Family-level) are the next KI-6 risk.
    priority = [rec for rec in drafts if rec["matched_rank"] in ("Family", "Order", "Class", "Phylum", "Kingdom", "")]
    lines.append("## Priority spot-checks (remaining above-genus draft matches — verify first)")
    lines.append("")
    lines.append(
        f"{len(priority)} SHIPPING draft taxa still resolved above genus (family-level and below the "
        "blanked ranks); each ID is a coarse DRAFT approximation — verify or blank these first at 18-03."
    )
    lines.append("")
    lines.append("| raw_label | matched_rank | wikidata_ID | aphia_ID | NCBI_ID | BOLD_ID | cox |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['raw_label']} | {rec['matched_rank']} | "
        f"{rec[WIKIDATA_ID]} | {rec['aphia_ID']} | {rec['NCBI_ID']} | {rec['BOLD_ID']} | {rec['cox']} |"
        for rec in priority
    )
    lines.append("")

    lines.append("## Overlap reuse (verbatim existing IDs)")
    lines.append("")
    lines.append("| proposed_label | provenance | matched_rank | wikidata_ID | aphia_ID | NCBI_ID | BOLD_ID | cox |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['proposed_label']} | {rec['provenance']} | {rec['matched_rank']} | "
        f"{rec[WIKIDATA_ID]} | {rec['aphia_ID']} | {rec['NCBI_ID']} | {rec['BOLD_ID']} | {rec['cox']} |"
        for rec in reused
    )
    lines.append("")

    lines.append("## Draft resolutions (Wikidata — DRAFT, gated by 18-03)")
    lines.append("")
    lines.append("| proposed_label | provenance | matched_rank | wikidata_ID | aphia_ID | NCBI_ID | BOLD_ID | cox |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['proposed_label']} | {rec['provenance']} | {rec['matched_rank']} | "
        f"{rec[WIKIDATA_ID]} | {rec['aphia_ID']} | {rec['NCBI_ID']} | {rec['BOLD_ID']} | {rec['cox']} |"
        for rec in drafts
    )
    lines.append("")

    lines.append("## Machine-readable mapping (backfill source of truth)")
    lines.append("")
    lines.append(
        "The `--backfill` step reads the block below (keyed by byte-exact `raw_label`) "
        "to fill ONLY the four ID cells of the matching `frepj` rows. Do not hand-edit."
    )
    lines.append("")
    lines.append(TSV_FENCE_OPEN)
    lines.append(_machine_block(mapping, order).rstrip("\n"))
    lines.append(TSV_FENCE_CLOSE)
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote drafted-ID summary -> {SUMMARY_PATH}")


def parse_summary() -> dict:
    """Read the machine-readable TSV block back into ``mapping[raw_label]``.

    This lets ``--backfill`` apply the committed summary without any network call.
    """
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    if TSV_FENCE_OPEN not in text:
        raise ValueError(f"No {TSV_FENCE_OPEN} mapping block found in {SUMMARY_PATH}")
    block = text.split(TSV_FENCE_OPEN, 1)[1].split(TSV_FENCE_CLOSE, 1)[0].strip("\n")
    reader = csv.DictReader(io.StringIO(block), delimiter="\t")
    mapping: dict[str, dict] = {}
    for row in reader:
        mapping[row["raw_label"]] = row
    return mapping


# ── CSV backfill (idempotent, frepj-rows-only) ───────────────────────────────────


def backfill_csv(csv_path: Path, mapping: dict) -> int:
    """Fill the four ID cells of every frepj row from ``mapping`` (idempotent).

    Edits ONLY lines beginning ``frepj,`` and, within them, ONLY the last-but-one
    through last-but-four fields (wikidata/aphia/NCBI/BOLD). ``ecotaxa_ID`` (the
    final field) and all non-frepj lines stay byte-identical, so the append-only
    sha256 of the first 1486 lines is preserved. Re-running is a no-op.

    Returns:
        The number of frepj rows rewritten.
    """
    raw = csv_path.read_bytes()
    out_lines: list[bytes] = []
    changed = 0
    missing: list[str] = []
    for physical in raw.splitlines(keepends=True):
        text = physical.decode("utf-8")
        if not text.startswith(f"{DATASET},"):
            out_lines.append(physical)
            continue
        newline = "\n" if text.endswith("\n") else ""
        content = text[: -len(newline)] if newline else text
        raw_label = next(csv.reader([content]))[1]
        rec = mapping.get(raw_label)
        if rec is None:
            missing.append(raw_label)
            out_lines.append(physical)
            continue
        # The trailing five fields never contain commas, so rsplit isolates them
        # without touching the quoted Raw_Labels commas earlier in the line.
        head, _w, _a, _n, _b, ecotaxa = content.rsplit(",", 5)
        rebuilt = f"{head},{rec[WIKIDATA_ID]},{rec['aphia_ID']},{rec['NCBI_ID']},{rec['BOLD_ID']},{ecotaxa}{newline}"
        out_lines.append(rebuilt.encode("utf-8"))
        changed += 1

    if missing:
        raise ValueError(f"{len(missing)} frepj rows have no mapping entry (first: {missing[0]!r})")

    csv_path.write_bytes(b"".join(out_lines))
    logger.info(f"Backfilled {changed} frepj rows in {csv_path}")
    return changed


# ── Orchestration ────────────────────────────────────────────────────────────────


def frepj_order() -> list[str]:
    """The frepj Raw_Labels in CSV row order (mapping/summary ordering)."""
    return [r["Raw_Labels"] for r in read_csv_rows(CSV_PATH) if r["Dataset"] == DATASET]


def resolve() -> dict:
    """Run the resolution (overlap reuse + Wikidata draft) -> mapping in CSV order.

    COX1 corroboration is deliberately NOT done here — it is a separate ``--finalize``
    step that needs ``NCBI_EMAIL`` and must not gate the network-free draft.
    """
    rows = read_csv_rows(CSV_PATH)
    frepj_rows = [r for r in rows if r["Dataset"] == DATASET]
    logger.info(f"{len(frepj_rows)} frepj rows to resolve.")

    ex_species, ex_genus = build_existing_lookups(rows)
    mapping, draft_rows = resolve_overlaps(frepj_rows, ex_species, ex_genus)
    logger.info(f"Overlap reuse: {len(mapping)} rows; draft: {len(draft_rows)} rows.")

    mapping.update(resolve_drafts(draft_rows))

    shared = blank_cross_genus_shared_ids(mapping, frepj_rows)
    if shared:
        logger.warning(f"Blanked {len(shared)} draft rows sharing a wikidata_ID across differing genera.")

    order = [r["Raw_Labels"] for r in frepj_rows]
    missing = [raw for raw in order if raw not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} frepj rows unresolved by both paths (first: {missing[0]!r})")
    return {raw: mapping[raw] for raw in order}


def load_mapping_from_summary() -> dict:
    """Load the committed mapping, re-attaching proposed_label/Genus/Phylum from the CSV.

    ``proposed_label`` restores the human-facing label for the summary tables;
    ``_genus``/``_phylum`` (transient, not written back to the TSV) are the row's
    parsed taxonomy the NCBI lineage guard checks the resolved taxid against (WR-01).
    CSV row order is preserved.
    """
    parsed = parse_summary()
    csv_by_raw = {r["Raw_Labels"]: r for r in read_csv_rows(CSV_PATH) if r["Dataset"] == DATASET}
    order = frepj_order()
    missing = [raw for raw in order if raw not in parsed]
    if missing:
        raise ValueError(f"{len(missing)} frepj rows absent from the summary (first: {missing[0]!r})")
    for raw, rec in parsed.items():
        src = csv_by_raw.get(raw, {})
        rec["proposed_label"] = src.get("proposed_label", "")
        rec["_genus"] = (src.get("Genus") or "").strip().lower()
        rec["_phylum"] = (src.get("Phylum") or "").strip().lower()
    return {raw: parsed[raw] for raw in order}


def finalize() -> tuple[list[str], list[str], dict[str, str]]:
    """Blank too-coarse KI-6 + lineage-guard wrong-taxon draft IDs + COX1 corroborate.

    Network step (NCBI Entrez). In order: (1) blank the KI-6 too-coarse (above-genus)
    draft matches; (2) verify every draft NCBI_ID's taxon lineage against the row's
    Genus/Phylum and BLANK wrong-taxon substitutions (WR-01); (3) COX1-corroborate the
    surviving distinct NCBI_IDs (advisory). Returns
    ``(coarse_blanked, guard_blanked, ncbi_statuses)`` for the report. The CSV write
    itself stays a separate network-free ``--backfill``.
    """
    from planktonzilla.planktonzilla_dataset.utils import extract_cox

    extract_cox.configure_entrez()  # raises if NCBI_EMAIL unset — verified before any efetch.

    mapping = load_mapping_from_summary()

    coarse = blank_coarse_matches(mapping)
    logger.info(f"Blanked {len(coarse)} too-coarse (Order/Class/Phylum) draft rows.")

    guarded = verify_ncbi_lineage(mapping)
    logger.info(f"Lineage-guard blanked {len(guarded)} draft rows whose NCBI taxid contradicts the row.")

    # Corroborate every distinct NCBI_ID that survives the blanking.
    surviving = sorted({rec["NCBI_ID"] for rec in mapping.values() if rec["NCBI_ID"]})
    logger.info(f"Corroborating {len(surviving)} distinct surviving NCBI_IDs against NCBI.")
    statuses = corroborate_ncbi_ids(surviving)
    apply_corroboration(mapping, statuses)

    write_summary(mapping, frepj_order())
    return coarse, guarded, statuses


def main() -> None:
    """CLI: resolve (default) | --finalize (COX + blank) | --backfill (CSV write)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--backfill",
        action="store_true",
        help="Network-free: backfill the frepj CSV rows from the committed FREPJ_DRAFTED_IDS.md.",
    )
    mode.add_argument(
        "--finalize",
        action="store_true",
        help="Network (NCBI): blank the too-coarse KI-6 matches + run COX1 corroboration, rewrite the summary.",
    )
    args = parser.parse_args()

    if args.backfill:
        mapping = parse_summary()
        logger.info(f"Loaded {len(mapping)} taxa from the committed summary.")
        backfill_csv(CSV_PATH, mapping)
        return

    if args.finalize:
        coarse, guarded, statuses = finalize()
        flagged = [k for k, v in statuses.items() if v == "cox:invalid-id"]
        logger.info(
            f"Finalize done: {len(coarse)} too-coarse blanked; {len(guarded)} lineage-guard blanked; "
            f"{len(flagged)} NCBI_IDs flagged invalid."
        )
        return

    mapping = resolve()
    write_summary(mapping, frepj_order())


if __name__ == "__main__":
    main()
