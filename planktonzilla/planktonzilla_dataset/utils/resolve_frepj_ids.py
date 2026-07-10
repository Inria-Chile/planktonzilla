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
    deepest known rank). Every network-derived ID is DRAFT and carries the
    KI-3/5/6 substring-match / marine-bias weakness, so it is gated by the
    BLOCKING human-verify checkpoint in Plan 18-03 (auto-accept is Out-of-Scope).
    Freshwater taxa with no marine WoRMS hit keep a blank ``aphia_ID`` (expected).

Optional corroboration:
    For a small sample of drafted ``NCBI_ID`` values, ``extract_cox.get_cox_sequences``
    is queried; a non-empty COX1 result is weak positive evidence the id is plausible.
    Skipped gracefully when ``NCBI_EMAIL`` is unset (it is only corroboration).

This module is a CURATION script, exactly like the Phase-15 download and Phase-17
crosswalk — the committed tests stay network-free. Two entry modes keep the
network step and the CSV write separate and auditable:

    # Resolve + write the drafted-ID summary (network; no CSV write):
    python -m planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids

    # Backfill the frepj rows from the committed summary (network-free, idempotent):
    python -m planktonzilla.planktonzilla_dataset.utils.resolve_frepj_ids --backfill

Requirements:
    polars, requests (Wikidata) — biopython only for the optional COX cross-check.
"""

import argparse
import csv
import io
import logging
import os
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


def build_existing_lookups(rows: list[dict]) -> tuple[dict, dict]:
    """Index the non-frepj rows by species-binomial and by genus-level id.

    Returns:
        ``(ex_species, ex_genus)`` where
        ``ex_species[(genus, species)] = (ids, source)`` for rows carrying both a
        Genus and a Species, and ``ex_genus[genus] = (ids, source)`` for
        genus-level rows (Species blank). ``ids`` is the ``ID_FIELDS`` tuple copied
        verbatim (``.0`` floats untouched). First occurrence wins; the overlap
        sources are known to agree byte-for-byte (18-01 reconciliation Section A).
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
            ex_species.setdefault((genus, species), (ids, r["Dataset"]))
        elif genus:
            ex_genus.setdefault(genus, (ids, r["Dataset"]))
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


def _unique_rank_frame(rows: list[dict]) -> pl.DataFrame:
    """Build the unique (Kingdom..Species) frame the Wikidata resolver consumes."""
    seen: dict[tuple[str, ...], None] = {}
    for r in rows:
        key = tuple((r[c] or "").strip().lower() for c in RANK_COLS)
        seen.setdefault(key, None)
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

    # Index the resolved unique taxa by their rank tuple.
    by_rank: dict[tuple[str, ...], dict] = {}
    for row in resolved.iter_rows(named=True):
        key = tuple((row[c] or "") for c in RANK_COLS)
        by_rank[key] = {
            WIKIDATA_ID: row.get(WIKIDATA_ID) or "",
            "aphia_ID": format_numeric_id(row.get("aphia_ID")),
            "NCBI_ID": format_numeric_id(row.get("NCBI_ID")),
            "BOLD_ID": format_numeric_id(row.get("BOLD_ID")),
            "matched_rank": row.get("Matched Rank") or "",
        }

    for r in draft_rows:
        key = tuple((r[c] or "").strip().lower() for c in RANK_COLS)
        res = by_rank.get(key, {})
        wikidata = res.get(WIKIDATA_ID, "")
        mapping[r["Raw_Labels"]] = {
            "raw_label": r["Raw_Labels"],
            "proposed_label": r["proposed_label"],
            "matched_rank": res.get("matched_rank", ""),
            WIKIDATA_ID: wikidata,
            "aphia_ID": res.get("aphia_ID", ""),
            "NCBI_ID": res.get("NCBI_ID", ""),
            "BOLD_ID": res.get("BOLD_ID", ""),
            "provenance": PROV_DRAFT if wikidata else PROV_UNRESOLVED,
            "cox": "",
        }
    return mapping


def cox_crosscheck(mapping: dict, sample: int) -> None:
    """Annotate a sample of drafted NCBI_IDs with a weak COX1 plausibility signal.

    Corroboration only: sets ``cox`` to ``cox:<n>`` (n COX1 records found),
    ``cox:0`` (none), or ``cox:skipped`` when ``NCBI_EMAIL`` is unset / the query
    errors. Never raises — a failed cross-check must not fail the curation run.
    """
    if sample <= 0:
        return
    if not os.environ.get("NCBI_EMAIL"):
        logger.warning("[cox] NCBI_EMAIL unset; skipping COX1 corroboration (Wikidata NCBI_IDs kept as-is).")
        return

    from planktonzilla.planktonzilla_dataset.utils import extract_cox

    try:
        extract_cox.configure_entrez()
    except Exception as e:  # corroboration only, never fatal
        logger.warning(f"[cox] could not configure Entrez, skipping: {e}")
        return

    drafted = [rec for rec in mapping.values() if rec["provenance"] == PROV_DRAFT and rec["NCBI_ID"]]
    for rec in drafted[:sample]:
        tax_id = str(int(float(rec["NCBI_ID"])))
        try:
            records = extract_cox.get_cox_sequences(tax_id, expand_to_children=True, max_results=5)
            rec["cox"] = f"cox:{len(records)}"
        except Exception as e:  # corroboration only, never fatal
            logger.warning(f"[cox] lookup failed for NCBI_ID {tax_id}: {e}")
            rec["cox"] = "cox:error"


# ── Summary emission + machine-readable mapping ──────────────────────────────────


def _counts(mapping: dict) -> dict:
    """Compute per-source resolved/blank tallies for the summary header."""
    reused = sum(1 for r in mapping.values() if r["provenance"].startswith("reused:"))
    draft = sum(1 for r in mapping.values() if r["provenance"] == PROV_DRAFT)
    unresolved = sum(1 for r in mapping.values() if r["provenance"] == PROV_UNRESOLVED)
    filled = {c: sum(1 for r in mapping.values() if r[c]) for c in ID_FIELDS}
    return {"reused": reused, "draft": draft, "unresolved": unresolved, "filled": filled}


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
    lines.append("- `unresolved` — Wikidata returned no biological match; all four IDs left blank.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Taxa total: **{len(mapping)}** (must equal 229).")
    lines.append(f"- Reused verbatim (overlap): **{c['reused']}**.")
    lines.append(f"- Draft-resolved via Wikidata: **{c['draft']}**.")
    lines.append(f"- Unresolved (all IDs blank): **{c['unresolved']}**.")
    lines.append(
        f"- Filled cells — wikidata_ID: **{c['filled'][WIKIDATA_ID]}**, "
        f"aphia_ID: **{c['filled']['aphia_ID']}**, "
        f"NCBI_ID: **{c['filled']['NCBI_ID']}**, "
        f"BOLD_ID: **{c['filled']['BOLD_ID']}**."
    )
    lines.append("")
    lines.append("## Caveats — KI-3 / KI-5 / KI-6 (what to spot-check at 18-03)")
    lines.append("")
    lines.append(
        "- **KI-3 (substring / fuzzy match):** the Wikidata search matches the first "
        "biological hit for the deepest known rank; a same-spelling homonym in another "
        "kingdom can be picked. Verify each `draft:wikidata` genus/species is the intended taxon."
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
    drafts = [mapping[r] for r in order if not mapping[r]["provenance"].startswith("reused:")]

    # The coarsest draft matches (resolved above genus) are the highest KI-6 risk —
    # surface them first so the 18-03 reviewer checks these before anything else.
    coarse_ranks = ("Family", "Order", "Class", "Phylum", "Kingdom", "")
    priority = [rec for rec in drafts if rec["matched_rank"] in coarse_ranks]
    lines.append("## Priority spot-checks (coarsest draft matches — verify first)")
    lines.append("")
    lines.append(
        f"{len(priority)} draft taxa resolved ABOVE genus level (Wikidata fell back up the ranks); "
        "each ID below is a coarse DRAFT approximation for a species/genus row and is the most likely "
        "KI-6 error. Verify or blank these first at 18-03."
    )
    lines.append("")
    lines.append("| raw_label | matched_rank | wikidata_ID | aphia_ID | NCBI_ID | BOLD_ID |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['raw_label']} | {rec['matched_rank']} | "
        f"{rec[WIKIDATA_ID]} | {rec['aphia_ID']} | {rec['NCBI_ID']} | {rec['BOLD_ID']} |"
        for rec in priority
    )
    lines.append("")

    lines.append("## Overlap reuse (verbatim existing IDs)")
    lines.append("")
    lines.append("| proposed_label | provenance | matched_rank | wikidata_ID | aphia_ID | NCBI_ID | BOLD_ID |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {rec['proposed_label']} | {rec['provenance']} | {rec['matched_rank']} | "
        f"{rec[WIKIDATA_ID]} | {rec['aphia_ID']} | {rec['NCBI_ID']} | {rec['BOLD_ID']} |"
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


def resolve(cox_sample: int) -> dict:
    """Run the full resolution (overlap reuse + Wikidata draft + COX) -> mapping."""
    rows = read_csv_rows(CSV_PATH)
    frepj_rows = [r for r in rows if r["Dataset"] == DATASET]
    logger.info(f"{len(frepj_rows)} frepj rows to resolve.")

    ex_species, ex_genus = build_existing_lookups(rows)
    mapping, draft_rows = resolve_overlaps(frepj_rows, ex_species, ex_genus)
    logger.info(f"Overlap reuse: {len(mapping)} rows; draft: {len(draft_rows)} rows.")

    mapping.update(resolve_drafts(draft_rows))
    cox_crosscheck(mapping, cox_sample)

    # Preserve CSV row order in the summary/mapping.
    order = [r["Raw_Labels"] for r in frepj_rows]
    missing = [raw for raw in order if raw not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} frepj rows unresolved by both paths (first: {missing[0]!r})")
    return {raw: mapping[raw] for raw in order}


def main() -> None:
    """CLI: resolve+summarize (default, network) or backfill the CSV (--backfill)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Network-free: backfill the frepj CSV rows from the committed FREPJ_DRAFTED_IDS.md.",
    )
    parser.add_argument("--cox-sample", type=int, default=5, help="Sample size for the optional COX1 cross-check.")
    args = parser.parse_args()

    if args.backfill:
        mapping = parse_summary()
        logger.info(f"Loaded {len(mapping)} taxa from the committed summary.")
        backfill_csv(CSV_PATH, mapping)
        return

    order = read_csv_rows(CSV_PATH)
    order = [r["Raw_Labels"] for r in order if r["Dataset"] == DATASET]
    mapping = resolve(args.cox_sample)
    write_summary(mapping, order)


if __name__ == "__main__":
    main()
