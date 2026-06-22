"""
(c) Inria

Download COX1 gene sequences from NCBI for plankton species.

Steps per species:
    1. build_query        -> builds the Entrez query (txid + COX filters).
    2. search_nuccore     -> gets the matching GenBank accession IDs.
    3. fetch_sequences    -> downloads the FASTA records in batches.
    4. save_fasta         -> saves one .fasta per species + a summary.csv.

Where this fits in the pipeline:
    taxonomy -> [extract_taxon_ids] wikidata_ID/aphia/NCBI/BOLD
             -> [extract_cox] COX1 sequences (FASTA)

Credentials (NCBI Entrez requires you to identify yourself):
    Set them through environment variables (NEVER hardcoded in the code):
        export NCBI_EMAIL="your_email@example.com"   # required
        export NCBI_API_KEY="your_api_key"           # optional, raises the rate limit 3->10 req/s
    Get a free API key at: https://www.ncbi.nlm.nih.gov/account/
    The email can also be passed with --email (it takes priority over the env var).

Usage (a single species):
    python extract_cox.py --ncbi_id 124140

Usage (batch from a CSV):
    python extract_cox.py --csv data/taxonomy_wiki_and_ids.csv --nb_rows 10 --clean

Requirements:
    pip install biopython polars tqdm
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm

from planktonzilla.utils.logger import get_pylogger

try:
    from Bio import Entrez, SeqIO
    from Bio.SeqRecord import SeqRecord
except ImportError:
    raise ImportError("Install biopython: pip install biopython")

# ── Configuration ─────────────────────────────────────────────────────────────

# NCBI Entrez credentials. Read from environment variables; never hardcoded.
# The email is required (NCBI demands it); the API key is optional.
ENTREZ_EMAIL = os.environ.get("NCBI_EMAIL")
ENTREZ_API_KEY = os.environ.get("NCBI_API_KEY")

# COX terms (match the title, gene name and product fields).
COX_TERMS = [
    "COI",
    "CO1",
    "COX1",
    "COXI",
    "cytochrome c oxidase subunit I",
    "cytochrome c oxidase subunit 1",
    "cytochrome oxidase subunit I",
]

# Query fragment with the COX filters joined by OR.
COX_FILTER = " OR ".join(f'"{t}"[All Fields]' for t in COX_TERMS)

MAX_SEQS_PER_SPECIES = 500  # Safety cap per species; raise it if needed.
BATCH_SIZE = 50  # Records downloaded per Entrez request.
SLEEP_BETWEEN_CALLS = 0.4  # seconds; respects the NCBI rate limit (~3/s without an API key).

log = get_pylogger(__name__)


class EntrezConfigError(RuntimeError):
    """Raised when Entrez cannot be configured (e.g. missing NCBI email)."""


# ── NCBI helpers ─────────────────────────────────────────────────────────────────


def configure_entrez(email: str | None = None):
    """Set up Entrez with the email (required) and the API key (optional)."""
    resolved_email = email or ENTREZ_EMAIL
    if not resolved_email:
        raise EntrezConfigError(
            "Missing NCBI email. Set NCBI_EMAIL in the environment or pass it with --email.\n"
            '  export NCBI_EMAIL="your_email@example.com"'
        )
    Entrez.email = resolved_email
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY


def build_query(ncbi_tax_id: int | str, expand_to_children: bool = True) -> str:
    """
    Build an Entrez nuccore query for a taxonomy ID.

    expand_to_children=True  → txid{ID}[Organism:exp]
        Includes all child taxa (useful at the genus/family level).
    expand_to_children=False → txid{ID}[Organism:noexp]
        Only the exact taxon (when you want strictly that species).
    """
    scope = "exp" if expand_to_children else "noexp"
    return f"(txid{ncbi_tax_id}[Organism:{scope}]) AND ({COX_FILTER})"


def search_nuccore(query: str, max_results: int = MAX_SEQS_PER_SPECIES) -> list[str]:
    """Return the list of GenBank accession IDs that match the query."""
    time.sleep(SLEEP_BETWEEN_CALLS)
    try:
        handle = Entrez.esearch(
            db="nuccore",
            term=query,
            retmax=max_results,
            usehistory="y",
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        count = int(record.get("Count", 0))
        log.info(f"  Found {count} total matches, retrieving up to {max_results}.")
        return ids
    except Exception as e:
        log.warning(f"  esearch failed for query {query!r}, returning no IDs: {e}")
        log.error(f"  esearch failed: {e}")
        return []


def fetch_sequences(id_list: list[str], label: str = "") -> list[SeqRecord]:
    """Download GenBank records in batches and return SeqRecord objects."""
    records = []
    if not id_list:
        return records

    for start in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[start : start + BATCH_SIZE]
        time.sleep(SLEEP_BETWEEN_CALLS)
        try:
            handle = Entrez.efetch(
                db="nuccore",
                id=",".join(batch),
                rettype="fasta",
                retmode="text",
            )
            batch_records = list(SeqIO.parse(handle, "fasta"))
            handle.close()
            records.extend(batch_records)
            log.info(f"  [{label}] Fetched {len(records)}/{len(id_list)} sequences…")
        except Exception as e:
            log.warning(f"  efetch batch {start}-{start + BATCH_SIZE} failed for [{label}], skipping batch: {e}")
            log.error(f"  efetch batch {start}-{start + BATCH_SIZE} failed: {e}")

    return records


def get_cox_sequences(
    ncbi_tax_id: int | str,
    expand_to_children: bool = True,
    max_results: int = MAX_SEQS_PER_SPECIES,
) -> list[SeqRecord]:
    """
    Given a taxonomy ID, return the COX SeqRecords.
    Tries the expanded search first; if there are no results, retries with noexp.
    """
    query = build_query(ncbi_tax_id, expand_to_children=expand_to_children)
    log.info(f"Query: {query}")
    ids = search_nuccore(query, max_results=max_results)

    if not ids and expand_to_children:
        log.info("  No results with expanded search, retrying with noexp…")
        query = build_query(ncbi_tax_id, expand_to_children=False)
        ids = search_nuccore(query, max_results=max_results)

    return fetch_sequences(ids, label=str(ncbi_tax_id))


# ── Output helpers ───────────────────────────────────────────────────────────────


def save_fasta(records: list[SeqRecord], filepath: str | Path):
    """Write the sequences to a FASTA file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        SeqIO.write(records, f, "fasta")
    log.info(f"  Saved {len(records)} sequences → {filepath}")


# ── Batch processing ─────────────────────────────────────────────────────────────


def process_csv(
    csv_path: str,
    out_dir: str,
    ncbi_col: str = "NCBI_ID",
    label_col: str = "proposed_label",
    max_results: int = MAX_SEQS_PER_SPECIES,
    skip_empty: bool = True,
    nb_rows: int | None = None,
):
    """
    Read the plankton CSV, loop over the rows, download the COX sequences
    for every species with a valid NCBI_ID and write the outputs.

    Output per species:   out_dir/{label}_{ncbi_id}.fasta
    Summary:              out_dir/summary.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if nb_rows is None:
        df = pl.read_csv(csv_path, separator=";")
    else:
        df = pl.read_csv(csv_path, separator=";", n_rows=nb_rows)

    log.info(f"Loaded {df.height} rows from {csv_path}")

    summary_rows = []

    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Species"):
        ncbi_id = row.get(ncbi_col)
        label = str(row.get(label_col, "unknown")).strip().replace(" ", "_")

        # Only NCBI IDs are currently used to drive the COX search.
        if ncbi_id is not None and str(ncbi_id).strip() not in ("", "nan", "NaN"):
            tax_id = str(int(float(ncbi_id)))
            source = "ncbi"
        else:
            if skip_empty:
                log.debug(f"Skipping {label}: no ID found.")
                summary_rows.append(
                    {
                        "label": label,
                        "ncbi_tax_id": None,
                        "source": None,
                        "n_sequences": 0,
                        "status": "no_id",
                    }
                )
            continue

        log.info(f"[{label}] NCBI tax ID: {tax_id} (source: {source})")

        records = get_cox_sequences(
            tax_id,
            expand_to_children=True,
            max_results=max_results,
        )

        n = len(records)
        status = "ok" if n > 0 else "no_sequences"

        if n > 0:
            fasta_path = out_dir / f"{label}_{tax_id}.fasta"
            save_fasta(records, fasta_path)

        summary_rows.append(
            {
                "label": label,
                "ncbi_tax_id": tax_id,
                "source": source,
                "n_sequences": n,
                "status": status,
            }
        )

    summary_df = pl.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.write_csv(summary_path)
    log.info(f"\nDone. Summary saved to {summary_path}")

    status_counts = summary_df["status"].value_counts()
    log.info(f"\nStatus counts:\n{status_counts}")

    return summary_df


# ── Single-ID processing ───────────────────────────────────────────────────────


def process_single(
    ncbi_id: int | str,
    out_dir: str | Path,
    expand_to_children: bool = True,
    max_results: int = MAX_SEQS_PER_SPECIES,
):
    """Fetch the COX sequences for a single NCBI Taxonomy ID and save them.

    Output:   out_dir/{ncbi_id}.fasta (only written when there are sequences).

    Faithful extraction of the former inline single-``--ncbi_id`` branch of
    ``main`` — same logging, same expand_to_children/max_results semantics, same
    "save only when records exist" guard.
    """
    log.info(f"Fetching COX sequences for NCBI Taxonomy ID: {ncbi_id}")
    records = get_cox_sequences(
        ncbi_id,
        expand_to_children=expand_to_children,
        max_results=max_results,
    )
    log.info(f"\nFound {len(records)} COX sequences for taxid {ncbi_id}\n")
    for r in records:
        log.info(f"  {r.id}  len={len(r.seq)} bp  {r.description[:100]}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_out = out_dir / f"{ncbi_id}.fasta"
    if records:
        save_fasta(records, fasta_out)


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI args and fetch COX1 sequences for a single ID or a CSV batch."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Fetch COX1 sequences from NCBI for plankton species.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ncbi_id", type=str, help="A single NCBI Taxonomy ID")
    mode.add_argument("--csv", type=str, help="Plankton CSV for batch processing")

    parser.add_argument("--out_dir_b", type=str, default="cox_output_batch", help="Output directory (batch)")
    parser.add_argument("--out_dir_s", type=str, default="cox_output_single", help="Output directory (single)")
    parser.add_argument("--ncbi_col", type=str, default="NCBI_ID", help="Column with the NCBI IDs in the CSV")
    parser.add_argument("--label_col", type=str, default="proposed_label", help="Column used to name the files")
    parser.add_argument("--max_seqs", type=int, default=MAX_SEQS_PER_SPECIES, help="Max sequences per species")
    parser.add_argument("--noexp", action="store_true", help="Use noexp (exact taxon, no children)")
    parser.add_argument("--email", type=str, default=None, help="Email for NCBI Entrez (otherwise uses NCBI_EMAIL)")
    parser.add_argument("--nb_rows", type=int, default=None, help="Number of CSV rows to process (default: all)")
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before processing")

    args = parser.parse_args()

    # Set up Entrez with the resolved email (--email takes priority over the env var).
    configure_entrez(email=args.email)

    if args.ncbi_id:
        # ── Single NCBI ID mode ──
        process_single(
            args.ncbi_id,
            out_dir=args.out_dir_s,
            expand_to_children=not args.noexp,
            max_results=args.max_seqs,
        )

    elif args.csv:
        # ── Batch CSV mode ──
        out_dir = Path(args.out_dir_b)

        if args.clean and out_dir.exists():
            log.info(f"Removing existing output directory: {out_dir}")
            shutil.rmtree(out_dir)

        process_csv(
            csv_path=args.csv,
            out_dir=args.out_dir_b,
            ncbi_col=args.ncbi_col,
            label_col=args.label_col,
            max_results=args.max_seqs,
            nb_rows=args.nb_rows,
        )


if __name__ == "__main__":
    main()
