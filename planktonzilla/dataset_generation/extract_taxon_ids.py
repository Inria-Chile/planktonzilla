"""
extract_taxon_ids.py
=====================
Two-step pipeline that, starting from the planktonzilla taxonomy, resolves the
external identifiers for each taxon.

Step 1 (Wikidata):
    For each unique taxon (Kingdom -> Species) it looks up its Wikidata Qcode,
    keeping the deepest rank available (Species first, and if it does not exist
    it goes up to Kingdom).

Step 2 (External databases):
    From each Qcode it queries Wikidata (wbgetentities) and extracts the IDs for
    WoRMS (P850 -> aphia_ID), NCBI Taxonomy (P685 -> NCBI_ID) and
    BOLD Systems (P3606 -> BOLD_ID).

Input:
    data/planktonzilla_taxonomy_v20.csv   (separator ",")

Outputs (in data/):
    taxonomy_and_wikidata.csv     -> unique taxa + wikidata_ID
    taxonomy_wiki_and_ids.csv     -> unique taxa + wikidata_ID + aphia/NCBI/BOLD

Usage:
    python extract_taxon_ids.py
    python extract_taxon_ids.py --limit 10        # quick test with 10 taxa
    python extract_taxon_ids.py --input data/another_taxonomy.csv

Requirements:
    pip install polars requests
"""

import argparse
import os
import time

import polars as pl
import requests

from .constants import TAXONOMY_CSV_FILENAME, TAXONOMY_RANKS

# ── Paths (relative to the repo, no hardcoded absolute paths) ───────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, TAXONOMY_CSV_FILENAME)
WIKIDATA_CSV = os.path.join(DATA_DIR, "taxonomy_and_wikidata.csv")
IDS_CSV = os.path.join(DATA_DIR, "taxonomy_wiki_and_ids.csv")

# Separator for the input/output CSV (planktonzilla_taxonomy_v20 uses ",").
SEP = ","

# Taxonomy columns, from most general to most specific.
COLS = list(TAXONOMY_RANKS)

HEADERS = {"User-Agent": "plankton-script/1.0"}

# Keywords to check that a Wikidata result is biological.
BIOLOGICAL_KEYWORDS = ["taxon", "species", "genus", "family", "order", "phylum", "organism"]

# Wikidata properties we want to extract.
WIKIDATA_PROPERTIES = {
    "aphia_ID": "P850",  # WoRMS ID
    "NCBI_ID": "P685",  # NCBI Taxonomy ID
    "BOLD_ID": "P3606",  # BOLD Systems taxon ID
}

session = requests.Session()
_SEARCH_CACHE: dict[str, dict | None] = {}


# ── Step 1: Wikidata Qcode per taxon ──────────────────────────────────────────


def search_wikidata_taxon(taxon: str) -> dict | None:
    """Search a taxon on Wikidata and return {qid, label, description, url} or None.

    Caches per taxon and retries on rate limit (HTTP 429).
    """
    if taxon in _SEARCH_CACHE:
        return _SEARCH_CACHE[taxon]

    params = {
        "action": "wbsearchentities",
        "search": taxon,
        "language": "en",
        "format": "json",
        "limit": 5,
    }

    try:
        r = session.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code == 429:
            time.sleep(2)
            return search_wikidata_taxon(taxon)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    for result in data.get("search", []):
        description = result.get("description", "").lower()
        if any(k in description for k in BIOLOGICAL_KEYWORDS):
            qid = result["id"]
            output = {
                "qid": qid,
                "label": result.get("label", ""),
                "description": result.get("description", ""),
                "url": f"https://www.wikidata.org/wiki/{qid}",
            }
            _SEARCH_CACHE[taxon] = output
            return output

    _SEARCH_CACHE[taxon] = None
    return None


def fetch_wikidata_ids(taxa: pl.DataFrame) -> pl.DataFrame:
    """Add Wikidata URL/Matched Taxon/Matched Rank/wikidata_ID to each unique taxon."""
    wikidata_urls, matched_taxon, matched_rank = [], [], []

    total = taxa.height
    for idx, row in enumerate(taxa.iter_rows(named=True), start=1):
        print(f"[wikidata] {idx}/{total}")

        # Taxa present, from Species down to Kingdom (deepest rank first).
        taxons = [(row[c], c) for c in COLS if row[c] != ""]

        found_url = found_taxon = found_rank = ""
        for taxon, rank in reversed(taxons):
            result = search_wikidata_taxon(taxon)
            if result:
                found_url, found_taxon, found_rank = result["url"], taxon, rank
                break

        wikidata_urls.append(found_url)
        matched_taxon.append(found_taxon)
        matched_rank.append(found_rank)

    return taxa.with_columns(
        [
            pl.Series("Wikidata URL", wikidata_urls),
            pl.Series("Matched Taxon", matched_taxon),
            pl.Series("Matched Rank", matched_rank),
        ]
    ).with_columns(pl.col("Wikidata URL").str.extract(r"(Q\d+)", 1).alias("wikidata_ID"))


# ── Step 2: WoRMS / NCBI / BOLD per Qcode ───────────────────────────────────────


def _extract_property(claims: dict, prop: str) -> str | None:
    """Extract a property value from the claims of a Wikidata entity."""
    if prop not in claims:
        return None
    try:
        return claims[prop][0]["mainsnak"]["datavalue"]["value"]
    except Exception:
        return None


def fetch_external_ids(taxa_wiki: pl.DataFrame, batch_size: int = 50) -> pl.DataFrame:
    """Query Wikidata in batches and return a DF with wikidata_ID + aphia/NCBI/BOLD."""
    qcodes = taxa_wiki.select("wikidata_ID").drop_nulls().unique().to_series().to_list()

    results = []
    for i in range(0, len(qcodes), batch_size):
        batch = qcodes[i : i + batch_size]
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={'|'.join(batch)}&format=json"
        print(f"[ids] batch {i // batch_size + 1} ({len(batch)} Qcodes)")

        success = False
        for attempt in range(5):
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
                if r.status_code == 429:
                    wait = 2**attempt
                    print(f"  429 -> waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                entities = r.json().get("entities", {})
                for qcode, entity in entities.items():
                    claims = entity.get("claims", {})
                    results.append(
                        {
                            "wikidata_ID": qcode,
                            **{col: _extract_property(claims, prop) for col, prop in WIKIDATA_PROPERTIES.items()},
                        }
                    )
                success = True
                break
            except Exception as e:
                print(f"  error in batch {i}: {e}")
                time.sleep(2)

        if not success:
            for qcode in batch:
                results.append({"wikidata_ID": qcode, **{col: None for col in WIKIDATA_PROPERTIES}})
        time.sleep(1)

    df_ids = pl.DataFrame(results)
    return taxa_wiki.join(df_ids, on="wikidata_ID", how="left")


# ── Orchestration ───────────────────────────────────────────────────────────────


def load_unique_taxa(input_csv: str, limit: int | None) -> pl.DataFrame:
    """Read the taxonomy CSV and return the unique taxonomy combinations."""
    df = pl.read_csv(input_csv, separator=SEP).fill_null("")
    if limit is not None:
        df = df[:limit]
    return (
        df.select(COLS)
        .filter(pl.any_horizontal([pl.col(c) != "" for c in COLS]))
        .with_columns([pl.col(c).str.to_lowercase() for c in COLS])
        .unique()
    )


def main() -> None:
    """Resolve Wikidata Qcodes and external IDs for each taxon and write CSVs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_CSV, help="Input taxonomy CSV.")
    parser.add_argument("--wikidata-out", default=WIKIDATA_CSV, help="Step 1 output (taxa + wikidata_ID).")
    parser.add_argument("--ids-out", default=IDS_CSV, help="Final output (taxa + all the IDs).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (test).")
    args = parser.parse_args()

    # Step 0: unique taxa.
    taxa = load_unique_taxa(args.input, args.limit)
    print(f"{taxa.height} unique taxa to resolve.")

    # Step 1: Wikidata Qcodes.
    taxa_wiki = fetch_wikidata_ids(taxa)
    taxa_wiki.write_csv(args.wikidata_out, separator=SEP)
    print(f"Step 1 done -> {args.wikidata_out}")

    # Step 2: WoRMS / NCBI / BOLD.
    taxa_ids = fetch_external_ids(taxa_wiki)
    # Normalize empty strings to null before saving.
    taxa_ids = taxa_ids.with_columns(pl.when(pl.col(pl.String) == "").then(None).otherwise(pl.col(pl.String)).name.keep())
    taxa_ids.write_csv(args.ids_out, separator=SEP)
    print(f"Step 2 done -> {args.ids_out}")

    print("DONE")


if __name__ == "__main__":
    main()
