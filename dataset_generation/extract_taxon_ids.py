"""
extract_taxon_ids.py
=====================
Pipeline en dos pasos que, partiendo de la taxonomia de planktonzilla, resuelve
los identificadores externos de cada taxon.

Paso 1 (Wikidata):
    Para cada taxon unico (Kingdom -> Species) busca su Qcode de Wikidata,
    quedandose con el rango mas profundo disponible (primero Species, si no
    existe sube hasta Kingdom).

Paso 2 (Bases externas):
    A partir de cada Qcode consulta Wikidata (wbgetentities) y extrae los IDs de
    WoRMS (P850 -> aphia_ID), NCBI Taxonomy (P685 -> NCBI_ID) y
    BOLD Systems (P3606 -> BOLD_ID).

Entrada:
    data/planktonzilla_taxonomy_v20.csv   (separador ",")

Salidas (en data/):
    taxonomy_and_wikidata.csv     -> taxones unicos + wikidata_ID
    taxonomy_wiki_and_ids.csv     -> taxones unicos + wikidata_ID + aphia/NCBI/BOLD

Uso:
    python extract_taxon_ids.py
    python extract_taxon_ids.py --limit 10        # prueba rapida con 10 taxones
    python extract_taxon_ids.py --input data/otra_taxonomia.csv

Requisitos:
    pip install polars requests
"""

import argparse
import os
import time

import polars as pl
import requests

# ── Rutas (relativas al repo, sin rutas absolutas hardcodeadas) ─────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "planktonzilla_taxonomy_v20.csv")
WIKIDATA_CSV = os.path.join(DATA_DIR, "taxonomy_and_wikidata.csv")
IDS_CSV = os.path.join(DATA_DIR, "taxonomy_wiki_and_ids.csv")

# Separador del CSV de entrada/salida (planktonzilla_taxonomy_v20 usa ",").
SEP = ","

# Columnas taxonomicas, de mas general a mas especifico.
COLS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

HEADERS = {"User-Agent": "plankton-script/1.0"}

# Palabras clave para validar que un resultado de Wikidata es biologico.
BIOLOGICAL_KEYWORDS = ["taxon", "species", "genus", "family", "order", "phylum", "organism"]

# Propiedades de Wikidata que queremos extraer.
WIKIDATA_PROPERTIES = {
    "aphia_ID": "P850",   # WoRMS ID
    "NCBI_ID": "P685",    # NCBI Taxonomy ID
    "BOLD_ID": "P3606",   # BOLD Systems taxon ID
}

session = requests.Session()
_SEARCH_CACHE: dict[str, dict | None] = {}


# ── Paso 1: Wikidata Qcode por taxon ────────────────────────────────────────────

def search_wikidata_taxon(taxon: str) -> dict | None:
    """Busca un taxon en Wikidata y devuelve {qid, label, description, url} o None.

    Cachea por taxon y reintenta ante rate-limit (HTTP 429).
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
    """Anade Wikidata URL/Matched Taxon/Matched Rank/wikidata_ID a cada taxon unico."""
    wikidata_urls, matched_taxon, matched_rank = [], [], []

    total = taxa.height
    for idx, row in enumerate(taxa.iter_rows(named=True), start=1):
        print(f"[wikidata] {idx}/{total}")

        # Taxones presentes, de Species hacia Kingdom (rango mas profundo primero).
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

    return taxa.with_columns([
        pl.Series("Wikidata URL", wikidata_urls),
        pl.Series("Matched Taxon", matched_taxon),
        pl.Series("Matched Rank", matched_rank),
    ]).with_columns(
        pl.col("Wikidata URL").str.extract(r"(Q\d+)", 1).alias("wikidata_ID")
    )


# ── Paso 2: WoRMS / NCBI / BOLD por Qcode ───────────────────────────────────────

def _extract_property(claims: dict, prop: str):
    """Extrae el valor de una propiedad de los claims de una entidad Wikidata."""
    if prop not in claims:
        return None
    try:
        return claims[prop][0]["mainsnak"]["datavalue"]["value"]
    except Exception:
        return None


def fetch_external_ids(taxa_wiki: pl.DataFrame, batch_size: int = 50) -> pl.DataFrame:
    """Consulta Wikidata en lotes y devuelve un DF con wikidata_ID + aphia/NCBI/BOLD."""
    qcodes = (
        taxa_wiki.select("wikidata_ID")
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    results = []
    for i in range(0, len(qcodes), batch_size):
        batch = qcodes[i:i + batch_size]
        url = (
            "https://www.wikidata.org/w/api.php"
            "?action=wbgetentities"
            f"&ids={'|'.join(batch)}"
            "&format=json"
        )
        print(f"[ids] lote {i // batch_size + 1} ({len(batch)} Qcodes)")

        success = False
        for attempt in range(5):
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
                if r.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  429 -> espera {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                entities = r.json().get("entities", {})
                for qcode, entity in entities.items():
                    claims = entity.get("claims", {})
                    results.append({
                        "wikidata_ID": qcode,
                        **{col: _extract_property(claims, prop) for col, prop in WIKIDATA_PROPERTIES.items()},
                    })
                success = True
                break
            except Exception as e:
                print(f"  error lote {i}: {e}")
                time.sleep(2)

        if not success:
            for qcode in batch:
                results.append({"wikidata_ID": qcode, **{col: None for col in WIKIDATA_PROPERTIES}})
        time.sleep(1)

    df_ids = pl.DataFrame(results)
    return taxa_wiki.join(df_ids, on="wikidata_ID", how="left")


# ── Orquestacion ────────────────────────────────────────────────────────────────

def load_unique_taxa(input_csv: str, limit: int | None) -> pl.DataFrame:
    """Lee el CSV de taxonomia y devuelve las combinaciones taxonomicas unicas."""
    df = pl.read_csv(input_csv, separator=SEP).fill_null("")
    if limit is not None:
        df = df[:limit]
    return (
        df.select(COLS)
        .filter(pl.any_horizontal([pl.col(c) != "" for c in COLS]))
        .with_columns([pl.col(c).str.to_lowercase() for c in COLS])
        .unique()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_CSV, help="CSV de taxonomia de entrada.")
    parser.add_argument("--wikidata-out", default=WIKIDATA_CSV, help="Salida del paso 1 (taxones + wikidata_ID).")
    parser.add_argument("--ids-out", default=IDS_CSV, help="Salida final (taxones + todos los IDs).")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N filas (prueba).")
    args = parser.parse_args()

    # Paso 0: taxones unicos.
    taxa = load_unique_taxa(args.input, args.limit)
    print(f"{taxa.height} taxones unicos a resolver.")

    # Paso 1: Wikidata Qcodes.
    taxa_wiki = fetch_wikidata_ids(taxa)
    taxa_wiki.write_csv(args.wikidata_out, separator=SEP)
    print(f"Paso 1 listo -> {args.wikidata_out}")

    # Paso 2: WoRMS / NCBI / BOLD.
    taxa_ids = fetch_external_ids(taxa_wiki)
    # Normaliza cadenas vacias a null antes de guardar.
    taxa_ids = taxa_ids.with_columns(
        pl.when(pl.col(pl.String) == "").then(None).otherwise(pl.col(pl.String)).name.keep()
    )
    taxa_ids.write_csv(args.ids_out, separator=SEP)
    print(f"Paso 2 listo -> {args.ids_out}")

    print("DONE")


if __name__ == "__main__":
    main()
