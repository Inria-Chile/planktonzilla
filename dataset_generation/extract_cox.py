"""
Descarga secuencias genéticas COX1 desde NCBI para especies de plancton.

Flujo por especie:
    1. build_query        -> arma la consulta Entrez (txid + filtros COX).
    2. search_nuccore     -> obtiene los accession IDs de GenBank que coinciden.
    3. fetch_sequences    -> descarga los registros FASTA en lotes.
    4. save_fasta         -> guarda un .fasta por especie + un summary.csv.

Lugar en el pipeline:
    taxonomía -> [extract_taxon_ids] wikidata_ID/aphia/NCBI/BOLD
              -> [extract_cox] secuencias COX1 (FASTA)

Credenciales (NCBI Entrez exige identificarte):
    Configúralas por variable de entorno (NUNCA hardcodeadas en el código):
        export NCBI_EMAIL="tu_email@ejemplo.com"     # obligatorio
        export NCBI_API_KEY="tu_api_key"             # opcional, sube el rate-limit 3->10 req/s
    Consigue una API key gratis en: https://www.ncbi.nlm.nih.gov/account/
    El email también puede pasarse con --email (tiene prioridad sobre la env var).

Uso (una sola especie):
    python extract_cox.py --ncbi_id 124140

Uso (batch desde CSV):
    python extract_cox.py --csv data/taxonomy_wiki_and_ids.csv --nb_rows 10 --clean

Requisitos:
    pip install biopython polars requests tqdm
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm

try:
    from Bio import Entrez, SeqIO
    from Bio.SeqRecord import SeqRecord
except ImportError:
    raise ImportError("Install biopython: pip install biopython")

# ── Configuración ───────────────────────────────────────────────────────────────

# Credenciales NCBI Entrez. Se leen de variables de entorno; nunca se hardcodean.
# El email es obligatorio (NCBI lo exige); la API key es opcional.
ENTREZ_EMAIL = os.environ.get("NCBI_EMAIL")
ENTREZ_API_KEY = os.environ.get("NCBI_API_KEY")

# Términos COX (matchea título, nombre de gen y campos de producto).
COX_TERMS = [
    "COI", "CO1", "COX1", "COXI",
    "cytochrome c oxidase subunit I",
    "cytochrome c oxidase subunit 1",
    "cytochrome oxidase subunit I",
]

# Fragmento de query OR-joined con los filtros COX.
COX_FILTER = " OR ".join(f'"{t}"[All Fields]' for t in COX_TERMS)

MAX_SEQS_PER_SPECIES = 500   # Tope de seguridad por especie; súbelo si hace falta.
BATCH_SIZE = 50              # Registros descargados por petición Entrez.
SLEEP_BETWEEN_CALLS = 0.4    # segundos; respeta el rate-limit de NCBI (~3/s sin API key).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── NCBI helpers ─────────────────────────────────────────────────────────────────

def configure_entrez(email: str | None = None):
    """Configura Entrez con el email (obligatorio) y la API key (opcional)."""
    resolved_email = email or ENTREZ_EMAIL
    if not resolved_email:
        raise SystemExit(
            "Falta el email de NCBI. Define NCBI_EMAIL en el entorno o pásalo con --email.\n"
            '  export NCBI_EMAIL="tu_email@ejemplo.com"'
        )
    Entrez.email = resolved_email
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY


def build_query(ncbi_tax_id: int | str, expand_to_children: bool = True) -> str:
    """
    Arma una query Entrez nuccore para un taxonomy ID.

    expand_to_children=True  → txid{ID}[Organism:exp]
        Incluye todos los taxones descendientes (útil a nivel de género/familia).
    expand_to_children=False → txid{ID}[Organism:noexp]
        Solo el taxón exacto (cuando quieres estrictamente esa especie).
    """
    scope = "exp" if expand_to_children else "noexp"
    return f"(txid{ncbi_tax_id}[Organism:{scope}]) AND ({COX_FILTER})"


def search_nuccore(query: str, max_results: int = MAX_SEQS_PER_SPECIES) -> list[str]:
    """Devuelve la lista de accession IDs de GenBank que coinciden con la query."""
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
        log.error(f"  esearch failed: {e}")
        return []


def fetch_sequences(id_list: list[str], label: str = "") -> list[SeqRecord]:
    """Descarga registros GenBank en lotes y devuelve objetos SeqRecord."""
    records = []
    if not id_list:
        return records

    for start in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[start: start + BATCH_SIZE]
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
            log.error(f"  efetch batch {start}-{start + BATCH_SIZE} failed: {e}")

    return records


def get_cox_sequences(
    ncbi_tax_id: int | str,
    expand_to_children: bool = True,
    max_results: int = MAX_SEQS_PER_SPECIES,
) -> list[SeqRecord]:
    """
    Dado un taxonomy ID, devuelve los SeqRecords COX.
    Prueba primero la búsqueda expandida; si no hay resultados, reintenta noexp.
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
    """Escribe las secuencias a un archivo FASTA."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        SeqIO.write(records, f, "fasta")
    log.info(f"  Saved {len(records)} sequences → {filepath}")


def records_to_dataframe(records: list[SeqRecord], ncbi_tax_id=None) -> pl.DataFrame:
    """Convierte una lista de SeqRecords en un DataFrame ordenado."""
    rows = []
    for rec in records:
        # Parsea accession y descripción desde el header FASTA.
        parts = rec.description.split(" ", 1)
        accession = parts[0]
        description = parts[1] if len(parts) > 1 else ""
        rows.append({
            "ncbi_tax_id": ncbi_tax_id,
            "accession": accession,
            "description": description,
            "length_bp": len(rec.seq),
            "sequence": str(rec.seq),
        })
    return pl.DataFrame(rows)


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
    Lee el CSV de plancton, itera sobre las filas, descarga las secuencias COX
    de cada especie con NCBI_ID válido y escribe las salidas.

    Salidas por especie:   out_dir/{label}_{ncbi_id}.fasta
    Resumen:               out_dir/summary.csv
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

        # Determina qué ID usar.
        if ncbi_id is not None and str(ncbi_id).strip() not in ("", "nan", "NaN"):
            tax_id = str(int(float(ncbi_id)))
            source = "ncbi"
        else:
            if skip_empty:
                log.debug(f"Skipping {label}: no ID found.")
                summary_rows.append({
                    "label": label,
                    "ncbi_tax_id": None,
                    "source": None,
                    "n_sequences": 0,
                    "status": "no_id",
                })
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

        summary_rows.append({
            "label": label,
            "ncbi_tax_id": tax_id,
            "source": source,
            "n_sequences": n,
            "status": status,
        })

    summary_df = pl.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.write_csv(summary_path)
    log.info(f"\nDone. Summary saved to {summary_path}")

    status_counts = summary_df["status"].value_counts()
    log.info(f"\nStatus counts:\n{status_counts}")

    return summary_df


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch COX1 sequences from NCBI for plankton species.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ncbi_id", type=str, help="NCBI Taxonomy ID único")
    mode.add_argument("--csv", type=str, help="CSV de plancton para procesamiento batch")

    parser.add_argument("--out_dir_b", type=str, default="cox_output_batch", help="Directorio de salida (batch)")
    parser.add_argument("--out_dir_s", type=str, default="cox_output_single", help="Directorio de salida (single)")
    parser.add_argument("--ncbi_col", type=str, default="NCBI_ID", help="Columna de NCBI IDs en el CSV")
    parser.add_argument("--label_col", type=str, default="proposed_label", help="Columna usada para nombrar los archivos")
    parser.add_argument("--max_seqs", type=int, default=MAX_SEQS_PER_SPECIES, help="Máx. secuencias por especie")
    parser.add_argument("--noexp", action="store_true", help="Usar noexp (taxón exacto, sin descendientes)")
    parser.add_argument("--email", type=str, default=None, help="Email para NCBI Entrez (si no, usa NCBI_EMAIL)")
    parser.add_argument("--nb_rows", type=int, default=None, help="Nº de filas a procesar del CSV (default: todas)")
    parser.add_argument("--clean", action="store_true", help="Borra el directorio de salida antes de procesar")

    args = parser.parse_args()

    # Configura Entrez con el email resuelto (--email tiene prioridad sobre la env var).
    configure_entrez(email=args.email)

    if args.ncbi_id:
        # ── Modo single NCBI ID ──
        log.info(f"Fetching COX sequences for NCBI Taxonomy ID: {args.ncbi_id}")
        records = get_cox_sequences(
            args.ncbi_id,
            expand_to_children=not args.noexp,
            max_results=args.max_seqs,
        )
        print(f"\nFound {len(records)} COX sequences for taxid {args.ncbi_id}\n")
        for r in records:
            print(f"  {r.id}  len={len(r.seq)} bp  {r.description[:100]}")

        out_dir = Path(args.out_dir_s)
        out_dir.mkdir(parents=True, exist_ok=True)

        fasta_out = out_dir / f"{args.ncbi_id}.fasta"
        if records:
            save_fasta(records, fasta_out)

    elif args.csv:
        # ── Modo batch CSV ──
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
