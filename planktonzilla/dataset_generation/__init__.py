"""Dataset generation helpers for the planktonzilla package."""

from .extract_cox import (
    configure_entrez,
    fetch_sequences,
    get_cox_sequences,
    process_csv,
    save_fasta,
)
from .extract_taxon_ids import (
    fetch_external_ids as fetch_taxon_external_ids,
)
from .extract_taxon_ids import (
    fetch_wikidata_ids,
    load_unique_taxa,
    search_wikidata_taxon,
)
from .gen_planktonzilla import main as gen_planktonzilla_main
from .gen_planktonzilla_only_plankton import main as gen_planktonzilla_only_plankton_main
from .save_planktonzilla_for_clip import export_to_tar_shards
from .save_planktonzilla_for_clip import main as save_planktonzilla_for_clip_main
from .update_planktonzilla import main as update_planktonzilla_main

__all__ = [
    "configure_entrez",
    "export_to_tar_shards",
    "fetch_sequences",
    "fetch_taxon_external_ids",
    "fetch_wikidata_ids",
    "gen_planktonzilla_main",
    "gen_planktonzilla_only_plankton_main",
    "get_cox_sequences",
    "load_unique_taxa",
    "process_csv",
    "save_fasta",
    "save_planktonzilla_for_clip_main",
    "search_wikidata_taxon",
    "update_planktonzilla_main",
]
