"""
(c) Inria

build_frepj_taxonomy.py
=======================
Deterministic, network-free curation engine that turns the frozen 229 FREPJ
image-folder class-dir tuples into complete ``planktonzilla_taxonomy.csv`` rows
and APPENDS them to the master CSV — leaving the pre-existing header + 1485 rows
byte-unchanged (append-only / output-preserving milestone invariant).

What it does
------------
1. Reads the frozen class-dir list from ``tests/fixtures/frepj/frepj_class_dirs.tsv``
   (column ``class_dir``). These strings are the exact ``Raw_Labels`` join keys,
   byte-for-byte, commas preserved — the same fixture ``tests/test_frepj_layout.py``
   reads. They are never regenerated or normalised.
2. Parses each class-dir into Kingdom -> Species ranks using the SENTINEL CASCADE
   (TAX-01): scan Order -> Family -> Genus -> Species and at the FIRST column whose
   value is an unknown-rank sentinel (``Or./Fa./Ge./Sp._unk`` optionally with a
   ``_*stage`` suffix, or the bare open-nomenclature forms ``sp``/``sp2``), null
   THAT rank and every rank below it — regardless of which literal sentinel string
   the downstream columns echo. This is essential: when a rank is unknown the
   downstream columns echo the SAME sentinel string (Genus ``Ge._unk`` echoes
   ``Ge._unk`` into Species, NOT ``Sp._unk``), so a per-column matcher that only
   recognises ``Sp._unk``/``sp``/``sp2`` for Species would leak ``ge._unk`` as a
   literal epithet into 9 real class-dirs.
3. Fills Phylum/Kingdom by PREFERRING the existing-CSV Class anchor (grouping the
   non-frepj rows by lowercased Class), so cross-source consistency is automatic;
   only ``lobosa`` lacks an anchor and uses a curated proposal (flagged).
4. Reconciles higher-rank spellings against the existing CSV (TAX-04/TAX-06): when
   a FREPJ Genus already exists from another source, the existing lineage's
   Kingdom/Phylum/Class/Order/Family spellings are reused verbatim (e.g. Bosmina
   Order -> ``anomopoda`` not ``diplostraca``; Paradileptus -> ``haptorida`` /
   ``tracheliidae``). Genuine granularity/spelling conflicts and judgment calls are
   NOT silently resolved — they are enumerated in ``FREPJ_TAXONOMY_RECONCILIATION.md``
   for the Phase-18 human-verify checkpoint.
5. Sets the controlled-vocab flags (TAX-03): every frepj row is
   ``plankton=True, living=True, root_class=living, qualifier=full_body`` (FREPJ is
   curated individual live-organism crops).
6. Leaves ALL four external-ID columns (wikidata/aphia/NCBI/BOLD) AND ecotaxa_ID
   BLANK for every frepj row here — the single authoritative external-ID fill is
   Plan 18-02; ecotaxa_ID stays blank always (FREPJ has none).
7. Emits ``FREPJ_TAXONOMY_RECONCILIATION.md`` (Section A = overlap reuse table,
   Section B = flagged-conflict list).

The append is IDEMPOTENT: re-running rewrites the frepj block in place so a second
run leaves the CSV byte-identical.

Network-free BY CONSTRUCTION: reads only the committed TSV fixture and the committed
CSV. No HTTP, no downloads.
"""

import argparse
import csv
import io
import logging
import re
from collections import Counter
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

DATASET_NAME = "frepj"

# Repository-root-relative location of the frozen class-dir fixture (shared with
# tests/test_frepj_layout.py). Resolved from this file: utils/ -> dataset -> pkg ->
# repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLASS_DIRS_TSV = _REPO_ROOT / "tests" / "fixtures" / "frepj" / "frepj_class_dirs.tsv"

DEFAULT_RECONCILIATION_MD = Path(__file__).parent / "FREPJ_TAXONOMY_RECONCILIATION.md"

# The full 19-column schema, in the exact order of the master CSV header.
CSV_COLUMNS = (
    "Dataset",
    "Raw_Labels",
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Species",
    "proposed_label",
    "plankton",
    "living",
    "root_class",
    "qualifier",
    "wikidata_ID",
    "aphia_ID",
    "NCBI_ID",
    "BOLD_ID",
    "ecotaxa_ID",
)

# Controlled-vocab flags — identical for every frepj row (TAX-03).
FREPJ_FLAGS = {
    "plankton": "True",
    "living": "True",
    "root_class": "living",
    "qualifier": "full_body",
}

# Unknown-rank sentinels: Or./Fa./Ge./Sp. with an optional _*stage suffix, plus the
# bare open-nomenclature species forms sp / sp2. Matched case-insensitively.
_SENTINEL_UNK_RE = re.compile(r"^(or|fa|ge|sp)\._unk(_.*stage)?$")
_BARE_SP_FORMS = ("sp", "sp2")

# Class spellings that must be reconciled to the existing-CSV spelling (TAX-06):
# the FREPJ tuple spells the ciliate class "Oligophymenophorea" (a typo).
CLASS_NORMALIZE = {"oligophymenophorea": "oligohymenophorea"}

# Curated Class -> (Kingdom, Phylum) fallback, used only when the existing CSV holds
# no anchor for a Class. In practice only ``lobosa`` (amoebozoa/protozoa) has no
# anchor; the rest are belt-and-suspenders that mirror the existing anchors.
CURATED_CLASS_MAP = {
    "branchiopoda": ("animalia", "arthropoda"),
    "copepoda": ("animalia", "arthropoda"),
    "eurotatoria": ("animalia", "rotifera"),
    "arachnida": ("animalia", "arthropoda"),
    "ostracoda": ("animalia", "arthropoda"),
    "insecta": ("animalia", "arthropoda"),
    "dinophyceae": ("chromista", "myzozoa"),
    "heterotrichea": ("chromista", "ciliophora"),
    "litostomatea": ("chromista", "ciliophora"),
    "oligohymenophorea": ("chromista", "ciliophora"),
    "lobosa": ("protozoa", "amoebozoa"),
}

# Classes with no existing anchor -> curated proposal that needs sign-off (flag 7).
CURATED_ONLY_CLASSES = {"lobosa"}

# Strain suffixes that are metadata, not taxa (dropped for rank/proposed_label).
_STRAIN_SUFFIXES = (" jpn1", " jpn2")


def is_unk_sentinel(value: str) -> bool:
    """Return ``True`` if a raw rank-column value is an unknown-rank sentinel.

    Covers ``Or./Fa./Ge./Sp._unk`` (optionally with a ``_*stage`` suffix) and the
    bare open-nomenclature species forms ``sp`` / ``sp2``. This is the predicate the
    cascade uses on Order/Family/Genus/Species columns uniformly.
    """
    v = value.strip().lower()
    return bool(_SENTINEL_UNK_RE.match(v)) or v in _BARE_SP_FORMS


def _load_existing(csv_path: Path):
    """Read the master CSV and return the reconciliation references.

    Returns ``(class_anchor, genus_lineage)`` where:
      * ``class_anchor``: lowercased Class -> (Kingdom, Phylum) from non-frepj rows
        (majority vote, so a stray blank never wins).
      * ``genus_lineage``: lowercased Genus -> dict with the unique existing
        (Kingdom, Phylum, Class, Order, Family) spelling, the source datasets, and
        the representative external IDs (for the Plan 18-02 reuse note).

    frepj rows are excluded so re-runs never reconcile against their own output.
    """
    class_kp = {}
    genus_rows = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["Dataset"] == DATASET_NAME:
                continue
            cls = row["Class"].strip().lower()
            king = row["Kingdom"].strip()
            phy = row["Phylum"].strip()
            if cls and king and phy:
                class_kp.setdefault(cls, Counter())[(king, phy)] += 1
            gen = row["Genus"].strip().lower()
            if gen:
                genus_rows.setdefault(gen, []).append(row)

    class_anchor = {cls: counts.most_common(1)[0][0] for cls, counts in class_kp.items()}

    genus_lineage = {}
    for gen, rows in genus_rows.items():
        lineages = {(r["Kingdom"], r["Phylum"], r["Class"], r["Order"], r["Family"]) for r in rows}
        genus_lineage[gen] = {
            "lineages": lineages,
            "kpc_of": sorted(lineages)[0],
            "sources": sorted({r["Dataset"] for r in rows}),
            "id_rows": [
                {
                    "proposed_label": r["proposed_label"],
                    "wikidata_ID": r["wikidata_ID"],
                    "aphia_ID": r["aphia_ID"],
                    "NCBI_ID": r["NCBI_ID"],
                    "BOLD_ID": r["BOLD_ID"],
                }
                for r in rows
            ],
        }
    return class_anchor, genus_lineage


def _load_class_dirs(tsv_path: Path) -> list[str]:
    """Return the frozen class-dir strings (byte-exact) from the TSV fixture.

    Loaded exactly like ``tests/test_frepj_layout.py``: split each line on tab and
    take field 0. The commas inside each class-dir are preserved verbatim.
    """
    lines = tsv_path.read_text().splitlines()
    return [line.split("\t")[0] for line in lines[1:]]


def _split_ranks(class_dir: str) -> tuple[list[str], bool]:
    """Split a class-dir into the five provisional rank strings.

    Returns ``(ranks, is_six_tuple)``. A normal class-dir is a 5-tuple
    ``Class,Order,Family,Genus,Species``. The single 6-tuple anomaly
    ``Eurotatoria,Ploima,Flosculariaceae,Flosculariidae,Floscularia,Floscularia sp``
    is provisional-parsed by dropping the spurious extra Order token (``Ploima``):
    ``Flosculariaceae`` is the Order elsewhere in the TSV, so the resolved nesting
    is Class=Eurotatoria, Order=Flosculariaceae, Family=Flosculariidae,
    Genus=Floscularia, Species="Floscularia sp". The anomaly is recorded in the
    reconciliation report (flag category 3); Raw_Labels keeps the full 6-tuple.
    """
    parts = class_dir.split(",")
    if len(parts) == 5:
        return parts, False
    if len(parts) == 6:
        cls, _spurious_order, order, family, genus, species = parts
        return [cls, order, family, genus, species], True
    raise ValueError(f"Unexpected class-dir arity ({len(parts)}): {class_dir!r}")


def _parse_species(genus: str | None, raw_species: str) -> tuple[str | None, str | None]:
    """Resolve the Species epithet from the raw Species column.

    Returns ``(epithet, ambiguity_kind)``. ``epithet`` is ``None`` when the species
    is unknown / open-nomenclature / ambiguous. ``ambiguity_kind`` is a short tag
    (``"or"``, ``"cf"``, ``"strain"``) when the label is flagged for the checkpoint,
    else ``None``.

    Note: sentinel ``Sp._unk*`` values are already nulled by the cascade before this
    runs; here we additionally handle ``Genus sp``/``Genus sp2`` open nomenclature,
    the ``... or ...`` two-species labels, ``cf.`` uncertain IDs, and ``jpn1/jpn2``
    strain suffixes.
    """
    if genus is None:
        return None, None

    label = raw_species.strip()
    # Strip the genus prefix when present (case-insensitive) to isolate the epithet.
    if label.lower().startswith(genus.lower() + " "):
        epithet = label[len(genus) + 1 :].strip()
    else:
        epithet = label

    epithet_l = epithet.lower()

    # Open nomenclature: unspecified species -> fall back to genus.
    if epithet_l in _BARE_SP_FORMS:
        return None, None

    # Ambiguous "species A or species B" -> genuinely ambiguous, flag + genus level.
    if " or " in epithet_l:
        return None, "or"

    # "cf." (confer) = uncertain identification -> flag + genus level.
    if epithet_l.startswith("cf.") or epithet_l.startswith("cf "):
        return None, "cf"

    # Strain annotations (jpn1/jpn2) are metadata: drop the suffix, keep the species.
    strain = None
    for suffix in _STRAIN_SUFFIXES:
        if epithet_l.endswith(suffix):
            epithet = epithet[: -len(suffix)].strip()
            strain = "strain"
            break

    return (epithet.lower() or None), strain


class Parsed:
    """A fully curated frepj taxonomy row plus the reconciliation bookkeeping."""

    __slots__ = (
        "_id_rows",
        "class_",
        "class_curated",
        "class_typo",
        "family",
        "genus",
        "is_six_tuple",
        "kingdom",
        "order",
        "phylum",
        "proposed_label",
        "raw_labels",
        "reused_genus",
        "reused_sources",
        "species",
        "species_flag",
        "tuple_family",
        "tuple_order",
    )

    def as_csv_row(self) -> dict:
        """Materialise the 19-column CSV row (external IDs blank, per this plan)."""
        return {
            "Dataset": DATASET_NAME,
            "Raw_Labels": self.raw_labels,
            "Kingdom": self.kingdom,
            "Phylum": self.phylum,
            "Class": self.class_,
            "Order": self.order or "",
            "Family": self.family or "",
            "Genus": self.genus or "",
            "Species": self.species or "",
            "proposed_label": self.proposed_label,
            "plankton": FREPJ_FLAGS["plankton"],
            "living": FREPJ_FLAGS["living"],
            "root_class": FREPJ_FLAGS["root_class"],
            "qualifier": FREPJ_FLAGS["qualifier"],
            "wikidata_ID": "",
            "aphia_ID": "",
            "NCBI_ID": "",
            "BOLD_ID": "",
            "ecotaxa_ID": "",
        }


def parse_class_dir(class_dir: str, class_anchor: dict, genus_lineage: dict) -> Parsed:
    """Curate a single frozen class-dir into a complete taxonomy row."""
    ranks, is_six = _split_ranks(class_dir)
    raw_class, raw_order, raw_family, raw_genus, raw_species = ranks

    p = Parsed()
    p.raw_labels = class_dir  # BYTE-EXACT join key (commas preserved).
    p.is_six_tuple = is_six
    p.reused_genus = None
    p.reused_sources = []
    p.class_curated = False
    p.class_typo = None
    p.species_flag = None

    # --- Sentinel cascade over Order -> Family -> Genus -> Species -----------------
    lowered = [raw_order.strip().lower(), raw_family.strip().lower(), raw_genus.strip().lower()]
    cascade_vals = [raw_order, raw_family, raw_genus, raw_species]
    cut = None
    for idx, val in enumerate(cascade_vals):
        if is_unk_sentinel(val):
            cut = idx
            break

    order = None if (cut is not None and cut <= 0) else (lowered[0] or None)
    family = None if (cut is not None and cut <= 1) else (lowered[1] or None)
    genus = None if (cut is not None and cut <= 2) else (lowered[2] or None)

    # --- Class normalisation + Kingdom/Phylum anchor ------------------------------
    class_raw_l = raw_class.strip().lower()
    class_norm = CLASS_NORMALIZE.get(class_raw_l, class_raw_l)
    if class_norm != class_raw_l:
        p.class_typo = (class_raw_l, class_norm)
    p.class_ = class_norm

    if class_norm in class_anchor and class_norm not in CURATED_ONLY_CLASSES:
        p.kingdom, p.phylum = class_anchor[class_norm]
    else:
        p.kingdom, p.phylum = CURATED_CLASS_MAP[class_norm]
        p.class_curated = class_norm in CURATED_ONLY_CLASSES

    p.tuple_order = order
    p.tuple_family = family

    # --- Species epithet ----------------------------------------------------------
    if cut is not None and cut <= 3:
        species = None
    else:
        species, sp_flag = _parse_species(genus, raw_species)
        p.species_flag = sp_flag

    # --- Cross-source reconciliation (reuse existing lineage for shared genus) -----
    if genus is not None and genus in genus_lineage:
        king, phy, cls, ord_, fam = genus_lineage[genus]["kpc_of"]
        p.kingdom, p.phylum, p.class_ = king, phy, cls
        order, family = ord_, fam
        p.reused_genus = genus
        p.reused_sources = genus_lineage[genus]["sources"]

    p.order, p.family, p.genus, p.species = order, family, genus, species

    # --- proposed_label = most-specific KNOWN rank, lowercased (never a sentinel) --
    if p.species:
        p.proposed_label = f"{p.genus} {p.species}"
    elif p.genus:
        p.proposed_label = p.genus
    elif p.family:
        p.proposed_label = p.family
    elif p.order:
        p.proposed_label = p.order
    else:
        p.proposed_label = p.class_

    return p


def build_rows(tsv_path: Path, csv_path: Path) -> tuple[list[Parsed], list[dict]]:
    """Parse every frozen class-dir; return ``(parsed, csv_rows)``."""
    class_anchor, genus_lineage = _load_existing(csv_path)
    parsed = [parse_class_dir(cd, class_anchor, genus_lineage) for cd in _load_class_dirs(tsv_path)]
    # Attach the genus_lineage reference so the report can quote external IDs.
    for p in parsed:
        if p.reused_genus is not None:
            p._id_rows = genus_lineage[p.reused_genus]["id_rows"]  # type: ignore[attr-defined]
    return parsed, [p.as_csv_row() for p in parsed]


def _encode_rows(csv_rows: list[dict]) -> str:
    """Encode the frepj rows as a pure-LF CSV block (Raw_Labels auto-quoted)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for row in csv_rows:
        writer.writerow([row[col] for col in CSV_COLUMNS])
    return buf.getvalue()


def write_csv(csv_path: Path, csv_rows: list[dict]) -> None:
    """Idempotently append the frepj rows after the pristine 1486-line prefix.

    The pre-existing header + 1485 rows are preserved byte-for-byte: the prefix is
    recovered as everything up to (and including the newline before) the first
    ``frepj,`` line, so a re-run rewrites only the frepj block and leaves the file
    byte-identical.
    """
    raw = csv_path.read_text()
    marker = "\nfrepj,"
    idx = raw.find(marker)
    prefix = raw if idx == -1 else raw[: idx + 1]
    csv_path.write_text(prefix + _encode_rows(csv_rows))


def _fmt_ids(id_rows: list[dict]) -> str:
    """Render the external IDs a genus contributes (for the 18-02 reuse note)."""
    chunks = []
    for r in id_rows:
        ids = (
            f"wikidata={r['wikidata_ID'] or '-'}, aphia={r['aphia_ID'] or '-'}, "
            f"NCBI={r['NCBI_ID'] or '-'}, BOLD={r['BOLD_ID'] or '-'}"
        )
        chunks.append(f"{r['proposed_label'] or '(genus)'} → {ids}")
    return "<br>".join(chunks)


def render_report(parsed: list[Parsed]) -> str:
    """Build the ``FREPJ_TAXONOMY_RECONCILIATION.md`` contents."""
    # --- Section A: overlaps ------------------------------------------------------
    overlaps: dict[str, Parsed] = {}
    for p in parsed:
        if p.reused_genus is not None and p.reused_genus not in overlaps:
            overlaps[p.reused_genus] = p

    overlap_lines = [
        "| FREPJ genus | Reused source(s) | Reused lineage (Kingdom/Phylum/Class/Order/Family) "
        "| External IDs to copy verbatim in Plan 18-02 |",
        "| --- | --- | --- | --- |",
    ]
    for gen in sorted(overlaps):
        p = overlaps[gen]
        lineage = f"{p.kingdom}/{p.phylum}/{p.class_}/{p.order}/{p.family}"
        ids = _fmt_ids(getattr(p, "_id_rows", []))
        overlap_lines.append(f"| {gen} | {', '.join(p.reused_sources)} | {lineage} | {ids} |")

    # --- Section B dynamic lists --------------------------------------------------
    new_cladocerans = sorted(
        {p.genus for p in parsed if p.class_ == "branchiopoda" and p.order == "diplostraca" and p.genus is not None}
    )
    reconciled_cladocerans = sorted(
        {f"{p.genus} → {p.order}" for p in parsed if p.reused_genus is not None and p.class_ == "branchiopoda"}
    )
    daphniida_rows = sorted({p.raw_labels for p in parsed if p.family == "daphniida"})
    ambiguous_species = sorted({f"{p.raw_labels}  [{p.species_flag}]" for p in parsed if p.species_flag is not None})
    open_nomenclature = sorted(
        {p.raw_labels for p in parsed if p.genus is not None and p.species is None and p.species_flag is None}
    )
    six_tuple = [p.raw_labels for p in parsed if p.is_six_tuple]
    class_typos = sorted({f"{a} → {b}" for p in parsed if (t := p.class_typo) for a, b in [t]})
    curated_classes = sorted({p.class_ for p in parsed if p.class_curated})
    genus_sentinel_rows = sorted(
        {f"{p.raw_labels} → proposed_label `{p.proposed_label}`" for p in parsed if p.genus is None and p.family is not None}
    )

    lines: list[str] = []
    lines.append("<!--")
    lines.append("(c) Inria")
    lines.append("-->")
    lines.append("")
    lines.append("# FREPJ Taxonomy Reconciliation Report")
    lines.append("")
    lines.append(
        "Generated deterministically by `build_frepj_taxonomy.py`. Documents how the 229 "
        "frozen FREPJ class-dir tuples were reconciled against the existing "
        "`planktonzilla_taxonomy.csv` sources (TAX-04) and enumerates the genuine "
        "granularity / spelling / judgment-call conflicts (TAX-06) for the Phase-18 "
        "human-verify checkpoint. External-ID columns are left blank in Plan 18-01; "
        "Plan 18-02 performs the single authoritative fill."
    )
    lines.append("")
    lines.append("## Section A — Cross-source overlap reuse (TAX-04)")
    lines.append("")
    lines.append(
        f"{len(overlaps)} FREPJ genera already exist in the CSV from another source. For each, "
        "the existing higher-rank spellings are reused VERBATIM (this is what makes the Order "
        "reconciliation below deterministic), and the existing external IDs are the values Plan "
        "18-02 will copy in."
    )
    lines.append("")
    lines.extend(overlap_lines)
    lines.append("")
    lines.append("## Section B — Flagged conflicts & judgment calls (TAX-06)")
    lines.append("")
    lines.append(
        "These are NOT silently resolved into a single spelling by the engine. Where a shared "
        "anchor exists the existing spelling is applied; the genuinely ambiguous new-to-FREPJ "
        "cases are recorded here for developer sign-off."
    )
    lines.append("")

    lines.append("### 1. Order granularity: Diplostraca vs anomopoda/ctenopoda/haplopoda")
    lines.append("")
    lines.append(
        "The FREPJ tuple uses the GBIF-2024 order `Diplostraca` for all cladocerans; the existing "
        "CSV (zoolake / sykezooscan2024) uses the finer orders below for the OVERLAPPING genera, "
        "which are reused verbatim:"
    )
    lines.append("")
    lines.extend(f"- {line}" for line in reconciled_cladocerans)
    lines.append("")
    lines.append(
        "New-to-FREPJ cladoceran genera have NO existing anchor, so their Order stays the tuple's "
        "`diplostraca` (genuinely ambiguous — decide whether to keep `diplostraca` or map each to "
        "anomopoda/ctenopoda/haplopoda/etc.):"
    )
    lines.append("")
    lines.append(f"- {', '.join(new_cladocerans)}")
    lines.append("")

    lines.append("### 2. Paradileptus: Dileptida/Dileptidae vs haptorida/tracheliidae")
    lines.append("")
    lines.append(
        "FREPJ tuple = `Litostomatea,Dileptida,Dileptidae,Paradileptus`; existing zoolake = "
        "`litostomatea,haptorida,tracheliidae,paradileptus`. Genus overlaps, so the existing "
        "`haptorida`/`tracheliidae` spellings are applied. Confirm this is the intended rank vocab."
    )
    lines.append("")

    lines.append("### 3. Six-tuple anomaly (ambiguous rank nesting)")
    lines.append("")
    lines.append(
        "One class-dir carries six comma fields instead of five (Flosculariaceae appears as an "
        "Order elsewhere in the TSV). Raw_Labels keeps the full 6-tuple byte-exact; the parse "
        "drops the spurious extra Order token (`Ploima`) and nests Order=Flosculariaceae, "
        "Family=Flosculariidae, Genus=Floscularia, Species=(open nomenclature `sp`) → "
        "proposed_label `floscularia`:"
    )
    lines.append("")
    lines.extend(f"- `{row}`" for row in six_tuple)
    lines.append("")

    lines.append("### 4. Suspected Class typo: Oligophymenophorea → oligohymenophorea")
    lines.append("")
    lines.extend(f"- {line}" for line in class_typos)
    lines.append("")

    lines.append("### 5. FREPJ-internal Family inconsistency: Daphniida vs Daphniidae")
    lines.append("")
    lines.append(
        "Simocephalus rows spell the family `Daphniida` while Ceriodaphnia/Daphnia/Scapholeberis "
        "rows spell it `Daphniidae`. Simocephalus has no existing anchor, so its family stays the "
        "tuple's literal `daphniida` (flagged; decide whether to normalise to `daphniidae`):"
    )
    lines.append("")
    lines.extend(f"- `{row}`" for row in daphniida_rows)
    lines.append("")

    lines.append("### 6. Ambiguous species labels")
    lines.append("")
    lines.append(
        "The following species strings are open-nomenclature / typo / uncertain / strain "
        "annotations. `or` and `cf.` labels resolve to genus-level proposed_label (species "
        "nulled); `strain` labels drop the jpn1/jpn2 suffix and keep the species:"
    )
    lines.append("")
    lines.extend(f"- `{row}`" for row in ambiguous_species)
    lines.append("")
    lines.append(
        f"Additionally, {len(open_nomenclature)} rows use plain `Genus sp`/`Sp._unk` open "
        "nomenclature and resolve to genus-level proposed_label (species nulled) — expected, not "
        "individually flagged."
    )
    lines.append("")

    lines.append("### 7. Class map entry with no existing anchor")
    lines.append("")
    lines.append(
        "The following Class has no anchor in the existing CSV and uses a curated Kingdom/Phylum proposal that needs sign-off:"
    )
    lines.append("")
    for cls in curated_classes:
        king, phy = CURATED_CLASS_MAP[cls]
        lines.append(f"- `{cls}` → Kingdom `{king}` / Phylum `{phy}` (curated proposal)")
    lines.append("")

    lines.append("### Appendix — Genus-sentinel cascade rows (family-level proposed_label)")
    lines.append("")
    lines.append(
        "The 9 rows whose Genus column is a `Ge._unk*` sentinel cascade to NULL Species with a "
        "family-level proposed_label (no echoed `_unk` leaks):"
    )
    lines.append("")
    lines.extend(f"- `{row}`" for row in genus_sentinel_rows)
    lines.append("")

    return "\n".join(lines)


def write_report(md_path: Path, parsed: list[Parsed]) -> None:
    """Write the reconciliation report deterministically."""
    md_path.write_text(render_report(parsed))


def main() -> None:
    """Curate the 229 FREPJ rows, append them, and emit the reconciliation report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_CLASS_DIRS_TSV)
    parser.add_argument("--csv", type=Path, default=constants.DEFAULT_TAXONOMY_CSV_FILENAME)
    parser.add_argument("--report", type=Path, default=DEFAULT_RECONCILIATION_MD)
    args = parser.parse_args()

    logger.info(f"Reading frozen class-dirs from «{args.tsv}».")
    parsed, csv_rows = build_rows(args.tsv, args.csv)
    logger.info(f"Curated {len(csv_rows)} «frepj» rows.")

    write_csv(args.csv, csv_rows)
    logger.info(f"Appended {len(csv_rows)} rows to «{args.csv}» (idempotent, append-only).")

    write_report(args.report, parsed)
    logger.info(f"Wrote reconciliation report «{args.report}».")


if __name__ == "__main__":
    main()
