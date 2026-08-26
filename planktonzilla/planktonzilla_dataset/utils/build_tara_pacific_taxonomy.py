"""
(c) Inria

build_tara_pacific_taxonomy.py
==============================
Deterministic, network-free curation engine that turns the 600 frozen Tara Pacific
``(dataset, class_dir, ecotaxa_taxon_id)`` tuples into complete
``planktonzilla_taxonomy.csv`` rows and APPENDS them to the master CSV, leaving every
pre-existing line byte-unchanged (append-only / output-preserving milestone invariant).

Inputs (both committed, both read-only)
---------------------------------------
* ``planktonzilla/dataset_import/tara_pacific_classes.tsv`` — the frozen class-dir map the
  IMPORTER also reads, so the ``Raw_Labels`` written here are byte-identical to the
  directory names the imagefolder will carry. There is no second list to keep in sync.
* ``tests/fixtures/tara_pacific/tara_pacific_taxa.tsv`` — the frozen EcoTaxa taxon records
  (name, display name, ``type`` P/M, rank, WoRMS aphia id, id lineage, name lineage) for
  the 356 leaf taxa AND their full ancestor closure, captured from ``/api/taxon/{id}`` on
  2026-08-26. Frozen rather than fetched so this engine is offline and so a later upstream
  edit cannot silently rewrite already-published rows.

How a row is decided — four cascading rules, most-specific first
----------------------------------------------------------------
1. VERBATIM REUSE. If the class dir already exists as a ``Raw_Labels`` value anywhere in
   the CSV, copy that row's 17 non-key columns wholesale. 323 of the 600 rows take this
   path, which is the point: these four sources share EcoTaxa's label vocabulary with
   ``global_uvp5``, ``zooscan``, ``uvp6net``, ``flowcamnet``, ``zoocamnet`` and
   ``planktoscope``, so every decision the maintainers already made about
   ``part<Crustacea`` or ``dark<sphere`` is inherited instead of re-litigated.
2. ANCHOR REUSE. Otherwise resolve the row's ANCHOR — the taxon whose ranks describe the
   specimen (the taxon itself for a phylogenetic label; the nearest phylogenetic ancestor
   for a morphological one like ``part<Copepoda``) — and, if the anchor's name already
   exists as a ``proposed_label``, reuse that label, its seven ranks and its five external
   IDs verbatim. Reusing the IDs (rather than EcoTaxa's aphia id) is what keeps
   ``tests/test_taxonomy_known_issues.py::test_forward_id_mapping_is_clean`` green: one
   taxon must never carry two values of one ID column.
3. DERIVE. Otherwise build the seven ranks from the EcoTaxa lineage, then RECONCILE the
   higher ranks against the CSV: the deepest ancestor that already exists as a
   ``proposed_label`` donates its own rank and everything above it, so a genus new to this
   repository still lands under the spellings its family already uses (the TAX-04/TAX-06
   rule ``build_frepj_taxonomy`` applies). ``aphia_ID`` is filled from EcoTaxa for these —
   and only these — brand-new labels.
4. FLAGS. ``root_class``/``qualifier``/``plankton``/``living``/``proposed_label`` for a
   morphological label come from :data:`BRANCH_RULES` (which non-living branch of the
   EcoTaxa tree it hangs from) and :data:`MORPH_RULES` (which morphology or life stage the
   token names). Neither table was invented: every entry was read off the decision the
   master CSV already records for that same token under another source, and the handful
   with no precedent are marked ``NEW`` in the generated reconciliation report so a human
   can check exactly those.

``ecotaxa_ID`` is never WRITTEN by this engine — only inherited. That column holds a
small-integer legacy crosswalk from a different id space (its values do not resolve against
today's ``/api/taxon/{id}``: 328, 49, 625 are all 404s), so stamping modern EcoTaxa taxon
ids into it would silently mix two numbering schemes in one column. A row that inherits an
existing mapping keeps whatever that row recorded, because the same taxon must not read one
way under ``global_uvp5`` and blank here; a row with nothing to inherit leaves it empty. The
authoritative EcoTaxa taxon id per label lives in ``tara_pacific_classes.tsv``, which is
where the importer reads it from anyway.

The append is IDEMPOTENT: re-running rewrites the Tara Pacific block in place, so a second
run leaves the CSV byte-identical.
"""

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

from planktonzilla.planktonzilla_dataset import constants
from planktonzilla.utils.logger import get_pylogger

logger = get_pylogger(__name__)

# The four sources, in the order they are appended (= their order in cfg.datasets).
DATASET_NAMES = ("tara_pacific_bongo", "tara_pacific_decknet", "tara_pacific_hsn", "tara_pacific_manta")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLASSES_TSV = _REPO_ROOT / "planktonzilla" / "dataset_import" / "tara_pacific_classes.tsv"
DEFAULT_TAXA_TSV = _REPO_ROOT / "tests" / "fixtures" / "tara_pacific" / "tara_pacific_taxa.tsv"
DEFAULT_RECONCILIATION_MD = Path(__file__).parent / "TARA_PACIFIC_TAXONOMY_RECONCILIATION.md"

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

RANKS = constants.TAXONOMY_RANKS
ID_COLUMNS = ("wikidata_ID", "aphia_ID", "NCBI_ID", "BOLD_ID", "ecotaxa_ID")

# EcoTaxa rank -> master-CSV rank column. Only the seven the CSV models are mapped; every
# intermediate rank EcoTaxa carries (Subkingdom, Infraorder, Tribe, ...) is dropped, which
# is what the CSV has always done.
ECOTAXA_RANK_TO_COLUMN = {rank: rank for rank in RANKS}


# --- Non-living branches of the EcoTaxa tree (rule 4a) --------------------------------
#
# Checked against the ROOT of a label's lineage before the token table, because the branch
# outranks the token: `fiber` under `not-living>detritus` is marine detritus, the same
# `fiber` under `not-living>plastic` is an anthropogenic microplastic, and the master CSV
# already records them as two different things (`fiber` -> detritus, `fiber_plastic` ->
# inert / "plastic fiber"). Each entry is (root_class, qualifier, plankton, living).
BRANCH_RULES = {
    "not-living>plastic": ("inert", "", False, False),
    "not-living>detritus": ("detritus", "", False, False),
    "not-living>artefact": ("artefact", "", False, False),
    # EcoTaxa's holding pen for labels an annotator has not yet resolved. `global_uvp5`
    # records its t001..t019 as artefact / proposed_label "unknown"; t005/t008/t012/t014-17
    # arrive here for the first time and follow it.
    "temporary": ("artefact", "", False, False),
}

# `proposed_label` for a plastic label, keyed by its EcoTaxa token. Follows the plain
# -English naming the CSV already uses for plastics (`plastic`, `plastic fiber`,
# `other plastic`) rather than EcoTaxa's bare morphology word, so `fiber<plastic` cannot be
# confused with detrital `fiber` by a reader scanning the label column.
PLASTIC_LABELS = {
    "fiber": "plastic fiber",
    "filament": "plastic filament",
    "film": "plastic film",
    "fragment": "plastic fragment",
    "pellet": "plastic pellet",
    "polystyrene": "polystyrene",
    "multiple": "plastic",
    "other": "other plastic",
}

# Anything on the plastic branch with no entry above. `plastic` (not `plastic <token>`) so
# an unmapped morphology word cannot mint a new label on its own; global_uvp5's
# `other<plastic` already uses exactly this one.
PLASTIC_LABEL_DEFAULT = "plastic"

# The three labels whose branch and token disagree about what they are, resolved by hand
# against the master CSV. Keyed by the class dir, so the decision is visible rather than
# hidden in a rule interaction. Value: (root_class, qualifier, plankton, living, label).
#
#   multiple<other   EcoTaxa hangs it from `Biota>other`, so the token rule would call it a
#                    living mixture. planktoscope's `multiple_other` records the opposite —
#                    artefact, `other` — and it is the same label; follow the CSV.
#   *rods<rods       EcoTaxa hangs `rods` from `not-living>plastic>other`, so unlike
#                    `darkrods<othertocheck` (Biota>other, an imaging artefact) these two
#                    really are plastic. One label for both: they are brightness variants
#                    of one morphotype, and Raw_Labels keeps them distinct anyway.
SPECIAL_CASES = {
    "multiple<other": ("artefact", "", False, False, "other"),
    "darkrods<rods": ("inert", "", False, False, "plastic rod"),
    "lightrods<rods": ("inert", "", False, False, "plastic rod"),
}

# Every `temporary>tNNN` label resolves to this, exactly as global_uvp5's do.
TEMPORARY_LABEL = "unknown"


# --- Morphology / life-stage tokens (rule 4b) -----------------------------------------
#
# EcoTaxa spells a morphological label `<token><Parent taxon>`; `type == "M"` in the frozen
# taxa table is the authoritative marker, so the token is never guessed from the string.
# Each entry is (root_class, qualifier, plankton, living), and the comment on each group
# names the master-CSV rows the decision was read off. `qualifier` values are all members
# of constants.QUALIFIERS (pinned by tests/test_taxonomy_known_issues.py::test_ki11...).

# A whole, live organism: either a morphotype of the taxon (a diatom chain, a polar view)
# or a developmental stage the CSV already records as full_body. Precedent, one per line
# group: zoocamnet/global_uvp5 `nauplii`, `zoea`, `megalopa`, `veliger`, `ephyra`,
# `calyptopsis`, `cypris`, `pluteus`, `echinopluteus`, `ophiopluteus`, `juvenile`;
# zooscan `nectophore_*`, `bract_*`, `gonophore_*`, `eudoxie_*`; planktoscope `centric`,
# `pennate`, `chain*`, `polar view`, `cyano a/b`, `Pseudo-Nitzschia chain`, `cirrus`,
# `nucleus`, `empty`, `metanauplii`, `protozoea`.
_LIVE_WHOLE = (
    "Pseudo-Nitzschia chain",
    "attached",
    "bract",
    "calyptopsis",
    "centric",
    "chain",
    "chainlarge",
    "chainthin",
    "cirrus",
    "cyano a",
    "cyano b",
    "cyphonautes",
    "cypris",
    "damaged",
    "echinopluteus",
    "empty",
    "ephyra",
    "eudoxid",
    "gonophore",
    "juvenile",
    "megalopa",
    "metanauplii",
    "nauplii",
    "nectophore",
    "nucleus",
    "ophiopluteus",
    "pennate",
    "phyllosoma",
    "pluteus",
    "polar view",
    "polype",
    "protozoea",
    "rhizosolenia inter richelia tmp i",
    "rhizosolenia tmp i",
    "veliger",
    "zoea",
)

# A larva named as such. Precedent: zoocamnet `larvae_Annelida`, global_uvp5
# `larvae<Echinodermata`, whoi `trochophore` -> qualifier `larvae`.
_LIVE_LARVAE = ("larvae", "trochophore", "trochozoa")

# A detached body part: not plankton, not living, root_class detritus. Precedent:
# zooscan/global_uvp5 `part_*`/`part<*`, `head<*`, `tail<*`, `trunk_Appendicularia`,
# `double spike`/`hairy capsule` (acantharian spines) -> detritus / part.
_PARTS = {
    "head": "part_head",
    "part": "part",
    "part diatom": "part",
    "scale": "part_skin",
    "spines": "part",
    "tail": "part_tail",
    "trunk": "part_trunk",
    "wing": "part",
}

# Imaging or annotation artefacts. Precedent: zooscan/global_uvp5 `artefact`, `badfocus`,
# `bubble`, `duplicate`, `light aggregates`, `gelatinous`, `darkrods`, `othertocheck`.
_ARTEFACTS = (
    "aggregates",
    "artefact",
    "badfocus",
    "bubble",
    "crumple sphere",
    "dark",
    "darkrods",
    "duplicate",
    "gelatinous",
    "light",
    "lightrods",
    "othersphere",
    "othertocheck",
)

MORPH_RULES = {
    **{token: ("living", "full_body", True, True) for token in _LIVE_WHOLE},
    **{token: ("living", "larvae", True, True) for token in _LIVE_LARVAE},
    **{token: ("detritus", qualifier, False, False) for token, qualifier in _PARTS.items()},
    **{token: ("artefact", "", False, False) for token in _ARTEFACTS},
    # global_uvp5 `egg` -> living / egg / plankton False; zoocamnet `egg_Chordata` ->
    # plankton True. The flag follows whether a real taxon anchors the row, so it is set in
    # `_flags_for` rather than here.
    "egg": ("living", "egg", None, True),
    "egg sac": ("living", "egg", None, True),
    # global_uvp5 `like<Acantharea`, uvp6net `like<Copepoda` -> living / like.
    "like": ("living", "like", True, True),
    # zooscan `multiple<Copepoda` -> living / mix. Under `other` or `plastic` the branch
    # rule wins first, which is exactly what the CSV does with `multiple<other`.
    "multiple": ("living", "mix", True, True),
    # planktoscope/whoi `pollen`, zooscan/planktoscope `seaweed` -> living, but not
    # plankton: they are terrestrial or benthic material caught in a plankton sample.
    "pollen": ("living", "full_body", False, True),
    "seaweed": ("living", "full_body", False, True),
    # `cyst<unicellular` hangs from `Biota>other`, the branch the CSV records as
    # living / full_body / plankton False (`other<Biota`).
    "cyst": ("living", "full_body", False, True),
    "other": ("living", "full_body", False, True),
    "detritus": ("detritus", "", False, False),
    "feces": ("detritus", "", False, False),
    "fiber": ("detritus", "", False, False),
    "filament": ("detritus", "", False, False),
    "borax": ("detritus", "", False, False),
}

# Tokens whose `proposed_label` is a controlled word rather than the anchor taxon's name.
# All read off the master CSV: `badfocus<artefact` -> "bad focus", `duplicate` -> "error",
# `othertocheck` -> "other", `sphere<othertocheck` -> "shape".
# Applied ONLY when the label has no phylogenetic anchor, so `egg<other` becomes "egg"
# (planktoscope `egg_other`) while `egg<Actinopterygii` still takes its anchor's label
# (zoocamnet `egg_Actinopterygii` -> "chordata").
MORPH_LABELS = {
    "aggregates": "other",
    "artefact": "artefact",
    "badfocus": "bad focus",
    "borax": "detritus",
    "bubble": "bubble",
    "crumple sphere": "shape",
    "dark": "shape",
    "darkrods": "shape",
    "detritus": "detritus",
    "duplicate": "error",
    "egg": "egg",
    "egg sac": "egg",
    "feces": "feces",
    "fiber": "fiber",
    "filament": "filament",
    "gelatinous": "other",
    "light": "shape",
    "lightrods": "shape",
    "other": "other",
    "othersphere": "shape",
    "othertocheck": "other",
    "pollen": "pollen",
    "seaweed": "seaweed",
}

# Tokens with no precedent anywhere in the master CSV. Listed so the reconciliation report
# can mark exactly these rows NEW for the human-verify checkpoint, instead of leaving a
# reviewer to diff 600 rows against six other sources by hand.
# Rows whose final lineage differs from what EcoTaxa's own tree says, with the reason.
#
# All three are the SAME mechanism, and it is deliberate: rules 1 and 2 take the lineage
# the master CSV already records for a `proposed_label`, because the table's own invariant
# is that one label has exactly one lineage (KNOWN_ISSUES: "Each proposed_label has exactly
# one lineage"). Where EcoTaxa and the CSV disagree, following the CSV keeps these rows
# consistent with the six other EcoTaxa-derived sources; minting a second lineage for a
# label that already has one would break that invariant across the whole table.
#
# They are listed here — and asserted by tests/test_tara_pacific_taxonomy.py to be the ONLY
# ones — so the divergence is a recorded decision rather than an accident of rule ordering,
# and so a NEW one cannot slip in unremarked.
HOMONYM_NOTES = {
    "Odontella sp.": (
        "EcoTaxa hangs this node under `Hexapoda>Collembola>Odontella` — the springtail genus, not the "
        "diatom. `Odontella` is a homonym (WoRMS 148963 the diatom, ITIS the collembolan), and a FlowCam "
        "micro-plankton sample cannot contain a springtail. Taking the existing `odontella` row "
        "(Chromista > Heterokontophyta > Bacillariophyceae) is the correct reading AND the consistent one."
    ),
    "Ctenophora<Animalia": (
        "INHERITED ISSUE, not introduced here. EcoTaxa says Animalia > Ctenophora (the comb jellies), but "
        "the master CSV has mapped `ctenophora` to the DIATOM genus Ctenophora (aphia 163921) since long "
        "before this milestone — nine rows across zooscan, uvp6net, zoocamnet, isiisnet, global_uvp5 and "
        "planktoscope, all of them zooplankton imagers where the comb jelly is the only plausible reading. "
        "These rows follow the table rather than contradict it: a second `ctenophora` lineage would break "
        "the one-label-one-lineage invariant across all of them. Correcting the homonym is a separate "
        "change to those nine rows, gated on a golden-output diff like every other KNOWN_ISSUES data item."
    ),
    "part<Ctenophora": "Same inherited `ctenophora` homonym as `Ctenophora<Animalia` above.",
}

# A rank the frozen EcoTaxa lineage leaves blank while asserting DEEPER ones, filled by
# hand. The master CSV has no such hole in any of its 1714 pre-existing rows, and
# tests/test_sankey.py enforces that: a ribbon that stops must not resume, or a rank is
# silently skipped in the published label graph. `_assert_no_rank_gaps` fails the build if a
# hole survives, so this table can never be quietly outgrown.
#
# Keyed by proposed_label -> {rank: value}. One entry today:
#
#   branchiostoma lanceolatum — EcoTaxa goes Leptocardii > Branchiostomatidae with no order,
#     and so do WoRMS (AphiaID 104906 reports `"order": null`) and GBIF (usageKey 5227671
#     returns no `order` key), both checked 2026-08-26. NCBI (taxid 2682553) and ITIS do
#     assert `Amphioxiformes`, which is the universally used ordinal name for lancelets.
#     Filling it keeps the row's certain family/genus/species instead of truncating them
#     away, and the CSV's rank columns are already a pragmatic seven-level ladder rather
#     than a WoRMS transcription (it records `hexapoda` as a Class and the legacy diatom
#     orders `centrales`/`pennales` the same way).
RANK_GAP_FILLS = {
    "branchiostoma lanceolatum": {"Order": "amphioxiformes"},
}

TOKENS_WITHOUT_PRECEDENT = frozenset(
    {
        "attached",
        "borax",
        "cyphonautes",
        "cyst",
        "damaged",
        "eudoxid",
        "film",
        "fragment",
        "othersphere",
        "part diatom",
        "pellet",
        "phyllosoma",
        "polype",
        "polystyrene",
        "scale",
        "spines",
        "trochozoa",
        "wing",
    }
)


# --- Reading the frozen inputs --------------------------------------------------------


def read_taxa(path=DEFAULT_TAXA_TSV) -> dict[int, dict]:
    """The frozen EcoTaxa taxon records, keyed by taxon id."""
    taxa = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            taxon_id = int(row["ecotaxa_taxon_id"])
            taxa[taxon_id] = {
                "id": taxon_id,
                "name": row["name"],
                "display_name": row["display_name"],
                "type": row["type"],
                "rank": row["rank"] or None,
                "aphia_id": row["aphia_id"] or None,
                # Stored root -> leaf, inclusive of the taxon itself.
                "id_lineage": [int(part) for part in row["id_lineage"].split(";") if part],
                "lineage": row["lineage"].split(">"),
            }
    return taxa


def read_class_map(path=DEFAULT_CLASSES_TSV) -> list[dict]:
    """The frozen ``(dataset, class_dir, ecotaxa_taxon_id)`` tuples, in file order."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {"dataset": row["dataset"], "class_dir": row["class_dir"], "taxon_id": int(row["ecotaxa_taxon_id"])}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def read_master_csv(path=None) -> list[dict]:
    """The committed master taxonomy CSV, as dict rows."""
    path = path or constants.DEFAULT_TAXONOMY_CSV_FILENAME
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Resolution -----------------------------------------------------------------------


def _existing_indexes(rows):
    """``(by Raw_Labels, by proposed_label)`` over the rows of the OTHER sources.

    Tara Pacific rows are excluded so a re-run reads the same pre-existing table the first
    run did — that is what makes the append idempotent rather than self-reinforcing.
    """
    by_raw, by_label = {}, defaultdict(list)
    for row in rows:
        if row["Dataset"] in DATASET_NAMES:
            continue
        by_raw.setdefault(row["Raw_Labels"], row)
        label = (row["proposed_label"] or "").strip().lower()
        if label:
            by_label[label].append(row)
    return by_raw, by_label


def _branch(taxon) -> str | None:
    """Which non-living branch of the EcoTaxa tree a taxon hangs from, if any."""
    lineage = taxon["lineage"]
    for depth in (2, 1):
        key = ">".join(lineage[:depth])
        if key in BRANCH_RULES:
            return key
    return None


def anchor_taxon(taxon, taxa):
    """The taxon whose ranks describe the specimen, or ``None``.

    A phylogenetic label anchors on itself. A morphological one (``part<Copepoda``,
    ``larvae<Polychaeta``) anchors on its nearest phylogenetic ancestor, so the row carries
    Copepoda's or Polychaeta's ranks and records the morphology in ``qualifier`` — which is
    exactly how the master CSV already models ``part_Crustacea`` and ``larvae_Annelida``.
    A label with no phylogenetic ancestor at all (``detritus``, ``t005``, ``film``) anchors
    on nothing and carries seven blank ranks, like its counterparts in the CSV.
    """
    if taxon["type"] == "P":
        return taxon
    for ancestor_id in reversed(taxon["id_lineage"][:-1]):
        ancestor = taxa.get(ancestor_id)
        if ancestor is None or ancestor["type"] != "P":
            continue
        # `Biota` itself says nothing (`larvae<Biota` is a larva of an unknown organism),
        # and neither do the non-living roots.
        if ancestor["name"] in ("Biota", "not-living", "temporary"):
            continue
        return ancestor
    return None


def _derived_ranks(anchor, taxa) -> dict:
    """The seven rank columns read straight off the EcoTaxa lineage of ``anchor``."""
    ranks = {rank: "" for rank in RANKS}
    if anchor is None:
        return ranks
    for taxon_id in anchor["id_lineage"]:
        node = taxa.get(taxon_id)
        if node is None or node["rank"] is None:
            continue
        column = ECOTAXA_RANK_TO_COLUMN.get(node["rank"])
        if column is None:
            continue
        if column == "Species":
            # EcoTaxa names a species by its binomial; the CSV stores the epithet alone
            # and the genus separately (KNOWN_ISSUES: "the Species column holds the
            # epithet only, not the binomial").
            genus, _, epithet = node["name"].partition(" ")
            ranks["Species"] = (epithet or node["name"]).lower()
            if not ranks["Genus"] and epithet:
                ranks["Genus"] = genus.lower()
        else:
            ranks[column] = node["name"].lower()
    return ranks


def _inherit_higher_ranks(anchor, taxa, by_label):
    """Higher ranks donated by the deepest ancestor the CSV already knows.

    Returns ``(ranks, donor name)``. The donor contributes its own rank column and every
    column above it, so a genus new to this repository inherits the exact Kingdom..Family
    spellings its family already carries here — the cross-source consistency rule
    ``build_frepj_taxonomy`` calls TAX-04/TAX-06.
    """
    if anchor is None:
        return {}, None
    for ancestor_id in reversed(anchor["id_lineage"][:-1]):
        ancestor = taxa.get(ancestor_id)
        if ancestor is None:
            continue
        candidates = by_label.get(ancestor["name"].strip().lower())
        if not candidates:
            continue
        donor = candidates[0]
        for index, rank in enumerate(RANKS):
            if (donor[rank] or "").strip().lower() == ancestor["name"].strip().lower():
                return {rank: (donor[rank] or "").strip().lower() for rank in RANKS[: index + 1]}, ancestor["name"]
        # The ancestor is a known label but sits in no rank column of its own row
        # (`artefact`, `detritus`, `other`): it donates nothing and the walk continues.
    return {}, None


def _flags_for(taxon, anchor):
    """``(root_class, qualifier, plankton, living)`` for one label.

    Branch first, token second: which part of the EcoTaxa tree a label hangs from outranks
    the morphology word, because the same word means different things on different
    branches (detrital ``fiber`` vs. plastic ``fiber``).
    """
    branch = _branch(taxon)
    if branch is not None:
        return BRANCH_RULES[branch]

    if taxon["type"] == "P":
        return "living", "full_body", True, True

    rule = MORPH_RULES.get(taxon["name"])
    if rule is None:
        # An unmapped morphology token: the safest reading of an unknown non-taxonomic
        # label is "something in the sample we cannot characterise", which is what the CSV
        # calls an artefact. Reported as NEEDS-RULE in the reconciliation report.
        return "artefact", "", False, False
    root_class, qualifier, plankton, living = rule
    if plankton is None:
        # `egg` / `egg sac`: plankton only when a real taxon anchors the row, matching
        # `egg` (False) vs `egg_Chordata` (True) in the master CSV.
        plankton = anchor is not None
    return root_class, qualifier, plankton, living


def _anchor_donor(anchor, by_raw, by_label):
    """The existing CSV row that already describes ``anchor``, or ``None``.

    Looked up three ways, most specific first: the anchor's EcoTaxa display name as a
    ``Raw_Labels``, its bare name as a ``Raw_Labels``, then its bare name as a
    ``proposed_label``. The ``Raw_Labels`` lookups matter because the master CSV records
    deliberately COARSE readings for several high-rank taxa — ``Rhizaria`` maps to
    ``chromista``, ``Crustacea`` to ``arthropoda``, ``Insecta`` to ``hexapoda`` — and a row
    anchored on one of them must say the same thing rather than mint a finer label beside
    the existing one.
    """
    if anchor is None:
        return None
    for key in (anchor["display_name"], anchor["name"]):
        row = by_raw.get(key)
        if row is not None:
            return row
    candidates = by_label.get(anchor["name"].strip().lower())
    return candidates[0] if candidates else None


def _label_for(taxon, anchor, donor):
    """The ``proposed_label`` for one label, lowercased."""
    branch = _branch(taxon)
    if branch == "not-living>plastic":
        return PLASTIC_LABELS.get(taxon["name"], PLASTIC_LABEL_DEFAULT).lower()
    if branch == "temporary":
        return TEMPORARY_LABEL
    if anchor is not None:
        if donor is not None:
            return (donor["proposed_label"] or "").strip().lower()
        return anchor["name"].strip().lower()
    if taxon["type"] == "M" and taxon["name"] in MORPH_LABELS:
        return MORPH_LABELS[taxon["name"]]
    if branch == "not-living>detritus":
        return "detritus"
    if branch == "not-living>artefact":
        return "artefact"
    return taxon["name"].strip().lower()


def _float_id(value) -> str:
    """Serialize an integer external ID the way the master CSV already does (``X.0``).

    KI-12: every ``aphia_ID``/``NCBI_ID``/``BOLD_ID`` value in the CSV is float-formatted.
    That is pinned by ``tests/test_taxonomy_known_issues.py``, so a new row must match it —
    a bare ``582419`` would turn that test red.
    """
    if value in (None, ""):
        return ""
    return f"{int(value)}.0"


def _assert_no_rank_gaps(rows) -> None:
    """Fail the build if any row asserts a rank below one it left blank.

    The published label graph walks Kingdom -> Species as a ladder; a hole in the middle
    silently skips a rank (``tests/test_sankey.py`` enforces the same invariant over the
    committed CSV). :data:`RANK_GAP_FILLS` closes the one hole the frozen EcoTaxa lineages
    contain — this makes a NEW one an error at build time rather than a broken ribbon.
    """
    offenders = []
    for row in rows:
        values = [(row[rank] or "").strip() for rank in RANKS]
        first_gap = next((index for index, value in enumerate(values) if not value), len(values))
        if any(values[first_gap:]):
            offenders.append((row["Dataset"], row["Raw_Labels"], values))
    if offenders:
        listed = "; ".join(f"{dataset}/{label} {values}" for dataset, label, values in offenders[:5])
        raise ValueError(
            f"{len(offenders)} row(s) skip a taxonomic rank while asserting a deeper one: {listed}. "
            "Add the missing rank to RANK_GAP_FILLS (with its provenance) or drop the deeper ranks."
        )


def build_rows(class_map, taxa, master_rows):
    """Turn the frozen tuples into complete CSV rows. Returns ``(rows, decisions)``."""
    by_raw, by_label = _existing_indexes(master_rows)
    rows, decisions = [], []

    for entry in class_map:
        taxon = taxa[entry["taxon_id"]]
        class_dir = entry["class_dir"]
        anchor = anchor_taxon(taxon, taxa)

        existing = by_raw.get(class_dir)
        if existing is not None:
            row = {column: existing[column] for column in CSV_COLUMNS}
            row["Dataset"] = entry["dataset"]
            row["Raw_Labels"] = class_dir
            rule, donor = "verbatim", existing["Dataset"]
        else:
            anchor_donor = _anchor_donor(anchor, by_raw, by_label)
            special = SPECIAL_CASES.get(class_dir)
            if special is not None:
                root_class, qualifier, plankton, living, label = special
            else:
                root_class, qualifier, plankton, living = _flags_for(taxon, anchor)
                label = _label_for(taxon, anchor, anchor_donor)
            # Prefer the row that already carries this exact label; fall back to the row
            # that describes the anchor. Either way the ranks AND the five external IDs
            # come from one existing row, which is what keeps a taxon from acquiring a
            # second aphia/NCBI/BOLD value (test_forward_id_mapping_is_clean).
            candidates = by_label.get(label)
            donor_row = candidates[0] if candidates else anchor_donor
            if donor_row is not None:
                ranks = {rank: (donor_row[rank] or "").strip().lower() for rank in RANKS}
                ids = {column: (donor_row[column] or "").strip() for column in ID_COLUMNS}
                rule, donor = "anchor", donor_row["Dataset"]
            else:
                ranks = _derived_ranks(anchor, taxa)
                inherited, inherited_from = _inherit_higher_ranks(anchor, taxa, by_label)
                ranks.update(inherited)
                ids = {column: "" for column in ID_COLUMNS}
                ids["aphia_ID"] = _float_id(anchor["aphia_id"]) if anchor else ""
                rule, donor = "derived", inherited_from
            ranks.update(RANK_GAP_FILLS.get(label, {}))

            row = {
                "Dataset": entry["dataset"],
                "Raw_Labels": class_dir,
                **{rank: ranks[rank] for rank in RANKS},
                "proposed_label": label,
                "plankton": str(plankton),
                "living": str(living),
                "root_class": root_class,
                "qualifier": qualifier,
                **ids,
            }
            # ecotaxa_ID is inherited with the rest of the donor's IDs above and left blank
            # when there is no donor — it is never minted here, because it is a legacy
            # crosswalk in a DIFFERENT id space from today's EcoTaxa taxon ids. See the
            # module docstring; the authoritative id lives in tara_pacific_classes.tsv.
            row.setdefault("ecotaxa_ID", "")

        rows.append(row)
        decisions.append(
            {
                "dataset": entry["dataset"],
                "class_dir": class_dir,
                "taxon_id": taxon["id"],
                "type": taxon["type"],
                "rule": rule,
                "donor": donor or "",
                "anchor": anchor["name"] if anchor else "",
                "label": row["proposed_label"],
                "root_class": row["root_class"],
                "qualifier": row["qualifier"],
                "new_token": taxon["type"] == "M" and taxon["name"] in TOKENS_WITHOUT_PRECEDENT,
                "needs_rule": taxon["type"] == "M" and _branch(taxon) is None and taxon["name"] not in MORPH_RULES,
            }
        )

    _assert_no_rank_gaps(rows)
    return rows, decisions


# --- Writing --------------------------------------------------------------------------


def _serialize(rows) -> str:
    """Render rows with the master CSV's dialect (``\\n``, minimal quoting)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def append_to_master(rows, path=None) -> int:
    """Append (or, on a re-run, replace in place) the Tara Pacific block. Idempotent.

    Every line that is not a Tara Pacific row is passed through byte-for-byte, so the
    frozen header + 1485 rows and the 229 frepj rows are untouched — the append-only
    invariant ``tests/test_frepj_taxonomy_coverage.py::test_existing_rows_byte_frozen``
    pins.

    Returns:
        The number of rows written.
    """
    path = Path(path or constants.DEFAULT_TAXONOMY_CSV_FILENAME)
    original = path.read_text(encoding="utf-8")

    kept = []
    for line in original.splitlines(keepends=True):
        # A Tara Pacific row is recognised by its Dataset column, which is the first field
        # and never quoted; nothing else in the file starts with these names.
        if any(line.startswith(f"{name},") for name in DATASET_NAMES):
            continue
        kept.append(line)

    body = "".join(kept)
    if body and not body.endswith("\n"):
        body += "\n"

    path.write_text(body + _serialize(rows), encoding="utf-8")
    return len(rows)


def write_reconciliation(decisions, path=DEFAULT_RECONCILIATION_MD) -> Path:
    """Emit the human-verify report: how each row was decided and what is genuinely new."""
    by_rule = Counter(decision["rule"] for decision in decisions)
    new_tokens = sorted({decision["class_dir"] for decision in decisions if decision["new_token"]})
    needs_rule = sorted({decision["class_dir"] for decision in decisions if decision["needs_rule"]})

    lines = [
        "# Tara Pacific taxonomy reconciliation",
        "",
        "Generated by `build_tara_pacific_taxonomy.py`. Records how each of the "
        f"{len(decisions)} `tara_pacific_*` rows of `planktonzilla_taxonomy.csv` was decided, so the "
        "human-verify checkpoint knows exactly what to spot-check instead of re-reading every row.",
        "",
        "`ecotaxa_ID` is never written here, only inherited with the rest of a donor row's IDs: that "
        "column holds a legacy crosswalk in a DIFFERENT id space from today's EcoTaxa taxon ids, so a row "
        "with nothing to inherit leaves it blank rather than minting one. The authoritative EcoTaxa taxon "
        "id per label lives in `planktonzilla/dataset_import/tara_pacific_classes.tsv`.",
        "",
        "## Section A — how many rows took each rule",
        "",
        "| rule | rows | meaning |",
        "| --- | ---: | --- |",
        f"| verbatim | {by_rule['verbatim']} | the class dir already existed as a `Raw_Labels` value; "
        "the whole mapping was copied from that row |",
        f"| anchor | {by_rule['anchor']} | the anchor taxon already existed as a `proposed_label`; its "
        "ranks and external IDs were reused |",
        f"| derived | {by_rule['derived']} | built from the frozen EcoTaxa lineage, with higher ranks "
        "reconciled against the CSV |",
        "",
        "## Section B — rows to spot-check",
        "",
        f"### B1. Morphology tokens with no precedent in the master CSV ({len(new_tokens)})",
        "",
        "These carry a `root_class`/`qualifier` decision that was reasoned from the closest analogue "
        "rather than copied from an existing row.",
        "",
    ]
    lines.extend(f"- `{class_dir}`" for class_dir in new_tokens)
    lines.extend(
        [
            "",
            f"### B2. Morphology tokens with no rule at all ({len(needs_rule)})",
            "",
            "These fell through to the artefact default. A non-empty list means `MORPH_RULES` needs an "
            "entry before the next build.",
            "",
        ]
    )
    lines.extend(f"- `{class_dir}`" for class_dir in needs_rule)
    lines.extend(
        [
            "",
            f"### B3. Rows whose lineage departs from EcoTaxa's tree ({len(HOMONYM_NOTES)})",
            "",
            "All three take the lineage the master CSV already records for their `proposed_label`, because "
            "the table's invariant is one lineage per label. Two of them repair an upstream misplacement; "
            "one inherits an issue the table already had. A test asserts these are the only three.",
            "",
        ]
    )
    for class_dir, note in sorted(HOMONYM_NOTES.items()):
        lines.append(f"- `{class_dir}` — {note}")
    lines.extend(
        [
            "",
            f"### B4. Ranks filled by hand to close a ladder gap ({len(RANK_GAP_FILLS)})",
            "",
        ]
    )
    for label, fills in sorted(RANK_GAP_FILLS.items()):
        lines.append(f"- `{label}` — {', '.join(f'{rank} = {value}' for rank, value in sorted(fills.items()))}")
    lines.extend(
        [
            "",
            "### B5. EcoTaxa's two diatom lineages",
            "",
            "EcoTaxa carries diatoms under BOTH `Chromista>Bacillariophyta` and "
            "`Chromista>Heterokontophyta>Bacillariophytina>Bacillariophyceae`. Rows anchored on the first "
            "(`centric`, `pennate<Bacillariophyta`, `Coscinodiscids`, `Rhizosolenids`) therefore carry "
            "`Phylum=bacillariophyta`, while every diatom row with a genus carries the "
            "`heterokontophyta`/`bacillariophyceae` spelling the rest of the CSV uses. Both spellings are "
            "kept as upstream states them; no row was rewritten to hide the duality.",
            "",
            "## Section C — every derived row",
            "",
            "| dataset | Raw_Labels | anchor | proposed_label | root_class | qualifier | higher ranks from |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for decision in decisions:
        if decision["rule"] != "derived":
            continue
        lines.append(
            f"| {decision['dataset']} | `{decision['class_dir']}` | {decision['anchor'] or '—'} | "
            f"{decision['label']} | {decision['root_class']} | {decision['qualifier'] or '—'} | "
            f"{decision['donor'] or '—'} |"
        )
    lines.append("")

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    """CLI: rebuild the Tara Pacific taxonomy block and its reconciliation report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes-tsv", default=DEFAULT_CLASSES_TSV, type=Path)
    parser.add_argument("--taxa-tsv", default=DEFAULT_TAXA_TSV, type=Path)
    parser.add_argument("--csv", default=None, type=Path, help="Master taxonomy CSV (default: the committed one).")
    parser.add_argument("--reconciliation", default=DEFAULT_RECONCILIATION_MD, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written, write nothing.")
    args = parser.parse_args(argv)

    taxa = read_taxa(args.taxa_tsv)
    class_map = read_class_map(args.classes_tsv)
    master_rows = read_master_csv(args.csv)

    rows, decisions = build_rows(class_map, taxa, master_rows)
    by_rule = Counter(decision["rule"] for decision in decisions)
    logger.info(f"Built {len(rows)} Tara Pacific taxonomy row(s): {dict(by_rule)}.")

    needs_rule = sorted({decision["class_dir"] for decision in decisions if decision["needs_rule"]})
    if needs_rule:
        logger.warning(f"{len(needs_rule)} label(s) fell through to the artefact default: {needs_rule}")

    if args.dry_run:
        logger.info("Dry run: nothing written.")
        return 0

    written = append_to_master(rows, args.csv)
    report = write_reconciliation(decisions, args.reconciliation)
    logger.info(f"Wrote {written} row(s) to the master CSV and the report to {report}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
