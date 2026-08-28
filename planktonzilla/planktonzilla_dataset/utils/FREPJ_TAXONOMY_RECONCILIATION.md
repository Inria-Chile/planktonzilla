<!--
(c) Inria
-->

# FREPJ Taxonomy Reconciliation Report

Generated deterministically by `build_frepj_taxonomy.py`. Documents how the 229 frozen FREPJ class-dir tuples were reconciled against the existing `planktonzilla_taxonomy.csv` sources (TAX-04) and enumerates the genuine granularity / spelling / judgment-call conflicts (TAX-06) for the Phase-18 human-verify checkpoint. External-ID columns are left blank in Plan 18-01; Plan 18-02 performs the single authoritative fill.

> **2026-08-27 — historical record.** The open questions below were settled by the KI-31
> repair (see `RESOLVED_ISSUES.md`): the `diplostraca` orders of Section B.1 were mapped to
> the finer cladoceran orders the table already used (Chydoridae and the other anomopod
> families → `anomopoda`; Holopediidae/Sididae → `ctenopoda`), Section B.5's `daphniida` was
> corrected to `daphniidae`, and Section B.7's curated `lobosa` proposal was superseded by
> the table's own `tubulinea` anchor. None of those values occur in the CSV any more; the
> body below is kept verbatim as the record of the Plan 18-01 reconciliation.

## Section A — Cross-source overlap reuse (TAX-04)

18 FREPJ genera already exist in the CSV from another source. For each, the existing higher-rank spellings are reused VERBATIM (this is what makes the Order reconciliation below deterministic), and the existing external IDs are the values Plan 18-02 will copy in.

| FREPJ genus | Reused source(s) | Reused lineage (Kingdom/Phylum/Class/Order/Family) | External IDs to copy verbatim in Plan 18-02 |
| --- | --- | --- | --- |
| asplanchna | zoolake | animalia/rotifera/eurotatoria/ploima/asplanchnidae | asplanchna → wikidata=Q737675, aphia=134935.0, NCBI=84396.0, BOLD=31555.0 |
| bosmina | sykezooscan2024, zoolake | animalia/arthropoda/branchiopoda/anomopoda/bosminidae | bosmina → wikidata=Q618980, aphia=106265.0, NCBI=27402.0, BOLD=4568.0<br>bosmina → wikidata=Q618980, aphia=106265.0, NCBI=27402.0, BOLD=4568.0 |
| brachionus | zoolake | animalia/rotifera/eurotatoria/ploima/brachionidae | brachionus → wikidata=Q3516967, aphia=134937.0, NCBI=10194.0, BOLD=24694.0 |
| ceratium | whoi, zoolake | chromista/myzozoa/dinophyceae/gonyaulacales/ceratiaceae | ceratium → wikidata=Q287907, aphia=109506.0, NCBI=2915.0, BOLD=317197.0<br>ceratium → wikidata=Q287907, aphia=109506.0, NCBI=2915.0, BOLD=317197.0 |
| ceriodaphnia | sykezooscan2024 | animalia/arthropoda/branchiopoda/anomopoda/daphniidae | ceriodaphnia → wikidata=Q6549092, aphia=148376.0, NCBI=77671.0, BOLD=4592.0 |
| chaoborus | zoolake | animalia/arthropoda/insecta/diptera/chaoboridae | chaoborus → wikidata=Q2707905, aphia=118088.0, NCBI=41812.0, BOLD=169991.0 |
| conochilus | zoolake | animalia/rotifera/eurotatoria/flosculariaceae/conochilidae | conochilus → wikidata=Q4195327, aphia=134930.0, NCBI=360647.0, BOLD=281615.0 |
| cyclops | sykezooscan2024, zoolake | animalia/arthropoda/copepoda/cyclopoida/cyclopidae | cyclops → wikidata=Q1421606, aphia=149782.0, NCBI=263416.0, BOLD=302138.0<br>cyclops → wikidata=Q1421606, aphia=149782.0, NCBI=263416.0, BOLD=302138.0<br>cyclops → wikidata=Q1421606, aphia=149782.0, NCBI=263416.0, BOLD=302138.0 |
| daphnia | sykezooscan2024, zoolake | animalia/arthropoda/branchiopoda/anomopoda/daphniidae | daphnia → wikidata=Q269354, aphia=148370.0, NCBI=6668.0, BOLD=4582.0<br>daphnia → wikidata=Q269354, aphia=148370.0, NCBI=6668.0, BOLD=4582.0<br>daphnia → wikidata=Q269354, aphia=148370.0, NCBI=6668.0, BOLD=4582.0 |
| diaphanosoma | zoolake | animalia/arthropoda/branchiopoda/ctenopoda/sididae | diaphanosoma → wikidata=Q4561167, aphia=234062.0, NCBI=117531.0, BOLD=5262.0 |
| kellicottia | zoolake | animalia/rotifera/eurotatoria/ploima/brachionidae | kellicottia → wikidata=Q1738282, aphia=134940.0, NCBI=1213226.0, BOLD=284195.0 |
| keratella | zoolake | animalia/rotifera/eurotatoria/ploima/brachionidae | keratella cochlearis → wikidata=Q150192, aphia=134990.0, NCBI=204738.0, BOLD=31550.0<br>keratella quadrata → wikidata=Q2683607, aphia=134992.0, NCBI=204742.0, BOLD=31552.0 |
| leptodora | zoolake | animalia/arthropoda/branchiopoda/haplopoda/leptodoridae | leptodora → wikidata=Q15886001, aphia=247922.0, NCBI=77706.0, BOLD=4564.0 |
| oithona | global_uvp5, planktonset1.0 | animalia/arthropoda/copepoda/cyclopoida/oithonidae | oithona → wikidata=Q6556536, aphia=106485.0, NCBI=136190.0, BOLD=142597.0<br>oithona → wikidata=Q6556536, aphia=106485.0, NCBI=136190.0, BOLD=142597.0<br>oithona → wikidata=Q6556536, aphia=106485.0, NCBI=136190.0, BOLD=142597.0 |
| paradileptus | zoolake | chromista/ciliophora/litostomatea/haptorida/tracheliidae | paradileptus → wikidata=Q25360965, aphia=425542.0, NCBI=2778690.0, BOLD=72834.0 |
| polyarthra | zoolake | animalia/rotifera/eurotatoria/ploima/synchaetidae | polyarthra → wikidata=Q2102727, aphia=134957.0, NCBI=3024083.0, BOLD=1314672.0 |
| synchaeta | sykezooscan2024, zoolake | animalia/rotifera/eurotatoria/ploima/synchaetidae | synchaeta → wikidata=Q1254189, aphia=134958.0, NCBI=204744.0, BOLD=31546.0<br>synchaeta → wikidata=Q1254189, aphia=134958.0, NCBI=204744.0, BOLD=31546.0 |
| trichocerca | zoolake | animalia/rotifera/eurotatoria/ploima/trichocercidae | trichocerca → wikidata=Q2452640, aphia=134959.0, NCBI=360703.0, BOLD=281611.0 |

## Section B — Flagged conflicts & judgment calls (TAX-06)

These are NOT silently resolved into a single spelling by the engine. Where a shared anchor exists the existing spelling is applied; the genuinely ambiguous new-to-FREPJ cases are recorded here for developer sign-off.

### 1. Order granularity: Diplostraca vs anomopoda/ctenopoda/haplopoda

The FREPJ tuple uses the GBIF-2024 order `Diplostraca` for all cladocerans; the existing CSV (zoolake / sykezooscan2024) uses the finer orders below for the OVERLAPPING genera, which are reused verbatim:

- bosmina → anomopoda
- ceriodaphnia → anomopoda
- daphnia → anomopoda
- diaphanosoma → ctenopoda
- leptodora → haplopoda

New-to-FREPJ cladoceran genera have NO existing anchor, so their Order stays the tuple's `diplostraca` (genuinely ambiguous — decide whether to keep `diplostraca` or map each to anomopoda/ctenopoda/haplopoda/etc.):

- alona, alonella, bosminopsis, camptocercus, chydorus, disparalona, holopedium, leydigia, macrothrix, moina, monospilus, pleuroxus, scapholeberis, sida, simocephalus

### 2. Paradileptus: Dileptida/Dileptidae vs haptorida/tracheliidae

FREPJ tuple = `Litostomatea,Dileptida,Dileptidae,Paradileptus`; existing zoolake = `litostomatea,haptorida,tracheliidae,paradileptus`. Genus overlaps, so the existing `haptorida`/`tracheliidae` spellings are applied. Confirm this is the intended rank vocab.

### 3. Six-tuple anomaly (ambiguous rank nesting)

One class-dir carries six comma fields instead of five (Flosculariaceae appears as an Order elsewhere in the TSV). Raw_Labels keeps the full 6-tuple byte-exact; the parse drops the spurious extra Order token (`Ploima`) and nests Order=Flosculariaceae, Family=Flosculariidae, Genus=Floscularia, Species=(open nomenclature `sp`) → proposed_label `floscularia`:

- `Eurotatoria,Ploima,Flosculariaceae,Flosculariidae,Floscularia,Floscularia sp`

### 4. Suspected Class typo: Oligophymenophorea → oligohymenophorea

- oligophymenophorea → oligohymenophorea

### 5. FREPJ-internal Family inconsistency: Daphniida vs Daphniidae

Simocephalus rows spell the family `Daphniida` while Ceriodaphnia/Daphnia/Scapholeberis rows spell it `Daphniidae`. Simocephalus has no existing anchor, so its family stays the tuple's literal `daphniida` (flagged; decide whether to normalise to `daphniidae`):

- `Branchiopoda,Diplostraca,Daphniida,Simocephalus,Simocephalus serrulatus`
- `Branchiopoda,Diplostraca,Daphniida,Simocephalus,Simocephalus vetulus`

### 6. Ambiguous species labels

The following species strings are open-nomenclature / typo / uncertain / strain annotations. `or` and `cf.` labels resolve to genus-level proposed_label (species nulled); `strain` labels drop the jpn1/jpn2 suffix and keep the species:

- `Branchiopoda,Diplostraca,Daphniidae,Daphnia,Daphnai galeata or daphnia dentifera  [or]`
- `Branchiopoda,Diplostraca,Moinidae,Moina,Moina micrura jpn1  [strain]`
- `Branchiopoda,Diplostraca,Moinidae,Moina,Moina micrura jpn2  [strain]`
- `Branchiopoda,Diplostraca,Sididae,Diaphanosoma,Diaphanosoma cf. amurensis  [cf]`
- `Eurotatoria,Ploima,Brachionidae,Keratella,Keratella valga or tropica  [or]`

Additionally, 55 rows use plain `Genus sp`/`Sp._unk` open nomenclature and resolve to genus-level proposed_label (species nulled) — expected, not individually flagged.

### 7. Class map entry with no existing anchor

The following Class has no anchor in the existing CSV and uses a curated Kingdom/Phylum proposal that needs sign-off:

- `lobosa` → Kingdom `protozoa` / Phylum `amoebozoa` (curated proposal)

### Appendix — Genus-sentinel cascade rows (family-level proposed_label)

The 9 rows whose Genus column is a `Ge._unk*` sentinel cascade to NULL Species with a family-level proposed_label (no echoed `_unk` leaks):

- `Arachnida,Trombidiformes,Hydrachnidia,Ge._unk,Ge._unk → proposed_label `hydrachnidia``
- `Branchiopoda,Diplostraca,Chydoridae,Ge._unk,Ge._unk → proposed_label `chydoridae``
- `Copepoda,Cyclopoida,Cyclopidae,Ge._unk,Ge._unk → proposed_label `cyclopidae``
- `Eurotatoria,Bdelloidea,Habrotrochidae,Ge._unk,Ge._unk → proposed_label `habrotrochidae``
- `Eurotatoria,Bdelloidea,Philodinidae,Ge._unk,Ge._unk → proposed_label `philodinidae``
- `Eurotatoria,Flosculariaceae,Flosculariidae,Ge._unk,Ge._unk → proposed_label `flosculariidae``
- `Eurotatoria,Ploima,Lecanidae,Ge._unk,Ge._unk → proposed_label `lecanidae``
- `Eurotatoria,Ploima,Notommatidae,Ge._unk,Ge._unk → proposed_label `notommatidae``
- `Insecta,Diptera,Chironomidae,Ge._unk_larva_stage,Ge._unk_larva_stage → proposed_label `chironomidae``
