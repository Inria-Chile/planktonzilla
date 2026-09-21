# Dataset ingest priority

A ranking of the 80 missing-dataset issues (#39-#118) by their actual impact on the composite
dataset. Companion to [`MISSING_DATASETS.md`](MISSING_DATASETS.md), which surveys and
deduplicates them; this file says which to do first.

## What "impact" means here

Not the headline image count. Each dataset is scored on **net usable labelled images** — what
would actually land in the build — after three deductions:

1. **Labelled only.** Unvalidated EcoTaxa objects and `unclassifiable` folders do not count.
2. **Obtainable only.** Where a deposit publishes more than it serves, only the servable slice counts.
3. **Minus measured overlap**, from the cross-check in `MISSING_DATASETS.md` — both against the 21
   registry sources and against the other 79 issues.

Across the 76 issues with a measurable count this takes 16,011,703 headline images down to
**10,894,667**, a 32% shrink. The remaining 4 publish no usable count and are listed unranked.

> **This is not the same figure as `MISSING_DATASETS.md`'s ~17.26M.** That number counts net-new
> *images*; this one counts net-new *labelled and obtainable* images. The gap is unvalidated
> EcoTaxa objects, unlabelled partitions, and #43 — excluded here entirely because its validated
> fraction was never measured.

## The ranking

`*` marks a row where the headline count overstates the real contribution by more than 40%.

| Rank | Issue | Dataset | Headline | Net usable | Licence | Note |
| ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | #40 | LOG IFCB campaigns (6) | 2,335,758 | **1,957,717** | NC | validated fraction |
| 2 | #41 | Tara Europa / TREC | 1,751,236 | **1,751,236** | NC | 100% validated |
| 3 | #39 | PELGAS ZooScan Biscay | 1,153,507 | **1,153,507** | ready |  |
| 4 | #45 | Point B ZooScan 1966-2025 | 2,023,876 | **709,128** `*` | ready | validated fraction |
| 5 | #46 | UCSC IFCB Santa Cruz | 559,046 | **559,046** | **blocked** — no licence |  |
| 6 | #47 | UVP6 SeaExplorer glider | 1,123,123 | **458,757** `*` | ready | −74,449 registry, validated |
| 7 | #48 | DFO VPR St. Lawrence | 512,323 | **451,286** | **blocked** — no licence | validated |
| 8 | #42 | LifeWatch BPNS FlowCam | 337,567 | **337,541** | ready | full library >2M, no exact total |
| 9 | #49 | eHFCM fluorescence | 332,976 | **332,976** | licence split |  |
| 10 | #50 | CSIRO CAPMD | 270,889 | **270,889** | NC | strain codes not field taxa |
| 11 | #51 | CytoSense Lexplore/MAP-IO | 232,597 | **232,597** | **blocked** — no licence |  |
| 12 | #44 | UVP5 + MorphoCluster | 1,588,120 | **224,550** `*` | NC | −649,123 registry; labelled new only |
| 13 | #53 | Plonus VPR | 219,062 | **219,062** | ready |  |
| 14 | #54 | PlanktonFlow76 FlowCAM | 198,569 | **198,569** | ready |  |
| 15 | #55 | Tropical Indian forams | 185,222 | **185,222** | NC | CNN labels |
| 16 | #56 | USGS FlowCam Mississippi | 165,759 | **165,759** | ready | CC0 |
| 17 | #57 | Luo ISIIS libraries | 160,805 | **160,805** | ready | 108 classes |
| 18 | #58 | VLIZ Pi-10 OSPAR | 155,761 | **155,698** | ready | −63 |
| 19 | #61 | 61k Recent forams (Elder) | 124,230 | **124,230** | ready | 16 coarse categories |
| 20 | #60 | Womack UW38 + HBOI19 | 122,933 | **122,933** | ready | scoped to 2 components |
| 21 | #63 | PlanktonLake-CEREEP | 88,000 | **88,000** | ready | count not independently verified |
| 22 | #64 | SMHI IFCB library | 86,232 | **86,232** | ready | 146 classes |
| 23 | #52 | IMR salmon-lice shadowgraph | 222,460 | **86,088** `*` | ready | binary classes; 100k unlabelled |
| 24 | #65 | UDE Diatoms in the Wild | 83,570 | **83,570** | ready | 611 taxa, CC0 |
| 25 | #66 | ISIIS N. California Current | 82,736 | **82,736** | **blocked** — no licence | 100% validated |
| 26 | #68 | FAIR ISIIS-DPI + CPICS | 61,990 | **61,990** | ready | superset of #85 |
| 27 | #67 | NOC LISST-Holo | 79,247 | **61,211** | **blocked** — no licence | validated |
| 28 | #59 | SYKE IFCB Utö 2021 | 151,235 | **57,207** `*` | ready | 94k unclassifiable |
| 29 | #70 | PlanktoShare Pi-10 | 53,325 | **53,313** | ready | upstream of #74/#76 |
| 30 | #62 | NES Plankton 2022 | 97,026 | **41,997** `*` | ready | −55,029 registry |
| 31 | #71 | Baikal zooplankton | 40,111 | **40,111** | ready | polygon masks |
| 32 | #72 | DIATLAS | 38,719 | **38,719** | ready | 866 taxa |
| 33 | #73 | Lac du Luitel PlanktoScope | 175,930 | **36,691** `*` | NC | validated |
| 34 | #69 | Endless Forams + MD cores | 56,004 | **36,328** | ready | −19,676 to #61 |
| 35 | #75 | WDEC China diatoms | 29,163 | **29,163** | ready | 213 species |
| 36 | #78 | NOAA CPICS SFER | 26,251 | **26,251** | **blocked** — no licence |  |
| 37 | #82 | ZooScan segmentation masks | 19,239 | **19,239** | ready | no taxonomic labels |
| 38 | #83 | Bering Sea ZOOVIS | 17,920 | **17,920** | ready | CC0, 7 classes |
| 39 | #74 | VLIZ iMagine Pi-10 | 35,598 | **16,106** `*` | NC | −19,492 to #70 |
| 40 | #84 | Lille radiolarians | 15,456 | **15,456** | ready | Eocene fossil |
| 41 | #86 | AMT14 coccolith SEM | 10,649 | **10,649** | ready | labels in companion deposit |
| 42 | #88 | Bloombio microscopy | 10,432 | **10,432** | ready |  |
| 43 | #89 | Paradinium-Oithona SPC | 8,744 | **8,744** | ready | condition labels |
| 44 | #80 | Pu diatoms (Kaggle) | 49,843 | **7,983** `*` | **blocked** — no licence | only 7,983 original |
| 45 | #90 | Planktivore cluster-sorted | 7,895 | **7,895** | ready |  |
| 46 | #91 | LMFM-12 microalgae | 7,555 | **7,555** | ready |  |
| 47 | #92 | ZooplanktonBench | 7,470 | **7,470** | **blocked** — licence conflict |  |
| 48 | #81 | Rotifer/cryptomonad video | 21,221 | **7,200** `*` | ready | 2 taxa |
| 49 | #93 | Fish early life stages | 7,142 | **7,142** | ready |  |
| 50 | #94 | MesoZ FlowCam Macro | 6,653 | **6,653** | ready |  |
| 51 | #96 | MIO SEM reference | 6,247 | **6,242** | **blocked** — no licence | species-level SEM |
| 52 | #87 | Synthetic diatom (UADENQ) | 10,512 | **5,427** `*` | ready | superseded by #72 |
| 53 | #97 | ASTRAL HAB | 75,371 | **5,249** `*` | ready | only ~5,249 are plankton |
| 54 | #98 | SAG Göttingen strains | 4,823 | **4,823** | ready (SA) | scrape-only |
| 55 | #99 | FlowCam Pseudo-nitzschia | 4,703 | **4,703** | ready | 2 taxa |
| 56 | #102 | ADIAC diatoms | 3,452 | **3,452** | **blocked** — no licence | terms stated, no SPDX |
| 57 | #103 | Roscoff RCC strains | 3,408 | **3,408** | **blocked** — no licence | scrape-only |
| 58 | #101 | Polarstern SO diatoms | 3,319 | **3,319** | NC |  |
| 59 | #76 | Hovenkamp CPICS/Pi-10 | 27,218 | **3,240** `*` | ready (SA) | 88% covered by #70+#68 |
| 60 | #104 | Diatoms of Bonaire | 3,161 | **3,161** | **blocked** — licence conflict |  |
| 61 | #105 | ZooID hierarchy | 2,706 | **2,706** | ready | hierarchy is the value |
| 62 | #106 | PlanktoScope Santa Cruz | 2,574 | **2,574** | ready |  |
| 63 | #79 | Iroise ZooScan MPA | 655,930 | **2,307** `*` | ready | only 25,747 obtainable, −23,440 |
| 64 | #95 | IFCB-PAD anomaly | 6,314 | **2,228** `*` | ready | −4,086 registry |
| 65 | #107 | Gündüz diatom benchmark | 2,197 | **2,197** | NC + SA | 68 species |
| 66 | #108 | NY ichthyoplankton | 2,137 | **2,137** | ready | DNA-barcoded |
| 67 | #109 | AlgaeDet-HR | 1,664 | **1,664** | ready | 30,552 boxes |
| 68 | #112 | LabelChecker FlowCam | 1,004 | **1,004** | ready | collages, de-collage needed |
| 69 | #113 | VisAlgae 2023 | 1,000 | **1,000** | **blocked** — licence conflict |  |
| 70 | #115 | Siliceous microfossil YOLO | 791 | **791** | ready |  |
| 71 | #116 | CyanoPampas | 382 | **382** | ready | CC0 |
| 72 | #117 | Verrucodesmus | 368 | **368** | ready | 1 species |
| 73 | #114 | Cyanobacteria lensfree | 874 | **360** `*` | ready | 437 unique, 360 labelled |
| 74 | #118 | FMPD Lake Doniños | 293 | **293** | **blocked** — ND | ND = cannot redistribute |
| 75 | #85 | Dogger Bank mDPI | 13,417 | **73** `*` | ready (SA) | 99.5% inside #68 |
| 76 | #77 | 26k machine-ID forams | 26,663 | **0** `*` | ready | 100% inside #61 |
| — | #43 | EcoTaxa LOKI projects | unknown | unknown | **blocked** — no licence | validated fraction UNMEASURED |
| — | #100 | Scripps SPC HAB | unknown | unknown | ready | count unknown; obtainability uncertain |
| — | #110 | Plymouth iCPR holography | unknown | unknown | ready | count unknown |
| — | #111 | Diatoms of North America | unknown | unknown | **blocked** — not open | licence blocker |

## Where the headline count misleads

The `*` rows above. The worst offenders, and why:

| Issue | Headline | Real | Cause |
| ---: | ---: | ---: | --- |
| #79 Iroise ZooScan MPA | 655,930 | 2,307 | only 25,747 obtainable, −23,440 |
| #77 26k machine-ID forams | 26,663 | 0 | 100% inside #61 |
| #85 Dogger Bank mDPI | 13,417 | 73 | 99.5% inside #68 |
| #97 ASTRAL HAB | 75,371 | 5,249 | only ~5,249 are plankton |
| #80 Pu diatoms (Kaggle) | 49,843 | 7,983 | only 7,983 original |
| #44 UVP5 + MorphoCluster | 1,588,120 | 224,550 | −649,123 registry; labelled new only |
| #73 Lac du Luitel PlanktoScope | 175,930 | 36,691 | validated |
| #59 SYKE IFCB Utö 2021 | 151,235 | 57,207 | 94k unclassifiable |
| #52 IMR salmon-lice shadowgraph | 222,460 | 86,088 | binary classes; 100k unlabelled |
| #62 NES Plankton 2022 | 97,026 | 41,997 | −55,029 registry |

**#44 is the subtlest.** Its net-new image count is 938,997, but only **224,550 carry labels** —
the expert-labelled partition is 61.8% duplicated against `global_uvp5` while the unlabelled one is
only 28.6%. Ranked on raw net-new images it looks top-five; ranked on labelled yield it is twelfth.

## The licence picture

| Status | Net usable | Share |
| --- | ---: | ---: |
| Ready to ingest | 4,667,628 | 42.8% |
| Non-commercial | 4,780,903 | 43.9% |
| Blocked | 1,446,136 | 13.3% |

**Nearly half the ingestable yield is NC-encumbered**, and it is concentrated in four issues —
#40, #41, #44, #50. `README.md` records NC at 17.1% of the published corpus; working down this
ranking without filtering would raise that sharply. #39 is the highest-impact dataset with a clean
licence. #56 and #65 are the only sizeable CC0 ones.

**1.45M images are blocked on paperwork, not availability.** #46, #48, #51, #66, #67, #78 and #96
are all anonymously downloadable today and simply declare no licence. Seven emails is plausibly the
highest-leverage action on this list.

## What to measure first

**#43 (public EcoTaxa LOKI projects, 2,274,213 objects)** is the largest open question in the
survey. If its validated fraction is high it ranks first or second; if its undeclared licence
cannot be resolved it drops out entirely. Measuring it changes the top of this table more than any
other single action.

#100 and #110 publish no count at all. #111 is licence-blocked regardless of size.

---

*Generated with [Claude Code](https://claude.ai/code). Figures derive from the verification and
cross-check recorded in [`MISSING_DATASETS.md`](MISSING_DATASETS.md).*
