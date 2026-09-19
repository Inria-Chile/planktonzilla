# Missing datasets survey

A systematic sweep for public plankton image datasets **absent from the 21-source build registry** in [`configs/generate_planktonzilla.yaml`](../configs/generate_planktonzilla.yaml), run 2026-09-18/19. It found **80 distinct datasets**, each filed as its own `dataset`-labelled issue (#39-#118).

This file is the index. The issues carry the detail; nothing here supersedes them.

## Method

Sixteen independent discovery lenses swept the space: Hugging Face; Zenodo/figshare/Dryad/OSF;
PANGAEA/SEANOE/Dataverse/Fairdata; Kaggle/IEEE DataPort/Roboflow/Meta-Album; data-descriptor
journals (ESSD, Scientific Data, Data in Brief, ICES JMS); benchmark and survey papers harvested
for the datasets they evaluate on; GitHub/GitLab; in-situ instruments; laboratory instruments;
freshwater and regional collections; detection/segmentation and plankton-adjacent particles; and
2025-2026 releases only. A completeness critic then named four gaps the sweep had missed — live
instrument dashboards, taxonomic reference atlases, systematic EcoTaxa enumeration, and
non-English national repositories — and a second round chased each.

That produced 165 distinct candidates. **Every one was then attacked by two independent
verifiers**: one trying to prove the dataset does not exist or is not obtainable (resolving the
DOI, fetching the files, parsing ZIP central directories to count images rather than trusting a
description), and one trying to prove it is already covered by a registry source or an open issue,
or is unfit. 96 survived as confirmed and 24 as partially uncertain; those 120 were consolidated
into 83 clusters, of which 80 were filed.

Counts in the issues are what a verifier **measured**, not what a paper claims. Several headline
figures did not survive contact: FMPD's archive holds 293 TIFFs, LMFM-12 ships 7,555 images rather
than the advertised 8,555, the Polarstern diatom collection has 3,319 unique objects rather than
~25,550, and the ASTRAL deposit's 75,371 files yield only ~5,249 plankton images.

## The 80 datasets

Ranked by value to the project — image count, instrument/geographic/taxonomic novelty against
the 21 sources, label quality, and licence cleanliness. `⚠️` marks a licence that is absent,
conflicting, or restrictive enough to block redistribution; read the issue before ingesting.

| # | Dataset | Issue | Instrument | Images | Licence | Region |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 1 | PELGAS Bay of Biscay ZooScan zooplankton dataset (2004-2016) | #39 | ZooScan (WP2 net samples, 300 um - 3.39… | 1,153,507 | `cc-by-4.0` | Bay of Biscay, NE Atlantic |
| 2 | LOG IFCB campaigns — Phytoplankton diversity at high spatial and temporal resolution (six GBIF/EcoTaxa deposits) | #40 | IFCB (Imaging FlowCytoBot) | 2,335,758 | `cc-by-nc-4.0` | Eastern English Channel, southern… |
| 3 | Tara Europa / TREC plankton community imagery (FlowCam 20 um, ZooScan 200 um, ZooScan 680 um) | #41 | FlowCam (20 um WP2 net); ZooScan (200 um… | 1,751,236 | `cc-by-nc-4.0` | European coastal seas (Tara Europa… |
| 4 | LifeWatch Belgian Part of the North Sea FlowCam phytoplankton image library and annotated training set | #42 | FlowCam VS-4 at 4x magnification (Sony… | 337,567 | `cc-by-4.0` | Belgian Part of the North Sea… |
| 5 | Public EcoTaxa LOKI projects (Arctic, Antarctic and Atlantic zooplankton) | #43 | LOKI (Lightframe On-sight Keyspecies… | 2,274,213 | `none ⚠️` | Canadian Arctic, Central Arctic… |
| 6 | UVP5 data sorted with EcoTaxa and MorphoCluster (Kiko & Schröder 2020) | #44 | UVP5 (Underwater Vision Profiler 5) | 1,588,120 | `cc-by-nc-4.0` | Global ocean, deployments… |
| 7 | Point B Villefranche-sur-Mer Juday-Bogorov 330 um ZooScan time series (1966-2025) | #45 | ZooScan (Juday-Bogorov 330 um net) | 2,023,876 | `cc-by-4.0` | Ligurian Sea, Point B,… |
| 8 | UCSC / Santa Cruz Wharf IFCB phytoplankton (Hugging Face, patcdaniel) | #46 | Imaging FlowCytobot (IFCB) | 559,046 | `none ⚠️` | Monterey Bay / Santa Cruz Municipal… |
| 9 | UVP6 on a SeaExplorer glider, 2021 spring bloom, NW Mediterranean | #47 | UVP6 mounted on a SeaExplorer glider… | 1,123,123 | `cc-by-4.0` | North-Western Mediterranean Sea |
| 10 | Fisheries and Oceans Canada VPR surveys, Gulf of St. Lawrence (public EcoTaxa projects) | #48 | VPR (Video Plankton Recorder) | 512,323 | `none ⚠️` | Gulf of St. Lawrence / Scotian… |
| 11 | eHFCM high-content fluorescence microscopy of live plankton (Tara HCS1 and TREC 2023/24) | #49 | eHFCM (environmental high-content… | 332,976 | `split ⚠️` | Global (Tara HCS1) and the European… |
| 12 | CSIRO Australian Phytoplankton Microscopy Dataset (CAPMD 2021/2022/2023) and its ML detection derivative | #50 | Zeiss Axio Observer and Zeiss Axio Plan… | 270,889 | `cc-by-nc-4.0` | Australia (ANACC culture… |
| 13 | CytoSense imaging flow cytometry — Lexplore (Lake Geneva) and MAP-IO (Indian Ocean) | #51 | CytoSense / CytoBuoy imaging flow… | 232,597 | `none ⚠️` | Lake Geneva (freshwater, Lexplore… |
| 14 | Automated shadowgraph imaging for planktonic salmon lice (IMR Bergen) | #52 | Automated winch-profiling in-situ… | 222,460 | `cc-by-4.0` | Austevoll and Flodevigen, Norway |
| 15 | Plonus et al. 2021 Video Plankton Recorder dataset (North Sea / Baltic dataset-shift benchmark) | #53 | VPR (Video Plankton Recorder) | 219,062 | `cc-by-4.0` | Inner German Bight (North Sea, ~94%… |
| 16 | PlanktonFlow76 — curated FlowCAM mesocosm dataset (INRAE UMR DECOD) | #54 | FlowCAM, segmented with ZooProcess | 198,569 | `cc-by-4.0` | INRAE UMR DECOD mesocosms, Rennes,… |
| 17 | Tropical Indian Ocean annotated planktonic foraminifera from surface sediments | #55 | MiSo automated microfossil… | 185,222 | `cc-by-nc-4.0` | Tropical Indian Ocean coretop… |
| 18 | USGS FlowCam phytoplankton CNN training images (Upper Mississippi River) | #56 | FlowCam imaging flow cytometer | 165,759 | `cc0-1.0` | Upper Mississippi River region… |
| 19 | Luo et al. 2018 ISIIS plankton training, test and validation libraries | #57 | ISIIS (In Situ Ichthyoplankton Imaging… | 160,805 | `cc-by-4.0` | not stated on the record (Cowen lab… |
| 20 | VLIZ Pi-10 Plankton Imager dataset for OSPAR-relevant taxonomic classification (Belgian North Sea) | #58 | Plankton Imager Pi-10 | 155,761 | `cc-by-4.0` | Belgian Part of the North Sea (BPNS) |
| 21 | SYKE-plankton_IFCB_Utö_2021 | #59 | IFCB (McLane Research Laboratories) | 151,235 | `cc-by-4.0` | Baltic Sea, Uto Atmospheric and… |
| 22 | Womack et al. Zenodo deposit — UW38 (UW ZooCam, Hood Canal) and HBOI19 (HOLOCAM holography) | #60 | University of Washington ZooCam… | 15,966 | `cc-by-4.0` | Hood Canal, Puget Sound,… |
| 23 | Sixty-one thousand Recent planktonic foraminifera from the Atlantic Ocean (Hsiang et al., AutoMorph) | #61 | Automated-stage light microscope with the… | 124,230 | `cc-by-4.0` | Atlantic Ocean, primarily North… |
| 24 | NES Plankton Classifier 2022 training data (WHOI Sosik Lab, Hugging Face) | #62 | IFCB, annotated with the IFCB Analysis… | 97,026 | `cc-by-4.0` | Northeast U.S. Shelf (WHOI IFCB Lab… |
| 25 | PlanktonLake-CEREEP (FlowCAM 8000, experimental freshwater ponds) | #63 | FlowCAM 8000 | 88,000 | `cc-by-4.0` | CEREEP-Ecotron IDF experimental… |
| 26 | SMHI IFCB Plankton Image Reference Library (Skagerrak, Kattegat, Baltic Proper) | #64 | IFCB (McLane Research Laboratories) | 86,232 | `cc-by-4.0` | Skagerrak, Kattegat and Baltic… |
| 27 | UDE Diatoms in the Wild 2024 | #65 | Automated light-microscope whole-slide… | 83,570 | `cc0-1.0` | German ecological monitoring and… |
| 28 | ISIIS Northern California Current training set (EcoTaxa project 3070) | #66 | ISIIS (In Situ Ichthyoplankton Imaging… | 82,736 | `none ⚠️` | Northern California Current, NE… |
| 29 | NOC LISST-Holo holographic plankton imagery (COMICS and AMT30 cruises) | #67 | LISST-Holo / LISST-Holo2 digital in-line… | 79,247 | `none ⚠️` | Southern Ocean (COMICS cruises… |
| 30 | A FAIR annotated set of 60k+ in situ zooplankton images from the Greater North Sea and Greenland (ISIIS-DPI and CPICS) | #68 | ISIIS-DPI in-situ shadowgraph imager and… | 61,990 | `cc-by-4.0` | Greater North Sea and Greenland… |
| 31 | Endless Forams plus the MD022508 and MD972138 foraminifera training sets | #69 | AutoMorph high-throughput reflected-light… | 56,004 | `cc-by-4.0` | Global modern ocean coretops… |
| 32 | PlanktoShare — a large FAIR learning set for the Plankton Imager Pi-10 | #70 | Pi-10 Plankton Imager (Plankton… | 52,882 | `cc-by-4.0` | Greater North Sea and NE Atlantic |
| 33 | Baikal_Dataset — Lake Baikal zooplankton with polygon annotations | #71 | Microscope camera with consistent… | 40,111 | `cc0-1.0 ⚠️` | Lake Baikal, Russia (freshwater,… |
| 34 | DIATLAS — the French freshwater diatom atlas image dataset | #72 | Optical light microscopy (bright field /… | 38,719 | `cc-by-4.0` | France — freshwater streams, 14… |
| 35 | Lac du Luitel freshwater PlanktoScope observations (Lacoscope, summer 2024) | #73 | PlanktoScope | 175,930 | `cc-by-nc-4.0` | Lac du Luitel, reserve naturelle,… |
| 36 | Pre-trained models and learning set for Southern North Sea Pi-10 plankton (VLIZ / iMagine) | #74 | Plankton Imager (Pi-10) | 35,605 | `cc-by-nc-4.0` | Southern North Sea (Belgium /… |
| 37 | Wetland Diatoms from Eastern China (WDEC) | #75 | Light microscopy (model not named on the… | 29,163 | `cc-by-4.0` | Lakes and peatlands of eastern… |
| 38 | Annotated CPICS and Pi-10 coastal zooplankton training sets (Hovenkamp, 'Optimizing automated classification') | #76 | CPICS and Pi-10 are the new… | 27,243 | `cc-by-sa-4.0` | North Sea, southwest Netherlands,… |
| 39 | 26,000+ machine-identified Modern planktonic foraminifera from 23 Atlantic coretops | #77 | AutoMorph imaging of picked foraminifer… | 26,663 | `cc-by-4.0` | 23 Atlantic Ocean coretops |
| 40 | NOAA AOML / SFER-SEMBON CPICS projects (public EcoTaxa) | #78 | CPICS (Continuous Particle Imaging and… | 26,251 | `none ⚠️` | South Florida / SE United States… |
| 41 | Iroise Marine Natural Park long-term ZooScan zooplankton monitoring | #79 | ZooScan (WP2-type net samples) | 655,930 | `cc-by-4.0` | Iroise Sea, North Atlantic — Iroise… |
| 42 | Pu et al. diatom image collection (Kaggle siyuepu/diatom-datasets) | #80 | Light microscopy | 49,843 | `none ⚠️` | not stated in the record; authors… |
| 43 | Predator-prey encounter detection dataset (rotifer and cryptomonad interaction videos) | #81 | Zeiss SteREO Discovery v20 dissection… | 21,221 | `apache-2.0` | laboratory culture (freshwater),… |
| 44 | Segmentation masks of ZooScan images with hand-separated touching objects | #82 | ZooScan | 19,245 | `cc-by-4.0` | Villefranche-sur-Mer / Sorbonne… |
| 45 | Dataset_BeringSea — ZOOVIS in situ plankton dataset | #83 | ZOOVIS in situ plankton imager | 17,920 | `cc0-1.0` | Southeastern Bering Sea, May 2017 |
| 46 | Lille radiolarian microfossil image datasets (Carlsson et al., ODP Leg 207) | #84 | Automatic microscope imaging with ImageJ… | 12,215 | `etalab-2.0` | ODP Leg 207, Demerara Rise, western… |
| 47 | North Sea Dogger Bank mDPI (ISIIS-DPI) zooplankton training set | #85 | mDPI, also called ISIIS-DPI (in situ Dual… | 13,417 | `cc-by-sa-4.0` | North Sea, Dogger Bank frontal… |
| 48 | AMT14 scanning electron microscopy images of the coccolithophore community | #86 | Scanning electron microscopy (SEM), TIF… | 10,649 | `cc-by-4.0` | Atlantic Ocean, Falkland Islands to… |
| 49 | Synthetic diatom detection dataset (Usefulness of synthetic datasets, UADENQ) | #87 | Optical microscopy (atlas-sourced… | 10,512 | `etalab-2.0` | France (public diatom atlases) |
| 50 | Bloombio phytoplankton microscopy dataset | #88 | High-resolution light microscopy (~3584 x… | 10,432 | `cc-by-4.0` | unknown — the card does not state a… |
| 51 | Paradinium-Oithona annotated images and count data (Scripps Plankton Camera) | #89 | Scripps Plankton Camera System (SPC) | 8,744 | `cc-by-4.0` | Scripps Pier, La Jolla, California,… |
| 52 | Planktivore Imaging System cluster-sorted high-magnification set (synchro April 2025) | #90 | Planktivore Imaging System | 7,895, | `cc-by-4.0` | unknown — the card does not state a… |
| 53 | LMFM-12 — Light Microscopy Freshwater Microalgae (12 species) | #91 | Light microscopy | 7,555 | `cc-by-4.0` | freshwater; specific region not… |
| 54 | ZooplanktonBench — geo-aware zooplankton recognition benchmark | #92 | not identified on the paper page or… | 7,470 | `cc-by-4.0 ⚠️` | marine, depth-stratified at 10 m,… |
| 55 | Optical imaging and machine learning for early developmental stages of fish | #93 | unknown — the record describes optical… | 7,142 | `cc-by-4.0` | unknown from the record |
| 56 | Mesozooplankton (MesoZ) FlowCam Macro images, Norwegian and North Seas | #94 | FlowCam Macro (150-3000 um) | 6,653 | `cc-by-4.0` | Norwegian Sea and North Sea (WP2… |
| 57 | IFCB Phytoplankton Anomaly Dataset (IFCB-PAD) | #95 | IFCB | 6,317 | `cc-by-4.0` | Baltic Sea (anomalous images from… |
| 58 | MIO scanning-electron-microscopy plankton reference series (public EcoTaxa) | #96 | Scanning electron microscope (EcoTaxa… | 6,247 | `none ⚠️` | Mediterranean / reference cultures |
| 59 | ASTRAL phytoplankton dataset (open-hardware HAB monitoring) | #97 | Cost-effective open-hardware imaging… | 75,371 | `cc-by-4.0` | IMTA (integrated multi-trophic… |
| 60 | SAG Culture Collection of Algae at Göttingen — imaged strains | #98 | Light microscopy, per strain | 4,823 | `cc-by-sa-4.0` | Global (culture collection holdings) |
| 61 | FlowCam 4X images of Pseudo-nitzschia spp. and Phaeocystis globosa (English Channel / North Sea) | #99 | Benchtop FlowCam VS-IV, 4x objective (40x… | 4,703, | `cc-by-4.0` | English Channel and North Sea… |
| 62 | Scripps Plankton Camera training images for Southern California Bight HAB drivers | #100 | Scripps Plankton Camera — micro-SPC (5x… | 2023 | `cc-by-4.0` | Scripps Pier, La Jolla, California,… |
| 63 | Annotated Southern Ocean diatom LM micrographs from Polarstern PS79 and PS103 | #101 | Light microscopy, focus-enhanced /… | 6,638 | `cc-by-nc-4.0` | Southern Ocean / South Atlantic… |
| 64 | ADIAC diatom image database | #102 | Bright-field light microscope | 3,452 | `none, terms stated ⚠️` | Europe (freshwater/marine diatom… |
| 65 | Roscoff Culture Collection (RCC) strain image library | #103 | Direct light microscopy (e.g. Olympus… | 3,408, | `none ⚠️` | Global marine (strains from cruises… |
| 66 | Diatoms of Bonaire (sediment-core virtual slides) | #104 | Light microscopy of sediment-core slides… | 3,161 | `conflict ⚠️` | Salina Bartol, Bonaire, Caribbean… |
| 67 | ZooID zooplankton species hierarchy dataset | #105 | not stated on the record — the project's… | 2,706 | `cc-by-4.0` | not stated on the record; authors… |
| 68 | PlanktoScope phytoplankton, Santa Cruz Wharf (Hugging Face) | #106 | PlanktoScope | 2,574 | `cc-by-4.0` | Santa Cruz Municipal Wharf,… |
| 69 | Gündüz et al. diatom detection, segmentation and classification benchmark | #107 | Light microscopy, 2112x1584 px colour | 2,197 | `cc-by-nc-sa-4.0` | not stated |
| 70 | NY Offshore Ichthyoplankton Images | #108 | Specimen photography of sorted embryos… | 2,137 | `cc-by-4.0` | New York offshore waters, NW… |
| 71 | AlgaeDet-HR — high-resolution bright-field multi-class algae detection dataset | #109 | Bright-field light microscopy, 10x,… | 1,664 | `cc-by-4.0` | Coastal waters around Pingtan,… |
| 72 | Prototyping deep transfer learning classification of holographic plankton imagery (iCPR, Plymouth) | #110 | Bespoke holographic camera deployed on… | unknown | `cc-by-4.0` | Continuous Plankton Recorder survey… |
| 73 | Diatoms of North America (diatoms.org) | #111 | Light microscopy (LM) and scanning… | unknown | `none ⚠️` | North America (freshwater) |
| 74 | LabelChecker FlowCam classification training data (SYKE and LakeLab) | #112 | FlowCam VS (100 um flow cell, 10x) and… | 1,004 | `cc-by-4.0` | Bay of Finland water in indoor… |
| 75 | VisAlgae 2023 — Vision Meets Algae challenge detection dataset | #113 | Microfluidic-chip high-throughput… | 1,000 | `cc-by-4.0 ⚠️` | Laboratory cultures (China) |
| 76 | Cyanobacteria lensfree holographic microscopy dataset | #114 | Lensfree (lensless) holographic microscope | 874 | `cc-by-4.0` | unknown — laboratory cultures, not… |
| 77 | Siliceous microfossil detection on sediment microscope slides (YOLO annotations) | #115 | Light microscopy of sediment microscope… | 791 | `cc-by-4.0` | not stated on the record (see the… |
| 78 | CyanoPampas — filamentous bloom-forming cyanobacteria from Pampean shallow lakes | #116 | Inverted light microscope, Utermohl… | 382 | `cc0-1.0` | Pampean shallow lakes, Buenos Aires… |
| 79 | Microscopic image dataset of Verrucodesmus verrucosus for microalgae detection and quantification | #117 | Optical microscopy of photobioreactor… | 368 | `cc-by-4.0` | TecNM / Instituto Tecnologico de… |
| 80 | FMPD — Freshwater Microscopy Phytoplankton Dataset (Lake Doniños) | #118 | Digital light microscopy at fixed… | 293 | `cc-by-nc-nd-4.0` | Lake Doninos, Ferrol, Galicia,… |

Where a row shows `unknown` images, the source genuinely publishes no count — that is the
finding, not a gap in the survey.

## What was deliberately not filed

Three verified datasets were dropped. The reasons are recorded so they are not re-proposed:

- **Fjord benthic foraminifera (Skagerrak)** — the specimens are picked from fjord sediment and
  are bottom-dwelling, not plankton. The deposit itself is fine (7,089 JPEGs, 29 species,
  CC-BY-4.0, `10.5281/zenodo.17608881`); it is simply out of domain.
- **DYB-PlanktonNet (Daya Bay Imaging Plankton Probe)** — real and live, but the imagery sits
  behind a paid IEEE DataPort subscription, so nobody can download it. CC BY-NC-ND 4.0 would be
  a second blocker.
- **In situ marine plankton holographic image dataset (HOLOCAM, Eastsound WA)** — the record and
  its CC-BY-4.0 licence are genuine, but the files are under Zenodo embargo until **2027-07-15**.
  Worth revisiting after that date.

Beyond those, 45 candidates were rejected outright by the verifiers. The recurring failure modes:

- **Metadata-only deposits.** A resolvable DOI and a real landing page, but zero files behind it —
  e.g. the ZooImage northern Baltic training set, whose Fairdata file listing returns `count: 0`.
- **Dead or restricted data links.** DeepLOKI's only public share URL returns HTTP 404, and its own
  Data Availability Statement declares the data restricted. (LOKI imagery is still reachable
  through EcoTaxa — that is issue #43.)
- **Re-packaged registry sources.** One Hugging Face candidate proved to be 100% re-packaged from
  three sources already in the build; another was byte-identical to a second candidate (same
  sha256 on row 0) and a strict subset of it.
- **Out of domain.** Benthic foraminifera, ostracods picked from marine sediment, and microplastic
  fragments from a wastewater sludge line are not plankton.
- **Operational dashboards.** Live IFCB dashboards serve raw ROIs and model predictions, not
  published human labels.

### One finding that belongs on an existing issue

`SYKE-plankton_CytoSense_2025` (Fairdata `ef66099a-…`, CC-BY-4.0, published 2026-04-01) is real,
open and downloadable, but it is **not** a new source. Its own paper states that the labelled LAB
and SEA subsets share source material with `DAPlankton_LAB` and `DAPlankton_SEA` — already registry
entry 17, whose CytoSense slices are preserved under the `lab_cs_` / `sea_cs_` filename prefixes.
What is new is finer label granularity (24 vs 15 LAB classes, 38 vs 31 SEA) and a non-image
modality: forward and sideward scatter plus four fluorescence channels. Its one genuinely new
subset, UTO (32,930 CytoSense samples, Utö, March-October 2024), is entirely unlabelled and
intended for self-supervised pre-training. That makes it a re-label / extra-modality enhancement of
an existing source rather than a missing dataset — worth a note on the closed issue #17 if the team
later wants the cross-modal profiles or a finer `DAPlankton_CS` label mapping.

## Cross-cutting notes for whoever picks these up

- **The licence spread widens.** The registry today holds `cc-by-4.0`, `cc-by-nc-4.0`,
  `cc-by-sa-4.0`, `cc0-1.0`, `mit` and `other`. These 80 add `apache-2.0`, `etalab-2.0`,
  `cc-by-nc-sa-4.0` and one `cc-by-nc-nd-4.0`. **ND is the one condition a derived-corpus
  redistribution cannot satisfy at all** — the build re-encodes source imagery, which is adapted
  material, and `README.md` licenses the composite per-image with no aggregate override. See #118.
- **Undeclared licences are common on EcoTaxa.** Nine clusters are anonymously harvestable today
  but declare no licence anywhere (#43 LOKI, #48 DFO VPR, #67 NOC LISST-Holo, #96 MIO SEM,
  #66 ISIIS NCC, #78 NOAA CPICS, #51 CytoSense, #46 UCSC IFCB, #103 RCC). They are filed because
  the imagery is real and obtainable, but each needs its owner contacted before redistribution.
- **Dedup is the main technical risk.** Several clusters overlap registry sources or each other:
  #44 UVP5/MorphoCluster against `global_uvp5`, #47 glider UVP6 against `uvp6net`, #62 NES 2022
  against `whoi`, #45 Point B against `zooscan`, #95 IFCB-PAD's healthy half against
  `syke_ifcb_2022`, and four separate North Sea Pi-10 deposits (#58, #74, #70, #76) against one
  another. Each issue names its own overlap risk.
- **EcoTaxa harvesting is already solved here.** Eleven clusters need the per-object API walk the
  four `tara_pacific_*` importers already implement — manifest first, then one vignette per object,
  resumable, with class folders keyed by taxon id rather than display name. That machinery is
  reusable rather than new work.
- **Five sources are CC0**, so they carry no attribution obligation at all: #56 USGS FlowCam,
  #65 UDE Diatoms in the Wild, #83 Bering Sea ZOOVIS, #116 CyanoPampas, and the CC0 side of the
  #71 Baikal licence conflict.

---

*Survey and issues generated with [Claude Code](https://claude.ai/code).*
