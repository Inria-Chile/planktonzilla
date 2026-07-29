<div align="center">
<img src="https://raw.githubusercontent.com/Inria-Chile/planktonzilla/main/docs/banner.jpg" width="100%" alt="planktonzilla banner"/><br/>

# 🪸 🦠 🪼 🦐 🦖 🐙 🫧 🌊<br/>`planktonzilla`

Multimodal deep learning framework, datasets, and models for plankton identification.

**Part of [Inria Challenge OcéanIA](https://oceania.inria.cl/).**

[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org)
[![CI](https://img.shields.io/github/actions/workflow/status/Inria-Chile/planktonzilla/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/Inria-Chile/planktonzilla/actions/workflows/ci.yml)
![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-Models-FF9D00?logo=huggingface&logoColor=white&link=https://huggingface.co/project-oceania/models&style=for-the-badge)
![Hugging Face Datasets](https://img.shields.io/badge/Hugging_Face-Datasets-FF9D00?logo=huggingface&logoColor=white&link=https://huggingface.co/project-oceania/models&style=for-the-badge)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b8cd?logo=hexo&logoColor=white&style=for-the-badge&label=Hydra)](https://hydra.cc/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge&logo=uv)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge&logo=ruff)](https://github.com/astral-sh/ruff)
![Discord](https://img.shields.io/discord/956298015335927839?style=for-the-badge&logo=Discord&logoColor=white&color=%235865F2&link=https%3A%2F%2Fdiscord.gg%2FkksV2htk)
[![Paper DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2606.00080-FAB70C?style=for-the-badge&logo=doi)](https://doi.org/10.48550/arXiv.2606.00080)
[![License: MIT](https://img.shields.io/github/license/Inria-Chile/planktonzilla?style=for-the-badge)](LICENSE)

</div>

`planktonzilla` is a framework for managing datasets, training computer vision models, and evaluating performance on various plankton image identification tasks. Built on top of Hugging Face Transformers and Hydra for configuration management, it offers specialized tools for handling imbalanced plankton datasets and state-of-the-art imbalance learning loss functions.

## Online Resources

- `planktonzilla-17M` dataset: 17.4 million plankton images drawn from 15 source datasets, all standardized and preprocessed for deep learning applications: [`project-oceania/planktonzilla-17M`](https://huggingface.co/datasets/project-oceania/planktonzilla-17m). To explore how those source labels map onto one taxonomy, build the Sankey locally with [`pz_sankey`](#explore-the-label-space-sankey).
- Models trained on [`project-oceania/planktonzilla-17M`](https://huggingface.co/datasets/project-oceania/planktonzilla-17m):
  - [`project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt`](https://huggingface.co/project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt)
  - [`project-oceania/CLIP-ViT-B-16.bioclip-pt.planktonzilla-pt`](https://huggingface.co/project-oceania/CLIP-ViT-B-16.bioclip-pt.planktonzilla-pt)
  - [`project-oceania/CLIP-ViT-L-14.bioclip2-pt.planktonzilla-pt`](https://huggingface.co/project-oceania/CLIP-ViT-L-14.bioclip2-pt.planktonzilla-pt)
  - [`project-oceania/CLIP-ViT-L-14.laion2b-pt.planktonzilla-pt`](https://huggingface.co/project-oceania/CLIP-ViT-L-14.laion2b-pt.planktonzilla-pt)
- Planktonzilla Hugging Face Collection: <https://huggingface.co/collections/project-oceania/planktonzilla>
- Project OcéanIA project website: <https://oceania.inria.cl>.
- Project OcéanIA on Hugging Face Hub (more datasets, trained models, and demos): <https://huggingface.co/project-oceania>.

## Citation

If you use Planktonzilla in your research, please cite as:

> A. G. Contreras Montanares, L. Valenzuela, L. Martí, and N. Sanchez‑Pi, **Planktonzilla: Multimodal dataset and models
for understanding plankton ecosystems,** Inria Chile Research Center, Tech. Rep., May 2026,
doi: [10.48550/arXiv.2606.00080](https://doi.org/10.48550/arXiv.2606.00080), arXiv: 2606.00080 [cs.CV]. url: <https://arxiv.org/abs/2606.00080>

```bibtex
@techreport{contrerasmontanares:hal-05621003,
  title         = {Planktonzilla: {M}ultimodal dataset and models for understanding plankton ecosystems},
  author        = {Contreras Montanares, Alan Gerson and Valenzuela, Luis and Mart{\'i}, Luis and Sanchez-Pi, Nayat},
  year          = 2026,
  month         = {May},
  keywords      = {Explainable AI; XAI ; Plankton Classification ; CLIPS ; Multimodal Classification},
  eprinttype    = {arxiv},
  hal_id        = {hal-05621003},
  hal_version   = {v1},
  eprint        = {2606.00080},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.00080},
  doi           = {10.48550/arXiv.2606.00080},
  institution   = {Inria Chile Research Center},
}
```

## Load a pre-trained model

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

model_id = "project-oceania/<model-name>"  # see https://huggingface.co/project-oceania
processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)

image = Image.open("plankton.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_idx = outputs.logits.argmax(-1).item()
print(model.config.id2label[predicted_idx])
```

## Project Structure

```text
planktonzilla/                          # repo root
├── configs/                            # Hydra configuration tree (bundled into wheel)
│   ├── train.yaml                      # root config for pz_train
│   ├── import_dataset.yaml             # root config for pz_import_dataset
│   ├── generate_planktonzilla.yaml     # root config for dataset generation
│   ├── update_planktonzilla.yaml       # root config for dataset update
│   ├── augmentation/                   # data augmentation strategies
│   ├── custom_loss/                    # imbalance-aware loss configs
│   ├── dataset/                        # dataset-specific configs
│   ├── dataset_import/                 # per-source import configs
│   ├── debug/                          # debug-run configs
│   ├── experiment/                     # composed experiment configs
│   ├── extras/                         # misc extras (e.g. print config tree)
│   ├── hparams_search/                 # hyperparameter-search configs
│   ├── hydra/                          # Hydra runtime (help/, launcher/ for SLURM)
│   ├── local/                          # machine-local overrides
│   ├── model/                          # model architecture configs
│   ├── paths/                          # path configs (PROJECT_ROOT etc.)
│   ├── peft/                           # LoRA / PEFT adapter configs
│   ├── tracking/                       # experiment tracking (W&B, MLflow, trackio)
│   ╰── training_arguments/             # HF TrainingArguments configs
├── planktonzilla/                      # main package
│   ├── train.py                        # pz_train entry point (HF Trainer pipeline)
│   ├── dataset.py                      # DatasetWrapper: load/split/transform
│   ├── loss.py                         # imbalance-aware loss functions
│   ├── clip_model.py                   # ClipClassifier (open_clip encoder + head)
│   ├── dataset_import/                 # pz_import_dataset entry point + DatasetImporter subclasses
│   │   ╰── public_data/                # bundled source-dataset metadata
│   ├── clip_train/                     # SLURM contrastive CLIP pretraining (main.py, train.py)
│   ├── open_clip_ext/                  # forward-compat seam around open_clip factory/transform
│   │   ╰── model_configs/              # open_clip model JSON configs
│   ├── planktonzilla_dataset/          # builds the master composite dataset from external sources
│   │   ├── generate_planktonzilla.py        # main dataset build (Hydra entry)
│   │   ├── gen_planktonzilla_only_plankton.py
│   │   ├── update_planktonzilla.py          # incremental dataset update (Hydra entry)
│   │   ├── save_planktonzilla_for_clip.py   # export to WebDataset for CLIP
│   │   ├── sankey.py                        # pz_sankey — live label-space Sankey (self-contained HTML)
│   │   ├── templates/sankey_flow.html       # the page pz_sankey fills in
│   │   ├── constants.py                     # shared constants
│   │   ├── planktonzilla_taxonomy.csv       # taxonomy mapping table
│   │   ╰── utils/                            # extract_cox.py, extract_taxon_ids.py, KNOWN_ISSUES.md
│   ╰── utils/                           # hydra.py, resolvers.py, logger.py, rich_utils.py
├── scripts/                            # train.sh, train_clip.sh, push_dataset.sh (SLURM launchers)
├── notebooks/                          # exploratory analysis (metrics paper, sampling map)
├── docs/                               # banner + figures used by this README
├── .devcontainer/                      # CUDA 12.5 + cuDNN dev container
├── .github/workflows/ci.yml            # CI: lint · test · dependency-isolation guard
╰── tests/                              # pytest suite (mocks all network)
```

### Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Inria-Chile/planktonzilla.git
cd planktonzilla

# Install dependencies (creates .venv automatically)
uv sync

# Install with development dependencies
uv sync --group dev

# Activate the virtual environment (optional — `uv run` works without it)
source .venv/bin/activate
```

`uv run <command>` runs any project script inside the project venv without needing
to activate it manually. If you prefer an activated shell, run
`source .venv/bin/activate`.

### Import a public dataset as a Hugging Face dataset

```bash
# Import ISIISNET dataset
uv run pz_import_dataset dataset_import=isiisnet

# Import other available datasets
uv run pz_import_dataset dataset_import=flowcamnet
uv run pz_import_dataset dataset_import=lensless
```

Every importable source has a config in `configs/dataset_import/` — pass its filename (without
the `.yaml`) as `dataset_import=`.

### Build the composite dataset

`planktonzilla-17M` is assembled from the imported sources by mapping each source's own labels
onto the shared taxonomy in `planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv`.
Both entry points are Hydra-configured (`configs/generate_planktonzilla.yaml`,
`configs/update_planktonzilla.yaml`):

```bash
# Full build of the master composite dataset
uv run pz_generate_planktonzilla

# Incremental update of an existing build
uv run pz_update_planktonzilla
```

> The published dataset and the models trained on it are frozen artifacts. These commands are
> reproduction tooling — changing what they emit means republishing, not patching.

Both entry points stamp every image with its source's `license` / `license_url`
(see [Licensing](#licensing-of-the-composite-dataset)). `pz_update_planktonzilla` appends the
two columns with a zero-copy Arrow column concat, without ever reading the image column.
Note that the same run also re-syncs the taxonomy from the CSV, which is a full-table rewrite.
Because the added columns change the published schema, publish onto a new revision rather than
over the one the paper and the released models are pinned to:

```bash
# Add the license columns and publish them on a v1.1 branch, leaving the default revision alone
uv run pz_update_planktonzilla push_to_hub=true push_revision=v1.1
```

Then tag that revision on the Hub (`HfApi().create_tag(..., repo_type="dataset")`), leaving the
existing tag on the frozen bytes.

### Explore the label space (Sankey)

`pz_sankey` writes one self-contained HTML file — no server, no CDN, no build step — that
follows every source label from the dataset that produced it, through `root_class`, and down
the Linnaean ranks:

```text
Source dataset → root_class → Domain → Kingdom → Phylum → Class → Order → Family → Genus → Species
```

Rows whose `root_class` is not `living` have no lineage, so their `proposed_label` sits at the
**Domain** column and the ribbon ends there. Living ribbons also stop at the deepest rank the
taxonomy actually fills, so nothing drains into a fictitious "blank" node.

```bash
# Defaults: bundled taxonomy CSV + ./samples.json if present
uv run pz_sankey

# Explicit counts and output, opened when done
uv run pz_sankey --samples-json samples.json --out flow.html --open

# No image counts: ribbons are weighted by label mappings instead
uv run pz_sankey --no-samples

# Rescan the published dataset for fresh per-class counts and cache them
uv run pz_sankey --dataset-repo project-oceania/planktonzilla-17M --save-samples samples.json

# Name a different dataset on the page, with its version pinned instead of read from the Hub
uv run pz_sankey --dataset-name org/plankton-9K --dataset-version v1.2

# Fully offline: no font/logo fetch and no Hub lookup
uv run pz_sankey --no-assets
```

The page names the dataset it describes, links it back to the Hub, and stamps its own
provenance — dataset version, revision and the UTC build time — so a downloaded copy still says
which data it came from and when.

Everything in the page recomputes in the browser: show or hide any column, pick the dimension
that colours the ribbons, drag the **merge threshold** slider to pool small classes into a grey
*Other* node per column, click any node to focus on that branch, and search for a taxon. Flow
is conserved at every node on every change.

### Train a model

```bash
# Basic training with default configuration
uv run pz_train

# Train with specific dataset and model
uv run pz_train dataset=isiisnet model=resnet18

# Use specialized loss for imbalanced data
uv run pz_train dataset=isiisnet model=resnet50 custom_loss=focal

# Override training parameters
uv run pz_train dataset=isiisnet model=resnet18 training_arguments.num_train_epochs=10 training_arguments.learning_rate=1e-4
```

### Configuration system

Planktonzilla uses Hydra for hierarchical configuration management. You can override any configuration parameter:

```bash
# Use different model architecture
uv run pz_train model=efficientnet

# Apply different augmentation strategy
uv run pz_train augmentation=autoaugment

# Combine multiple overrides
uv run pz_train dataset=isiisnet model=resnet50 custom_loss=ldam training_arguments.learning_rate=1e-4
```

### Architecture

The training pipeline composes Hydra-configured datasets, models, and losses through the Hugging Face `Trainer`, then publishes the resulting checkpoint to the Hub — where external users load it with `AutoModelForImageClassification.from_pretrained`.

```mermaid
flowchart TB
  subgraph Configure["1 · Configure"]
    direction TB
    CLI["CLI<br/>pz_import_dataset · pz_train"]:::entry
    CFG["Hydra configs<br/>configs/"]:::cfg
  end

  subgraph Ingest["2 · Ingest"]
    direction TB
    DATA_IMPORT["planktonzilla/dataset_import/<br/>DatasetImporter subclasses"]:::code
    HF_DATA[("HF Hub<br/>project-oceania datasets")]:::ext
  end

  subgraph Train["3 · Train"]
    direction TB
    DATA["planktonzilla/dataset.py<br/>DatasetWrapper"]:::code
    MODEL["Model<br/>timm · HF · open_clip"]:::code
    LOSS["planktonzilla/loss.py<br/>AbstractHFLoss subclasses"]:::code
    TRAIN_LOOP["HF Trainer<br/>planktonzilla/train.py"]:::code
    TRACK["Tracking<br/>W&B · MLflow · trackio"]:::ext
    OUTPUTS["Local outputs<br/>logs/ · checkpoints/"]:::code
  end

  subgraph Publish["4 · Publish"]
    direction TB
    HF_MODEL[("HF Hub<br/>project-oceania models")]:::ext
  end

  SCRIPTS["scripts/*.sh<br/>SLURM launchers"]:::code
  TESTS["tests/<br/>smoke runs"]:::code
  CONSUMER(["AutoModelForImageClassification<br/>.from_pretrained"]):::consumer

  SCRIPTS --> CLI
  CLI --> CFG
  CFG -.->|configures| DATA_IMPORT
  CFG -.->|configures| TRAIN_LOOP
  CFG -.->|selects| MODEL
  CFG -.->|selects + params| LOSS
  DATA_IMPORT --> HF_DATA
  HF_DATA --> DATA
  DATA --> TRAIN_LOOP
  MODEL --> TRAIN_LOOP
  LOSS --> TRAIN_LOOP
  TRAIN_LOOP --> OUTPUTS
  TRAIN_LOOP -.->|metrics| TRACK
  TRAIN_LOOP --> HF_MODEL
  HF_MODEL --> CONSUMER
  TESTS -.->|smoke| TRAIN_LOOP

  classDef entry fill:#27348b,stroke:#1b2461,color:#fff
  classDef cfg fill:#e8eaf3,stroke:#27348b,color:#1b2461
  classDef code fill:#f4f5f7,stroke:#8a8f98,color:#2b2f36
  classDef ext fill:#fff3e0,stroke:#c9191e,color:#7a1013
  classDef consumer fill:#ffffff,stroke:#2b2f36,color:#2b2f36,stroke-dasharray:4 3
```

### Source datasets

Fifteen public plankton-imaging sources are assembled into `planktonzilla-17M`. Each has an
importer config in `configs/dataset_import/`:

| Source | `dataset` value | Images | Description | License |
| --- | --- | ---: | --- | --- |
| **Global UVP5** | `global_uvp5` | 7,414,467 | Underwater Vision Profiler 5, global deployment (largest contributor) | `cc-by-4.0` |
| **WHOI-Plankton** | `whoi` | 3,563,595 | Woods Hole Oceanographic Institution IFCB imagery | `mit` ⚠️ |
| **JEDI-Oceans** | `jedioceans` | 1,915,882 | JEDI oceanic plankton (CPICS) | `cc-by-sa-4.0` |
| **ZooScanNet** | `zooscan` | 1,451,745 | ZooScan scanned-sample plankton | `cc-by-nc-4.0` |
| **ZooCamNet** | `zoocamnet` | 1,286,590 | ZooCam in-situ imaging | `cc-by-4.0` |
| **UVP6Net** | `uvp6net` | 634,459 | Underwater Vision Profiler 6 | `cc-by-nc-4.0` |
| **ISIISNET** | `isiisnet` | 408,166 | In-Situ Ichthyoplankton Imaging System Network | `cc-by-nc-4.0` |
| **FlowCamNet** | `flowcamnet` | 301,247 | FlowCam imaging flow cytometry | `cc-by-nc-4.0` |
| **PlanktoScope** | `planktoscope` | 179,720 | PlanktoScope open-hardware microscopy | `cc-by-nc-4.0` |
| **MedPlanktonSet** | `medplanktonset` | 77,271 | Mediterranean plankton set | `cc-by-4.0` |
| **SYKE IFCB 2022** | `syke_ifcb_2022` | 63,074 | Finnish Environment Institute, Imaging FlowCytobot | `cc-by-4.0` |
| **PlanktonSet 1.0** | `planktonset1.0` | 60,736 | NOAA/Kaggle PlanktonSet | `other` ⚠️ |
| **SYKE ZooScan 2024** | `sykezooscan2024` | 22,753 | Finnish Environment Institute, ZooScan | `cc-by-4.0` |
| **ZooLake** | `zoolake` | 17,942 | Lake Greifensee (Switzerland) zooplankton | `cc-by-4.0` |
| **Lensless** | `lensless` | 6,400 | Lensless plankton microscopy (lab culture) | `cc-by-4.0` |

Note that the `dataset` column value does not always match the importer config stem (`whoi` vs
`whoi-plankton.yaml`, `zooscan` vs `zooscannet.yaml`, and three more). The mapping is recorded in
`constants.DATASET_IMPORT_CONFIGS`.

For training, `configs/dataset/` selects either the composite `planktonzilla` dataset or a single
source; **CIFAR-10** is also configured there as a generic sanity-check/smoke-test target.

### Licensing of the composite dataset

`planktonzilla-17M` aggregates sources under **five different sets of terms**, so it has no single
license. Every image therefore carries its source's terms in two columns — `license` (the slug,
verbatim from that source's importer config) and `license_url` (where those terms are stated):

| License | Images | Share | Reuse |
| --- | ---: | ---: | --- |
| `cc-by-4.0` | 8,888,497 | 51.07% | attribution |
| `mit` ⚠️ | 3,563,595 | 20.48% | attribution — but see KI-14 |
| `cc-by-nc-4.0` | 2,975,337 | 17.10% | attribution, **non-commercial only** |
| `cc-by-sa-4.0` | 1,915,882 | 11.01% | attribution, **share-alike** |
| `other` ⚠️ | 60,736 | 0.35% | unstated — see KI-15 |

The practical consequence: **17.1% of the corpus may not be used commercially** and a further
11.0% imposes share-alike on derivatives. Filter before you train:

```python
from datasets import load_dataset

ds = load_dataset("project-oceania/planktonzilla-17M", split="train")
commercial = ds.filter(lambda row: row["license"] in {"cc-by-4.0", "mit"})  # 12,452,092 images
```

Two entries deserve a second look before you rely on them — `whoi` (`mit` is the license of a
*code* repository, and it covers a fifth of the corpus) and `planktonset1.0` (`other` states
nothing at all). Both are recorded exactly as their importer config states them, and both are
written up as **KI-14 / KI-15** in
[`KNOWN_ISSUES.md`](planktonzilla/planktonzilla_dataset/utils/KNOWN_ISSUES.md).

The slugs live in `constants.DATASET_LICENSES`, transcribed from `configs/dataset_import/*.yaml`,
which stay the upstream source of truth: `tests/test_dataset_licenses.py` fails if the two ever
drift apart, or if a source in the published dataset has no recorded terms.

### Loss functions for imbalanced learning

Planktonzilla includes specialized loss functions designed for imbalanced plankton classification:

- **FocalLoss**: Addresses class imbalance through dynamic loss weighting
- **LDAMLoss**: Label-Distribution-Aware Margin loss
- **AsymmetricLoss**: For multi-label classification scenarios
- **RobustAsymmetricLoss**: Enhanced version of asymmetric loss
- **MaximumMarginLoss**: Margin-based learning approach
- **BalancedMetaSoftmaxLoss**: Meta-learning approach for class balance

### Experiment tracking

Integrate with popular experiment tracking tools:

```bash
# Enable Weights & Biases tracking
uv run pz_train tracking.use_wandb=true

# Enable MLflow tracking
uv run pz_train tracking.use_mlflow=true

# Enable Trackio
uv run pz_train tracking.use_trackio=true
```

### Development

#### Running Tests

```bash
# Run all tests
uv run pytest

# What CI runs — skips the slow HF Trainer / Hub integration matrices
uv run pytest tests/ --ignore=tests/test_train.py --ignore=tests/test_datasets.py

# Run with coverage
uv run pytest --cov=planktonzilla

# Run specific test file
uv run pytest tests/test_datasets.py
```

All tests mock the network: no run reaches NCBI, Wikidata, WHOI, EcoTaxa or the Hugging Face Hub.

#### Code Quality

```bash
# Lint and format — the paths CI checks
uv run ruff check planktonzilla/ tests/
uv run ruff format planktonzilla/ tests/
```

Pass the paths explicitly: `notebooks/` is inside ruff's `include` (so notebooks *can* be linted
on demand) but CI checks only `planktonzilla/` and `tests/`, and the notebooks are not currently
clean.

#### Dependency isolation

`tests/test_dependency_isolation.py` asserts that heavy visualization packages (`gradio`,
`plotly`, `kaleido`) appear in **no** dependency group and at **no** module scope under
`planktonzilla/`. `pz_sankey` renders its own SVG and embeds its own assets, so the training and
dataset core stays free of a viz stack. Function-local imports remain compliant if an opt-in
surface is ever reintroduced.

#### Adding New Datasets

1. Create a dataset configuration in `configs/dataset/your_dataset.yaml`
2. Ensure your dataset is available on Hugging Face Hub
3. Test with: `uv run pz_train dataset=your_dataset`

#### Custom Loss Functions

1. Implement your loss class inheriting from `AbstractHFLoss` in `planktonzilla/loss.py`
2. Add configuration file in `configs/custom_loss/your_loss.yaml`  
3. Loss functions must handle `ImageClassifierOutputWithNoAttention` input format
4. Test with: `uv run pz_train custom_loss=your_loss`

<div align="center">
  <strong>Built with ❤️ by <a href="https://inria.cl/">Inria</a>.</strong>
</div>
