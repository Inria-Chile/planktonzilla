<div align="center">
<img src="docs/images/planktonzilla-logo.gif" width="29%" alt="planktonzilla logo"/>

# `planktonzilla`

Deep learning framework, datasets, and models for plankton identification.

**Part of [Inria Challenge OcéanIA](https://oceania.inria.cl/).**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.*-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![transformers](https://img.shields.io/badge/HuggingFace_transformers-1.0-FF9D00?logo=huggingface&logoColor=white&label=transformers&link=https%3A%2F%2Fgithub.com%2Fhuggingface%2F)
![timm](https://img.shields.io/badge/HuggingFace_datasets-1.0-FF9D00?logo=huggingface&logoColor=white&label=datasets&link=https%3A%2F%2Fgithub.com%2Fhuggingface%2Fpytorch-image-models)
![huggingface_hub](https://img.shields.io/badge/HuggingFace_Hub-0.23-FF9D00?logo=huggingface&logoColor=white&label=hub&link=https%3A%2F%2Fhuggingface.co%2Fdocs%2Fhuggingface_hub)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b8cd?logo=hexo&logoColor=white)](https://hydra.cc/)
![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)
![Discord](https://img.shields.io/discord/956298015335927839?logo=Discord&logoColor=white&color=%235865F2&link=https%3A%2F%2Fdiscord.gg%2FkksV2htk)
</div>

Planktonzilla provides a comprehensive toolkit for importing datasets, training computer vision models, and evaluating performance on various plankton image classification tasks. Built on top of Hugging Face Transformers and Hydra for configuration management, it offers specialized tools for handling imbalanced plankton datasets and state-of-the-art loss functions.

- OcéanIA project website: <https://oceania.inria.cl>.
- OcéanIA Hugging Face organization (datasets, trained models and demos): <https://huggingface.co/project-oceania>.

## ✨ Features

- **🔧 Modular Configuration**: Hydra-based hierarchical configuration system
- **📊 Multi-Dataset Support**: Built-in support for ISIISNET, FlowCamNet, Lensless, UVP6Net, WHOI-Plankton, and more
- **🎯 Specialized Loss Functions**: Advanced loss functions for imbalanced classification (Focal, LDAM, Asymmetric, etc.)
- **🚀 Model Hub Integration**: Seamless integration with Hugging Face Hub for model sharing
- **📈 Experiment Tracking**: Built-in support for Weights & Biases and MLflow
- **🔄 Flexible Training Pipeline**: Based on Hugging Face Transformers Trainer with custom enhancements
- **📱 Easy CLI Interface**: Simple command-line tools for all operations

## 🚀 Quick Start

### Prerequisites

- Python 3.11-3.13
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Inria-Chile/deep_plankton.git
cd planktonzilla

# Install dependencies
poetry install

# Install with development dependencies
poetry install --with dev

# Activate the virtual environment
poetry shell
```

### Basic Usage

#### 1. Import a Dataset

```bash
# Import ISIISNET dataset
poetry run pz_import_dataset dataset_import=isiisnet

# Import other available datasets
poetry run pz_import_dataset dataset_import=flowcamnet
poetry run pz_import_dataset dataset_import=lensless
```

#### 2. Train a Model

```bash
# Basic training with default configuration
poetry run pz_train

# Train with specific dataset and model
poetry run pz_train dataset=isiisnet model=resnet18

# Use specialized loss for imbalanced data
poetry run pz_train dataset=isiisnet model=resnet50 custom_loss=focal

# Override training parameters
poetry run pz_train dataset=isiisnet model=resnet18 training_arguments.num_train_epochs=10 training_arguments.learning_rate=1e-4
```

#### 3. Push Model to Hub

```bash
# Push trained model to Hugging Face Hub
poetry run pz_push_model
```

## 📁 Project Structure

```
planktonzilla/
├── configs/                    # Hydra configuration files
│   ├── dataset/               # Dataset-specific configs
│   ├── model/                 # Model architecture configs  
│   ├── training_arguments/    # Training hyperparameters
│   ├── augmentation/          # Data augmentation strategies
│   ├── custom_loss/           # Loss function configurations
│   └── tracking/              # Experiment tracking setup
├── planktonzilla/             # Main package
│   ├── dataset.py             # Dataset loading and preprocessing
│   ├── train.py               # Training pipeline
│   ├── loss.py                # Custom loss functions
│   ├── dataset_import/        # Dataset import utilities
│   └── utils/                 # Logging, Hydra helpers
└── tests/                     # Test suite
```

## 🎯 Advanced Usage

### Configuration System

Planktonzilla uses Hydra for hierarchical configuration management. You can override any configuration parameter:

```bash
# Use different model architecture
poetry run pz_train model=efficientnet

# Apply different augmentation strategy
poetry run pz_train augmentation=autoaugment

# Combine multiple overrides
poetry run pz_train dataset=isiisnet model=resnet50 custom_loss=ldam training_arguments.learning_rate=1e-4
```

### Supported Datasets

- **ISIISNET**: In-Situ Ichthyoplankton Imaging System Network
- **FlowCamNet**: FlowCam plankton dataset
- **Lensless**: Lensless plankton microscopy dataset
- **UVP6Net**: Underwater Vision Profiler 6 dataset
- **WHOI-Plankton**: Woods Hole Oceanographic Institution plankton dataset
- **ZooLake**: Lake Zurich zooplankton dataset
- **JEDI-Oceans**: JEDI oceanic plankton dataset

### Loss Functions for Imbalanced Learning

Planktonzilla includes specialized loss functions designed for imbalanced plankton classification:

- **FocalLoss**: Addresses class imbalance through dynamic loss weighting
- **LDAMLoss**: Label-Distribution-Aware Margin loss
- **AsymmetricLoss**: For multi-label classification scenarios
- **RobustAsymmetricLoss**: Enhanced version of asymmetric loss
- **MaximumMarginLoss**: Margin-based learning approach
- **BalancedMetaSoftmaxLoss**: Meta-learning approach for class balance

### Experiment Tracking

Integrate with popular experiment tracking tools:

```bash
# Enable Weights & Biases tracking
poetry run pz_train tracking.use_wandb=true

# Enable MLflow tracking  
poetry run pz_train tracking.use_mlflow=true
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=planktonzilla

# Run specific test file
poetry run pytest tests/test_datasets.py
```

### Code Quality

```bash
# Lint code
poetry run ruff check

# Format code
poetry run ruff format
```

### Adding New Datasets

1. Create a dataset configuration in `configs/dataset/your_dataset.yaml`
2. Ensure your dataset is available on Hugging Face Hub
3. Test with: `poetry run pz_train dataset=your_dataset`

### Custom Loss Functions

1. Implement your loss class inheriting from `AbstractHFLoss` in `planktonzilla/loss.py`
2. Add configuration file in `configs/custom_loss/your_loss.yaml`  
3. Loss functions must handle `ImageClassifierOutputWithNoAttention` input format

## 📊 Performance

Planktonzilla has been tested on various plankton datasets and demonstrates strong performance on imbalanced classification tasks. The framework's specialized loss functions and data handling strategies are particularly effective for marine organism identification challenges.

## 🤝 Contributing

We welcome contributions to Planktonzilla! Please feel free to:

- Report bugs and request features via [GitHub Issues](https://github.com/Inria-Chile/deep_plankton/issues)
- Submit pull requests for improvements
- Add new datasets or model architectures
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏛️ Citation

If you use Planktonzilla in your research, please cite:

```bibtex
@software{planktonzilla,
  title={Planktonzilla: A Deep Learning Framework for Plankton Identification},
  author={Inria Chile},
  year={2024},
  url={https://github.com/Inria-Chile/deep_plankton},
  version={0.1.1}
}
```

## 📞 Support

- **Homepage**: [https://oceania.inria.cl/](https://oceania.inria.cl/)
- **Issues**: [GitHub Issues](https://github.com/Inria-Chile/deep_plankton/issues)
- **Email**: [info@inria.cl](mailto:info@inria.cl)

---

<div align="center">
  <strong>Built with ❤️ by <a href="https://oceania.inria.cl/">Inria Chile</a></strong>
</div>
