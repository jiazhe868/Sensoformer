# Sensoformer: Physics-Informed Deep Learning for Moment Tensor Inversion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

**Sensoformer** is a state-of-the-art deep learning framework designed to invert earthquake **Moment Tensors (MT)** and **Magnitude ($M_w$)** directly from single-station waveforms.

By combining a **Siamese 1D-ResNet** for local feature extraction with a **Transformer Encoder** for global event aggregation, Sensoformer solves the geometric ambiguity problem inherent in traditional inversion methods. It uniquely integrates high-performance legacy **Fortran** code for physics calculations with modern **PyTorch** deep learning workflows.

---

## 🌟 Key Features

*   **Hybrid Computing**: Seamless integration of PyTorch tensors and Fortran-based physics kernels (via `ctypes`) for rigorous Kagan angle evaluation.
*   **Permutation Invariance**: Transformer-based aggregation allows processing variable numbers of seismic stations dynamically (Dynamic Batching).
*   **Unified Data Pipeline**: A single, robust `SeismicDataset` handling both Synthetic (Pretraining) and Real (Finetuning) data with on-the-fly augmentation.
*   **Experiment Management**: Fully configurable experiments using **Hydra**, supporting rapid switching between pretraining and finetuning modes without code changes.
*   **Production Ready**: Packaged as a standard Python library (`src` layout) with CI/CD-ready Makefiles and unit tests.

---

## Models and Data

Models and data are available upon request. 

---

## 🛠️ Project Structure

The project follows a modern `src`-layout for better packaging and testing isolation.

```text
Sensoformer/
├── Makefile                 # Automation entry points (build, test, clean)
├── pyproject.toml           # Dependency and package management
├── configs/                 # Hydra configuration center
│   ├── config.yaml          # Global defaults
│   ├── model/               # Network architecture params
│   ├── data/                # Data paths and augmentation settings
│   └── training/            # LR, Loss functions, Epochs
├── src/
│   └── sensoformer/           # Core Python package
│       ├── models/          # Neural Network definitions (Sensoformer)
│       ├── data/            # Unified Dataset & Collate functions
│       ├── utils/           # Physics (Beachballs) & Metrics (Focal Loss)
│       └── ext/             # Fortran extensions (mtdcmp.f)
├── scripts/                 # Execution drivers
│   ├── train.py             # Unified Training/Finetuning script
│   └── inference.py         # Evaluation & Visualization
└── tests/                   # Unit and Integration tests
```

---

## 🚀 Installation

### Prerequisites
*   Python >= 3.9
*   GFortran (for compiling physics extensions)
*   CUDA Toolkit (optional, for GPU training)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <this_path>/Sensoformer.git
    cd Sensoformer
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    conda create -n sensoformer python=3.9
    conda activate sensoformer
    ```

3.  **Build and Install:**
    Use the `Makefile` to compile the Fortran extensions and install the package in editable mode.
    ```bash
    make build    # Compiles src/sensoformer/ext/mtdcmp.f -> mtdcmp.so
    make install  # Installs python dependencies via pip
    ```

4.  **Verify Installation:**
    Run the unit tests to ensure physics kernels and neural networks are functioning correctly.
    ```bash
    make test
    ```

---

## 🏃 Usage

Sensoformer uses **Hydra** for configuration management. You can override any parameter from the command line.

### 1. Pretraining (Synthetic Data)

By default, the system runs in pretraining mode using MSE loss and high learning rates.

```bash
python scripts/train.py data=synthetic training=pretrain
```

### 2. Finetuning (Real Data)

Switch to Transfer Learning mode (Focal Loss, Differential Learning Rates) by simply changing the config group.

```bash
python scripts/train.py \
    data=real_socal \
    training=finetune \
    training.pretrained_ckpt=/path/to/best_sensoformer_pretrain.pth
```

### 3. Inference & Visualization

Run evaluation on the test set to generate scatter plots and beachball comparisons.

> **Note:** If `model_path` is not defined in your `config.yaml`, use `+model_path` to append it dynamically.

```bash
python scripts/inference.py \
    data=real_socal \
    +model_path=outputs/debug_finetune/best_finetuned_model.pth \
    device='cuda'
```

**Output Artifacts (saved in `outputs/YYYY-MM-DD/HH-MM-SS/`):**
*   `scatter_results.pdf`: Comparison of predicted vs true MT components.
*   `beachball_grid.pdf`: Side-by-side comparison of focal mechanisms with Kagan angle annotation.
*   `kagan_histogram.pdf`: Distribution of angular errors.

---

## ⚙️ Configuration

Hyperparameters are managed in `configs/`. Key files:

*   **`configs/model/sensoformer.yaml`**:
    *   `embed_dim`: Dimension of station embeddings (Default: 128).
    *   `layers`: Number of Transformer layers (Default: 3).
*   **`configs/training/finetune.yaml`**:
    *   `lr_backbone`: Learning rate for the CNN encoder (Default: 2e-6).
    *   `lr_head`: Learning rate for regression heads (Default: 2e-6).
    *   `focal_gamma`: Focusing parameter for robust regression (Default: 1.5).

To change the batch size dynamically:
```bash
python scripts/train.py data.batch_size=128
```

---

## 🧪 Development

We adhere to strict engineering standards.

*   **Linting**: Code is formatted using `black` and imports sorted by `isort`.
*   **Testing**: All core logic (Physics, Dataset, Model) is covered by `pytest`.

To run the full test suite during development:
```bash
make test
```

To clean build artifacts:
```bash
make clean
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

