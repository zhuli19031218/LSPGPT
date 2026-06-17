# LSP-GPT: An XAI-driven Approach for Rainfall-induced Landslide Warning

> **Decoding Underlying Driving Mechanisms in the Poyang Lake Economic Zone**

A modular and reproducible experimental framework for evaluating **TabTransformer**-based models for landslide susceptibility prediction. The project supports three distinct operational modes — quick verification, fine-tuning, and full two-stage training — designed for deep learning research in geohazard applications.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Run Modes](#run-modes)
  - [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Key Results](#key-results)
- [Dependencies](#dependencies)

---

## Overview

This framework leverages transformer-based deep learning to predict rainfall-induced landslides. It was developed for the Poyang Lake Economic Zone and incorporates **21 conditioning factors** spanning topography, hydrology, land cover, and geology domains.

### Key Features

- **Three run modes** for flexible experimentation (quick eval / fine-tune / full reproduce)
- **TabTransformer** architecture adapted for tabular geospatial data with separate encoding paths for categorical and continuous features
- **Comprehensive evaluation suite**: ROC/PR curves, multi-metric comparison, and result persistence
- **Data screening pipeline**: Self-filtering mechanism with threshold-based sample refinement
- **Modular design**: Separated model definition, data utilities, evaluation tools, and experiment orchestration
- **Reproducibility**: Fixed random seeds, organized output directories (`save/logs`, `save/figs`)

---

## Project Structure

```
LSP-GPT/
├── models/
│   ├── __init__.py                   # Package init
│   └── tabtransformer.py             # Core TabTransformer model definition
├── utilis/
│   ├── __init__.py                   # Package init
│   ├── encapsulation_ipy_utilis_v26.py  # Main experiment orchestration functions
│   ├── evaluation_utils.py           # Evaluation metrics & visualization suite
│   ├── ipy_utilis.py                 # Extended utilities (training loops, data loading)
│   ├── batch_utilis_data.py          # Data processing, batching, t-SNE, NMTCriterion
│   └── utilis_file.py                # File I/O helpers (Excel, CSV, directory management)
├── Experiment Templates/
│   ├── checkpoints/                  # Pre-trained model weights (.pth)
│   └── data/                         # Training data (.pkl) and reference logs
├── save/
│   ├── figs/                         # Output visualizations (ROC, PR)
│   └── logs/                         # Experiment result logs
├── LSP-GPT_main.ipynb      # Main experiment notebook
└── README.md
```

### Core Module Details

| Module | Purpose |
|--------|---------|
| `models/tabtransformer.py` | Implements `MyTabTransformer` with Multi-Head Self-Attention, GEGLU feed-forward, and MLP classifier head |
| `utilis/encapsulation_ipy_utilis_v26.py` | High-level API: `run_quick_evaluation()`, `run_finetune_evaluation()`, `run_full_reproduction()` |
| `utilis/evaluation_utils.py` | `calculate_basic_metrics()`, `plot_roc_curve()`, `plot_pr_curve()`, `plot_comprehensive_evaluation()`, `compare_models()` |
| `utilis/batch_utilis_data.py` | `TabDataset`, `NMTCritierion` (label smoothing), data screening pipeline, t-SNE/PCA visualization |
| `utilis/ipy_utilis.py` | `load_data()`, `init_model()`, `load_pretrained_model()`, `train_model()`, training loop implementations |
| `utilis/utilis_file.py` | `read_excel()`, `write_excel()`, `mkdir()`, `save_file()`, directory naming |

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd LSP-GPT

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn einops openpyxl
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.10 (CUDA recommended for training)
- See [Dependencies](#dependencies) for the full list

---

## Usage

### Quick Start

1. Open the main notebook:
   ```bash
   jupyter notebook LSP-GPT_main.ipynb
   ```

2. Configure `RUN_MODES` in the **Setting Parameters and Paths** cell:
   ```python
   RUN_MODES = [1, 2, 3]  # Select modes to execute
   ```

3. Run all cells. Results are saved to `./save/logs/` and `./save/figs/`.

### Run Modes

| Mode | Code | Description |
|------|------|-------------|
| **Quick Verification** | `1` | Load pre-trained weights directly, perform batch prediction and evaluation |
| **Semi-complete Reproduction** | `2` | Load pre-trained weights, execute post-training fine-tuning, then evaluate |
| **Complete Reproduction** | `3` | Full two-stage process: pre-training → post-training → evaluation |


---

## Model Architecture

The **MyTabTransformer** processes tabular data through two parallel paths:

1. **Categorical Encoder**: `Linear → Reshape → Transformer (Multi-Head Self-Attention + GEGLU FFN)`
2. **Continuous Features**: `LayerNorm` (direct normalization, bypassing the second transformer for simplicity)

Both paths are concatenated and fed into a **3-layer MLP** with GELU activation, dropout, and softmax output for binary classification.

```
Input (21 features)
   ├── Continuous (18) ──→ LayerNorm ──┐
   └── Categorical (3) ──→ Linear → Transformer → Flatten ──┘
                                           │
                                    Concat → MLP → Output (2)
```

Key components:
- **Attention**: Scaled dot-product with 8 heads, dimension 16 per head
- **FeedForward**: GEGLU activation with 4× expansion ratio
- **Residual + PreNorm**: Standard transformer residual connections with pre-layer normalization
- **Regularization**: Dropout at attention and feed-forward layers

---

## Key Results

Performance on the Poyang Lake test set (2,128 samples, balanced classes):

| Mode | Accuracy | Precision | Recall | F1 Score | AUC |
|------|----------|-----------|--------|----------|-----|
| Quick Verification | 0.875 | 0.871 | 0.881 | 0.876 | 0.942 |
| Semi-complete | 0.852 | 0.854 | 0.851 | 0.852 | 0.934 |
| Complete Reproduction | 0.845 | 0.842 | 0.851 | 0.846 | 0.933 |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.10 | Deep learning framework |
| `numpy` | - | Numerical computation |
| `pandas` | - | Data manipulation |
| `matplotlib` | - | Visualization |
| `scikit-learn` | - | Metrics, preprocessing |
| `einops` | - | Tensor operations |
| `openpyxl` | - | Excel file I/O |

---

## Citation

If you use this work in your research, please cite:



---

## License

*To be specified.*
