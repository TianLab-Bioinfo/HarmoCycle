
# HarmoCycle

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
<!-- Uncomment the line below after publication -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx) -->

⚠️ This repository is under active development. Additional documentation, tutorials, and reproducibility materials will be gradually updated alongside the manuscript evaluation process.

[Introduction](#introduction) • [Installation](#installation) • [Usage](#usage) • [Reproducibility](#reproducibility) • [Citation](#citation)

---

## 📖 Introduction

(This section will be updated soon)

---

## 📂 Repository Structure

The repository is organized to facilitate both tool usage and result reproduction:

```text
.
├── HarmoCycle/       # Source code for the Python package
├── Tutorial/         # Jupyter notebooks demonstrating basic usage
├── Experiment/       # Scripts for benchmarking against other methods (Fig 2)
├── FigurePlot/       # Notebooks for generating figures (Fig 1-7)
└── Dataset/          # Documentation on data sources and preprocessing
```

---

## 🛠️ Installation

### Option 1: Quick Installation (Recommended)

HarmoCycle requires Python 3.8+ and PyTorch. We recommend using a Conda environment.

```bash
# Clone the repository
git clone https://github.com/TianLab-Bioinfo/HarmoCycle.git
cd HarmoCycle

# Create and activate environment
conda create -n harmocycle python=3.8 -y
conda activate harmocycle

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### Option 2: Step-by-step Installation (For Reproducibility)

If you need to exactly reproduce our analysis environment:

```bash
# Clone the repository
git clone https://github.com/TianLab-Bioinfo/HarmoCycle.git
cd HarmoCycle

# Create environment with specific Python version
conda create -n harmocycle python=3.8.17 -y
conda activate harmocycle

# Install core scientific packages
pip install numpy==1.23.5
pip install scipy==1.10.1
pip install pandas==1.5.3

# Install machine learning packages
pip install scikit-learn==1.3.2
pip install joblib==1.4.2

# Install single-cell analysis packages
pip install anndata==0.9.2
pip install scanpy==1.9.8

# Install PyTorch (CUDA 11.8 version)
pip install torch==2.0.0
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install visualization packages
pip install matplotlib==3.7.5
pip install seaborn==0.13.2

# Install utility packages
pip install tqdm==4.66.5
pip install gseapy==1.1.8

# Install optional but recommended packages
pip install leidenalg==0.10.2
pip install igraph==0.11.8
pip install umap-learn==0.5.6
pip install python-igraph==0.11.8

# Finally, install HarmoCycle package
pip install -e .
```

> **Note**: The PyTorch installation above is configured for CUDA 11.8. If you have a different CUDA version, please adjust the installation command accordingly from [PyTorch official site](https://pytorch.org/get-started/previous-versions/).

### Verification

To verify your installation, run:

```python
python -c "import harmocycle; print(harmocycle.__version__)"
```

---

## 🚀 Usage

### Basic Trajectory Inference

(This section will be updated soon)

For detailed demonstrations, please refer to the `Tutorial/` directory:

* `01_Basic_Usage.ipynb`: Standard pipeline for cyclic trajectory reconstruction
* `02_Advanced_Analysis.ipynb`: Examples of decoupling lineage and cell-cycle signals

---

## 📊 Reproducibility

This section describes how to reproduce the analysis presented in the manuscript.

### Data Availability
(This section will be updated soon)

### Generating Figures
(This section will be updated soon)

---

## ⚖️ License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📑 Citation

If you find HarmoCycle useful for your research, please consider citing our preprint/manuscript:

<!-- Placeholder for citation. Update this section after publication. -->
> Wen, B., et al. (2025). HarmoCycle: A Novel Method for Cyclic Trajectory Inference in Single-Cell Data.

---

## 💻 System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)
- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 (CUDA support optional but recommended for faster computation)
- **RAM**: 16GB minimum (32GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional)
