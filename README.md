# Quantum-Enhanced Multimodal Temporal Plant Disease Progression Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.34+-brightgreen.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A multimodal temporal deep-learning framework for predicting wheat foliar disease progression from longitudinal leaf observations using a hybrid **Vision Transformer (ViT-B/16)** and **Quantum Long Short-Term Memory (QLSTM)** recurrent architecture.

---

## 1. Project Overview

Foliar plant diseases develop progressively over time across leaf surfaces rather than appearing as isolated static events. Accurately modeling this disease progression requires capturing:
* **Visual characteristics:** Fine-grained necrosis, chlorosis, and lesion pigmentation across leaf surfaces.
* **Temporal dynamics:** Longitudinal trajectories of disease spread across sequential observation days.
* **Environmental context:** Canopy microclimate, temperature, humidity, and agronomic metadata.

This repository implements a leakage-free, multi-task temporal prediction framework evaluating whether a **Hybrid Quantum Long Short-Term Memory (QLSTM)** network can outperform classical recurrent architectures (LSTM and GRU) on longitudinal wheat leaf sequences from the ETH Zurich Sequential Wheat Dataset.

---

## 2. Architecture & Methodology

```text
Sequential Wheat Leaf Observations (T = 4 days)
┌───────────────────────────┐         ┌───────────────────────────┐
│ Sequential Leaf Images    │         │ Longitudinal Metadata     │
│ (4 consecutive days)      │         │ (23 canopy & microclimate)│
└─────────────┬─────────────┘         └─────────────┬─────────────┘
              │                                     │
              ▼ (ViT-B/16)                          ▼
      Visual Embeddings                      Tabular Features
       (768 features)                         (23 features)
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                     Multimodal Fusion Vector
                     (768 + 23 = 791 features)
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │     Hybrid Quantum LSTM (QLSTM)     │
              │                                     │
              │  4 Gates (f, i, o, g), each with:   │
              │   • Linear Encoder (823 -> 4)       │
              │   • 4-Qubit Variational Circuit     │
              │     (AngleEmbedding + 2 SELs)       │
              │   • Linear Decoder (4 -> 32)        │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ Dual Task-Specific MLP Heads        │
              │ (Linear(32,16) -> GELU -> Dropout) │
              └──────────┬───────────────┬──────────┘
                         │               │
                         ▼               ▼
                 Disease Severity    Lesion Area
                   (y_placl)       (y_lesion_area)
```

### Key Architectural Highlights:
* **Feature Extraction:** Pre-trained Vision Transformer (`ViT-B/16`) extracts 768-dimensional visual patch tokens from leaf photographs, concatenated with 23 longitudinal metadata features to produce a 791-dimensional representation at each timestep ($T=4$).
* **Quantum Gates:** The QLSTM cell features dedicated 4-qubit parameterized quantum circuits (`AngleEmbedding` + 2 `StronglyEntanglingLayers`) in all four recurrent gates ($f_t, i_t, o_t, g_t$), governed by standard LSTM dynamics:
  $$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$
* **Balanced Multi-Task Loss:** Optimized via:
  $$\mathcal{L}_{\text{total}} = 10.0 \times \mathcal{L}_{\text{disease}} + 0.5 \times \mathcal{L}_{\text{lesion}}$$
  This calibrated weighting balances gradient backpropagation between bounded disease severity ($[0, 1]$) and standardized lesion area ($\sigma=1$).

---

## 3. Leakage-Free Dataset Protocol

To prevent data leakage, all splits and feature scalers are isolated strictly at the **individual plant leaf level**:

* **Splitting Scheme:** Two-stage `GroupShuffleSplit` (random seed 42) grouped exclusively by `leaf_id`.
  * **Train Set:** 21 leaves (226 sliding sequences)
  * **Validation Set:** 4 leaves (47 sliding sequences)
  * **Held-Out Test Set:** 5 leaves (51 sliding sequences)
* **Zero Leaf Overlap:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. No sequence or image from a test leaf is ever seen during training or validation.
* **Strict Training-Only Preprocessing:**
  * Feature `StandardScaler` is fitted strictly on training sequence representations `X_train_2d`.
  * Lesion target normalization ($\mu_{\text{train}} = 168,567.97\text{ px}^2$, $\sigma_{\text{train}} = 384,944.75\text{ px}^2$) is computed exclusively on `y_lesion_train`.
  * All reported test metrics are calculated in original physical target units after inverse transformation.

---

## 4. Benchmark Performance & Comparative Results

All models were evaluated on the identical 51 held-out test sequences across 5 unseen wheat leaves:

| Model Architecture | Input Representation | Trainable Parameters | Disease Severity $R^2$ (↑) | Disease RMSE (↓) | Disease MAE (↓) | Lesion Area $R^2$ (↑) | Lesion RMSE (↓) | Lesion MAE (↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Multimodal QLSTM** | **ViT-B/16 + Metadata (791)** | **15,010** | **+0.8745** | **0.0292** | **0.0174** | **+0.6752** | **105,736** | **66,566** |
| Baseline Multimodal QLSTM | ViT-B/16 + Metadata (791) | 13,986 | +0.7263 | 0.0431 | 0.0303 | +0.5758 | 120,826 | 70,707 |
| QLSTM (Metadata-Only Ablation) | 23 Metadata Features | 2,722 | +0.6765 | 0.0468 | 0.0278 | +0.1728 | 168,737 | 134,564 |
| QLSTM (ViT-Only Ablation) | 768 ViT Features | 14,642 | +0.0054 | 0.0821 | 0.0593 | -0.0448 | 189,631 | 158,299 |
| Classical LSTM | ViT-B/16 + Metadata (791) | 105,666 | -0.9945 *(Collapsed)* | 0.1163 | 0.0948 | +0.1512 | 170,919 | 140,419 |
| Classical GRU | ViT-B/16 + Metadata (791) | 79,266 | -34.9794 *(Collapsed)* | 0.3032 | 0.2445 | -1.0106 | 205,640 | 176,159 |

### Key Scientific Findings:
1. **Quantum Inductive Bias Prevents Overfitting:** In high-dimensional multimodal regimes ($D=791$), classical LSTM (105k parameters) and GRU (79k parameters) severely overfit on longitudinal plant cohorts, collapsing on unseen leaves ($R^2 < 0$). In contrast, QLSTM achieves an **85.8% parameter reduction** (15k parameters), using unitary circuit constraints to achieve state-of-the-art test generalization (**$R^2 = 0.8745$**).
2. **Multimodal Synergy:** Fusing visual ViT embeddings with agronomic metadata yields substantial improvements over unimodal baselines:
   * Disease Severity $R^2$: $0.6765 \to \mathbf{0.8745}$ (**+0.1980 gain**)
   * Lesion Area $R^2$: $0.1728 \to \mathbf{0.6752}$ (**+0.5024 gain**)

---

## 5. Repository Structure

```text
├── checkpoints/             # Saved model weights (e.g. best_multimodal_qlstm.pth)
├── data/
│   └── sequences/          # Precomputed multimodal temporal sequences (.npz)
├── outputs/
│   ├── final_model_comparison.csv
│   ├── final_qlstm_audit.json
│   ├── multimodal_loss_history.csv
│   ├── multimodal_predictions.csv
│   ├── multimodal_test_metrics.json
│   └── plots/              # High-resolution benchmark and ablation plots
├── src/
│   ├── dataset/            # Leakage-free leaf-level GroupShuffleSplit loaders
│   ├── evaluation/         # Audit, metric calculation, and visualization tools
│   ├── quantum/            # PennyLane QLSTM cell, quantum layer, and model
│   └── train/              # Training scripts for QLSTM, classical LSTM, and ablations
└── README.md
```

---

## 6. Reproduction & Execution

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/sakshid455/Quantum-LSTM-for-plant-disease-prediction.git
cd Quantum-LSTM-for-plant-disease-prediction

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows (or source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

### Train Proposed Multimodal QLSTM
```bash
python -m src.train.train_multimodal_qlstm
```

### Generate Comparative Plots & Audit
```bash
python -m src.evaluation.generate_final_audit_and_plots
python -m src.evaluation.plot_multimodal_predictions
python -m src.evaluation.loss_plot
```

---

## 7. Citation & Acknowledgments

This research utilizes sequential leaf imagery and canopy data adapted from the **ETH Zurich Sequential Wheat Dataset**.
* **PennyLane:** Quantum Machine Learning Library developed by Xanadu.
* **PyTorch & HuggingFace:** Core deep-learning and Vision Transformer implementations.