# Clean Temporal Forecasting QLSTM Benchmark Report

> **CRITICAL SCIENTIFIC WARNING: LEAKY REFERENCE EXPERIMENT**  
> The previous 100-leaf metrics ($R^2 = 0.9339$ for disease severity, $R^2 = 0.8674$ for lesion area) were derived from an experimental setup containing **severe target leakage**:
> 1. **Contemporaneous Prediction**: The input window $[t-3, t-2, t-1, t]$ predicted the target at timestep $t$.
> 2. **Target-Derived Input Proxies**: The 23 metadata features contained contemporaneous disease and lesion segmentation masks (`total_lesion_area`, `la_damaged_f`, `pycn_density`, etc.) directly correlated with the target.
> 
> The results below represent the **true, scientifically valid temporal forecasting benchmark** predicting future disease progression at timestep $t+1$.

---

## 1. Experimental Configuration & Preprocessing Protocol

* **Research Objective**: Predict wheat foliar disease progression at future timestep $t+1$ from historical sequence observations $[t-3, t-2, t-1, t]$ using Hybrid Vision Transformer–QLSTM.
* **Temporal Sequence Window**: 4 historical steps ($T=4$) $\rightarrow$ strictly future target observation at $t+1$.
* **Leaf-Level Group Split**: `GroupShuffleSplit` (Seed 42) partitioned strictly by leaf identity:
  * **Train Cohort**: 69 unique leaves ($757$ sequences)
  * **Validation Cohort**: 15 unique leaves ($168$ sequences)
  * **Held-out Test Cohort**: 15 unique leaves ($166$ sequences)
  * **Leaf Overlap**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (Zero data contamination).
* **Leakage-Safe Scaling Protocol**:
  * Feature Normalization: `StandardScaler` fitted strictly on training leaves (`X[train]`).
  * Lesion Target Normalization: Z-score normalization ($\mu = 270,945.72 \text{ px}^2$, $\sigma = 448,571.62 \text{ px}^2$) fitted strictly on training targets (`y_lesion[train]`).
  * Validation & Test sets transformed exclusively via `transform()`.
  * All test evaluation metrics recomputed in original target units ($[0, 1]$ for disease, $\text{px}^2$ for lesion area).
* **Model & Hyperparameters**:
  * Architecture: Hybrid ViT-QLSTM with 4-qubit parameterized variational quantum circuit and dual MLP projection heads.
  * Epochs: 50 | Batch Size: 8 | Learning Rate: $1\times 10^{-4}$ with Cosine Annealing ($\eta_{\min} = 1\times 10^{-5}$).
  * Loss Formulation: Balanced Multi-Task MSE ($\mathcal{L} = 10.0 \cdot \mathcal{L}_{\text{disease}} + 0.5 \cdot \mathcal{L}_{\text{lesion}}$).
  * Early Stopping Patience: 10 epochs.

---

## 2. Dataset Formulations

### Dataset A — Clean Image-Only
* **Input Dimension**: $768$ features ($4 \text{ timesteps} \times 768 \text{ ViT-B/16 visual patch features}$).
* **Exogenous Metadata**: None ($0$ features).
* **Target**: Disease Severity (`placl`) and Lesion Area ($\text{px}^2$) at $t+1$.
* **File Path**: `data/sequences/clean_image_temporal_sequences_100leaves.npz`

### Dataset B — Clean Safe-Multimodal
* **Input Dimension**: $771$ features ($4 \text{ timesteps} \times [768 \text{ ViT} + 1 \text{ } la\_tot + 1 \text{ fungicide} + 1 \text{ inoculation}]$).
* **Legitimate Exogenous Metadata**:
  * `la_tot`: Total physical leaf blade pixel area from blade segmentation mask.
  * `fungicide_treatment`: Plot-level agronomic treatment (0 = No Fungicide, 1 = Fungicide).
  * `inoculation_treatment`: Plot-level artificial pathogen inoculation (0 = Natural/No Inoculation, 1 = Artificial Inoculation).
* **File Path**: `data/sequences/clean_multimodal_temporal_sequences_100leaves.npz`

---

## 3. Training & Validation Dynamics

| Metric | Clean Image-Only QLSTM ($D=768$) | Clean Safe-Multimodal QLSTM ($D=771$) |
| :--- | :---: | :---: |
| **Trainable Parameters** | 14,642 | 14,690 |
| **Best Validation Epoch** | Epoch 50 | Epoch 50 |
| **Best Validation Loss** | 0.176650 | 0.137131 |
| **Final Train Loss (Epoch 50)** | 0.092985 | 0.065613 |
| **Train Disease MSE** | 0.002438 | 0.001132 |
| **Train Lesion MSE** | 0.1372 | 0.1086 |
| **Validation Disease MSE** | 0.004316 | 0.003380 |
| **Validation Lesion MSE** | 0.2670 | 0.2067 |
| **Convergence Stability** | Highly stable monotonic descent | Highly stable monotonic descent |

---

## 4. Benchmark Performance Comparison Table

| Experiment | Input Dimension | Disease $R^2$ | Disease RMSE | Disease MAE | Lesion $R^2$ | Lesion RMSE ($\text{px}^2$) | Lesion MAE ($\text{px}^2$) | Status / Scientific Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Leaky Reference** | 791 | 0.9339 | 0.0349 | 0.0226 | 0.8674 | 165,905 | 78,799 | **LEAKY REFERENCE — NOT SCIENTIFICALLY VALID** |
| **Clean Image-Only** | 768 | **0.8690** | **0.0507** | 0.0333 | **0.7654** | **227,987** | **112,114** | **SCIENTIFICALLY VALID FORECASTING** |
| **Clean Safe-Multimodal** | 771 | 0.8343 | 0.0570 | **0.0316** | 0.7455 | 237,466 | 117,348 | **SCIENTIFICALLY VALID FORECASTING** |

---

## 5. Analytical Observations

1. **Clean Performance vs. Leaky Reference**:
   * On the clean future-forecasting task ($T \rightarrow t+1$), the model achieves **$R^2 = 0.8690$** for disease severity and **$R^2 = 0.7654$** for lesion area using image features alone.
   * While lower than the inflated leaky reference ($0.9339 / 0.8674$), these results demonstrate that the Hybrid ViT-QLSTM model possesses substantial genuine prognostic capacity to forecast future wheat disease progression 1 step ahead from historical images.
2. **Image-Only vs. Safe-Multimodal**:
   * Safe-Multimodal achieved slightly lower training and validation multi-task loss ($0.1371$ vs $0.1766$), but on the 15 held-out test leaves, Image-Only demonstrated superior generalization ($R^2 = 0.8690$ vs $0.8343$ for disease, $0.7654$ vs $0.7455$ for lesion area).
   * The 3 static/quasi-static exogenous metadata variables (`la_tot`, `fungicide`, `inoculation`) provided slight in-sample optimization advantage but slight out-of-distribution variance across unseen plots. Image representations alone carry richer, leaf-specific spatial-temporal dynamics.
3. **Disease Severity vs. Lesion Area Forecasting**:
   * Disease Severity ($R^2 \approx 0.83 - 0.87$) is consistently easier to forecast than absolute Lesion Area ($R^2 \approx 0.75 - 0.77$). Lesion area in raw pixels exhibits high variance due to leaf size variations, whereas disease severity (proportion of leaf area) is bounded in $[0, 1]$.
4. **Stability & Overfitting Analysis**:
   * Training and validation loss curves decreased monotonically across all 50 epochs under cosine learning rate decay.
   * Minimal overfitting occurred: validation losses tracked training losses closely without runaway divergence.
