# PROJECT DOCUMENTATION

## Quantum-Enhanced Multimodal Temporal Plant Disease Progression Prediction

---

# 1. Abstract

This project investigates temporal plant disease progression prediction using sequential wheat leaf observations and multimodal features.

Unlike conventional plant disease classification systems that operate on a single image, this project treats disease development as a temporal process.

The system combines visual/image-derived information with additional metadata and processes the resulting sequence using recurrent neural networks.

Three architectures are evaluated:

1. Classical LSTM
2. Classical GRU
3. Quantum Long Short-Term Memory (QLSTM)

The models perform two regression tasks:

1. Disease severity prediction
2. Lesion area prediction

To reduce data leakage, the dataset is split at the leaf level, ensuring that sequences from the same leaf cannot appear in multiple dataset partitions.

The current experiment uses 30 leaves:

- 21 training leaves
- 4 validation leaves
- 5 test leaves

The QLSTM currently achieves:

Disease Severity:

    R² = 0.696379

Lesion Area:

    R² = 0.618810

These results are better than the evaluated classical LSTM and GRU baselines under the current experimental setup.

---

# 2. Research Motivation

Plant disease progression is inherently temporal.

A single image may indicate the current disease state, but it does not necessarily indicate:

- How quickly the disease is developing
- Whether lesions are expanding
- Whether disease severity is increasing
- How environmental conditions influence progression

Sequential observations provide additional information.

For example:

    Day 1 -> Day 2 -> Day 3 -> Day 4 -> ...

can be treated as a temporal sequence.

A recurrent neural network can learn relationships between these observations.

The project investigates whether introducing a quantum-enhanced recurrent mechanism can improve this temporal learning process.

---

# 3. Research Question

The project's central research question is:

> Can a hybrid QLSTM improve temporal plant disease progression prediction over classical recurrent models when using sequential wheat leaf images and environmental metadata?

---

# 4. Hypothesis

The working hypothesis is:

> A quantum-enhanced recurrent architecture can learn useful temporal representations from multimodal plant disease sequences and potentially improve prediction performance compared with classical recurrent architectures.

This remains an experimental hypothesis and is not treated as proven quantum advantage.

---

# 5. Dataset Description

The project uses the ETH Zurich Sequential Wheat Dataset.

The dataset was designed for temporal plant phenotyping and disease progression analysis.

It contains:

    1,032 image sequences
    12,520 images

Average sequence length:

    approximately 12.1 images

Temporal duration:

    approximately 15 days

Image interval:

    approximately 24–30 hours

Disease categories include:

- Septoria tritici blotch
- Brown rust
- Yellow rust
- Mixed infections

The dataset contains temporal observations and lesion-related annotations.

---

# 6. Why Sequence-Level Modeling?

A conventional classification problem may be:

    Image -> Disease Class

This project instead models:

    Image_1
    Image_2
    Image_3
    ...
    Image_T

as:

    Sequence -> Future / Current Disease State

This allows temporal dependencies to be learned.

---

# 7. Multimodal Learning

The project does not rely exclusively on image features.

The multimodal representation combines:

    Visual features
          +
    Metadata/environmental features
          |
          v
    Fused temporal representation

The current recurrent input contains:

    791 features

The goal is to allow the model to learn both:

- visual disease characteristics
- environmental/metadata relationships

---

# 8. Feature Pipeline

The conceptual pipeline is:

    Raw Dataset
        |
        v
    Data Cleaning
        |
        v
    Feature Extraction
        |
        v
    Feature Fusion
        |
        v
    Temporal Ordering
        |
        v
    Sequence Construction
        |
        v
    Train/Validation/Test Split
        |
        v
    Normalization
        |
        v
    Recurrent Model

---

# 9. Target Variables

## 9.1 Disease Severity

Target variable:

    y_placl

The current training set statistics are:

    Mean = 0.049653333
    Std  = 0.12183375
    Min  = 0
    Max  = 0.9221841

---

## 9.2 Lesion Area

Target variable:

    y_lesion_area

The current training statistics are:

    Mean = 140909.33
    Std  = 361345.8

Because lesion area is much larger numerically than disease severity, it is normalized during training.

---

# 10. Leakage Prevention

This is one of the most important parts of the project.

A leaf can have multiple temporal observations.

Suppose:

    Leaf A:
       Day 1
       Day 2
       Day 3
       Day 4

If Day 1 and Day 2 are in training while Day 3 is in testing, the test set is no longer independent.

Therefore:

> Splitting is performed by leaf rather than by individual sequence.

---

# 11. Current Dataset Split

The latest run produced:

    Total leaves: 30

    Training: 21
    Validation: 4
    Testing: 5

Sequences:

    Training: 224
    Validation: 45
    Testing: 55

Therefore the final test results are calculated on:

    55 held-out sequences
    from 5 held-out leaves

---

# 12. Severity Stratification

The mean disease severity is calculated for every leaf.

Then leaves are approximately divided into three severity groups:

    Low
    Medium
    High

The train/validation/test split is stratified using these groups.

This helps maintain a reasonable distribution of disease severity across the partitions.

---

# 13. Feature Normalization

Input features are standardized.

For each feature:

    z = (x - μ_train) / σ_train

where:

    μ_train = training mean
    σ_train = training standard deviation

The scaler is fitted only using training samples.

Validation and test samples are transformed using the same training scaler.

---

# 14. Target Normalization

The lesion target is standardized using training statistics:

    z = (x - μ_lesion_train) / σ_lesion_train

The same training mean and standard deviation are used for:

- training
- validation
- testing

Final lesion metrics are reported in the original lesion-area scale.

---

# 15. Model 1 — Classical LSTM

The LSTM provides a classical recurrent baseline.

The LSTM processes:

    x_1, x_2, ..., x_T

and maintains:

    hidden state
    cell state

The final temporal representation is used for prediction.

The project uses the LSTM as the principal classical recurrent comparison.

---

# 16. Model 2 — Classical GRU

The GRU is another recurrent baseline.

It uses:

- update gate
- reset gate

Compared with LSTM, GRU has a simpler internal structure.

The current GRU configuration:

    Input size = 791
    Hidden size = 32
    Layers = 1
    Batch size = 8
    Epochs = 30
    Learning rate = 1e-4

---

# 17. Model 3 — QLSTM

The QLSTM is the proposed model.

The architecture combines classical neural-network layers with quantum circuits.

Conceptually:

    Input
      |
      v
    Classical Encoder
      |
      v
    Quantum Circuit
      |
      v
    Classical Decoder
      |
      v
    LSTM Gate

This is repeated for multiple LSTM gates.

---

# 18. QLSTM Gates

The current model contains four quantum-enhanced components:

    Forget gate
    Input gate
    Output gate
    Candidate gate

Each contains:

    Linear Encoder
         |
         v
    Quantum Layer
         |
         v
    Linear Decoder

The resulting values are used by the recurrent cell.

---

# 19. Multi-Task Architecture

The shared recurrent representation feeds two independent output heads.

    QLSTM
      |
      v
    Hidden Representation
       / \
      /   \
     v     v
 Disease  Lesion
  Head     Head
    |        |
    v        v
 Severity   Area

This enables shared representation learning.

---

# 20. Loss Function

The current training objective is:

    L_total = L_disease + 0.5 L_lesion

Both component losses use MSE.

Therefore:

    Disease Loss = MSE(y_disease, prediction_disease)

    Lesion Loss = MSE(y_lesion, prediction_lesion)

    Total Loss =
        Disease Loss + 0.5 × Lesion Loss

---

# 21. QLSTM Training

The QLSTM was trained for:

    30 epochs

Batch size:

    8

Device:

    CPU

The training loss decreased from:

    0.505541

to:

    0.332089

Validation loss decreased from:

    0.445413

to:

    0.283348

---

# 22. QLSTM Training History

Selected values:

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.505541 | 0.445413 |
| 5 | 0.459666 | 0.407209 |
| 10 | 0.427488 | 0.381665 |
| 15 | 0.401100 | 0.357648 |
| 20 | 0.376769 | 0.329591 |
| 25 | 0.353363 | 0.308383 |
| 30 | 0.332089 | 0.283348 |

The validation loss continued decreasing throughout the 30 epochs.

---

# 23. Task-Level Validation Loss

At epoch 30:

    Disease validation loss:
        0.008289

    Lesion validation loss:
        0.550118

This indicates that lesion-area prediction remains substantially harder than disease severity prediction under the current loss scaling.

---

# 24. QLSTM Test Results

## Disease Severity

    MSE:
        0.000776

    RMSE:
        0.027856

    MAE:
        0.020934

    R²:
        0.696379

---

## Lesion Area

    MSE:
        8,017,396,736

    RMSE:
        89,539.917

    MAE:
        58,388.473

    R²:
        0.618810

---

# 25. LSTM Test Results

## Disease Severity

    MSE:
        0.032316

    RMSE:
        0.179767

    MAE:
        0.148362

    R²:
        -11.645286

## Lesion Area

    MSE:
        8,227,408,896

    RMSE:
        90,705.066

    MAE:
        76,202.258

    R²:
        0.608825

---

# 26. GRU Test Results

## Disease Severity

    MSE:
        0.091949

    RMSE:
        0.303231

    MAE:
        0.244537

    R²:
        -34.979447

## Lesion Area

    MSE:
        42,287,951,872

    RMSE:
        205,640.346

    MAE:
        176,159.047

    R²:
        -1.010597

---

# 27. Overall Comparison

| Task | Metric | QLSTM | LSTM | GRU |
|---|---|---:|---:|---:|
| Disease | MSE | 0.000776 | 0.032316 | 0.091949 |
| Disease | RMSE | 0.027856 | 0.179767 | 0.303231 |
| Disease | MAE | 0.020934 | 0.148362 | 0.244537 |
| Disease | R² | 0.696379 | -11.645286 | -34.979447 |
| Lesion | MSE | 8.017B | 8.227B | 42.288B |
| Lesion | RMSE | 89,539.9 | 90,705.1 | 205,640.3 |
| Lesion | MAE | 58,388.5 | 76,202.3 | 176,159.0 |
| Lesion | R² | 0.618810 | 0.608825 | -1.010597 |

---

# 28. Interpretation of R²

The QLSTM disease R² of approximately:

    0.696

means the model explains a substantial portion of the variance in the held-out disease-severity observations under this test split.

The lesion R² of approximately:

    0.619

also indicates useful predictive performance.

The negative R² values of the classical models for disease severity indicate that those models performed worse than a mean-prediction baseline on the current held-out test set.

---

# 29. Important Scientific Caution

The results are promising but should not be described as definitive quantum advantage.

Reasons include:

- Only 30 leaves are currently used
- Only 5 leaves form the test set
- Only 55 test sequences are available
- Only one split has currently been evaluated
- The quantum circuit is simulated rather than executed on quantum hardware
- Classical baselines may require further tuning
- Multi-seed evaluation has not yet been completed

Therefore the correct scientific statement is:

> The QLSTM achieved the best performance among the evaluated models under the current experimental setup.

It is premature to state:

> QLSTM proves quantum advantage.

---

# 30. Existing Evaluation Files

The project currently contains evaluation scripts for:

    analyze_multimodal_predictions.py

    analyze_predictions.py

    analyze_split_distribution.py

    analyze_target_distribution.py

    check_leaf_split.py

    compare_models.py

    evaluate.py

    loss_plot.py

    metrics.py

    model_comparison.py

    plot_multimodal_predictions.py

    predict.py

---

# 31. Generated Outputs

Important generated files include:

    outputs/multimodal_test_metrics.json

    outputs/multimodal_lstm_test_metrics.json

    outputs/multimodal_gru_test_metrics.json

    outputs/multimodal_loss_history.csv

    outputs/multimodal_gru_loss_history.csv

    outputs/qlstm_multimodal_loss_curve.png

    outputs/qlstm_validation_task_loss.png

    outputs/qlstm_disease_actual_vs_predicted.png

    outputs/qlstm_disease_error_distribution.png

    outputs/qlstm_lesion_actual_vs_predicted.png

    outputs/qlstm_lesion_error_distribution.png

    outputs/model_comparison_disease_r2.png

    outputs/model_comparison_lesion_r2.png

---

# 32. Model Checkpoints

The project currently saves:

    checkpoints/best_multimodal_qlstm.pth

    checkpoints/best_multimodal_gru.pth

    checkpoints/best_lstm_model.pth

The best checkpoint is selected using validation loss.

---

# 33. Prediction Visualization

The prediction analysis generates four primary plots.

### Disease Actual vs Predicted

Used to determine whether predicted disease severity follows the ideal diagonal relationship:

    predicted = actual

### Disease Error Distribution

Shows:

    prediction - actual

A distribution centered near zero indicates lower systematic bias.

### Lesion Actual vs Predicted

Used to evaluate lesion-area prediction quality.

### Lesion Error Distribution

Shows the distribution of lesion-area prediction errors.

---

# 34. Experimental Reproducibility

The project uses a fixed random state for the current leaf-level split:

    random_state = 42

This ensures the current train/validation/test split can be reproduced.

However, a single seed is not enough for a strong scientific conclusion.

Multiple seeds should be evaluated later.

---

# 35. Recommended Robustness Experiment

Run the complete pipeline using:

    Seed 42
    Seed 123
    Seed 456
    Seed 789
    Seed 2026

For each model record:

    Disease R²
    Lesion R²
    Disease RMSE
    Lesion RMSE

Then report:

    Mean ± Standard Deviation

Example final table:

| Model | Disease R² | Lesion R² |
|---|---:|---:|
| LSTM | mean ± std | mean ± std |
| GRU | mean ± std | mean ± std |
| QLSTM | mean ± std | mean ± std |

This would provide a much stronger comparison.

---

# 36. Recommended Ablation Study

The next major experiment should determine what information is actually contributing to performance.

Experiments:

### A. Image Features Only

    Image features
        |
        v
      LSTM/QLSTM

### B. Metadata Only

    Metadata
       |
       v
    LSTM/QLSTM

### C. Image + Metadata

    Image + Metadata
          |
          v
       LSTM/QLSTM

This will demonstrate the contribution of multimodal fusion.

---

# 37. Recommended Hyperparameter Study

Potential parameters:

    Hidden size:
        16
        32
        64

    Learning rate:
        1e-4
        5e-4
        1e-3

    Batch size:
        4
        8
        16

    Quantum wires:
        different small configurations

The final configuration should be selected using validation performance, not test performance.

---

# 38. Future Explainability

Future analysis can include:

- Feature importance
- Temporal importance
- Error analysis
- Attention-like visualization if introduced
- Environmental-feature sensitivity
- Lesion progression visualization

The goal is to answer:

> Why did the model make this prediction?

rather than only:

> How accurate was the model?

---

# 39. Potential Deployment

After the research pipeline is finalized, the model could be integrated into a web application.

Possible workflow:

    User uploads sequential leaf images
              |
              v
    Feature extraction
              |
              v
    Metadata input
              |
              v
    QLSTM prediction
              |
       +------+------+
       |             |
       v             v
    Disease       Lesion
    Severity       Area
       |             |
       +------+------+
              |
              v
    Disease progression visualization

---

# 40. Expected Final System

The final system is intended to provide:

- Sequential disease analysis
- Disease severity prediction
- Lesion-area prediction
- Temporal progression modeling
- Multimodal feature integration
- Quantum-enhanced recurrent modeling
- Classical baseline comparison
- Quantitative evaluation
- Prediction visualization

---

# 41. Final Project Contribution

The project's primary contribution is an experimental framework that combines:

    Sequential plant observations
             +
    Multimodal features
             +
    Multi-task learning
             +
    Classical recurrent baselines
             +
    Quantum-enhanced recurrent learning

for temporal disease progression prediction.

The research contribution is not simply implementing QLSTM.

It is the systematic comparison of a hybrid quantum-classical recurrent architecture against classical recurrent architectures under a leakage-controlled temporal plant-disease prediction setup.

---

# 42. Current Completion Status

Completed:

    ✓ Dataset processing
    ✓ Temporal sequence preparation
    ✓ Leaf-level splitting
    ✓ Severity stratification
    ✓ Training-only feature scaling
    ✓ Training-only lesion target scaling
    ✓ Multimodal input pipeline
    ✓ Multi-task prediction
    ✓ Classical LSTM
    ✓ Classical GRU
    ✓ QLSTM
    ✓ Training
    ✓ Test evaluation
    ✓ Model comparison
    ✓ Loss plots
    ✓ Prediction plots

Remaining:

    ☐ Multiple random seeds
    ☐ Robust statistical comparison
    ☐ Ablation study
    ☐ Hyperparameter tuning
    ☐ Improved baseline tuning
    ☐ Final research tables
    ☐ Final discussion
    ☐ Final report
    ☐ Optional web deployment

---

# 43. Final Conclusion

The project currently has a working end-to-end experimental pipeline.

The most important result so far is that the QLSTM outperforms the tested classical LSTM and GRU models on the current held-out leaf test set.

Current results:

    Disease Severity:

        QLSTM R² = 0.696
        LSTM  R² = -11.645
        GRU  R² = -34.979

    Lesion Area:

        QLSTM R² = 0.619
        LSTM  R² = 0.609
        GRU  R² = -1.011

This is a strong preliminary result.

The next phase should focus on proving that the result is robust rather than simply adding more architecture.

The most important next experiment is therefore:

    MULTI-SEED EVALUATION

followed by:

    ABLATION STUDY

and:

    FAIR BASELINE HYPERPARAMETER TUNING

Only after these experiments should the final research conclusion be written.