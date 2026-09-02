import os
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.baselines.gru_model import GRUModel
from src.dataset.multimodal_dataloader import (
    create_multimodal_loaders
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8

INPUT_SIZE = 791
HIDDEN_SIZE = 32
NUM_LAYERS = 1

MODEL_PATH = (
    "checkpoints/best_multimodal_gru.pth"
)

OUTPUT_PATH = (
    "outputs/multimodal_gru_test_metrics.json"
)

PREDICTIONS_PATH = (
    "outputs/multimodal_gru_predictions.csv"
)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


print("=" * 60)
print("MULTIMODAL GRU TEST EVALUATION")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# Load multimodal data
# ============================================================

(
    train_loader,
    val_loader,
    test_loader,
    lesion_mean,
    lesion_std
) = create_multimodal_loaders(
    batch_size=BATCH_SIZE
)


# ============================================================
# Create model
# ============================================================

model = GRUModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# Load best checkpoint
# ============================================================

print("\nLoading best GRU model...")

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# Prediction containers
# ============================================================

disease_predictions = []
disease_targets = []

lesion_predictions = []
lesion_targets = []


# ============================================================
# Test predictions
# ============================================================

with torch.no_grad():

    for X, y_disease, y_lesion in test_loader:

        X = X.to(DEVICE)

        disease_output, lesion_output = model(X)

        disease_predictions.extend(
            disease_output
            .view(-1)
            .cpu()
            .numpy()
        )

        disease_targets.extend(
            y_disease
            .view(-1)
            .cpu()
            .numpy()
        )

        lesion_predictions.extend(
            lesion_output
            .view(-1)
            .cpu()
            .numpy()
        )

        lesion_targets.extend(
            y_lesion
            .view(-1)
            .cpu()
            .numpy()
        )


# ============================================================
# Convert to NumPy
# ============================================================

disease_predictions = np.asarray(
    disease_predictions,
    dtype=np.float32
)

disease_targets = np.asarray(
    disease_targets,
    dtype=np.float32
)

lesion_predictions = np.asarray(
    lesion_predictions,
    dtype=np.float32
)

lesion_targets = np.asarray(
    lesion_targets,
    dtype=np.float32
)


# ============================================================
# Reverse lesion normalization
# ============================================================

lesion_predictions_original = (
    lesion_predictions
    * lesion_std
    + lesion_mean
)

lesion_targets_original = (
    lesion_targets
    * lesion_std
    + lesion_mean
)


# ============================================================
# Disease metrics
# ============================================================

disease_mse = mean_squared_error(
    disease_targets,
    disease_predictions
)

disease_rmse = np.sqrt(
    disease_mse
)

disease_mae = mean_absolute_error(
    disease_targets,
    disease_predictions
)

disease_r2 = r2_score(
    disease_targets,
    disease_predictions
)


# ============================================================
# Lesion metrics
# ============================================================

lesion_mse = mean_squared_error(
    lesion_targets_original,
    lesion_predictions_original
)

lesion_rmse = np.sqrt(
    lesion_mse
)

lesion_mae = mean_absolute_error(
    lesion_targets_original,
    lesion_predictions_original
)

lesion_r2 = r2_score(
    lesion_targets_original,
    lesion_predictions_original
)


# ============================================================
# Correlations
# ============================================================

if (
    np.std(disease_targets) > 0
    and np.std(disease_predictions) > 0
):

    disease_correlation = np.corrcoef(
        disease_targets,
        disease_predictions
    )[0, 1]

else:

    disease_correlation = float("nan")


if (
    np.std(lesion_targets_original) > 0
    and np.std(lesion_predictions_original) > 0
):

    lesion_correlation = np.corrcoef(
        lesion_targets_original,
        lesion_predictions_original
    )[0, 1]

else:

    lesion_correlation = float("nan")


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame({

    "Disease_Actual": disease_targets,

    "Disease_Predicted": disease_predictions,

    "Disease_Error":
        disease_predictions
        - disease_targets,

    "Disease_Absolute_Error":
        np.abs(
            disease_predictions
            - disease_targets
        ),

    "Lesion_Actual":
        lesion_targets_original,

    "Lesion_Predicted":
        lesion_predictions_original,

    "Lesion_Error":
        lesion_predictions_original
        - lesion_targets_original,

    "Lesion_Absolute_Error":
        np.abs(
            lesion_predictions_original
            - lesion_targets_original
        )
})


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# Save metrics
# ============================================================

metrics = {

    "Disease Severity": {

        "MSE": float(disease_mse),

        "RMSE": float(disease_rmse),

        "MAE": float(disease_mae),

        "R2": float(disease_r2),

        "Correlation":
            float(disease_correlation)

    },

    "Lesion Area": {

        "MSE": float(lesion_mse),

        "RMSE": float(lesion_rmse),

        "MAE": float(lesion_mae),

        "R2": float(lesion_r2),

        "Correlation":
            float(lesion_correlation)

    },

    "Samples": int(len(disease_targets))

}


with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)
print("GRU TEST RESULTS")
print("=" * 60)


print("\nDisease Severity:")

print(
    f"MSE         : {disease_mse:.6f}"
)

print(
    f"RMSE        : {disease_rmse:.6f}"
)

print(
    f"MAE         : {disease_mae:.6f}"
)

print(
    f"R²          : {disease_r2:.6f}"
)

print(
    f"Correlation : {disease_correlation:.6f}"
)


print("\nLesion Area:")

print(
    f"MSE         : {lesion_mse:.6f}"
)

print(
    f"RMSE        : {lesion_rmse:.6f}"
)

print(
    f"MAE         : {lesion_mae:.6f}"
)

print(
    f"R²          : {lesion_r2:.6f}"
)

print(
    f"Correlation : {lesion_correlation:.6f}"
)


# ============================================================
# Prediction summaries
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print("\nDisease:")

print(
    f"Actual mean    : "
    f"{disease_targets.mean():.6f}"
)

print(
    f"Predicted mean : "
    f"{disease_predictions.mean():.6f}"
)

print(
    f"Actual max     : "
    f"{disease_targets.max():.6f}"
)

print(
    f"Predicted max  : "
    f"{disease_predictions.max():.6f}"
)


print("\nLesion Area:")

print(
    f"Actual mean    : "
    f"{lesion_targets_original.mean():.2f}"
)

print(
    f"Predicted mean : "
    f"{lesion_predictions_original.mean():.2f}"
)

print(
    f"Actual max     : "
    f"{lesion_targets_original.max():.2f}"
)

print(
    f"Predicted max  : "
    f"{lesion_predictions_original.max():.2f}"
)


# ============================================================
# Save confirmation
# ============================================================

print("\n" + "=" * 60)

print(
    "Saved -> "
    "outputs/multimodal_gru_test_metrics.json"
)

print(
    "Saved -> "
    "outputs/multimodal_gru_predictions.csv"
)

print("=" * 60)