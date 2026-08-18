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
    "checkpoints/best_gru_model.pth"
)

OUTPUT_PATH = (
    "outputs/gru_metrics.json"
)


# ============================================================
# Directories
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


print("=" * 60)
print("GRU TEST EVALUATION")
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
# Test predictions
# ============================================================

predictions = []
targets = []


with torch.no_grad():

    for X, y_disease, y_lesion in test_loader:

        X = X.to(DEVICE)

        output = model(X)

        output = (
            output
            .view(-1)
            .cpu()
            .numpy()
        )

        target = (
            y_disease
            .view(-1)
            .cpu()
            .numpy()
        )

        predictions.extend(
            output
        )

        targets.extend(
            target
        )


# ============================================================
# Convert to NumPy
# ============================================================

predictions = np.asarray(
    predictions,
    dtype=np.float32
)

targets = np.asarray(
    targets,
    dtype=np.float32
)


# ============================================================
# Metrics
# ============================================================

mse = mean_squared_error(
    targets,
    predictions
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    targets,
    predictions
)

r2 = r2_score(
    targets,
    predictions
)


# ============================================================
# Correlation
# ============================================================

if (
    np.std(targets) > 0
    and np.std(predictions) > 0
):

    correlation = np.corrcoef(
        targets,
        predictions
    )[0, 1]

else:

    correlation = float("nan")


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame({
    "Actual": targets,
    "Predicted": predictions
})

prediction_df["Error"] = (
    prediction_df["Predicted"]
    - prediction_df["Actual"]
)

prediction_df["Absolute_Error"] = (
    prediction_df["Error"]
    .abs()
)

prediction_df.to_csv(
    "outputs/gru_predictions.csv",
    index=False
)


# ============================================================
# Save metrics
# ============================================================

metrics = {

    "MSE": float(mse),

    "RMSE": float(rmse),

    "MAE": float(mae),

    "R2": float(r2),

    "Correlation": float(correlation),

    "Samples": int(len(targets))

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

print(
    f"Samples       : {len(targets)}"
)

print(
    f"MSE           : {mse:.6f}"
)

print(
    f"RMSE          : {rmse:.6f}"
)

print(
    f"MAE           : {mae:.6f}"
)

print(
    f"R²            : {r2:.6f}"
)

print(
    f"Correlation   : {correlation:.6f}"
)


print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print(
    f"Actual mean   : {targets.mean():.6f}"
)

print(
    f"Predicted mean: {predictions.mean():.6f}"
)

print(
    f"Actual max    : {targets.max():.6f}"
)

print(
    f"Predicted max : {predictions.max():.6f}"
)


print("\n" + "=" * 60)
print("WORST 10 PREDICTIONS")
print("=" * 60)

worst = prediction_df.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    worst[
        [
            "Actual",
            "Predicted",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved ->", OUTPUT_PATH)

print(
    "Saved -> outputs/gru_predictions.csv"
)