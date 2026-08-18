import os
import json

import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.quantum.lstm_model import LSTMModel
from src.dataset.multimodal_dataloader import (
    create_multimodal_loaders
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


print("=" * 60)
print("MULTIMODAL CLASSICAL LSTM TRAINING")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# Dataset
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
# Model
# ============================================================

model = LSTMModel(
    input_size=791,
    hidden_size=32
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# Loss
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training history
# ============================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_train_loss = 0.0

    for X, y_disease, y_lesion in train_loader:

        X = X.to(DEVICE)
        y_disease = y_disease.to(DEVICE)
        y_lesion = y_lesion.to(DEVICE)

        optimizer.zero_grad()

        disease_output, lesion_output = model(X)

        disease_loss = criterion(
            disease_output,
            y_disease
        )

        lesion_loss = criterion(
            lesion_output,
            y_lesion
        )

        # Same weighting as QLSTM
        loss = disease_loss + 0.5 * lesion_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_train_loss += loss.item()

    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    running_val_loss = 0.0
    running_val_disease_loss = 0.0
    running_val_lesion_loss = 0.0

    with torch.no_grad():

        for X, y_disease, y_lesion in val_loader:

            X = X.to(DEVICE)
            y_disease = y_disease.to(DEVICE)
            y_lesion = y_lesion.to(DEVICE)

            disease_output, lesion_output = model(X)

            disease_loss = criterion(
                disease_output,
                y_disease
            )

            lesion_loss = criterion(
                lesion_output,
                y_lesion
            )

            loss = disease_loss + 0.5 * lesion_loss

            running_val_loss += loss.item()
            running_val_disease_loss += disease_loss.item()
            running_val_lesion_loss += lesion_loss.item()

    val_loss = (
        running_val_loss /
        len(val_loader)
    )

    val_disease_loss = (
        running_val_disease_loss /
        len(val_loader)
    )

    val_lesion_loss = (
        running_val_lesion_loss /
        len(val_loader)
    )


    # ========================================================
    # Store history
    # ========================================================

    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" | Train Loss: {train_loss:.6f}"
        f" | Val Loss: {val_loss:.6f}"
        f" | Disease Val: {val_disease_loss:.6f}"
        f" | Lesion Val: {val_lesion_loss:.6f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_multimodal_lstm.pth"
        )

        print("  -> Best model saved")


# ============================================================
# Save history
# ============================================================

history = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses
})

history.to_csv(
    "outputs/multimodal_lstm_loss_history.csv",
    index=False
)


print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Saved model -> "
    "checkpoints/best_multimodal_lstm.pth"
)


# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        "checkpoints/best_multimodal_lstm.pth",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# Test evaluation
# ============================================================

disease_predictions = []
disease_targets = []

lesion_predictions = []
lesion_targets = []


with torch.no_grad():

    for X, y_disease, y_lesion in test_loader:

        X = X.to(DEVICE)

        disease_output, lesion_output = model(X)

        disease_predictions.extend(
            disease_output.cpu().numpy().flatten()
        )

        disease_targets.extend(
            y_disease.numpy().flatten()
        )

        lesion_predictions.extend(
            lesion_output.cpu().numpy().flatten()
        )

        lesion_targets.extend(
            y_lesion.numpy().flatten()
        )


# ============================================================
# Convert to arrays
# ============================================================

disease_predictions = pd.Series(
    disease_predictions
).to_numpy()

disease_targets = pd.Series(
    disease_targets
).to_numpy()

lesion_predictions = pd.Series(
    lesion_predictions
).to_numpy()

lesion_targets = pd.Series(
    lesion_targets
).to_numpy()


# ============================================================
# Reverse lesion normalization
# ============================================================

lesion_predictions_original = (
    lesion_predictions * lesion_std
    + lesion_mean
)

lesion_targets_original = (
    lesion_targets * lesion_std
    + lesion_mean
)


# ============================================================
# Disease metrics
# ============================================================

disease_mse = mean_squared_error(
    disease_targets,
    disease_predictions
)

disease_rmse = disease_mse ** 0.5

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

lesion_rmse = lesion_mse ** 0.5

lesion_mae = mean_absolute_error(
    lesion_targets_original,
    lesion_predictions_original
)

lesion_r2 = r2_score(
    lesion_targets_original,
    lesion_predictions_original
)


# ============================================================
# Results
# ============================================================

metrics = {

    "Disease Severity": {
        "MSE": float(disease_mse),
        "RMSE": float(disease_rmse),
        "MAE": float(disease_mae),
        "R2": float(disease_r2)
    },

    "Lesion Area": {
        "MSE": float(lesion_mse),
        "RMSE": float(lesion_rmse),
        "MAE": float(lesion_mae),
        "R2": float(lesion_r2)
    }

}


# ============================================================
# Save metrics
# ============================================================

with open(
    "outputs/multimodal_lstm_test_metrics.json",
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
print("FINAL LSTM TEST RESULTS")
print("=" * 60)

print("\nDisease Severity:")

print(
    f"MSE  : {disease_mse:.6f}"
)

print(
    f"RMSE : {disease_rmse:.6f}"
)

print(
    f"MAE  : {disease_mae:.6f}"
)

print(
    f"R²   : {disease_r2:.6f}"
)


print("\nLesion Area:")

print(
    f"MSE  : {lesion_mse:.6f}"
)

print(
    f"RMSE : {lesion_rmse:.6f}"
)

print(
    f"MAE  : {lesion_mae:.6f}"
)

print(
    f"R²   : {lesion_r2:.6f}"
)


print(
    "\nSaved -> "
    "outputs/multimodal_lstm_test_metrics.json"
)
