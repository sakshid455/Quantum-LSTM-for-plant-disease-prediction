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

from src.quantum.qlstm_model import QLSTMModel
from src.dataset.multimodal_dataloader import create_multimodal_loaders


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

# Multi-task loss weight
DISEASE_LOSS_WEIGHT = 1.0
LESION_LOSS_WEIGHT = 0.5

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Early stopping
PATIENCE = 5


# ============================================================
# Output directories
# ============================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# Start
# ============================================================

print("=" * 60)
print("MULTIMODAL QLSTM TRAINING")
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

model = QLSTMModel(
    input_size=791,
    hidden_size=32
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# Loss function
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

train_disease_losses = []
train_lesion_losses = []

val_disease_losses = []
val_lesion_losses = []

best_val_loss = float("inf")

epochs_without_improvement = 0


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0
    running_train_disease_loss = 0.0
    running_train_lesion_loss = 0.0


    # --------------------------------------------------------
    # Training batches
    # --------------------------------------------------------

    for X, y_disease, y_lesion in train_loader:

        X = X.to(DEVICE)
        y_disease = y_disease.to(DEVICE)
        y_lesion = y_lesion.to(DEVICE)

        # Reset gradients
        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        disease_output, lesion_output = model(X)


        # ----------------------------------------------------
        # Individual task losses
        # ----------------------------------------------------

        disease_loss = criterion(
            disease_output,
            y_disease
        )

        lesion_loss = criterion(
            lesion_output,
            y_lesion
        )


        # ----------------------------------------------------
        # Weighted multi-task loss
        # ----------------------------------------------------

        loss = (
            DISEASE_LOSS_WEIGHT * disease_loss
            +
            LESION_LOSS_WEIGHT * lesion_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM
        )


        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        running_train_loss += loss.item()
        running_train_disease_loss += disease_loss.item()
        running_train_lesion_loss += lesion_loss.item()


    # --------------------------------------------------------
    # Average training losses
    # --------------------------------------------------------

    train_loss = (
        running_train_loss /
        len(train_loader)
    )

    train_disease_loss = (
        running_train_disease_loss /
        len(train_loader)
    )

    train_lesion_loss = (
        running_train_lesion_loss /
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


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            disease_output, lesion_output = model(X)


            # ------------------------------------------------
            # Individual losses
            # ------------------------------------------------

            disease_loss = criterion(
                disease_output,
                y_disease
            )

            lesion_loss = criterion(
                lesion_output,
                y_lesion
            )


            # ------------------------------------------------
            # Combined loss
            # ------------------------------------------------

            loss = (
                DISEASE_LOSS_WEIGHT * disease_loss
                +
                LESION_LOSS_WEIGHT * lesion_loss
            )


            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            running_val_loss += loss.item()
            running_val_disease_loss += disease_loss.item()
            running_val_lesion_loss += lesion_loss.item()


    # --------------------------------------------------------
    # Average validation losses
    # --------------------------------------------------------

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

    train_disease_losses.append(
        train_disease_loss
    )

    train_lesion_losses.append(
        train_lesion_loss
    )

    val_disease_losses.append(
        val_disease_loss
    )

    val_lesion_losses.append(
        val_lesion_loss
    )


    # ========================================================
    # Print epoch results
    # ========================================================

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

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "checkpoints/best_multimodal_qlstm.pth"
        )

        print("  -> Best model saved")

    else:

        epochs_without_improvement += 1

        print(
            f"  -> No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # ========================================================
    # Early stopping
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# Save training history
# ============================================================

num_epochs_completed = len(train_losses)

history = pd.DataFrame({
    "Epoch": range(
        1,
        num_epochs_completed + 1
    ),

    "Train Loss": train_losses,

    "Validation Loss": val_losses,

    "Train Disease Loss": train_disease_losses,

    "Train Lesion Loss": train_lesion_losses,

    "Validation Disease Loss": val_disease_losses,

    "Validation Lesion Loss": val_lesion_losses
})


history.to_csv(
    "outputs/multimodal_loss_history.csv",
    index=False
)


# ============================================================
# Training finished
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    f"Epochs completed: "
    f"{num_epochs_completed}"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Saved model -> "
    "checkpoints/best_multimodal_qlstm.pth"
)

print(
    "Saved history -> "
    "outputs/multimodal_loss_history.csv"
)


# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        "checkpoints/best_multimodal_qlstm.pth",
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


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        disease_output, lesion_output = model(X)


        # ----------------------------------------------------
        # Disease
        # ----------------------------------------------------

        disease_predictions.extend(
            disease_output
            .cpu()
            .numpy()
            .flatten()
        )

        disease_targets.extend(
            y_disease
            .numpy()
            .flatten()
        )


        # ----------------------------------------------------
        # Lesion
        # ----------------------------------------------------

        lesion_predictions.extend(
            lesion_output
            .cpu()
            .numpy()
            .flatten()
        )

        lesion_targets.extend(
            y_lesion
            .numpy()
            .flatten()
        )


# ============================================================
# Convert to NumPy arrays
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

disease_rmse = (
    disease_mse ** 0.5
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

lesion_rmse = (
    lesion_mse ** 0.5
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
# Results dictionary
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
    "outputs/multimodal_test_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
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
    "outputs/multimodal_test_metrics.json"
)