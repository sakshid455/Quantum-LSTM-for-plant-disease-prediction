import os
import pandas as pd
import torch
import torch.nn as nn

from src.baselines.gru_model import GRUModel
from src.dataset.multimodal_dataloader import create_multimodal_loaders


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4

INPUT_SIZE = 791
HIDDEN_SIZE = 32
NUM_LAYERS = 1


# ============================================================
# Directories
# ============================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


print("=" * 60)
print("MULTIMODAL CLASSICAL GRU TRAINING")
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
# Model
# ============================================================

model = GRUModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
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

        # Same multitask weighting as LSTM
        loss = (
            disease_loss
            + 0.5 * lesion_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_train_loss += loss.item()


    train_loss = (
        running_train_loss
        / len(train_loader)
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

            loss = (
                disease_loss
                + 0.5 * lesion_loss
            )

            running_val_loss += loss.item()
            running_val_disease_loss += disease_loss.item()
            running_val_lesion_loss += lesion_loss.item()


    val_loss = (
        running_val_loss
        / len(val_loader)
    )

    val_disease_loss = (
        running_val_disease_loss
        / len(val_loader)
    )

    val_lesion_loss = (
        running_val_lesion_loss
        / len(val_loader)
    )


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
            "checkpoints/best_multimodal_gru.pth"
        )

        print("  -> Best GRU model saved")


# ============================================================
# Save loss history
# ============================================================

history = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses
})

history.to_csv(
    "outputs/multimodal_gru_loss_history.csv",
    index=False
)


print("\n" + "=" * 60)
print("GRU TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Saved -> "
    "checkpoints/best_multimodal_gru.pth"
)

print(
    "Saved -> "
    "outputs/multimodal_gru_loss_history.csv"
)