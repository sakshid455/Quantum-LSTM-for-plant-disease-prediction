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


# ============================================================
# Header
# ============================================================

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
# Training loop
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for X, y_disease, y_lesion in train_loader:

        # Move inputs and disease target to device
        X = X.to(DEVICE)

        y_disease = y_disease.to(DEVICE)

        # ----------------------------------------------------
        # IMPORTANT:
        # Make target explicitly 1-D
        # ----------------------------------------------------

        y_disease = y_disease.view(-1)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # GRU predicts disease severity only
        # ----------------------------------------------------

        output = model(X)

        # Explicitly flatten model output
        output = output.view(-1)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if output.shape != y_disease.shape:
            raise RuntimeError(
                f"Shape mismatch: "
                f"output={output.shape}, "
                f"target={y_disease.shape}"
            )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            y_disease
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_train_loss += loss.item()


    # --------------------------------------------------------
    # Average training loss
    # --------------------------------------------------------

    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for X, y_disease, y_lesion in val_loader:

            X = X.to(DEVICE)

            y_disease = y_disease.to(DEVICE)

            # Explicitly make target 1-D
            y_disease = y_disease.view(-1)

            # Prediction
            output = model(X)

            # Explicitly flatten prediction
            output = output.view(-1)

            # Safety check
            if output.shape != y_disease.shape:
                raise RuntimeError(
                    f"Validation shape mismatch: "
                    f"output={output.shape}, "
                    f"target={y_disease.shape}"
                )

            # Validation loss
            loss = criterion(
                output,
                y_disease
            )

            running_val_loss += loss.item()


    # --------------------------------------------------------
    # Average validation loss
    # --------------------------------------------------------

    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    # ========================================================
    # Store history
    # ========================================================

    train_losses.append(train_loss)
    val_losses.append(val_loss)


    # ========================================================
    # Print progress
    # ========================================================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_gru_model.pth"
        )

        print("  -> Best GRU model saved")


# ============================================================
# Save loss history
# ============================================================

loss_df = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses
})


loss_df.to_csv(
    "outputs/gru_loss_history.csv",
    index=False
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 60)
print("GRU TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Saved -> "
    "checkpoints/best_gru_model.pth"
)

print(
    "Saved -> "
    "outputs/gru_loss_history.csv"
)