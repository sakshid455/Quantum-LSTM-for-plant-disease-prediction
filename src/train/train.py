import os
import torch
import torch.nn as nn

from src.quantum.qlstm_model import QLSTMModel
from src.dataset.dataloader import get_dataloaders

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 8

os.makedirs("checkpoints", exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = QLSTMModel().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_loss = float("inf")
# Store loss history
train_losses = []
val_losses = []
# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            output = model(X)

            loss = criterion(output, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Save losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Train Loss: {train_loss:.6f}"
        f" | Val Loss: {val_loss:.6f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )
import pandas as pd

os.makedirs("outputs", exist_ok=True)

history = pd.DataFrame({
    "Epoch": list(range(1, EPOCHS + 1)),
    "Train Loss": train_losses,
    "Validation Loss": val_losses
})

history.to_csv(
    "outputs/loss_history.csv",
    index=False
)

print("\nTraining Finished!")
print(f"Best Validation Loss: {best_loss:.6f}")
print("Saved -> outputs/loss_history.csv")