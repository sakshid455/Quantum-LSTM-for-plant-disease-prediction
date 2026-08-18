import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.baselines.lstm_model import LSTMModel


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load data
# ============================================================

X_train = np.load("data/sequences/X_train.npy")
y_train = np.load("data/sequences/y_train.npy")

X_val = np.load("data/sequences/X_val.npy")
y_val = np.load("data/sequences/y_val.npy")


# ============================================================
# Convert to tensors
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Model
# ============================================================

model = LSTMModel(
    input_size=768,
    hidden_size=32,
    num_layers=1
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

loss_history = []


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(X)

        output = output.view(-1)
        y = y.view(-1)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_loader.dataset)


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            output = model(X)

            output = output.view(-1)
            y = y.view(-1)

            loss = criterion(output, y)

            val_loss += loss.item() * X.size(0)

    val_loss /= len(val_loader.dataset)


    loss_history.append(
        [epoch + 1, train_loss, val_loss]
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_lstm_model.pth"
        )


# ============================================================
# Save loss history
# ============================================================

np.savetxt(
    "outputs/lstm_loss_history.csv",
    np.array(loss_history),
    delimiter=",",
    header="Epoch,Train Loss,Validation Loss",
    comments=""
)


print("\n" + "=" * 60)
print("LSTM Training Finished!")
print("=" * 60)

print(f"Best Validation Loss: {best_val_loss:.6f}")

print(
    "Saved -> checkpoints/best_lstm_model.pth"
)

print(
    "Saved -> outputs/lstm_loss_history.csv"
)