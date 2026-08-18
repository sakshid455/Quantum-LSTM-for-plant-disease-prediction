import os
import json
import torch
import torch.nn as nn

from src.models.baseline_models import (
    RNNModel,
    LSTMModel,
    GRUModel
)

from src.dataset.dataloader import get_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 8

os.makedirs("checkpoints/baselines", exist_ok=True)


def train_model(model, name, train_loader, val_loader):

    model = model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_loss = float("inf")

    print("\n" + "=" * 50)
    print(f"Training {name}")
    print("=" * 50)

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

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

        # Validation
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE).unsqueeze(1)

                output = model(X)

                loss = criterion(output, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
            f" | Train Loss: {train_loss:.6f}"
            f" | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                f"checkpoints/baselines/{name.lower()}_best.pth"
            )

    print(f"{name} Best Validation Loss: {best_loss:.6f}")

    return model


# -----------------------------------------
# Dataset
# -----------------------------------------

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)


# -----------------------------------------
# Train RNN
# -----------------------------------------

train_model(
    RNNModel(),
    "RNN",
    train_loader,
    val_loader
)


# -----------------------------------------
# Train LSTM
# -----------------------------------------

train_model(
    LSTMModel(),
    "LSTM",
    train_loader,
    val_loader
)


# -----------------------------------------
# Train GRU
# -----------------------------------------

train_model(
    GRUModel(),
    "GRU",
    train_loader,
    val_loader
)


print("\n" + "=" * 50)
print("Baseline Training Completed")
print("=" * 50)