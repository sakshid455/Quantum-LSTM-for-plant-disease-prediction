import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.baselines.lstm_model import LSTMModel
from src.evaluation.metrics import evaluate


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8


# ============================================================
# Load test data
# ============================================================

X_test = np.load("data/sequences/X_test.npy")
y_test = np.load("data/sequences/y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Load LSTM
# ============================================================

model = LSTMModel(
    input_size=768,
    hidden_size=32,
    num_layers=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best_lstm_model.pth",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# Predictions
# ============================================================

predictions = []
targets = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        output = model(X)

        predictions.extend(
            output.view(-1).cpu().numpy().tolist()
        )

        targets.extend(
            y.view(-1).numpy().tolist()
        )


predictions = np.array(predictions)
targets = np.array(targets)


# ============================================================
# Evaluation
# ============================================================

results = evaluate(
    targets,
    predictions
)


# ============================================================
# Save metrics
# ============================================================

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/lstm_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# Display
# ============================================================

print("=" * 60)
print("LSTM TEST EVALUATION")
print("=" * 60)

print(f"MSE  : {results['MSE']:.6f}")
print(f"RMSE : {results['RMSE']:.6f}")
print(f"MAE  : {results['MAE']:.6f}")
print(f"R²   : {results['R2']:.6f}")

print("\nSaved -> outputs/lstm_metrics.json")