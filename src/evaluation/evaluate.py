import numpy as np
import torch
import json
import os

from src.quantum.qlstm_model import QLSTMModel
from src.dataset.dataloader import get_dataloaders
from src.evaluation.metrics import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data only
_, _, test_loader = get_dataloaders(batch_size=8)

# Load model
model = QLSTMModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

predictions = []
targets = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        # QLSTM returns:
        # disease_output, lesion_output
        output = model(X)

        pred = output.view(-1).cpu().numpy()
        true = y.view(-1).cpu().numpy()

        predictions.extend(pred.tolist())
        targets.extend(true.tolist())


predictions = np.array(predictions)
targets = np.array(targets)

# Calculate metrics
results = evaluate(targets, predictions)

# Save metrics
os.makedirs("outputs", exist_ok=True)

with open("outputs/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nTest Results:")
for key, value in results.items():
    print(f"{key}: {value:.6f}")

print("\nSaved -> outputs/metrics.json")