import os
import json
import numpy as np
import torch

from src.models.baseline_models import (
    RNNModel,
    LSTMModel,
    GRUModel
)

from src.dataset.dataloader import get_dataloaders
from src.evaluation.metrics import evaluate


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8

os.makedirs("outputs", exist_ok=True)

_, _, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)


def evaluate_model(model, checkpoint, name):

    model = model.to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(DEVICE)

            output = model(X)

            pred = output.view(-1).cpu().numpy()
            true = y.view(-1).cpu().numpy()

            predictions.extend(pred.tolist())
            targets.extend(true.tolist())

    predictions = np.array(predictions)
    targets = np.array(targets)

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    metrics = evaluate(
        targets,
        predictions
    )

    return metrics


results = {}

results["RNN"] = evaluate_model(
    RNNModel(),
    "checkpoints/baselines/rnn_best.pth",
    "RNN"
)

results["LSTM"] = evaluate_model(
    LSTMModel(),
    "checkpoints/baselines/lstm_best.pth",
    "LSTM"
)

results["GRU"] = evaluate_model(
    GRUModel(),
    "checkpoints/baselines/gru_best.pth",
    "GRU"
)


with open(
    "outputs/baseline_metrics.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 40)
print("Baseline evaluation completed")
print("=" * 40)

print("Saved -> outputs/baseline_metrics.json")