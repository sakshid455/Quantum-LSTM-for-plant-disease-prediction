import os
import numpy as np
import pandas as pd
import torch

from src.dataset.dataloader import get_dataloaders
from src.quantum.qlstm_model import QLSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

_, _, test_loader = get_dataloaders(batch_size=8)

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

        output = model(X)

        predictions.extend(output.view(-1).cpu().tolist())
        targets.extend(y.view(-1).cpu().tolist())

df = pd.DataFrame({
    "Actual": targets,
    "Predicted": predictions
})

df.to_csv(
    "outputs/predictions.csv",
    index=False
)

print(df.head())

print("\nSaved -> outputs/predictions.csv")