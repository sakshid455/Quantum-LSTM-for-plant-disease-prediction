import numpy as np
import pandas as pd
import torch

from src.quantum.qlstm_model import QLSTMModel
from src.dataset.multimodal_dataloader import create_multimodal_loaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_multimodal_qlstm.pth"


# ============================================================
# Load data
# ============================================================

(
    train_loader,
    val_loader,
    test_loader,
    lesion_mean,
    lesion_std
) = create_multimodal_loaders(
    batch_size=8
)


# ============================================================
# Load model
# ============================================================

model = QLSTMModel(
    input_size=791,
    hidden_size=32
).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()


# ============================================================
# Predictions
# ============================================================

rows = []

with torch.no_grad():

    for X, y_disease, y_lesion in test_loader:

        X = X.to(DEVICE)

        disease_pred, lesion_pred = model(X)

        disease_pred = (
            disease_pred
            .cpu()
            .numpy()
            .flatten()
        )

        lesion_pred = (
            lesion_pred
            .cpu()
            .numpy()
            .flatten()
        )

        disease_actual = (
            y_disease
            .cpu()
            .numpy()
            .flatten()
        )

        lesion_actual = (
            y_lesion
            .cpu()
            .numpy()
            .flatten()
        )

        for i in range(len(disease_pred)):

            actual_lesion = (
                lesion_actual[i] * lesion_std
                + lesion_mean
            )

            predicted_lesion = (
                lesion_pred[i] * lesion_std
                + lesion_mean
            )

            rows.append({
                "actual_disease": float(
                    disease_actual[i]
                ),

                "predicted_disease": float(
                    disease_pred[i]
                ),

                "actual_lesion": float(
                    actual_lesion
                ),

                "predicted_lesion": float(
                    predicted_lesion
                )
            })


# ============================================================
# Save
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    "outputs/test_predictions.csv",
    index=False
)


# ============================================================
# Display
# ============================================================

print("=" * 70)
print("QLSTM TEST PREDICTIONS")
print("=" * 70)

print(df.describe())

print()
print(df.head(20).to_string(index=False))

print()
print("Saved -> outputs/test_predictions.csv")