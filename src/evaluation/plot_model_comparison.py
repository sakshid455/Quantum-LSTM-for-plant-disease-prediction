import json
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Load metrics
# ============================================================

with open("outputs/multimodal_test_metrics.json", "r") as f:
    qlstm = json.load(f)

with open("outputs/multimodal_lstm_test_metrics.json", "r") as f:
    lstm = json.load(f)

with open("outputs/multimodal_gru_test_metrics.json", "r") as f:
    gru = json.load(f)


models = ["QLSTM", "LSTM", "GRU"]


# ============================================================
# Disease Severity R²
# ============================================================

disease_r2 = [
    qlstm["Disease Severity"]["R2"],
    lstm["Disease Severity"]["R2"],
    gru["Disease Severity"]["R2"]
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    models,
    disease_r2
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.ylabel("R²")
plt.xlabel("Model")
plt.title("Disease Severity Prediction: Model Comparison")

for bar, value in zip(bars, disease_r2):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom" if value >= 0 else "top"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison_disease_r2.png",
    dpi=300
)

plt.close()


# ============================================================
# Lesion Area R²
# ============================================================

lesion_r2 = [
    qlstm["Lesion Area"]["R2"],
    lstm["Lesion Area"]["R2"],
    gru["Lesion Area"]["R2"]
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    models,
    lesion_r2
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.ylabel("R²")
plt.xlabel("Model")
plt.title("Lesion Area Prediction: Model Comparison")

for bar, value in zip(bars, lesion_r2):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom" if value >= 0 else "top"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison_lesion_r2.png",
    dpi=300
)

plt.close()


# ============================================================
# Print
# ============================================================

print("=" * 60)
print("MODEL COMPARISON PLOTS GENERATED")
print("=" * 60)

print("Saved:")
print("  outputs/model_comparison_disease_r2.png")
print("  outputs/model_comparison_lesion_r2.png")