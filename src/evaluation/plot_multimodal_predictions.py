import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = "outputs/test_predictions.csv"


# ============================================================
# Load predictions
# ============================================================

df = pd.read_csv(INPUT_PATH)


# ============================================================
# Disease: Actual vs Predicted
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["actual_disease"],
    df["predicted_disease"],
    alpha=0.7
)

min_val = min(
    df["actual_disease"].min(),
    df["predicted_disease"].min()
)

max_val = max(
    df["actual_disease"].max(),
    df["predicted_disease"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Disease Severity")
plt.ylabel("Predicted Disease Severity")
plt.title("QLSTM: Disease Severity — Actual vs Predicted")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/qlstm_disease_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# Disease: Error Distribution
# ============================================================

disease_error = (
    df["predicted_disease"]
    - df["actual_disease"]
)

plt.figure(figsize=(7, 6))

plt.hist(
    disease_error,
    bins=20,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("QLSTM: Disease Severity — Error Distribution")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/qlstm_disease_error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# Lesion: Actual vs Predicted
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["actual_lesion"],
    df["predicted_lesion"],
    alpha=0.7
)

min_val = min(
    df["actual_lesion"].min(),
    df["predicted_lesion"].min()
)

max_val = max(
    df["actual_lesion"].max(),
    df["predicted_lesion"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Lesion Area")
plt.ylabel("Predicted Lesion Area")
plt.title("QLSTM: Lesion Area — Actual vs Predicted")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/qlstm_lesion_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# Lesion: Error Distribution
# ============================================================

lesion_error = (
    df["predicted_lesion"]
    - df["actual_lesion"]
)

plt.figure(figsize=(7, 6))

plt.hist(
    lesion_error,
    bins=20,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("QLSTM: Lesion Area — Error Distribution")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/qlstm_lesion_error_distribution.png",
    dpi=300
)

plt.close()


print("=" * 60)
print("QLSTM PREDICTION PLOTS GENERATED")
print("=" * 60)

print("Saved:")
print("  outputs/qlstm_disease_actual_vs_predicted.png")
print("  outputs/qlstm_disease_error_distribution.png")
print("  outputs/qlstm_lesion_actual_vs_predicted.png")
print("  outputs/qlstm_lesion_error_distribution.png")