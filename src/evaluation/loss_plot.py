import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

INPUT_PATH = "outputs/multimodal_loss_history.csv"
OUTPUT_PATH = "outputs/qlstm_multimodal_loss_curve.png"

df = pd.read_csv(INPUT_PATH)

# ------------------------------------------------------------
# Overall loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Epoch"],
    df["Train Loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    df["Epoch"],
    df["Validation Loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multimodal QLSTM Training and Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("QLSTM LOSS CURVE GENERATED")
print("=" * 60)
print(f"Saved -> {OUTPUT_PATH}")


# ------------------------------------------------------------
# Disease and lesion validation losses
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["Epoch"],
    df["Validation Disease Loss"],
    marker="o",
    label="Disease Validation Loss"
)

plt.plot(
    df["Epoch"],
    df["Validation Lesion Loss"],
    marker="o",
    label="Lesion Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("QLSTM Validation Loss by Prediction Task")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

TASK_OUTPUT = "outputs/qlstm_validation_task_loss.png"

plt.savefig(
    TASK_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved -> {TASK_OUTPUT}")