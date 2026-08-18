import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/loss_history.csv")

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
plt.ylabel("MSE Loss")
plt.title("QLSTM Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "plots/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved -> plots/loss_curve.png")