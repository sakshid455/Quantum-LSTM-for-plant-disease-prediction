import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# Load metrics
# -----------------------------------------

with open("outputs/metrics.json", "r") as f:
    qlstm_metrics = json.load(f)

with open("outputs/baseline_metrics.json", "r") as f:
    baseline_metrics = json.load(f)

# -----------------------------------------
# Combine results
# -----------------------------------------

results = {
    "RNN": baseline_metrics["RNN"],
    "LSTM": baseline_metrics["LSTM"],
    "GRU": baseline_metrics["GRU"],
    "QLSTM": qlstm_metrics
}

df = pd.DataFrame(results).T

# Make sure column order is consistent
df = df[["MSE", "RMSE", "MAE", "R2"]]

# -----------------------------------------
# Save CSV
# -----------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df.to_csv(
    "outputs/model_comparison.csv"
)

print("Saved -> outputs/model_comparison.csv")

# -----------------------------------------
# Create comparison plot
# -----------------------------------------

ax = df[["MSE", "RMSE", "MAE"]].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_xlabel("Model")
ax.set_ylabel("Error")
ax.set_title("Model Error Comparison")
ax.legend(title="Metric")
ax.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "plots/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved -> plots/model_comparison.png")

# -----------------------------------------
# R² plot
# -----------------------------------------

ax = df["R2"].plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_xlabel("Model")
ax.set_ylabel("R²")
ax.set_title("R² Comparison")
ax.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "plots/r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved -> plots/r2_comparison.png")

# -----------------------------------------
# Display table
# -----------------------------------------

print("\nModel Comparison")
print("=" * 60)
print(df)