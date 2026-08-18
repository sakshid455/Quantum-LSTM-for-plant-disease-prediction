import os
import pandas as pd
import matplotlib.pyplot as plt

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Load predictions
df = pd.read_csv("outputs/predictions.csv")

actual = df["Actual"]
predicted = df["Predicted"]

# Clamp predictions to valid PLACL range
predicted = predicted.clip(lower=0, upper=1)

# -------------------------------
# Actual vs Predicted Plot
# -------------------------------
plt.figure(figsize=(7, 7))

plt.scatter(actual, predicted, alpha=0.7)

plt.plot([0, 1], [0, 1], 'r--', linewidth=2)

plt.xlabel("Actual PLACL")
plt.ylabel("Predicted PLACL")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.close()

# -------------------------------
# Residual Plot
# -------------------------------
residual = actual - predicted

plt.figure(figsize=(7, 6))

plt.scatter(predicted, residual, alpha=0.7)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted PLACL")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.grid(True)

plt.savefig("plots/residual_plot.png", dpi=300)
plt.close()

# -------------------------------
# Prediction Distribution
# -------------------------------
plt.figure(figsize=(7, 6))

plt.hist(actual, bins=20, alpha=0.6, label="Actual")
plt.hist(predicted, bins=20, alpha=0.6, label="Predicted")

plt.xlabel("PLACL")
plt.ylabel("Frequency")

plt.title("Prediction Distribution")

plt.legend()

plt.grid(True)

plt.savefig("plots/distribution.png", dpi=300)
plt.close()

print("=" * 40)
print("Plots Saved Successfully")
print("=" * 40)