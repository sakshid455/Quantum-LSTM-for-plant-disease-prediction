import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load comprehensive comparison
df = pd.read_csv("outputs/final_model_comparison.csv")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

models = df["Model"].tolist()
d_r2 = df["Disease_R2"].tolist()
l_r2 = df["Lesion_R2"].tolist()

# Labels for plotting
labels = [m.replace(" (Multimodal)", "\n(Multimodal)").replace(" Ablation", "") for m in models]
colors = ["#999999", "#7570b3", "#4A90E2", "#56B4E9", "#E69F00", "#009E73"]

# 1. Disease Severity R2
plt.figure(figsize=(10, 5.5), dpi=300)
d_r2_clipped = [max(v, -1.5) for v in d_r2]
bars = plt.bar(labels, d_r2_clipped, color=colors, edgecolor="black", width=0.55)
plt.axhline(0, linestyle="--", linewidth=1.2, color="gray")
plt.ylabel("Disease Severity R²", fontsize=12, fontweight="bold")
plt.title("Disease Severity Prediction: Comprehensive Model Benchmark", fontsize=13, fontweight="bold")
plt.ylim(-1.6, 1.0)
for bar, val in zip(bars, d_r2):
    y = max(bar.get_height(), -1.45)
    lbl = f"{val:.3f}" if val >= 0 else (f"{val:.2f}" if val > -5 else f"{val:.1f}*")
    plt.text(bar.get_x() + bar.get_width() / 2, y + (0.04 if val >= 0 else -0.12), lbl,
             ha="center", va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model_comparison_disease_r2.png", dpi=300)
plt.close()

# 2. Lesion Area R2
plt.figure(figsize=(10, 5.5), dpi=300)
l_r2_clipped = [max(v, -1.2) for v in l_r2]
bars = plt.bar(labels, l_r2_clipped, color=colors, edgecolor="black", width=0.55)
plt.axhline(0, linestyle="--", linewidth=1.2, color="gray")
plt.ylabel("Lesion Area R²", fontsize=12, fontweight="bold")
plt.title("Lesion Area Prediction: Comprehensive Model Benchmark", fontsize=13, fontweight="bold")
plt.ylim(-1.2, 0.8)
for bar, val in zip(bars, l_r2):
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, y + (0.04 if val >= 0 else -0.12), f"{val:.3f}",
             ha="center", va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model_comparison_lesion_r2.png", dpi=300)
plt.close()

print("Saved updated comparison plots -> outputs/model_comparison_disease_r2.png & outputs/model_comparison_lesion_r2.png")