import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_ablation_plots():
    csv_path = "outputs/qlstm_ablation_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs("outputs/plots", exist_ok=True)

    models = ["Metadata-Only\n(23 feats)", "ViT-Only\n(768 feats)", "Multimodal Fusion\n(791 feats)"]
    disease_r2 = df["Disease R2"].values
    lesion_r2 = df["Lesion R2"].values
    disease_rmse = df["Disease RMSE"].values
    lesion_rmse = df["Lesion RMSE"].values

    colors = ["#4A90E2", "#E28743", "#2CA02C"]

    # Figure 1: R2 Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(models))
    width = 0.55

    # Disease R2
    bars1 = ax1.bar(x, disease_r2, width, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax1.set_title("Disease Severity (PLACL) - Test $R^2$", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylabel("$R^2$ Score", fontsize=11)
    ax1.set_ylim(-0.1, 1.0)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, max(yval + 0.02, 0.03), f"{yval:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Lesion R2
    bars2 = ax2.bar(x, lesion_r2, width, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax2.set_title("Lesion Area ($px^2$) - Test $R^2$", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.set_ylabel("$R^2$ Score", fontsize=11)
    ax2.set_ylim(-0.2, 0.8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02 if yval >= 0 else yval - 0.06, f"{yval:.3f}",
                 ha="center", va="bottom" if yval >= 0 else "top", fontsize=11, fontweight="bold")

    plt.suptitle("Modality Ablation Study: Hybrid ViT-QLSTM Architecture", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plot_path = "outputs/plots/qlstm_ablation_study_r2.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation plot -> {plot_path}")

if __name__ == "__main__":
    generate_ablation_plots()
