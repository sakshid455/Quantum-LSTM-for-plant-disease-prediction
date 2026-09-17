"""
Generate Publication-Quality Visualizations for Clean Future Forecasting QLSTM Experiments
Cleans up obsolete plots and generates:
1. clean_loss_curves.png: Epoch loss curves for Image-Only vs Safe-Multimodal.
2. clean_actual_vs_predicted.png: 4-panel scatter plots for Disease and Lesion predictions.
3. clean_model_comparison.png: Rigorous benchmark comparison including Leaky Reference.
4. clean_residuals.png: Error distributions for Disease and Lesion targets.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

PLOTS_DIR = "plots"
APP_PLOTS_DIR = "app/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(APP_PLOTS_DIR, exist_ok=True)

# File paths
IMG_HISTORY_PATH = "outputs/clean_image_100leaves_loss_history.csv"
MULTI_HISTORY_PATH = "outputs/clean_multimodal_100leaves_loss_history.csv"
IMG_PREDS_PATH = "outputs/clean_image_100leaves_predictions.csv"
MULTI_PREDS_PATH = "outputs/clean_multimodal_100leaves_predictions.csv"
IMG_METRICS_PATH = "outputs/clean_image_100leaves_test_metrics.json"
MULTI_METRICS_PATH = "outputs/clean_multimodal_100leaves_test_metrics.json"


def plot_clean_loss_curves():
    print("Generating clean_loss_curves.png...")
    df_img = pd.read_csv(IMG_HISTORY_PATH)
    df_multi = pd.read_csv(MULTI_HISTORY_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Total Multi-Task Loss
    ax = axes[0]
    ax.plot(df_img["Epoch"], df_img["Train Loss"], label="Image-Only Train", color="#3498db", lw=2)
    ax.plot(df_img["Epoch"], df_img["Validation Loss"], label="Image-Only Val", color="#2980b9", lw=2, linestyle="--")
    ax.plot(df_multi["Epoch"], df_multi["Train Loss"], label="Multimodal Train", color="#2ecc71", lw=2)
    ax.plot(df_multi["Epoch"], df_multi["Validation Loss"], label="Multimodal Val", color="#27ae60", lw=2, linestyle="--")
    ax.set_title("Multi-Task Total Loss (50 Epochs)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Weighted MSE Loss", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")
    ax.grid(True)

    # 2. Disease Severity Loss
    ax = axes[1]
    ax.plot(df_img["Epoch"], df_img["Train Disease Loss"], label="Image-Only Train", color="#3498db", lw=2)
    ax.plot(df_img["Epoch"], df_img["Validation Disease Loss"], label="Image-Only Val", color="#2980b9", lw=2, linestyle="--")
    ax.plot(df_multi["Epoch"], df_multi["Train Disease Loss"], label="Multimodal Train", color="#2ecc71", lw=2)
    ax.plot(df_multi["Epoch"], df_multi["Validation Disease Loss"], label="Multimodal Val", color="#27ae60", lw=2, linestyle="--")
    ax.set_title("Disease Severity MSE Loss", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Disease MSE", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")
    ax.grid(True)

    # 3. Lesion Area Loss
    ax = axes[2]
    ax.plot(df_img["Epoch"], df_img["Train Lesion Loss"], label="Image-Only Train", color="#3498db", lw=2)
    ax.plot(df_img["Epoch"], df_img["Validation Lesion Loss"], label="Image-Only Val", color="#2980b9", lw=2, linestyle="--")
    ax.plot(df_multi["Epoch"], df_multi["Train Lesion Loss"], label="Multimodal Train", color="#2ecc71", lw=2)
    ax.plot(df_multi["Epoch"], df_multi["Validation Lesion Loss"], label="Multimodal Val", color="#27ae60", lw=2, linestyle="--")
    ax.set_title("Lesion Area Normalized MSE Loss", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Lesion MSE (Z-score)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "clean_loss_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_clean_actual_vs_predicted():
    print("Generating clean_actual_vs_predicted.png...")
    df_img = pd.read_csv(IMG_PREDS_PATH)
    df_multi = pd.read_csv(MULTI_PREDS_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

    # 1. Image-Only Disease
    ax = axes[0, 0]
    ax.scatter(df_img["true_disease"], df_img["predicted_disease"], color="#2980b9", alpha=0.6, s=40, edgecolors="none")
    lims = [0, max(df_img["true_disease"].max(), df_img["predicted_disease"].max()) * 1.05]
    ax.plot(lims, lims, color="#e74c3c", lw=2, linestyle="--", label="Ideal Fit (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Clean Image-Only: Disease Severity at t+1\nR² = 0.8690 | RMSE = 0.0507 | MAE = 0.0333", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Disease Severity (placl)", fontsize=11)
    ax.set_ylabel("Predicted Disease Severity", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    # 2. Image-Only Lesion Area
    ax = axes[0, 1]
    ax.scatter(df_img["true_lesion_area"] / 1e5, df_img["predicted_lesion_area"] / 1e5, color="#16a085", alpha=0.6, s=40, edgecolors="none")
    lims = [0, max(df_img["true_lesion_area"].max(), df_img["predicted_lesion_area"].max()) / 1e5 * 1.05]
    ax.plot(lims, lims, color="#e74c3c", lw=2, linestyle="--", label="Ideal Fit (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Clean Image-Only: Lesion Area at t+1\nR² = 0.7654 | RMSE = 227,987 px² | MAE = 112,114 px²", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Lesion Area (×10⁵ px²)", fontsize=11)
    ax.set_ylabel("Predicted Lesion Area (×10⁵ px²)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    # 3. Multimodal Disease
    ax = axes[1, 0]
    ax.scatter(df_multi["true_disease"], df_multi["predicted_disease"], color="#8e44ad", alpha=0.6, s=40, edgecolors="none")
    lims = [0, max(df_multi["true_disease"].max(), df_multi["predicted_disease"].max()) * 1.05]
    ax.plot(lims, lims, color="#e74c3c", lw=2, linestyle="--", label="Ideal Fit (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Clean Multimodal: Disease Severity at t+1\nR² = 0.8343 | RMSE = 0.0570 | MAE = 0.0316", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Disease Severity (placl)", fontsize=11)
    ax.set_ylabel("Predicted Disease Severity", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    # 4. Multimodal Lesion Area
    ax = axes[1, 1]
    ax.scatter(df_multi["true_lesion_area"] / 1e5, df_multi["predicted_lesion_area"] / 1e5, color="#d35400", alpha=0.6, s=40, edgecolors="none")
    lims = [0, max(df_multi["true_lesion_area"].max(), df_multi["predicted_lesion_area"].max()) / 1e5 * 1.05]
    ax.plot(lims, lims, color="#e74c3c", lw=2, linestyle="--", label="Ideal Fit (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Clean Multimodal: Lesion Area at t+1\nR² = 0.7455 | RMSE = 237,466 px² | MAE = 117,348 px²", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Lesion Area (×10⁵ px²)", fontsize=11)
    ax.set_ylabel("Predicted Lesion Area (×10⁵ px²)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "clean_actual_vs_predicted.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_clean_model_comparison():
    print("Generating clean_model_comparison.png...")
    categories = ["Clean Image-Only\n(Valid Forecasting)", "Clean Multimodal\n(Valid Forecasting)", "Leaky Reference\n(Contemporaneous / Leaky)"]
    disease_r2 = [0.8690, 0.8343, 0.9339]
    lesion_r2 = [0.7654, 0.7455, 0.8674]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars1 = ax.bar(x - width/2, disease_r2, width, label="Disease Severity R²", color=["#2980b9", "#8e44ad", "#e74c3c"], alpha=0.85, edgecolor="#2c3e50")
    bars2 = ax.bar(x + width/2, lesion_r2, width, label="Lesion Area R²", color=["#16a085", "#d35400", "#c0392b"], alpha=0.85, edgecolor="#2c3e50")

    ax.set_ylabel("Coefficient of Determination (R²)", fontsize=12, fontweight="bold")
    ax.set_title("Wheat Foliar Disease Progression: Clean Forecasting vs Leaky Reference", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc="lower right", facecolor="white", edgecolor="#ccc")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add annotation about leakage
    ax.annotate("⚠️ LEAKY REFERENCE\nNot valid forecasting\n(Contemporaneous targets\n& lesion proxies in input)",
                xy=(2, 0.9339), xytext=(1.45, 0.50),
                arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=2, headwidth=8),
                fontsize=10, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", lw=1.5))

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "clean_model_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_clean_residuals():
    print("Generating clean_residuals.png...")
    df_img = pd.read_csv(IMG_PREDS_PATH)
    df_multi = pd.read_csv(MULTI_PREDS_PATH)

    res_img_d = df_img["predicted_disease"] - df_img["true_disease"]
    res_img_l = df_img["predicted_lesion_area"] - df_img["true_lesion_area"]
    res_multi_d = df_multi["predicted_disease"] - df_multi["true_disease"]
    res_multi_l = df_multi["predicted_lesion_area"] - df_multi["true_lesion_area"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Disease Residuals
    ax = axes[0]
    sns.kdeplot(res_img_d, ax=ax, label=f"Image-Only (μ={res_img_d.mean():.4f}, σ={res_img_d.std():.4f})", color="#2980b9", lw=2, fill=True, alpha=0.2)
    sns.kdeplot(res_multi_d, ax=ax, label=f"Multimodal (μ={res_multi_d.mean():.4f}, σ={res_multi_d.std():.4f})", color="#8e44ad", lw=2, fill=True, alpha=0.2)
    ax.axvline(0, color="#e74c3c", linestyle="--", lw=1.5)
    ax.set_title("Disease Severity Residual Distribution (Error = Pred - Actual)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Error in Disease Severity (placl)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    # 2. Lesion Residuals
    ax = axes[1]
    sns.kdeplot(res_img_l / 1e5, ax=ax, label=f"Image-Only (μ={res_img_l.mean()/1e5:.3f}, σ={res_img_l.std()/1e5:.3f})", color="#16a085", lw=2, fill=True, alpha=0.2)
    sns.kdeplot(res_multi_l / 1e5, ax=ax, label=f"Multimodal (μ={res_multi_l.mean()/1e5:.3f}, σ={res_multi_l.std()/1e5:.3f})", color="#d35400", lw=2, fill=True, alpha=0.2)
    ax.axvline(0, color="#e74c3c", linestyle="--", lw=1.5)
    ax.set_title("Lesion Area Residual Distribution (Error = Pred - Actual)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Error in Lesion Area (×10⁵ px²)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "clean_residuals.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved -> {out_path}")


def cleanup_and_sync():
    """Remove obsolete, deprecated plots and sync fresh clean plots to app/plots/"""
    print("\nCleaning up obsolete plots...")
    obsolete_files = [
        "plots/qlstm_100leaves_disease_actual_vs_predicted.png",
        "plots/qlstm_100leaves_lesion_actual_vs_predicted.png",
        "plots/qlstm_100leaves_loss_curves.png",
        "plots/scaling_study_30_vs_100_leaves.png",
        "plots/actual_vs_predicted.png",
        "plots/distribution.png",
        "plots/loss_curve.png",
        "plots/model_comparison.png",
        "plots/r2_comparison.png",
        "plots/residual_plot.png",
    ]
    for f in obsolete_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed obsolete: {f}")

    # Clean app/plots completely and copy only the latest clean plots
    if os.path.exists(APP_PLOTS_DIR):
        shutil.rmtree(APP_PLOTS_DIR)
    os.makedirs(APP_PLOTS_DIR, exist_ok=True)

    clean_plots = [
        "clean_loss_curves.png",
        "clean_actual_vs_predicted.png",
        "clean_model_comparison.png",
        "clean_residuals.png",
    ]

    for p in clean_plots:
        src = os.path.join(PLOTS_DIR, p)
        dst = os.path.join(APP_PLOTS_DIR, p)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Synced to dashboard -> {dst}")


if __name__ == "__main__":
    plot_clean_loss_curves()
    plot_clean_actual_vs_predicted()
    plot_clean_model_comparison()
    plot_clean_residuals()
    cleanup_and_sync()
    print("\nAll clean plots generated and synchronized successfully!")
