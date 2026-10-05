"""Generate manuscript figures from the leak-free benchmark outputs.

    python paper/make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "paper/figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
COLORS = {"Persistence": "#8a897f", "QLSTM": "#eb6834", "LSTM": "#2a78d6", "GRU": "#1baf7a"}
ORDER = ["Persistence", "QLSTM", "LSTM", "GRU"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def fig_benchmark():
    df = pd.read_csv("outputs/rds_benchmark_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, task, title in zip(axes, ["Disease R2", "Lesion R2"], ["(a) Disease severity at t+1", "(b) Lesion area at t+1"]):
        for i, m in enumerate(ORDER):
            v = df.loc[df.Model == m, task].to_numpy()
            ax.bar(i, v.mean(), 0.62, color=COLORS[m], alpha=0.85, zorder=3)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=INK, zorder=4, lw=0)
            ax.text(i, v.mean() + 0.025, f"{v.mean():.3f}\n(n={len(v)})", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(ORDER)), ORDER)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        tidy(ax)
    axes[0].set_ylabel("Test R² (150 unseen leaves)")
    fig.text(0.5, -0.02, "Bars: mean over leaf-level splits; dots: individual seeds (each seed = a different split of test leaves).",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rds_benchmark.png", dpi=300, bbox_inches="tight")


def fig_scatter(seed=42):
    q = pd.read_csv(f"outputs/ablation_multimodal_rds_seed{seed}_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, (a, p, label, scale) in zip(axes, [("actual_disease", "predicted_disease", "Disease severity (placl)", 1),
                                               ("actual_lesion", "predicted_lesion", "Lesion area (×10⁶ px²)", 1e6)]):
        x, y = q[a] / scale, q[p] / scale
        lim = max(x.max(), y.max()) * 1.04
        ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1, zorder=2)
        ax.scatter(x, y, s=9, color=COLORS["QLSTM"], alpha=0.55, lw=0, zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"Observed {label}"); ax.set_ylabel(f"Predicted {label}")
        tidy(ax); ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rds_qlstm_scatter.png", dpi=300, bbox_inches="tight")


def fig_curves(seed=42):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for m, name in [("QLSTM", f"multimodal_rds_seed{seed}"), ("LSTM", f"lstm_multimodal_rds_seed{seed}"),
                    ("GRU", f"gru_multimodal_rds_seed{seed}")]:
        h = pd.read_csv(f"outputs/ablation_{name}_loss_history.csv")
        for ax, col in zip(axes, ["Validation Disease Loss", "Validation Lesion Loss"]):
            ax.plot(h["Epoch"], h[col], color=COLORS[m], lw=1.8, label=m)
    axes[0].set_title("(a) Validation severity MSE", loc="left", fontsize=9.5)
    axes[1].set_title("(b) Validation lesion MSE (standardised)", loc="left", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("Epoch"); tidy(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rds_curves.png", dpi=300, bbox_inches="tight")


def fig_leakage():
    """Same models, leaky vs leak-free formulation of the 100-leaf data."""
    load = lambda p: json.load(open(p))
    arch = "outputs/archive_leaky/metadata_100leaves/"
    rows = [
        ("GRU, metadata only", load(f"{arch}ablation_gru_metadata_100leaves_qlstm_metrics.json"),
         load("outputs/ablation_gru_metadata_histfc100_qlstm_metrics.json")),
        ("LSTM, metadata only", load(f"{arch}ablation_lstm_metadata_100leaves_qlstm_metrics.json"),
         load("outputs/ablation_lstm_metadata_histfc100_qlstm_metrics.json")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, task, title in zip(axes, ["Disease Severity", "Lesion Area"], ["(a) Disease severity", "(b) Lesion area"]):
        for i, (lab, leaky, clean) in enumerate(rows):
            for off, d, col, name in [(-0.2, leaky, "#c0392b", "Contemporaneous target (leaky)"),
                                      (0.2, clean, "#2a78d6", "t+1 target (leak-free)")]:
                v = d[task]["R2"]
                ax.bar(i + off, v, 0.38, color=col, zorder=3, label=name if i == 0 else None)
                ax.text(i + off, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows])
        ax.set_ylim(0.6, 1.05); ax.set_title(title, loc="left", fontsize=9.5); tidy(ax)
    axes[0].set_ylabel("Test R² (15 unseen leaves)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_leakage_audit.png", dpi=300, bbox_inches="tight")




def fig_trajectories(seed=42, n=4):
    """Observed vs QLSTM one-step-ahead forecasts along individual test leaves."""
    q = pd.read_csv(f"outputs/ablation_multimodal_rds_seed{seed}_predictions.csv")
    top = q.groupby("leaf_id").actual_disease.max().sort_values(ascending=False)
    picks = list(top.index[[0, 5, 15, 40]])[:n]
    fig, axes = plt.subplots(1, n, figsize=(10, 2.8), sharey=True)
    for ax, leaf in zip(axes, picks):
        g = q[q.leaf_id == leaf].reset_index(drop=True)
        steps = np.arange(len(g)) + 5  # first forecast target is the 5th observation
        ax.plot(steps, g.actual_disease, "-o", color=INK, ms=3.5, lw=1.4, label="Observed")
        ax.plot(steps, g.predicted_disease, "-s", color=COLORS["QLSTM"], ms=3.5, lw=1.4, label="QLSTM forecast")
        ax.set_title(leaf, fontsize=8.5, loc="left"); ax.set_xlabel("Observation index"); tidy(ax)
    axes[0].set_ylabel("Disease severity (placl)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rds_trajectories.png", dpi=300, bbox_inches="tight")


def fig_qlstm_curves(seed=42):
    h = pd.read_csv(f"outputs/ablation_multimodal_rds_seed{seed}_loss_history.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(h["Epoch"], h["Train Loss"], color=COLORS["QLSTM"], lw=1.8, label="Training")
    ax.plot(h["Epoch"], h["Validation Loss"], color=COLORS["QLSTM"], lw=1.8, ls="--", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Multi-task loss"); tidy(ax); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rds_qlstm_curves.png", dpi=300, bbox_inches="tight")


def fig_visual_scatter():
    q = pd.read_csv("outputs/clean_image_100leaves_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, (a, p, label, scale) in zip(axes, [("true_disease", "predicted_disease", "Disease severity (placl)", 1),
                                               ("true_lesion_area", "predicted_lesion_area", "Lesion area (×10⁶ px²)", 1e6)]):
        x, y = q[a] / scale, q[p] / scale
        lim = max(x.max(), y.max()) * 1.04
        ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1, zorder=2)
        ax.scatter(x, y, s=14, color=COLORS["QLSTM"], alpha=0.7, lw=0, zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"Observed {label}"); ax.set_ylabel(f"Predicted {label}")
        tidy(ax); ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_visual_qlstm_scatter.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fig_benchmark(); fig_scatter(); fig_curves(); fig_leakage(); fig_trajectories(); fig_qlstm_curves(); fig_visual_scatter()
    print("figures written to", OUT)
