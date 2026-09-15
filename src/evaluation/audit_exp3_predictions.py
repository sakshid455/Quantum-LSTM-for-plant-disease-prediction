import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("outputs/qlstm_exp3_predictions.csv")

d_act = df["actual_disease"]
d_pred = df["predicted_disease"]
d_err = d_pred - d_act
d_abs_err = np.abs(d_err)
d_corr, _ = stats.pearsonr(d_act, d_pred)
d_spearman, _ = stats.spearmanr(d_act, d_pred)

l_act = df["actual_lesion"]
l_pred = df["predicted_lesion"]
l_err = l_pred - l_act
l_abs_err = np.abs(l_err)
l_corr, _ = stats.pearsonr(l_act, l_pred)
l_spearman, _ = stats.spearmanr(l_act, l_pred)

print("=" * 70)
print("1. PREDICTION VS TARGET CORRELATIONS")
print("=" * 70)
print(f"Disease Severity Pearson r : {d_corr:.6f} (r2 = {d_corr**2:.6f})")
print(f"Disease Severity Spearman rho: {d_spearman:.6f}")
print(f"Lesion Area Pearson r      : {l_corr:.6f} (r2 = {l_corr**2:.6f})")
print(f"Lesion Area Spearman rho   : {l_spearman:.6f}")

print("\n" + "=" * 70)
print("2. DISEASE SEVERITY ERROR STATS")
print("=" * 70)
print(f"Mean Error (Systematic Bias): {d_err.mean():+.6f}")
print(f"Median Error                : {d_err.median():+.6f}")
print(f"Std of Error                : {d_err.std():.6f}")
print(f"Min Error                   : {d_err.min():+.6f}")
print(f"Max Error                   : {d_err.max():+.6f}")
print(f"MAE                         : {d_abs_err.mean():.6f}")
print(f"Median Absolute Error       : {d_abs_err.median():.6f}")
print(f"IQR of Absolute Error       : {stats.iqr(d_abs_err):.6f}")
print(f"Overpredicted Fraction      : {(d_err > 0).mean():.2%} ({sum(d_err > 0)}/{len(df)})")
print(f"Underpredicted Fraction     : {(d_err < 0).mean():.2%} ({sum(d_err < 0)}/{len(df)})")

print("\n" + "=" * 70)
print("3. LESION AREA ERROR STATS")
print("=" * 70)
print(f"Mean Error (Systematic Bias): {l_err.mean():+,.2f} pixels")
print(f"Median Error                : {l_err.median():+,.2f} pixels")
print(f"Std of Error                : {l_err.std():,.2f} pixels")
print(f"Min Error                   : {l_err.min():+,.2f} pixels")
print(f"Max Error                   : {l_err.max():+,.2f} pixels")
print(f"MAE                         : {l_abs_err.mean():,.2f} pixels")
print(f"Median Absolute Error       : {l_abs_err.median():,.2f} pixels")
print(f"IQR of Absolute Error       : {stats.iqr(l_abs_err):,.2f} pixels")
print(f"Overpredicted Fraction      : {(l_err > 0).mean():.2%} ({sum(l_err > 0)}/{len(df)})")
print(f"Underpredicted Fraction     : {(l_err < 0).mean():.2%} ({sum(l_err < 0)}/{len(df)})")

print("\n" + "=" * 70)
print("4. PER-LEAF PERFORMANCE BREAKDOWN")
print("=" * 70)

leaf_rows = []
for leaf, g in df.groupby("leaf_id"):
    ld_r2 = r2_score(g["actual_disease"], g["predicted_disease"])
    ld_rmse = mean_squared_error(g["actual_disease"], g["predicted_disease"]) ** 0.5
    ld_mae = mean_absolute_error(g["actual_disease"], g["predicted_disease"])
    ld_corr, _ = stats.pearsonr(g["actual_disease"], g["predicted_disease"]) if g["actual_disease"].std() > 0 else (np.nan, 1)

    ll_r2 = r2_score(g["actual_lesion"], g["predicted_lesion"])
    ll_rmse = mean_squared_error(g["actual_lesion"], g["predicted_lesion"]) ** 0.5
    ll_mae = mean_absolute_error(g["actual_lesion"], g["predicted_lesion"])
    ll_corr, _ = stats.pearsonr(g["actual_lesion"], g["predicted_lesion"]) if g["actual_lesion"].std() > 0 else (np.nan, 1)

    leaf_rows.append({
        "Leaf": leaf,
        "N": len(g),
        "Disease_R2": ld_r2,
        "Disease_RMSE": ld_rmse,
        "Disease_MAE": ld_mae,
        "Disease_r": ld_corr,
        "Disease_Max": g["actual_disease"].max(),
        "Lesion_R2": ll_r2,
        "Lesion_RMSE": ll_rmse,
        "Lesion_MAE": ll_mae,
        "Lesion_r": ll_corr,
        "Lesion_Max": g["actual_lesion"].max(),
    })

    print(f"\nLeaf: {leaf} (N = {len(g)} observations)")
    print(f"  Disease: R2={ld_r2:8.4f} | RMSE={ld_rmse:8.4f} | MAE={ld_mae:8.4f} | r={ld_corr:8.4f} | Target Range=[{g['actual_disease'].min():.4f}, {g['actual_disease'].max():.4f}]")
    print(f"  Lesion : R2={ll_r2:8.4f} | RMSE={ll_rmse:10,.1f} | MAE={ll_mae:10,.1f} | r={ll_corr:8.4f} | Target Range=[{g['actual_lesion'].min():,.0f}, {g['actual_lesion'].max():,.0f}]")

leaf_df = pd.DataFrame(leaf_rows)
leaf_df.to_csv("outputs/qlstm_exp3_per_leaf_metrics.csv", index=False)
print(f"\nSaved per-leaf breakdown -> outputs/qlstm_exp3_per_leaf_metrics.csv")

# ============================================================
# 5. Generate Per-Leaf Comparison Plots
# ============================================================
os.makedirs("outputs/plots", exist_ok=True)

# Plot 5A: Per-Leaf R2 (Clipped for visual clarity with labels)
fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)

x = np.arange(len(leaf_df))
width = 0.35

d_r2_clipped = np.clip(leaf_df["Disease_R2"], -1.5, 1.0)
l_r2_clipped = np.clip(leaf_df["Lesion_R2"], -1.5, 1.0)

rects1 = ax.bar(x - width/2, d_r2_clipped, width, label="Disease Severity R²", color="#2b5c8f", edgecolor="black")
rects2 = ax.bar(x + width/2, l_r2_clipped, width, label="Lesion Area R²", color="#d95f02", edgecolor="black")

ax.set_xlabel("Held-Out Test Leaves", fontsize=11, fontweight="bold")
ax.set_ylabel("R² Score (Clipped to [-1.5, 1.0] for Near-Zero Variance Leaves)", fontsize=10, fontweight="bold")
ax.set_title("Improved Multimodal QLSTM (Exp 3): Per-Leaf Test R²", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{row['Leaf']}\n(N={row['N']})" for _, row in leaf_df.iterrows()], fontsize=9.5)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.axhline(0.840074, color="#2b5c8f", linestyle=":", label="Overall Disease R² (0.84)")
ax.axhline(0.636625, color="#d95f02", linestyle=":", label="Overall Lesion R² (0.64)")
ax.set_ylim(-1.6, 1.05)
ax.legend(frameon=True, loc="lower left", fontsize=9.5)

for rect, orig_val in zip(rects1, leaf_df["Disease_R2"]):
    h = rect.get_height()
    lbl = f"{orig_val:.2f}" if orig_val > -10 else f"{orig_val:.0f}*"
    ax.annotate(lbl, xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for rect, orig_val in zip(rects2, leaf_df["Lesion_R2"]):
    h = rect.get_height()
    lbl = f"{orig_val:.2f}" if orig_val > -10 else f"{orig_val:.0f}*"
    ax.annotate(lbl, xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/plots/exp3_per_leaf_r2_comparison.png", dpi=300)
plt.close()
print("Saved -> outputs/plots/exp3_per_leaf_r2_comparison.png")

# Plot 5B: Per-Leaf Pearson Correlation and Absolute Errors (RMSE)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
leaf_labels = [row["Leaf"].replace("ESWW", "..") for _, row in leaf_df.iterrows()]

ax1.bar(x, leaf_df["Disease_r"], color="#2b5c8f", edgecolor="black", width=0.5)
ax1.set_title("Disease Pearson r per Leaf", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(leaf_labels, fontsize=8.5, rotation=20)
ax1.set_ylim(-0.6, 1.05)
ax1.axhline(0, color="gray", linestyle="--")
for i, v in enumerate(leaf_df["Disease_r"]):
    ax1.text(i, v + 0.04 if v >= 0 else v - 0.09, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")

ax2.bar(x, leaf_df["Disease_RMSE"], color="#1b9e77", edgecolor="black", width=0.5)
ax2.set_title("Disease RMSE per Leaf (Severity)", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(leaf_labels, fontsize=8.5, rotation=20)
for i, v in enumerate(leaf_df["Disease_RMSE"]):
    ax2.text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")

ax3.bar(x, leaf_df["Lesion_RMSE"], color="#d95f02", edgecolor="black", width=0.5)
ax3.set_title("Lesion RMSE per Leaf (Pixels)", fontsize=11, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(leaf_labels, fontsize=8.5, rotation=20)
for i, v in enumerate(leaf_df["Lesion_RMSE"]):
    ax3.text(i, v + 4000, f"{v:,.0f}", ha="center", fontsize=8, fontweight="bold")

plt.suptitle("Per-Leaf Diagnostic: Correlation and Absolute Prediction Error", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/plots/exp3_per_leaf_metrics.png", dpi=300)
plt.close()
print("Saved -> outputs/plots/exp3_per_leaf_metrics.png")
