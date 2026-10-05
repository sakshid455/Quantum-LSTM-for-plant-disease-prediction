"""
Summarise the multi-seed rds benchmark (scripts/run_rds_benchmark.py).

For every seed the leaf-level test split is rebuilt exactly as in training, a
persistence baseline (target at t+1 = observed value at t) is scored on the
same windows, and each model's saved predictions are re-scored. Outputs:

    outputs/rds_benchmark_per_seed.csv   one row per (model, seed)
    outputs/rds_benchmark_summary.csv    mean / std across seeds
    outputs/rds_benchmark_paired.csv     per-seed R2 differences vs QLSTM

    python -m src.evaluation.summarize_rds_benchmark
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data/sequences/rds_forecast_sequences.npz"
SEEDS = [42, 123, 456, 789, 2026]
MODELS = {"QLSTM": "multimodal_rds_seed{s}", "LSTM": "lstm_multimodal_rds_seed{s}", "GRU": "gru_multimodal_rds_seed{s}"}
TASKS = {"Disease": ("actual_disease", "predicted_disease"), "Lesion": ("actual_lesion", "predicted_lesion")}


def test_indices(leaf_ids, y, seed):
    """Same two-stage GroupShuffleSplit as train_qlstm_ablation.create_ablation_loaders."""
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, temp_idx = next(gss_1.split(np.zeros(len(y)), y, groups=leaf_ids))
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    _, rel_test = next(gss_2.split(temp_idx, y[temp_idx], groups=leaf_ids[temp_idx]))
    return temp_idx[rel_test]


def score(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred)}


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    features = list(data["feature_names"])
    X, y_d, y_l, leaves = data["X"], data["y_placl"], data["y_lesion_area"], data["leaf_ids"]
    last_placl = X[:, -1, features.index("placl")]
    last_lesion = last_placl * X[:, -1, features.index("la_tot")]

    rows = []
    for seed in SEEDS:
        te = test_indices(leaves, y_d, seed)
        truth = {"Disease": y_d[te], "Lesion": y_l[te]}
        persistence = {"Disease": last_placl[te], "Lesion": last_lesion[te]}
        row = {"Model": "Persistence", "Seed": seed, "Parameters": 0, "Test leaves": len(set(leaves[te])), "Test windows": len(te)}
        for task in TASKS:
            row.update({f"{task} {k}": v for k, v in score(truth[task], persistence[task]).items()})
        rows.append(row)

        for model, pattern in MODELS.items():
            name = pattern.format(s=seed)
            metrics_path = f"outputs/ablation_{name}_qlstm_metrics.json"
            preds_path = f"outputs/ablation_{name}_predictions.csv"
            if not (os.path.exists(metrics_path) and os.path.exists(preds_path)):
                continue
            preds = pd.read_csv(preds_path)
            # Guard: prediction rows must be exactly this seed's test windows, in order.
            assert (preds["leaf_id"].to_numpy() == leaves[te]).all(), f"{name}: leaf order mismatch"
            assert np.allclose(preds["actual_disease"].to_numpy(), truth["Disease"], atol=1e-6), f"{name}: split mismatch"
            with open(metrics_path) as f:
                meta = json.load(f)
            row = {"Model": model, "Seed": seed, "Parameters": meta["parameters"],
                   "Test leaves": len(set(leaves[te])), "Test windows": len(te),
                   "Best epoch": meta["best_validation_epoch"]}
            for task, (t_col, p_col) in TASKS.items():
                row.update({f"{task} {k}": v for k, v in score(preds[t_col], preds[p_col]).items()})
            rows.append(row)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv("outputs/rds_benchmark_per_seed.csv", index=False)

    metric_cols = [c for c in per_seed.columns if c.split(" ")[0] in TASKS]
    summary = per_seed.groupby("Model", sort=False).agg(
        Seeds=("Seed", "nunique"), Parameters=("Parameters", "first"),
        **{f"{c} mean": (c, "mean") for c in metric_cols}, **{f"{c} std": (c, "std") for c in metric_cols})
    summary.to_csv("outputs/rds_benchmark_summary.csv")

    # Paired comparison vs QLSTM on seeds where both exist.
    paired = []
    q = per_seed[per_seed.Model == "QLSTM"].set_index("Seed")
    for other in ["LSTM", "GRU", "Persistence"]:
        o = per_seed[per_seed.Model == other].set_index("Seed")
        common = q.index.intersection(o.index)
        if len(common) < 2:
            continue
        for task in TASKS:
            diff = q.loc[common, f"{task} R2"] - o.loc[common, f"{task} R2"]
            t_p = stats.ttest_rel(q.loc[common, f"{task} R2"], o.loc[common, f"{task} R2"]).pvalue
            paired.append({"Comparison": f"QLSTM - {other}", "Task": task, "Seeds": len(common),
                           "Mean dR2": diff.mean(), "Std dR2": diff.std(), "QLSTM wins": int((diff > 0).sum()),
                           "Paired t p": t_p})
    paired = pd.DataFrame(paired)
    paired.to_csv("outputs/rds_benchmark_paired.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda m, s: f"{m:.4f} +/- {s:.4f}" if pd.notna(s) else f"{m:.4f}"
    table = pd.DataFrame({
        "Seeds": summary["Seeds"], "Params": summary["Parameters"],
        **{f"{t} {k}": [fmt(m, s) for m, s in zip(summary[f"{t} {k} mean"], summary[f"{t} {k} std"])]
           for t in TASKS for k in ["R2", "RMSE"]},
    })
    print(table.to_string())
    if len(paired):
        print("\n" + paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
