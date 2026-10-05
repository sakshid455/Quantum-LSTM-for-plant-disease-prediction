"""
Build leak-free t+1 forecasting sequences from the ETH Zurich leaf-level
phenotype table (data_leaf.rds, doi:10.3929/ethz-b-000739194).

  Input  : per-observation leaf features at [t-3, t-2, t-1, t]
  Target : placl and lesion area (placl * la_tot, px^2) at [t+1]

The features are segmentation outputs (placl itself is a column), so they are
only valid as HISTORY: the target observation never appears in its own window.
"""
import argparse
import numpy as np
import pandas as pd
import pyreadr

DEFAULT_RDS = "F:/projectfiles/finalyear/dataset_2024/dataset_2024/data_leaf.rds"
DEFAULT_OUT = "data/sequences/rds_forecast_sequences.npz"

FEATURES = [
    "la_tot", "la_damaged", "la_healthy", "la_damaged_f", "la_healthy_f", "la_insect",
    "n_pycn", "n_rust", "n_lesion", "placl", "pycn_density", "rust_density",
    "mean_dist", "std_dist", "cv_dist",
    "mean_shortest_dist", "std_shortest_dist", "cv_shortest_dist",
]
NAN_FEATURES = ["mean_shortest_dist", "std_shortest_dist", "cv_shortest_dist"]


def build(rds_path, seq_len):
    df = pyreadr.read_r(rds_path)[None]
    df = df.sort_values(["leaf_UID", "timestamp"]).reset_index(drop=True)

    # Shortest-distance stats are undefined with <2 lesions: zero-fill and flag.
    df["has_shortest_dist"] = df["mean_shortest_dist"].notna().astype(np.float32)
    df[NAN_FEATURES] = df[NAN_FEATURES].fillna(0.0)

    # Observation spacing is irregular (median ~24 h, up to ~10 days).
    df["hours_since_prev"] = (
        df.groupby("leaf_UID")["timestamp"].diff().dt.total_seconds().div(3600).fillna(0.0)
    )
    df["lesion_area"] = df["placl"] * df["la_tot"]

    feature_cols = FEATURES + ["has_shortest_dist", "hours_since_prev"]

    X, y_placl, y_lesion, leaf_ids, seasons, horizon_h = [], [], [], [], [], []
    for leaf_id, g in df.groupby("leaf_UID", sort=True):
        feats = g[feature_cols].to_numpy(np.float32)
        placl = g["placl"].to_numpy(np.float32)
        lesion = g["lesion_area"].to_numpy(np.float32)
        hours = g["hours_since_prev"].to_numpy(np.float32)
        year = g["timestamp"].dt.year.iloc[0]
        for s in range(len(g) - seq_len):
            X.append(feats[s : s + seq_len])
            y_placl.append(placl[s + seq_len])
            y_lesion.append(lesion[s + seq_len])
            horizon_h.append(hours[s + seq_len])
            leaf_ids.append(leaf_id)
            seasons.append(year)

    return dict(
        X=np.stack(X),
        y_placl=np.array(y_placl, np.float32),
        y_lesion_area=np.array(y_lesion, np.float32),
        leaf_ids=np.array(leaf_ids),
        season=np.array(seasons),
        horizon_hours=np.array(horizon_h, np.float32),
        feature_names=np.array(feature_cols),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rds", default=DEFAULT_RDS)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()

    data = build(args.rds, args.seq_len)
    X, y = data["X"], data["y_placl"]

    # Shift guard: each window's target must be the last observation of the
    # NEXT window of the same leaf, i.e. strictly after its own inputs.
    placl_col = list(data["feature_names"]).index("placl")
    last_placl = X[:, -1, placl_col]
    same_leaf = data["leaf_ids"][1:] == data["leaf_ids"][:-1]
    assert np.allclose(y[:-1][same_leaf], X[1:, -1, placl_col][same_leaf])

    np.savez_compressed(args.output, **data)
    print(f"X {X.shape} | leaves {len(set(data['leaf_ids']))} | "
          f"seasons {dict(zip(*np.unique(data['season'], return_counts=True)))}")
    print(f"unchanged placl (t -> t+1) in {np.mean(np.isclose(last_placl, y)):.3f} of windows")
    print(f"Saved -> {args.output}")
