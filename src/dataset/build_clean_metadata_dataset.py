"""
Build Clean Metadata-Only Temporal Forecasting Dataset (100 Leaves Cohort)

Companion to build_clean_forecasting_datasets.py, using the same strict
future-forecasting formulation and leaf-disjoint integrity checks, but
producing a metadata-only (no ViT) input:

  Input  : Observations at [t-3, t-2, t-1, t] (Sequence Length = 4)
  Target : Observation at [t+1] (Strict Future Forecasting)

Output:
  data/sequences/clean_metadata_temporal_sequences_100leaves.npz
  - 3 legitimate exogenous features only, zero lesion/damage-derived features:
    * la_tot: Total physical leaf blade pixel area
    * fungicide_treatment: Plot-level fungicide treatment (0/1)
    * inoculation_treatment: Plot-level artificial inoculation (0/1)

NOTE: data/meta/experimental_design.csv is not present in this checkout, so
fungicide_treatment and inoculation_treatment fall back to a constant 0.0 for
every leaf (matching the same fallback behavior as the existing
clean_multimodal/clean_image datasets) -- la_tot is the only feature with
real variance here. This is a real data-completeness gap, not fixed by this
script; it is surfaced explicitly in the printed summary below.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Raw source only (leaky as-is); the output keeps just la_tot/fungicide/inoculation with t+1 targets.
RAW_100_PATH = "data/sequences/archive_leaky/multimodal_temporal_sequences_100leaves.npz"
EXP_DESIGN_PATH = "data/meta/experimental_design.csv"

OUT_PATH = "data/sequences/clean_metadata_temporal_sequences_100leaves.npz"

os.makedirs("data/sequences", exist_ok=True)


def build_clean_metadata_dataset():
    print("=" * 70)
    print("BUILDING CLEAN METADATA-ONLY FORECASTING DATASET (T=4 -> t+1)")
    print("=" * 70)

    assert os.path.exists(RAW_100_PATH), f"Source file {RAW_100_PATH} not found."
    data = np.load(RAW_100_PATH, allow_pickle=True)
    X_raw = data["X"]               # (1190, 4, 791)
    y_d_raw = data["y_placl"]       # (1190,)
    y_l_raw = data["y_lesion_area"] # (1190,)
    leaf_ids_raw = data["leaf_ids"] # (1190,)

    exp_design_present = os.path.exists(EXP_DESIGN_PATH)
    exp_design = pd.read_csv(EXP_DESIGN_PATH).set_index("plot_UID") if exp_design_present else None

    unique_leaves = []
    for lid in leaf_ids_raw:
        if lid not in unique_leaves:
            unique_leaves.append(lid)

    print(f"Total source unique leaves: {len(unique_leaves)}")
    print(f"experimental_design.csv present: {exp_design_present}"
          + ("" if exp_design_present else "  -> fungicide/inoculation will be constant 0.0 for ALL leaves"))

    clean_X = []
    clean_y_d = []
    clean_y_l = []
    clean_leaf_ids = []

    for leaf_id in unique_leaves:
        mask = (leaf_ids_raw == leaf_id)
        X_leaf = X_raw[mask]
        yd_leaf = y_d_raw[mask]
        yl_leaf = y_l_raw[mask]
        n_seqs = len(X_leaf)

        if n_seqs < 2:
            continue

        plot_uid = leaf_id.rsplit("_", 1)[0]
        fungicide = 0.0
        inoculation = 0.0
        if exp_design is not None and plot_uid in exp_design.index:
            row = exp_design.loc[plot_uid]
            f_val = str(row["fungicide_treatment"]) if not isinstance(row, pd.DataFrame) else str(row["fungicide_treatment"].iloc[0])
            i_val = str(row["inoculation_treatment"]) if not isinstance(row, pd.DataFrame) else str(row["inoculation_treatment"].iloc[0])
            fungicide = 1.0 if ("Fungicide" in f_val and "No" not in f_val) else 0.0
            inoculation = 1.0 if ("Inoculation" in i_val and "No" not in i_val) else 0.0

        for s in range(n_seqs - 1):
            # Input window: observations at [t-3, t-2, t-1, t], la_tot only (index 768)
            la_tot = X_leaf[s, :, 768:769]        # (4, 1) physical leaf blade pixel area
            treatments = np.tile(np.array([fungicide, inoculation], dtype=np.float32), (4, 1))  # (4, 2)
            inp = np.concatenate([la_tot, treatments], axis=-1)  # (4, 3)

            # Target: the (s+1)-th raw window's own t+1 target -- strictly further
            # ahead than window s's own target, so it never overlaps window s's
            # input frames (same anti-leakage margin as build_clean_forecasting_datasets.py)
            target_d = yd_leaf[s + 1]
            target_l = yl_leaf[s + 1]

            clean_X.append(inp)
            clean_y_d.append(target_d)
            clean_y_l.append(target_l)
            clean_leaf_ids.append(leaf_id)

    clean_X = np.array(clean_X, dtype=np.float32)
    clean_y_d = np.array(clean_y_d, dtype=np.float32)
    clean_y_l = np.array(clean_y_l, dtype=np.float32)
    clean_leaf_ids = np.array(clean_leaf_ids)

    np.savez_compressed(
        OUT_PATH,
        X=clean_X,
        y_placl=clean_y_d,
        y_lesion_area=clean_y_l,
        leaf_ids=clean_leaf_ids
    )
    print(f"\nSaved Clean Metadata-Only Dataset -> {OUT_PATH}")
    print(f"  X shape: {clean_X.shape} (la_tot, fungicide, inoculation) | Targets: {clean_y_d.shape} | Leaves: {len(np.unique(clean_leaf_ids))}")

    run_integrity_checks(clean_X, clean_y_d, clean_y_l, clean_leaf_ids)


def run_integrity_checks(X, y_d, y_l, leaf_ids):
    print("\n" + "=" * 70)
    print("DATASET INTEGRITY & TEMPORAL LEAKAGE VERIFICATION")
    print("=" * 70)

    print(f"X shape                         : {X.shape}")
    print(f"Total Temporal Forecasting Seqs : {len(y_d)}")
    print(f"Total Unique Leaves             : {len(np.unique(leaf_ids))}")

    nan_count = np.isnan(X).sum() + np.isnan(y_d).sum() + np.isnan(y_l).sum()
    inf_count = np.isinf(X).sum() + np.isinf(y_d).sum() + np.isinf(y_l).sum()
    print(f"NaN count                       : {nan_count}")
    print(f"Inf count                       : {inf_count}")
    assert nan_count == 0 and inf_count == 0

    # Feature variance check -- flag any constant (dead) columns explicitly
    print("\n" + "-" * 50)
    print("FEATURE VARIANCE CHECK")
    print("-" * 50)
    names = ["la_tot", "fungicide_treatment", "inoculation_treatment"]
    for i, name in enumerate(names):
        col = X[:, -1, i]
        print(f"{name:<24} min={col.min():.4f} max={col.max():.4f} std={col.std():.4f}"
              + ("  <-- CONSTANT (no signal)" if col.std() == 0 else ""))

    # Correlation with future targets -- the leakage check that flagged the
    # original 23-feature metadata set; should be much lower here.
    print("\n" + "-" * 50)
    print("LAST-TIMESTEP FEATURE vs FUTURE TARGET CORRELATION")
    print("-" * 50)
    last_t = X[:, -1, :]
    for i, name in enumerate(names):
        if last_t[:, i].std() == 0:
            print(f"{name:<24} corr undefined (constant column)")
            continue
        c_d = np.corrcoef(last_t[:, i], y_d)[0, 1]
        c_l = np.corrcoef(last_t[:, i], y_l)[0, 1]
        print(f"{name:<24} corr_disease(t+1)={c_d:.3f}  corr_lesion(t+1)={c_l:.3f}")

    # Leaf-level leakage-free split verification (same protocol as elsewhere)
    print("\n" + "-" * 50)
    print("LEAF-LEVEL LEAKAGE-FREE SPLIT VERIFICATION")
    print("-" * 50)
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss_1.split(X, y_d, groups=leaf_ids))
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    temp_rel_val, temp_rel_test = next(gss_2.split(temp_idx, y_d[temp_idx], groups=leaf_ids[temp_idx]))
    val_idx = temp_idx[temp_rel_val]
    test_idx = temp_idx[temp_rel_test]

    train_leaves = set(leaf_ids[train_idx])
    val_leaves = set(leaf_ids[val_idx])
    test_leaves = set(leaf_ids[test_idx])

    print(f"Train Leaves : {len(train_leaves)} ({len(train_idx)} sequences)")
    print(f"Val Leaves   : {len(val_leaves)} ({len(val_idx)} sequences)")
    print(f"Test Leaves  : {len(test_leaves)} ({len(test_idx)} sequences)")

    assert train_leaves.isdisjoint(val_leaves)
    assert train_leaves.isdisjoint(test_leaves)
    assert val_leaves.isdisjoint(test_leaves)
    print("Leakage-Free Leaf Disjoint Split: VERIFIED PASS (Zero overlap)")
    print("=" * 70)


if __name__ == "__main__":
    build_clean_metadata_dataset()
