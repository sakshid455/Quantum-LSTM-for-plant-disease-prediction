import os
import pandas as pd
import pyreadr


# ============================================================
# Paths
# ============================================================

LEAF_PATH = "data/processed/res/data_leaf.rds"
LESION_PATH = "data/processed/res/data_lesions.rds"

OUTPUT_DIR = "data/features"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_fusion.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load RDS files
# ============================================================

print("Loading leaf data...")

leaf_result = pyreadr.read_r(LEAF_PATH)
leaf_df = list(leaf_result.values())[0]

print("Leaf shape:", leaf_df.shape)


print("Loading lesion data...")

lesion_result = pyreadr.read_r(LESION_PATH)
lesion_df = list(lesion_result.values())[0]

print("Lesion shape:", lesion_df.shape)


# ============================================================
# Select leaf features
# ============================================================

leaf_columns = [
    "plot_UID",
    "leaf_UID",
    "timestamp",
    "la_tot",
    "la_damaged",
    "la_healthy",
    "la_damaged_f",
    "la_healthy_f",
    "la_insect",
    "n_pycn",
    "n_rust",
    "n_lesion",
    "placl",
    "pycn_density",
    "rust_density",
    "mean_dist",
    "std_dist",
    "cv_dist",
]

leaf = leaf_df[leaf_columns].copy()


# ============================================================
# Aggregate lesion data
# ============================================================

print("Aggregating lesion data...")

lesion_grouped = (
    lesion_df
    .groupby(
        ["plot_UID", "leaf_UID", "timestamp"],
        as_index=False
    )
    .agg(
        total_lesion_area=("area", "sum"),
        mean_lesion_area=("area", "mean"),
        max_lesion_area=("area", "max"),

        lesion_count=("lesion_UID", "count"),

        mean_lesion_perimeter=("perimeter", "mean"),
        mean_lesion_solidity=("solidity", "mean"),

        total_lesion_pycn=("n_pycn", "sum"),

        mean_lesion_pycn_density=(
            "pycn_density_lesion",
            "mean"
        ),

        mean_lesion_rust_density=(
            "rust_density_lesion",
            "mean"
        ),
    )
)


print(
    "Aggregated lesion shape:",
    lesion_grouped.shape
)


# ============================================================
# Merge leaf + lesion features
# ============================================================

print("Merging leaf and lesion features...")

fusion = pd.merge(
    leaf,
    lesion_grouped,
    on=[
        "plot_UID",
        "leaf_UID",
        "timestamp"
    ],
    how="left"
)


# ============================================================
# Sort temporally
# ============================================================

fusion["timestamp"] = pd.to_datetime(
    fusion["timestamp"]
)

fusion = fusion.sort_values(
    ["leaf_UID", "timestamp"]
).reset_index(drop=True)


# ============================================================
# Fill missing lesion values
# ============================================================

lesion_columns = [
    "total_lesion_area",
    "mean_lesion_area",
    "max_lesion_area",
    "lesion_count",
    "mean_lesion_perimeter",
    "mean_lesion_solidity",
    "total_lesion_pycn",
    "mean_lesion_pycn_density",
    "mean_lesion_rust_density",
]

fusion[lesion_columns] = fusion[
    lesion_columns
].fillna(0)


# ============================================================
# Save
# ============================================================

fusion.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("FEATURE FUSION DATASET CREATED")
print("=" * 60)

print("Shape:", fusion.shape)
print("Saved:", OUTPUT_FILE)

print()
print("Columns:")
print(fusion.columns.tolist())

print()
print("First rows:")
print(fusion.head().to_string())