import os
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "data/features/multimodal_fusion.csv"
OUTPUT_PATH = "data/sequences/multimodal_temporal_sequences.npz"

SEQUENCE_LENGTH = 4

# ============================================================
# Load dataset
# ============================================================

print("Loading multimodal fusion dataset...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset shape:", df.shape)

# ============================================================
# Identify ViT and metadata features
# ============================================================

vit_columns = [
    col for col in df.columns
    if col.startswith("vit_")
]

metadata_columns = [
    "la_tot",
    "la_damaged",
    "la_healthy",
    "la_damaged_f",
    "la_healthy_f",
    "la_insect",
    "n_pycn",
    "n_rust",
    "n_lesion",
    "pycn_density",
    "rust_density",
    "mean_dist",
    "std_dist",
    "cv_dist",
    "total_lesion_area",
    "mean_lesion_area",
    "max_lesion_area",
    "lesion_count",
    "mean_lesion_perimeter",
    "mean_lesion_solidity",
    "total_lesion_pycn",
    "mean_lesion_pycn_density",
    "mean_lesion_rust_density"
]

# ============================================================
# Verify features
# ============================================================

print("ViT features:", len(vit_columns))
print("Metadata features:", len(metadata_columns))

assert len(vit_columns) == 768
assert len(metadata_columns) == 23

feature_columns = vit_columns + metadata_columns

print("Total input features:", len(feature_columns))

# ============================================================
# Clean numeric data
# ============================================================

df[feature_columns] = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

df[feature_columns] = df[feature_columns].fillna(0)

# ============================================================
# Sort temporally
# ============================================================

df = df.sort_values(
    ["leaf_UID", "timestamp"]
).reset_index(drop=True)

# ============================================================
# Build sequences
# ============================================================

X = []
y_placl = []
y_lesion_area = []
leaf_ids = []
target_timestamps = []

print("Building multimodal temporal sequences...")

for leaf_id, group in df.groupby("leaf_UID"):

    group = group.sort_values("timestamp")

    if len(group) <= SEQUENCE_LENGTH:
        continue

    features = group[feature_columns].values.astype(np.float32)

    placl_values = (
        group["placl"]
        if "placl" in group.columns
        else None
    )

    # placl was removed from the fusion CSV,
    # so recover it from the original metadata.
    if placl_values is None:
        metadata_original = pd.read_csv(
            "data/features/feature_fusion.csv"
        )

        metadata_original["timestamp"] = pd.to_datetime(
            metadata_original["timestamp"]
        )

        target_map = metadata_original[
            ["leaf_UID", "timestamp", "placl"]
        ]

        group = group.merge(
            target_map,
            on=["leaf_UID", "timestamp"],
            how="left"
        )

        group = group.sort_values("timestamp")

        placl_values = group["placl"].values

    else:
        placl_values = placl_values.values

    # Lesion target
    if "total_lesion_area" in group.columns:
        lesion_values = group["total_lesion_area"].values
    else:
        lesion_values = np.zeros(len(group))

    # Create sliding windows
    for i in range(
        len(group) - SEQUENCE_LENGTH
    ):

        X.append(
            features[
                i:i + SEQUENCE_LENGTH
            ]
        )

        target_index = i + SEQUENCE_LENGTH

        y_placl.append(
            placl_values[target_index]
        )

        y_lesion_area.append(
            lesion_values[target_index]
        )

        leaf_ids.append(leaf_id)

        target_timestamps.append(
            group.iloc[target_index]["timestamp"]
        )

# ============================================================
# Convert to NumPy
# ============================================================

X = np.asarray(X, dtype=np.float32)
y_placl = np.asarray(y_placl, dtype=np.float32)
y_lesion_area = np.asarray(
    y_lesion_area,
    dtype=np.float32
)

leaf_ids = np.asarray(leaf_ids)

target_timestamps = np.asarray(
    target_timestamps,
    dtype="datetime64[ns]"
)

# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y_placl=y_placl,
    y_lesion_area=y_lesion_area,
    leaf_ids=leaf_ids,
    target_timestamps=target_timestamps,
    feature_names=np.asarray(feature_columns)
)

# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("MULTIMODAL TEMPORAL DATASET CREATED")
print("=" * 60)

print("X shape:", X.shape)
print("Disease target:", y_placl.shape)
print("Lesion target:", y_lesion_area.shape)
print("Number of sequences:", len(X))
print("Unique leaves:", len(set(leaf_ids)))
print("Input features:", len(feature_columns))
print("Sequence length:", SEQUENCE_LENGTH)

print()
print("Saved ->", OUTPUT_PATH)