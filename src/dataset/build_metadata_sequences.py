import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "data/features/multimodal_fusion.csv"
OUTPUT_PATH = "data/sequences/metadata_temporal_sequences.npz"

SEQUENCE_LENGTH = 4


# ============================================================
# Load dataset
# ============================================================

print("Loading multimodal fusion dataset...")

df = pd.read_csv(INPUT_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset shape:", df.shape)


# ============================================================
# Metadata features ONLY
# ============================================================

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

print("Metadata features:", len(metadata_columns))

assert len(metadata_columns) == 23

print("Total input features:", len(metadata_columns))


# ============================================================
# Clean numeric data
# ============================================================

df[metadata_columns] = df[metadata_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df[metadata_columns] = df[metadata_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

df[metadata_columns] = df[metadata_columns].fillna(0)


# ============================================================
# Load disease target
# ============================================================

print()
print("Loading disease targets...")

disease_path = "data/features/feature_fusion.csv"

disease_df = pd.read_csv(disease_path)

disease_df["timestamp"] = pd.to_datetime(
    disease_df["timestamp"]
)

print(
    "Disease target rows:",
    len(disease_df)
)


# ============================================================
# Merge disease target
# ============================================================

target_columns = [
    "leaf_UID",
    "timestamp",
    "placl"
]

target_map = disease_df[target_columns].drop_duplicates(
    subset=["leaf_UID", "timestamp"]
)

print("Merging disease targets...")

df = df.merge(
    target_map,
    on=["leaf_UID", "timestamp"],
    how="left"
)

missing_targets = df["placl"].isna().sum()

print(
    "Missing disease targets after merge:",
    missing_targets
)

assert missing_targets == 0


# ============================================================
# Lesion target
# ============================================================

assert "total_lesion_area" in df.columns


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
y_disease = []
y_lesion_area = []
leaf_ids = []
target_timestamps = []

print()
print("Building metadata temporal sequences...")
print(
    "Sequence length:",
    SEQUENCE_LENGTH
)


for leaf_id, group in df.groupby("leaf_UID"):

    group = group.sort_values("timestamp").reset_index(
        drop=True
    )

    if len(group) <= SEQUENCE_LENGTH:
        continue

    features = group[
        metadata_columns
    ].values.astype(np.float32)

    disease_values = group[
        "placl"
    ].values.astype(np.float32)

    lesion_values = group[
        "total_lesion_area"
    ].values.astype(np.float32)


    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for i in range(
        len(group) - SEQUENCE_LENGTH
    ):

        X.append(
            features[
                i:i + SEQUENCE_LENGTH
            ]
        )

        target_index = (
            i + SEQUENCE_LENGTH
        )

        y_disease.append(
            disease_values[target_index]
        )

        y_lesion_area.append(
            lesion_values[target_index]
        )

        leaf_ids.append(
            leaf_id
        )

        target_timestamps.append(
            group.iloc[target_index]["timestamp"]
        )


# ============================================================
# Convert to NumPy
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y_disease = np.asarray(
    y_disease,
    dtype=np.float32
)

y_lesion_area = np.asarray(
    y_lesion_area,
    dtype=np.float32
)

leaf_ids = np.asarray(
    leaf_ids
)

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
    y_placl=y_disease,
    y_lesion_area=y_lesion_area,
    leaf_ids=leaf_ids,
    target_timestamps=target_timestamps,
    feature_names=np.asarray(metadata_columns)
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("METADATA TEMPORAL DATASET CREATED")
print("=" * 60)

print(
    "X shape:",
    X.shape
)

print(
    "Disease target shape:",
    y_disease.shape
)

print(
    "Lesion target shape:",
    y_lesion_area.shape
)

print(
    "Number of sequences:",
    len(X)
)

print(
    "Unique leaves:",
    len(set(leaf_ids))
)

print(
    "Input features:",
    len(metadata_columns)
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)


print()
print("Disease target statistics:")

print(
    "  Mean:",
    y_disease.mean()
)

print(
    "  Std :",
    y_disease.std()
)

print(
    "  Min :",
    y_disease.min()
)

print(
    "  Max :",
    y_disease.max()
)


print()
print("Lesion area statistics:")

print(
    "  Mean:",
    y_lesion_area.mean()
)

print(
    "  Std :",
    y_lesion_area.std()
)

print(
    "  Min :",
    y_lesion_area.min()
)

print(
    "  Max :",
    y_lesion_area.max()
)

print(
    "  Zero targets:",
    np.sum(y_lesion_area == 0),
    f"({100 * np.mean(y_lesion_area == 0):.2f}%)"
)


print()
print(
    "Saved ->",
    OUTPUT_PATH
)