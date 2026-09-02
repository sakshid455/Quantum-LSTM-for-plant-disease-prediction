
import os
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "data/features/multimodal_fusion.csv"
TARGET_PATH = "data/features/feature_fusion.csv"
OUTPUT_PATH = "data/sequences/multimodal_temporal_sequences.npz"

SEQUENCE_LENGTH = 4

# ============================================================
# Load multimodal features
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
    "mean_lesion_rust_density",
]

# ============================================================
# Verify feature counts
# ============================================================

print("ViT features:", len(vit_columns))
print("Metadata features:", len(metadata_columns))

assert len(vit_columns) == 768, (
    f"Expected 768 ViT features, found {len(vit_columns)}"
)

assert len(metadata_columns) == 23, (
    f"Expected 23 metadata features, found {len(metadata_columns)}"
)

feature_columns = vit_columns + metadata_columns

print("Total input features:", len(feature_columns))

assert len(feature_columns) == 791

# ============================================================
# Load disease target
# ============================================================

print()
print("Loading disease targets...")

target_df = pd.read_csv(TARGET_PATH)

target_df["timestamp"] = pd.to_datetime(
    target_df["timestamp"]
)

required_target_columns = [
    "leaf_UID",
    "timestamp",
    "placl",
]

missing_targets = [
    col
    for col in required_target_columns
    if col not in target_df.columns
]

assert not missing_targets, (
    f"Missing target columns: {missing_targets}"
)

target_df = target_df[
    required_target_columns
].copy()

# ------------------------------------------------------------
# Check uniqueness of target keys
# ------------------------------------------------------------

duplicate_targets = target_df.duplicated(
    subset=["leaf_UID", "timestamp"]
).sum()

assert duplicate_targets == 0, (
    "Duplicate (leaf_UID, timestamp) rows found "
    "in feature_fusion.csv"
)

print(
    "Disease target rows:",
    len(target_df)
)

# ============================================================
# Merge disease target
# ============================================================

print("Merging disease targets...")

df = df.merge(
    target_df,
    on=["leaf_UID", "timestamp"],
    how="left",
    validate="one_to_one",
)

# ============================================================
# Verify target coverage
# ============================================================

missing_placl = df["placl"].isna().sum()

print(
    "Missing disease targets after merge:",
    missing_placl
)

assert missing_placl == 0, (
    "Some multimodal rows do not have a matching "
    "placl target in feature_fusion.csv"
)

# ============================================================
# Clean numeric features
# ============================================================

df[feature_columns] = df[
    feature_columns
].apply(
    pd.to_numeric,
    errors="coerce",
)

df[feature_columns] = df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan,
)

df[feature_columns] = df[
    feature_columns
].fillna(0)

# ============================================================
# Clean targets
# ============================================================

df["placl"] = pd.to_numeric(
    df["placl"],
    errors="coerce",
)

df["total_lesion_area"] = pd.to_numeric(
    df["total_lesion_area"],
    errors="coerce",
)

df["placl"] = df["placl"].fillna(0)

df["total_lesion_area"] = (
    df["total_lesion_area"].fillna(0)
)

# ============================================================
# Sort chronologically
# ============================================================

df = df.sort_values(
    ["leaf_UID", "timestamp"]
).reset_index(drop=True)

# ============================================================
# Build temporal sequences
# ============================================================

X = []
y_placl = []
y_lesion_area = []

leaf_ids = []
target_timestamps = []

print()
print("Building multimodal temporal sequences...")
print(
    f"Sequence length: {SEQUENCE_LENGTH}"
)

for leaf_id, group in df.groupby(
    "leaf_UID",
    sort=False,
):

    group = group.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Need at least SEQUENCE_LENGTH input
    # observations + 1 future target observation.
    if len(group) <= SEQUENCE_LENGTH:
        continue

    features = group[
        feature_columns
    ].values.astype(np.float32)

    placl_values = group[
        "placl"
    ].values.astype(np.float32)

    lesion_values = group[
        "total_lesion_area"
    ].values.astype(np.float32)

    timestamps = group[
        "timestamp"
    ].values

    # --------------------------------------------------------
    # Sliding windows
    #
    # Input:
    #   i, i+1, i+2, i+3
    #
    # Target:
    #   i+4
    # --------------------------------------------------------

    for i in range(
        len(group) - SEQUENCE_LENGTH
    ):

        target_index = (
            i + SEQUENCE_LENGTH
        )

        # Four previous observations
        X.append(
            features[
                i:target_index
            ]
        )

        # Future disease severity
        y_placl.append(
            placl_values[target_index]
        )

        # Future lesion area
        y_lesion_area.append(
            lesion_values[target_index]
        )

        # Leaf identity for leakage-free splitting
        leaf_ids.append(
            leaf_id
        )

        # Timestamp of prediction target
        target_timestamps.append(
            timestamps[target_index]
        )

# ============================================================
# Convert to NumPy
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32,
)

y_placl = np.asarray(
    y_placl,
    dtype=np.float32,
)

y_lesion_area = np.asarray(
    y_lesion_area,
    dtype=np.float32,
)

leaf_ids = np.asarray(
    leaf_ids
)

target_timestamps = np.asarray(
    target_timestamps,
    dtype="datetime64[ns]",
)

# ============================================================
# Sanity checks
# ============================================================

assert X.ndim == 3

assert X.shape[1] == SEQUENCE_LENGTH

assert X.shape[2] == 791

assert len(X) == len(y_placl)

assert len(X) == len(y_lesion_area)

assert len(X) == len(leaf_ids)

assert len(X) == len(target_timestamps)

# ============================================================
# Dataset summary
# ============================================================

print()
print("=" * 60)
print("MULTIMODAL TEMPORAL DATASET CREATED")
print("=" * 60)

print("X shape:", X.shape)

print(
    "Disease target shape:",
    y_placl.shape,
)

print(
    "Lesion target shape:",
    y_lesion_area.shape,
)

print(
    "Number of sequences:",
    len(X),
)

print(
    "Unique leaves:",
    len(np.unique(leaf_ids)),
)

print(
    "Input features:",
    X.shape[2],
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH,
)

print()
print("Disease target statistics:")
print("  Mean:", float(y_placl.mean()))
print("  Std :", float(y_placl.std()))
print("  Min :", float(y_placl.min()))
print("  Max :", float(y_placl.max()))

print()
print("Lesion area statistics:")
print("  Mean:", float(y_lesion_area.mean()))
print("  Std :", float(y_lesion_area.std()))
print("  Min :", float(y_lesion_area.min()))
print("  Max :", float(y_lesion_area.max()))
print(
    "  Zero targets:",
    int(np.sum(y_lesion_area == 0)),
    f"({100 * np.mean(y_lesion_area == 0):.2f}%)",
)

# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y_placl=y_placl,
    y_lesion_area=y_lesion_area,
    leaf_ids=leaf_ids,
    target_timestamps=target_timestamps,
    feature_names=np.asarray(
        feature_columns
    ),
)

print()
print(
    "Saved ->",
    OUTPUT_PATH,
)

