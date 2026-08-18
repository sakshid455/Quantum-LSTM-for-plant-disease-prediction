import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/features/feature_fusion.csv"

OUTPUT_DIR = "data/sequences"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "temporal_sequences.npz"
)

SEQUENCE_LENGTH = 4


# ============================================================
# Feature columns
# ============================================================

FEATURE_COLUMNS = [
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
# Load dataset
# ============================================================

print("Loading feature fusion dataset...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print("Dataset shape:", df.shape)


# ============================================================
# Sort temporally
# ============================================================

df = df.sort_values(
    ["leaf_UID", "timestamp"]
).reset_index(drop=True)


# ============================================================
# Handle missing numerical values
# ============================================================

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# Build sequences
# ============================================================

X_sequences = []
y_placl = []
y_lesion_area = []

leaf_ids = []
target_timestamps = []


print("Building temporal sequences...")


for leaf_id, group in df.groupby("leaf_UID"):

    group = group.sort_values("timestamp")

    values = group[FEATURE_COLUMNS].values

    placl_values = group["placl"].values

    lesion_area_values = (
        group["total_lesion_area"].values
    )

    timestamps = group["timestamp"].values

    # Need at least SEQUENCE_LENGTH + 1 observations
    if len(group) <= SEQUENCE_LENGTH:
        continue

    for i in range(
        len(group) - SEQUENCE_LENGTH
    ):

        # Historical observations
        X = values[
            i:i + SEQUENCE_LENGTH
        ]

        # Next observation
        target_index = (
            i + SEQUENCE_LENGTH
        )

        target_placl = (
            placl_values[target_index]
        )

        target_lesion_area = (
            lesion_area_values[target_index]
        )

        X_sequences.append(X)

        y_placl.append(
            target_placl
        )

        y_lesion_area.append(
            target_lesion_area
        )

        leaf_ids.append(
            leaf_id
        )

        target_timestamps.append(
            timestamps[target_index]
        )


# ============================================================
# Convert to NumPy
# ============================================================

X_sequences = np.asarray(
    X_sequences,
    dtype=np.float32
)

y_placl = np.asarray(
    y_placl,
    dtype=np.float32
)

y_lesion_area = np.asarray(
    y_lesion_area,
    dtype=np.float32
)


# ============================================================
# Save
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_FILE,

    X=X_sequences,

    y_placl=y_placl,

    y_lesion_area=y_lesion_area,

    leaf_ids=np.asarray(
        leaf_ids
    ),

    target_timestamps=np.asarray(
        target_timestamps,
        dtype="datetime64[ns]"
    ),

    feature_names=np.asarray(
        FEATURE_COLUMNS
    )
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 60)
print("TEMPORAL DATASET CREATED")
print("=" * 60)

print(
    "X shape:",
    X_sequences.shape
)

print(
    "Disease target shape:",
    y_placl.shape
)

print(
    "Lesion-area target shape:",
    y_lesion_area.shape
)

print(
    "Number of sequences:",
    len(X_sequences)
)

print(
    "Number of unique leaves:",
    len(set(leaf_ids))
)

print(
    "Number of input features:",
    len(FEATURE_COLUMNS)
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print()
print(
    "Saved ->",
    OUTPUT_FILE
)