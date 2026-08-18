import os
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================

METADATA_PATH = "data/features/feature_fusion.csv"
IMAGE_INDEX_PATH = "data/image_index.csv"
FEATURE_DIR = "data/features"
OUTPUT_PATH = "data/features/multimodal_fusion.csv"

# ============================================================
# Load metadata
# ============================================================

print("Loading metadata...")

metadata = pd.read_csv(METADATA_PATH)

metadata["timestamp"] = pd.to_datetime(metadata["timestamp"])

print("Metadata shape:", metadata.shape)

# ============================================================
# Load image index
# ============================================================

print("Loading image index...")

image_index = pd.read_csv(IMAGE_INDEX_PATH)

print("Image index shape:", image_index.shape)
print("Image leaves:", image_index["leaf_UID"].nunique())

# ============================================================
# Load ViT features
# ============================================================

print("Loading ViT features...")

records = []

for _, row in image_index.iterrows():

    feature_file = os.path.join(
        FEATURE_DIR,
        row["image_name"] + ".npy"
    )

    if not os.path.exists(feature_file):
        continue

    feature = np.load(feature_file)

    if feature.shape != (768,):
        print("WARNING:", feature_file, feature.shape)
        continue

    records.append({
        "leaf_UID": row["leaf_UID"],
        "image_name": row["image_name"],
        "image_feature": feature
    })

image_features = pd.DataFrame(records)

print("ViT feature records:", len(image_features))

# ============================================================
# Expand 768-D features into columns
# ============================================================

feature_matrix = np.vstack(
    image_features["image_feature"].values
)

vit_columns = [
    f"vit_{i}"
    for i in range(768)
]

vit_df = pd.DataFrame(
    feature_matrix,
    columns=vit_columns
)

image_features = pd.concat(
    [
        image_features[
            ["leaf_UID", "image_name"]
        ].reset_index(drop=True),
        vit_df
    ],
    axis=1
)

# ============================================================
# Extract timestamp from image name
# ============================================================

image_features["timestamp"] = pd.to_datetime(
    image_features["image_name"].str[:15],
    format="%Y%m%d_%H%M%S"
)

# ============================================================
# Merge image features with metadata
# ============================================================

print("Merging image and metadata features...")

merged = pd.merge(
    image_features,
    metadata,
    on=["leaf_UID", "timestamp"],
    how="inner"
)

# ============================================================
# Remove target leakage
# ============================================================

# placl is the disease target.
# It must NOT be included as an input feature.

if "placl" in merged.columns:
    merged = merged.drop(columns=["placl"])

# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

merged.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("MULTIMODAL FEATURE FUSION COMPLETE")
print("=" * 60)

print("Shape:", merged.shape)
print("Leaves:", merged["leaf_UID"].nunique())
print("Columns:", len(merged.columns))

print("ViT features:", len(vit_columns))
print("Metadata features:", len(metadata.columns) - 4)

print()
print("Saved ->", OUTPUT_PATH)