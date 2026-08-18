import pandas as pd
import pyreadr

print("=" * 60)
print("Loading Leaf Metadata...")
print("=" * 60)

leaf_df = pyreadr.read_r("data/processed/res/data_leaf.rds")[None]

print("Leaf records:", len(leaf_df))

print("\nLoading Image Index...")

image_df = pd.read_csv("data/image_index.csv")

print("Images:", len(image_df))

# -----------------------------------------
# Create common key
# -----------------------------------------

leaf_df["image_name"] = leaf_df["file_id"].str.replace(".txt", "", regex=False)

# -----------------------------------------
# Merge
# -----------------------------------------

merged = leaf_df.merge(
    image_df,
    on="image_name",
    how="inner"
)

print("\nMerged Records:", len(merged))

print("\nColumns")

print(merged.columns.tolist())

print("\nSample")

print(
    merged[
        [
            "leaf_UID_x",
            "timestamp",
            "placl",
            "la_damaged",
            "image_path"
        ]
    ].head()
)

merged.to_csv(
    "data/leaf_with_images.csv",
    index=False
)

print("\nSaved -> data/leaf_with_images.csv")