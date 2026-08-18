import os
import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 4

# ==========================================
# Load metadata
# ==========================================

df = pd.read_csv("data/leaf_with_images.csv")

feature_dir = "data/features"

feature_files = {
    f.replace(".npy", "")
    for f in os.listdir(feature_dir)
}

df["feature_name"] = df["image_name"].str.replace(".png", "", regex=False)

df = df[df["feature_name"].isin(feature_files)]

df = df.sort_values(["leaf_UID_x", "timestamp"])

# ==========================================
# Build sequences
# ==========================================

X = []
y = []

for leaf_id, group in df.groupby("leaf_UID_x"):

    group = group.reset_index(drop=True)

    if len(group) <= SEQUENCE_LENGTH:
        continue

    for i in range(len(group) - SEQUENCE_LENGTH):

        features = []

        for j in range(SEQUENCE_LENGTH):

            feature_path = os.path.join(
                feature_dir,
                group.loc[i+j, "feature_name"] + ".npy"
            )

            feature = np.load(feature_path)

            features.append(feature)

        X.append(np.array(features))

        y.append(group.loc[i+SEQUENCE_LENGTH, "placl"])

X = np.array(X)
y = np.array(y)

print("="*50)
print("Dataset Built")
print("="*50)

print("Input Shape :", X.shape)
print("Target Shape:", y.shape)

print("\nExample Target:", y[0])

# ==========================================
# Save Dataset
# ==========================================

os.makedirs("data/sequences", exist_ok=True)

np.save("data/sequences/X.npy", X)
np.save("data/sequences/y.npy", y)

print("\nDataset saved successfully!")

print("Saved:")
print("data/sequences/X.npy")
print("data/sequences/y.npy")