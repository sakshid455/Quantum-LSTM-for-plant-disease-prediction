import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


DATA_PATH = "data/sequences/multimodal_temporal_sequences.npz"


# ============================================================
# Load data
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

y_disease = data["y_placl"]
y_lesion = data["y_lesion_area"]
leaf_ids = data["leaf_ids"]


# ============================================================
# Reproduce EXACT same leaf split
# ============================================================

unique_leaves = np.unique(leaf_ids)

train_leaves, temp_leaves = train_test_split(
    unique_leaves,
    test_size=0.30,
    random_state=42
)

val_leaves, test_leaves = train_test_split(
    temp_leaves,
    test_size=0.50,
    random_state=42
)


# ============================================================
# Print leaves
# ============================================================

print("=" * 70)
print("LEAF DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nTrain leaves:")
print(train_leaves)

print("\nValidation leaves:")
print(val_leaves)

print("\nTest leaves:")
print(test_leaves)


# ============================================================
# Create dataframe
# ============================================================

df = pd.DataFrame({
    "leaf_id": leaf_ids,
    "disease": y_disease,
    "lesion": y_lesion
})


# ============================================================
# Per-leaf statistics
# ============================================================

leaf_stats = (
    df
    .groupby("leaf_id")
    .agg(
        sequences=("disease", "count"),

        disease_mean=("disease", "mean"),
        disease_max=("disease", "max"),

        lesion_mean=("lesion", "mean"),
        lesion_max=("lesion", "max")
    )
    .reset_index()
)


# ============================================================
# Assign split
# ============================================================

def get_split(leaf):

    if leaf in train_leaves:
        return "Train"

    if leaf in val_leaves:
        return "Validation"

    return "Test"


leaf_stats["split"] = leaf_stats["leaf_id"].apply(get_split)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 70)
print("PER-LEAF STATISTICS")
print("=" * 70)

print(
    leaf_stats.to_string(
        index=False
    )
)


# ============================================================
# Split statistics
# ============================================================

print("\n" + "=" * 70)
print("SPLIT TARGET DISTRIBUTIONS")
print("=" * 70)


for split in ["Train", "Validation", "Test"]:

    subset = leaf_stats[
        leaf_stats["split"] == split
    ]

    print(f"\n{split}")

    print(
        "Disease mean:",
        subset["disease_mean"].mean()
    )

    print(
        "Disease max :",
        subset["disease_max"].max()
    )

    print(
        "Lesion mean :",
        subset["lesion_mean"].mean()
    )

    print(
        "Lesion max  :",
        subset["lesion_max"].max()
    )


# ============================================================
# Sequence-level statistics
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE-LEVEL DISTRIBUTION")
print("=" * 70)


for split, leaves in [
    ("Train", train_leaves),
    ("Validation", val_leaves),
    ("Test", test_leaves)
]:

    mask = np.isin(
        leaf_ids,
        leaves
    )

    disease = y_disease[mask]
    lesion = y_lesion[mask]

    print(f"\n{split}")

    print(
        "Sequences:",
        len(disease)
    )

    print(
        "Disease mean:",
        disease.mean()
    )

    print(
        "Disease std :",
        disease.std()
    )

    print(
        "Disease min :",
        disease.min()
    )

    print(
        "Disease max :",
        disease.max()
    )

    print(
        "Lesion mean:",
        lesion.mean()
    )

    print(
        "Lesion std :",
        lesion.std()
    )

    print(
        "Lesion min :",
        lesion.min()
    )

    print(
        "Lesion max :",
        lesion.max()
    )