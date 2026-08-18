import numpy as np
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


unique_leaves = np.unique(leaf_ids)


# ============================================================
# Calculate leaf-level statistics
# ============================================================

leaf_stats = []

for leaf in unique_leaves:

    mask = leaf_ids == leaf

    disease_mean = np.mean(
        y_disease[mask]
    )

    lesion_mean = np.mean(
        y_lesion[mask]
    )

    leaf_stats.append({
        "leaf": leaf,
        "disease_mean": disease_mean,
        "lesion_mean": lesion_mean
    })


# ============================================================
# Sort by disease severity
# ============================================================

leaf_stats = sorted(
    leaf_stats,
    key=lambda x: x["disease_mean"]
)


print("=" * 70)
print("ALL LEAVES SORTED BY DISEASE SEVERITY")
print("=" * 70)

for item in leaf_stats:

    print(
        f"{item['leaf']:20s} "
        f"Disease={item['disease_mean']:.6f} "
        f"Lesion={item['lesion_mean']:.2f}"
    )


# ============================================================
# Current split
# ============================================================

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
# Print split
# ============================================================

print("\n" + "=" * 70)
print("CURRENT SPLIT")
print("=" * 70)

print("\nTRAIN:")
for leaf in sorted(train_leaves):
    print(" ", leaf)

print("\nVALIDATION:")
for leaf in sorted(val_leaves):
    print(" ", leaf)

print("\nTEST:")
for leaf in sorted(test_leaves):
    print(" ", leaf)


# ============================================================
# Split statistics
# ============================================================

def split_stats(name, leaves):

    mask = np.isin(
        leaf_ids,
        leaves
    )

    print("\n" + name)

    print(
        "Sequences:",
        mask.sum()
    )

    print(
        "Disease mean:",
        y_disease[mask].mean()
    )

    print(
        "Disease std:",
        y_disease[mask].std()
    )

    print(
        "Disease min:",
        y_disease[mask].min()
    )

    print(
        "Disease max:",
        y_disease[mask].max()
    )

    print(
        "Lesion mean:",
        y_lesion[mask].mean()
    )

    print(
        "Lesion std:",
        y_lesion[mask].std()
    )

    print(
        "Lesion min:",
        y_lesion[mask].min()
    )

    print(
        "Lesion max:",
        y_lesion[mask].max()
    )


print("\n" + "=" * 70)
print("SPLIT DISTRIBUTIONS")
print("=" * 70)

split_stats(
    "TRAIN",
    train_leaves
)

split_stats(
    "VALIDATION",
    val_leaves
)

split_stats(
    "TEST",
    test_leaves
)