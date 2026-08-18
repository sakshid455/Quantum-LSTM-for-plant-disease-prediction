import numpy as np

from src.dataset.multimodal_dataloader import DATA_PATH


data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y_disease = data["y_placl"]
y_lesion = data["y_lesion_area"]
leaf_ids = data["leaf_ids"]

unique_leaves = np.unique(leaf_ids)


print("=" * 70)
print("LEAF-LEVEL TARGET DISTRIBUTION")
print("=" * 70)


for leaf in unique_leaves:

    mask = leaf_ids == leaf

    disease = y_disease[mask]
    lesion = y_lesion[mask]

    print(
        f"\nLeaf {leaf}"
    )

    print(
        f"  Sequences : {len(disease)}"
    )

    print(
        f"  Disease   : "
        f"mean={disease.mean():.6f}, "
        f"std={disease.std():.6f}, "
        f"min={disease.min():.6f}, "
        f"max={disease.max():.6f}"
    )

    print(
        f"  Lesion    : "
        f"mean={lesion.mean():.2f}, "
        f"std={lesion.std():.2f}, "
        f"min={lesion.min():.2f}, "
        f"max={lesion.max():.2f}"
    )