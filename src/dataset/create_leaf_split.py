import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

INPUT = "data/sequences.csv"

TRAIN_OUT = "data/train.csv"
VAL_OUT = "data/val.csv"
TEST_OUT = "data/test.csv"


def main():

    print("=" * 70)
    print("CREATING SEVERITY-AWARE LEAF-LEVEL SPLIT")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    # --------------------------------------------------
    # Calculate one disease-severity value per leaf
    # --------------------------------------------------
    leaf_stats = (
        df.groupby("leaf_UID")["target_placl"]
        .mean()
        .reset_index()
        .rename(columns={"target_placl": "disease_mean"})
    )

    # --------------------------------------------------
    # Create severity groups
    #
    # Low    : < 0.02
    # Medium : 0.02 - < 0.10
    # High   : >= 0.10
    # --------------------------------------------------
    def severity_group(x):

        if x < 0.02:
            return "low"

        elif x < 0.10:
            return "medium"

        else:
            return "high"

    leaf_stats["severity_group"] = (
        leaf_stats["disease_mean"]
        .apply(severity_group)
    )

    print("\nLeaf severity groups:")
    print(
        leaf_stats[
            ["leaf_UID", "disease_mean", "severity_group"]
        ]
        .sort_values("disease_mean")
        .to_string(index=False)
    )

    print("\nSeverity counts:")
    print(leaf_stats["severity_group"].value_counts())

    # --------------------------------------------------
    # First split:
    #
    # 5 leaves -> TEST
    # 25 leaves -> remaining
    #
    # Stratified by severity
    # --------------------------------------------------
    train_val_leaves, test_leaves = train_test_split(
        leaf_stats,
        test_size=5,
        random_state=SEED,
        stratify=leaf_stats["severity_group"],
    )

    # --------------------------------------------------
    # Second split:
    #
    # 4 leaves -> VALIDATION
    # 21 leaves -> TRAIN
    #
    # Stratified by severity
    # --------------------------------------------------
    train_leaves, val_leaves = train_test_split(
        train_val_leaves,
        test_size=4,
        random_state=SEED,
        stratify=train_val_leaves["severity_group"],
    )

    train_ids = set(train_leaves["leaf_UID"])
    val_ids = set(val_leaves["leaf_UID"])
    test_ids = set(test_leaves["leaf_UID"])

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert len(train_ids) == 21
    assert len(val_ids) == 4
    assert len(test_ids) == 5

    # --------------------------------------------------
    # Build CSV files
    # --------------------------------------------------
    train = df[df["leaf_UID"].isin(train_ids)].copy()
    val = df[df["leaf_UID"].isin(val_ids)].copy()
    test = df[df["leaf_UID"].isin(test_ids)].copy()

    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    # --------------------------------------------------
    # Print final split
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    print("\nTRAIN")
    print("Leaves:", len(train_ids))
    print("Samples:", len(train))
    print("IDs:")
    for x in sorted(train_ids):
        print(" ", x)

    print("\nVALIDATION")
    print("Leaves:", len(val_ids))
    print("Samples:", len(val))
    print("IDs:")
    for x in sorted(val_ids):
        print(" ", x)

    print("\nTEST")
    print("Leaves:", len(test_ids))
    print("Samples:", len(test))
    print("IDs:")
    for x in sorted(test_ids):
        print(" ", x)

    # --------------------------------------------------
    # Distribution check
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("DISEASE SEVERITY DISTRIBUTION")
    print("=" * 70)

    for name, ids in [
        ("TRAIN", train_ids),
        ("VALIDATION", val_ids),
        ("TEST", test_ids),
    ]:

        stats = leaf_stats[
            leaf_stats["leaf_UID"].isin(ids)
        ]

        print(f"\n{name}")

        print(
            "Leaf disease mean:",
            round(stats["disease_mean"].mean(), 6)
        )

        print(
            "Leaf disease min :",
            round(stats["disease_mean"].min(), 6)
        )

        print(
            "Leaf disease max :",
            round(stats["disease_mean"].max(), 6)
        )

        print(
            "Low:",
            (stats["severity_group"] == "low").sum(),
            "| Medium:",
            (stats["severity_group"] == "medium").sum(),
            "| High:",
            (stats["severity_group"] == "high").sum(),
        )

    print("\n" + "=" * 70)
    print("OVERLAP CHECK")
    print("=" * 70)

    print("Train ∩ Validation:", train_ids & val_ids)
    print("Train ∩ Test:", train_ids & test_ids)
    print("Validation ∩ Test:", val_ids & test_ids)

    print("\nSaved:")
    print(TRAIN_OUT)
    print(VAL_OUT)
    print(TEST_OUT)


if __name__ == "__main__":
    main()