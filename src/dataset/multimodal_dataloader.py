import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/sequences/multimodal_temporal_sequences.npz"


class MultimodalTemporalDataset(Dataset):

    def __init__(self, X, y_disease, y_lesion):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y_disease = torch.tensor(
            y_disease,
            dtype=torch.float32
        ).unsqueeze(1)

        self.y_lesion = torch.tensor(
            y_lesion,
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y_disease[index],
            self.y_lesion[index]
        )


def create_multimodal_loaders(
    batch_size=8,
    random_state=42
):

    # ========================================================
    # Load data
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    y_disease = data["y_placl"]
    y_lesion = data["y_lesion_area"]
    leaf_ids = data["leaf_ids"]

    unique_leaves = np.unique(
        leaf_ids
    )


    # ========================================================
    # Calculate leaf-level disease severity
    # ========================================================

    leaf_disease_mean = {}

    for leaf in unique_leaves:

        mask = leaf_ids == leaf

        leaf_disease_mean[leaf] = np.mean(
            y_disease[mask]
        )


    # ========================================================
    # Create severity groups
    #
    # We use leaf-level statistics so sequences from the
    # same leaf can never cross train/val/test.
    # ========================================================

    severity_values = np.array([
        leaf_disease_mean[leaf]
        for leaf in unique_leaves
    ])


    # Three approximately balanced severity groups.
    #
    # qcut-like percentile boundaries are calculated manually
    # so that we can work directly with leaf IDs.

    q1 = np.percentile(
        severity_values,
        33.333333
    )

    q2 = np.percentile(
        severity_values,
        66.666667
    )


    severity_labels = []

    for leaf in unique_leaves:

        value = leaf_disease_mean[leaf]

        if value <= q1:
            severity_labels.append(0)

        elif value <= q2:
            severity_labels.append(1)

        else:
            severity_labels.append(2)


    severity_labels = np.array(
        severity_labels
    )


    # ========================================================
    # First split:
    #
    # 70% train
    # 30% temporary
    #
    # Stratified by leaf severity
    # ========================================================

    train_leaves, temp_leaves = train_test_split(
        unique_leaves,
        test_size=0.30,
        random_state=random_state,
        stratify=severity_labels
    )


    # ========================================================
    # Severity labels for temporary leaves
    # ========================================================

    severity_lookup = {
        leaf: severity
        for leaf, severity
        in zip(unique_leaves, severity_labels)
    }


    temp_severity = np.array([
        severity_lookup[leaf]
        for leaf in temp_leaves
    ])


    # ========================================================
    # Second split:
    #
    # 15% validation
    # 15% test
    #
    # Stratified by severity
    # ========================================================

    val_leaves, test_leaves = train_test_split(
        temp_leaves,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_severity
    )


    # ========================================================
    # Masks
    # ========================================================

    train_mask = np.isin(
        leaf_ids,
        train_leaves
    )

    val_mask = np.isin(
        leaf_ids,
        val_leaves
    )

    test_mask = np.isin(
        leaf_ids,
        test_leaves
    )


    # ========================================================
    # Split X
    # ========================================================

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]


    # ========================================================
    # Split disease targets
    # ========================================================

    disease_train = y_disease[train_mask]
    disease_val = y_disease[val_mask]
    disease_test = y_disease[test_mask]


    # ========================================================
    # Split lesion targets
    # ========================================================

    lesion_train = y_lesion[train_mask]
    lesion_val = y_lesion[val_mask]
    lesion_test = y_lesion[test_mask]


    # ========================================================
    # Normalize input features
    #
    # IMPORTANT:
    # scaler is fitted ONLY on training data.
    # ========================================================

    n_features = X_train.shape[-1]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(
        -1,
        n_features
    )

    scaler.fit(
        X_train_2d
    )


    X_train = scaler.transform(
        X_train_2d
    ).reshape(
        X_train.shape
    )


    X_val = scaler.transform(
        X_val.reshape(
            -1,
            n_features
        )
    ).reshape(
        X_val.shape
    )


    X_test = scaler.transform(
        X_test.reshape(
            -1,
            n_features
        )
    ).reshape(
        X_test.shape
    )


    # ========================================================
    # Normalize lesion target
    #
    # IMPORTANT:
    # mean/std calculated ONLY from training data.
    # ========================================================

    lesion_mean = lesion_train.mean()

    lesion_std = lesion_train.std()

    if lesion_std == 0:

        lesion_std = 1.0


    lesion_train = (
        lesion_train - lesion_mean
    ) / lesion_std


    lesion_val = (
        lesion_val - lesion_mean
    ) / lesion_std


    lesion_test = (
        lesion_test - lesion_mean
    ) / lesion_std


    # ========================================================
    # Create datasets
    # ========================================================

    train_dataset = MultimodalTemporalDataset(
        X_train,
        disease_train,
        lesion_train
    )


    val_dataset = MultimodalTemporalDataset(
        X_val,
        disease_val,
        lesion_val
    )


    test_dataset = MultimodalTemporalDataset(
        X_test,
        disease_test,
        lesion_test
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # Print split information
    # ========================================================

    print("=" * 60)
    print("SEVERITY-STRATIFIED MULTIMODAL DATA SPLIT")
    print("=" * 60)

    print(
        "Total leaves :",
        len(unique_leaves)
    )

    print(
        "Train leaves :",
        len(train_leaves)
    )

    print(
        "Val leaves   :",
        len(val_leaves)
    )

    print(
        "Test leaves  :",
        len(test_leaves)
    )


    print()

    print(
        "Train sequences:",
        len(train_dataset)
    )

    print(
        "Val sequences  :",
        len(val_dataset)
    )

    print(
        "Test sequences :",
        len(test_dataset)
    )


    # ========================================================
    # Disease statistics
    # ========================================================

    print()

    print("Disease target:")

    print(
        "  Train mean:",
        disease_train.mean()
    )

    print(
        "  Train std :",
        disease_train.std()
    )

    print(
        "  Train min :",
        disease_train.min()
    )

    print(
        "  Train max :",
        disease_train.max()
    )


    # ========================================================
    # Lesion statistics
    # ========================================================

    print()

    print("Lesion target:")

    print(
        "  Mean:",
        lesion_mean
    )

    print(
        "  Std :",
        lesion_std
    )


    # ========================================================
    # Print actual leaves
    # ========================================================

    print()

    print("TRAIN LEAVES:")

    for leaf in sorted(train_leaves):
        print(" ", leaf)


    print()

    print("VALIDATION LEAVES:")

    for leaf in sorted(val_leaves):
        print(" ", leaf)


    print()

    print("TEST LEAVES:")

    for leaf in sorted(test_leaves):
        print(" ", leaf)


    # ========================================================
    # Return
    # ========================================================

    return (
        train_loader,
        val_loader,
        test_loader,
        lesion_mean,
        lesion_std
    )