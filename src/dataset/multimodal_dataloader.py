import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
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

    print("=" * 60)
    print("LEAKAGE-FREE MULTIMODAL DATA SPLIT")
    print("=" * 60)

    print("Original X shape:", X.shape)
    print("Disease target shape:", y_disease.shape)
    print("Lesion target shape:", y_lesion.shape)


    # ========================================================
    # Group split by leaf
    #
    # IMPORTANT:
    # No target values are used to construct the split.
    # Every leaf belongs to exactly one partition.
    # ========================================================

    unique_leaves = np.unique(leaf_ids)

    print(
        "Unique leaves:",
        len(unique_leaves)
    )


    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    gss_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        gss_1.split(
            X,
            y_disease,
            groups=leaf_ids
        )
    )


    train_leaves = np.unique(
        leaf_ids[train_idx]
    )

    temp_leaves = np.unique(
        leaf_ids[temp_idx]
    )


    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state
    )

    temp_relative_train, temp_relative_test = next(
        gss_2.split(
            temp_idx,
            y_disease[temp_idx],
            groups=leaf_ids[temp_idx]
        )
    )

    val_idx = temp_idx[temp_relative_train]
    test_idx = temp_idx[temp_relative_test]


    val_leaves = np.unique(
        leaf_ids[val_idx]
    )

    test_leaves = np.unique(
        leaf_ids[test_idx]
    )


    # ========================================================
    # Safety checks
    # ========================================================

    train_leaf_set = set(train_leaves)
    val_leaf_set = set(val_leaves)
    test_leaf_set = set(test_leaves)

    assert train_leaf_set.isdisjoint(
        val_leaf_set
    )

    assert train_leaf_set.isdisjoint(
        test_leaf_set
    )

    assert val_leaf_set.isdisjoint(
        test_leaf_set
    )

    print("\nLeaf split verified: NO overlap.")


    # ========================================================
    # Split data
    # ========================================================

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]


    disease_train = y_disease[train_idx]
    disease_val = y_disease[val_idx]
    disease_test = y_disease[test_idx]


    lesion_train = y_lesion[train_idx]
    lesion_val = y_lesion[val_idx]
    lesion_test = y_lesion[test_idx]


    # ========================================================
    # Normalize input features
    #
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    n_features = X_train.shape[-1]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(
        -1,
        n_features
    )

    X_val_2d = X_val.reshape(
        -1,
        n_features
    )

    X_test_2d = X_test.reshape(
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
        X_val_2d
    ).reshape(
        X_val.shape
    )

    X_test = scaler.transform(
        X_test_2d
    ).reshape(
        X_test.shape
    )


    # ========================================================
    # Normalize lesion target
    #
    # FIT ONLY ON TRAINING TARGETS
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

    print()
    print("TRAIN leaves      :", len(train_leaves))
    print("VALIDATION leaves :", len(val_leaves))
    print("TEST leaves       :", len(test_leaves))

    print()

    print("TRAIN sequences      :", len(train_dataset))
    print("VALIDATION sequences :", len(val_dataset))
    print("TEST sequences       :", len(test_dataset))


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
        "  Train mean:",
        lesion_mean
    )

    print(
        "  Train std :",
        lesion_std
    )


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