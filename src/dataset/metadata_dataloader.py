import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/sequences/metadata_temporal_sequences.npz"


class MetadataTemporalDataset(Dataset):

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


def create_metadata_loaders(
    batch_size=8,
    random_state=42
):

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    y_disease = data["y_placl"]
    y_lesion = data["y_lesion_area"]
    leaf_ids = data["leaf_ids"]

    print("=" * 60)
    print("LEAKAGE-FREE METADATA LSTM DATA SPLIT")
    print("=" * 60)

    print("Original X shape:", X.shape)
    print("Disease target shape:", y_disease.shape)
    print("Lesion target shape:", y_lesion.shape)

    unique_leaves = np.unique(leaf_ids)

    print("Unique leaves:", len(unique_leaves))

    # --------------------------------------------------------
    # First split: 70% train / 30% temporary
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

    # --------------------------------------------------------
    # Second split: 15% validation / 15% test
    # --------------------------------------------------------

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state
    )

    val_relative, test_relative = next(
        gss_2.split(
            temp_idx,
            y_disease[temp_idx],
            groups=leaf_ids[temp_idx]
        )
    )

    val_idx = temp_idx[val_relative]
    test_idx = temp_idx[test_relative]

    # --------------------------------------------------------
    # Verify no leaf overlap
    # --------------------------------------------------------

    train_leaves = set(leaf_ids[train_idx])
    val_leaves = set(leaf_ids[val_idx])
    test_leaves = set(leaf_ids[test_idx])

    assert train_leaves.isdisjoint(val_leaves)
    assert train_leaves.isdisjoint(test_leaves)
    assert val_leaves.isdisjoint(test_leaves)

    print()
    print("Leaf split verified: NO overlap.")

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    disease_train = y_disease[train_idx]
    disease_val = y_disease[val_idx]
    disease_test = y_disease[test_idx]

    lesion_train = y_lesion[train_idx]
    lesion_val = y_lesion[val_idx]
    lesion_test = y_lesion[test_idx]

    # --------------------------------------------------------
    # Normalize metadata using TRAIN ONLY
    # --------------------------------------------------------

    n_features = X_train.shape[-1]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)

    scaler.fit(X_train_2d)

    X_train = scaler.transform(
        X_train_2d
    ).reshape(X_train.shape)

    X_val = scaler.transform(
        X_val.reshape(-1, n_features)
    ).reshape(X_val.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)

    # --------------------------------------------------------
    # Normalize lesion target using TRAIN ONLY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = MetadataTemporalDataset(
        X_train,
        disease_train,
        lesion_train
    )

    val_dataset = MetadataTemporalDataset(
        X_val,
        disease_val,
        lesion_val
    )

    test_dataset = MetadataTemporalDataset(
        X_test,
        disease_test,
        lesion_test
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    print()
    print("TRAIN leaves      :", len(train_leaves))
    print("VALIDATION leaves :", len(val_leaves))
    print("TEST leaves       :", len(test_leaves))

    print()
    print("TRAIN sequences      :", len(train_dataset))
    print("VALIDATION sequences :", len(val_dataset))
    print("TEST sequences       :", len(test_dataset))

    print()
    print("Metadata features:", n_features)

    print()
    print("Disease target:")
    print("  Train mean:", disease_train.mean())
    print("  Train std :", disease_train.std())
    print("  Train min :", disease_train.min())
    print("  Train max :", disease_train.max())

    print()
    print("Lesion target:")
    print("  Train mean:", lesion_mean)
    print("  Train std :", lesion_std)

    return (
        train_loader,
        val_loader,
        test_loader,
        lesion_mean,
        lesion_std
    )