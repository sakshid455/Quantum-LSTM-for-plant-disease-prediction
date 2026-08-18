import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DIR = Path("data/features")
SEQUENCE_LENGTH = 3
FEATURE_DIM = 768


def load_feature(image_path):
    """
    Convert an image path into its 768-dimensional feature vector.
    """
    stem = Path(str(image_path)).stem
    feature_path = FEATURE_DIR / f"{stem}.npy"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature not found for image:\n{image_path}\n"
            f"Expected:\n{feature_path}"
        )

    feature = np.load(feature_path)

    if feature.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Unexpected feature shape for {feature_path}: "
            f"{feature.shape}"
        )

    return feature.astype(np.float32)


def build_arrays(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():

        input_images = str(row["input_images"]).split("|")

        if len(input_images) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected {SEQUENCE_LENGTH} input images, "
                f"got {len(input_images)} in {csv_path}"
            )

        features = [
            load_feature(image_path)
            for image_path in input_images
        ]

        X.append(np.stack(features))
        y.append(float(row["target_placl"]))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def save_dataset(name, X, y):
    np.save(f"data/sequences/X_{name}.npy", X)
    np.save(f"data/sequences/y_{name}.npy", y)

    print(f"\n{name.upper()}:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")


def main():

    os.makedirs("data/sequences", exist_ok=True)

    print("=" * 60)
    print("Building Train / Validation / Test Arrays")
    print("=" * 60)

    # -----------------------------
    # Train
    # -----------------------------
    print("\nBuilding training data...")

    X_train, y_train = build_arrays("data/train.csv")

    save_dataset("train", X_train, y_train)

    # -----------------------------
    # Validation
    # -----------------------------
    print("\nBuilding validation data...")

    X_val, y_val = build_arrays("data/val.csv")

    save_dataset("val", X_val, y_val)

    # -----------------------------
    # Test
    # -----------------------------
    print("\nBuilding test data...")

    X_test, y_test = build_arrays("data/test.csv")

    save_dataset("test", X_test, y_test)

    # -----------------------------
    # Verification
    # -----------------------------
    print("\n" + "=" * 60)
    print("Final Verification")
    print("=" * 60)

    print("\nTrain:")
    print("  X:", X_train.shape)
    print("  y:", y_train.shape)

    print("\nValidation:")
    print("  X:", X_val.shape)
    print("  y:", y_val.shape)

    print("\nTest:")
    print("  X:", X_test.shape)
    print("  y:", y_test.shape)

    print("\nFeature dimension:", X_train.shape[-1])
    print("Sequence length:", X_train.shape[1])

    # -----------------------------
    # Safety checks
    # -----------------------------
    assert X_train.ndim == 3
    assert X_val.ndim == 3
    assert X_test.ndim == 3

    assert X_train.shape[1:] == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert X_val.shape[1:] == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert X_test.shape[1:] == (SEQUENCE_LENGTH, FEATURE_DIM)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    print("\nAll dataset checks passed!")


if __name__ == "__main__":
    main()