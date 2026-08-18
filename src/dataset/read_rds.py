import pyreadr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

LEAF_RDS = BASE_DIR / "data" / "processed" / "res" / "data_leaf.rds"
LESION_RDS = BASE_DIR / "data" / "processed" / "res" / "data_lesions.rds"

def load_leaf():
    result = pyreadr.read_r(str(LEAF_RDS))
    return result[None]

def load_lesions():
    result = pyreadr.read_r(str(LESION_RDS))
    return result[None]

if __name__ == "__main__":
    leaf_df = load_leaf()
    lesion_df = load_lesions()

    print("Leaf Shape:", leaf_df.shape)
    print("Lesion Shape:", lesion_df.shape)

    print("\nLeaf Columns:")
    print(leaf_df.columns.tolist())

    print("\nLesion Columns:")
    print(lesion_df.columns.tolist())