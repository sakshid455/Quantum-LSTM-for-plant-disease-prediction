import pyreadr
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

LEAF_RDS = BASE_DIR / "data" / "processed" / "res" / "data_leaf.rds"
LESION_RDS = BASE_DIR / "data" / "processed" / "res" / "data_lesions.rds"

leaf_df = pyreadr.read_r(str(LEAF_RDS))[None]
lesion_df = pyreadr.read_r(str(LESION_RDS))[None]

print("="*60)
print("LEAF DATASET SUMMARY")
print("="*60)

print("\nShape:", leaf_df.shape)

print("\nMissing Values:")
print(leaf_df.isnull().sum())

print("\nUnique Leaves:")
print(leaf_df["leaf_UID"].nunique())

print("\nUnique Plots:")
print(leaf_df["plot_UID"].nunique())

print("\nDate Range:")
print(leaf_df["date"].min(), "->", leaf_df["date"].max())

print("\nAverage Images per Leaf:")
print(round(leaf_df.groupby("leaf_UID").size().mean(),2))

print("\nTarget Statistics (PLACL):")
print(leaf_df["placl"].describe())

print("\nTarget Statistics (Damaged Area):")
print(leaf_df["la_damaged"].describe())


print("\n"+"="*60)
print("LESION DATASET SUMMARY")
print("="*60)

print("\nShape:", lesion_df.shape)

print("\nUnique Lesions:")
print(lesion_df["lesion_UID"].nunique())

print("\nAverage Lesions per Leaf:")
print(round(lesion_df.groupby("leaf_UID").size().mean(),2))

print("\nLesion Area Statistics:")
print(lesion_df["area"].describe())

print("\nMaximum Lesion Area:")
print(lesion_df["area"].max())

print("\nMinimum Lesion Area:")
print(lesion_df["area"].min())