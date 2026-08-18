from pathlib import Path
import pandas as pd

TS_DIR = Path("data/processed/ts")

records = []

for leaf_folder in TS_DIR.iterdir():

    overlay_dir = leaf_folder / "overlay"

    if not overlay_dir.exists():
        continue

    for img in overlay_dir.glob("*.png"):

        records.append({
            "leaf_UID": leaf_folder.name,
            "image_name": img.stem,
            "image_path": str(img.resolve())
        })

image_df = pd.DataFrame(records)

print("=" * 50)
print("Total Images:", len(image_df))
print("=" * 50)

print(image_df.head())

image_df.to_csv(
    "data/image_index.csv",
    index=False
)

print("\nSaved to data/image_index.csv")