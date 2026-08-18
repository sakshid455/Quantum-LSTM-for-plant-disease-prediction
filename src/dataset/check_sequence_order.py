import pandas as pd

df = pd.read_csv("data/leaf_with_images.csv")

# Sort by leaf and timestamp
df = df.sort_values(["leaf_UID_x", "timestamp"])

print("=" * 60)
print("FIRST 20 RECORDS")
print("=" * 60)

print(df[[
    "leaf_UID_x",
    "timestamp",
    "file_id",
    "placl"
]].head(20))

print("\n")

leaf = df["leaf_UID_x"].unique()[0]

print("=" * 60)
print(f"Timeline for {leaf}")
print("=" * 60)

sample = df[df["leaf_UID_x"] == leaf]

print(sample[[
    "timestamp",
    "file_id",
    "placl"
]])