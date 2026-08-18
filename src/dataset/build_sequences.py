import pandas as pd

SEQUENCE_LENGTH = 3

print("=" * 60)
print("Building Temporal Sequences")
print("=" * 60)

df = pd.read_csv("data/leaf_with_images.csv")

# Sort each leaf chronologically
df = df.sort_values(["leaf_UID_x", "timestamp"])

sequences = []

# Group by leaf
for leaf_id, group in df.groupby("leaf_UID_x"):

    group = group.reset_index(drop=True)

    if len(group) <= SEQUENCE_LENGTH:
        continue

    for i in range(len(group) - SEQUENCE_LENGTH):

        input_images = list(
            group.iloc[i:i+SEQUENCE_LENGTH]["image_path"]
        )

        target = group.iloc[i+SEQUENCE_LENGTH]["placl"]

        target_image = group.iloc[i+SEQUENCE_LENGTH]["image_path"]

        sequences.append({
            "leaf_UID": leaf_id,
            "input_images": "|".join(input_images),
            "target_image": target_image,
            "target_placl": target
        })

sequence_df = pd.DataFrame(sequences)

print()

print("Total Sequences:", len(sequence_df))

print()

print(sequence_df.head())

sequence_df.to_csv(
    "data/sequences.csv",
    index=False
)

print()

print("Saved -> data/sequences.csv")