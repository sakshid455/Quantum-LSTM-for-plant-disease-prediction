import os
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==========================================================
# Load image index
# ==========================================================

df = pd.read_csv("data/image_index.csv")

print(f"Images found : {len(df)}")

# ==========================================================
# Image preprocessing
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# Load ViT
# ==========================================================

print("Loading ViT...")

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0
)

model.eval()

# ==========================================================
# Output folder
# ==========================================================

os.makedirs("data/features", exist_ok=True)

# ==========================================================
# Extract Features
# ==========================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    image = Image.open(row["image_path"]).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze().numpy()

    save_path = os.path.join(
        "data/features",
        row["image_name"].replace(".png", ".npy")
    )

    np.save(save_path, feature)

print("\nFinished!")

print(f"Saved {len(df)} feature vectors.")