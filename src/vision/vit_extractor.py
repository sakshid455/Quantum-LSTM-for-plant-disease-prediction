import torch
import timm

print("=" * 50)
print("Loading Vision Transformer...")
print("=" * 50)

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0
)

model.eval()

print(model)

print("\nFeature Dimension:")

dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    feature = model(dummy)

print(feature.shape)