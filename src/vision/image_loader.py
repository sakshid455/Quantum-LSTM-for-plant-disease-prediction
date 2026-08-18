from PIL import Image
from torchvision import transforms

IMAGE_PATH = r"data/processed/ts/ESWW0070020_1/overlay/20230531_094332_ESWW0070020_1.png"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")

tensor = transform(image)

print("Tensor Shape:", tensor.shape)