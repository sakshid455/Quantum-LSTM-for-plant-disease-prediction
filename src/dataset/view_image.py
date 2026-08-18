import cv2
import matplotlib.pyplot as plt

image_path = r"data/processed/ts/ESWW0070020_1/overlay/20230531_094332_ESWW0070020_1.png"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Shape:", img.shape)

plt.imshow(img)
plt.title("Overlay Image")
plt.axis("off")
plt.show()