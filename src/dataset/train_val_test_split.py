import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load full dataset
X = np.load("data/sequences/X.npy")
y = np.load("data/sequences/y.npy")

print("Total Samples:", len(X))

# Train (70%) + Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Validation (15%) + Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print("\nTrain:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# Save files
os.makedirs("data/sequences", exist_ok=True)

np.save("data/sequences/X_train.npy", X_train)
np.save("data/sequences/y_train.npy", y_train)

np.save("data/sequences/X_val.npy", X_val)
np.save("data/sequences/y_val.npy", y_val)

np.save("data/sequences/X_test.npy", X_test)
np.save("data/sequences/y_test.npy", y_test)

print("\nDataset split saved successfully.")