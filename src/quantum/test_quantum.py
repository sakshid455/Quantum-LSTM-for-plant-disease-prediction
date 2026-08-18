import torch
from .quantum_layer import QuantumLayer

model = QuantumLayer()

x = torch.rand(4)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
print(output)