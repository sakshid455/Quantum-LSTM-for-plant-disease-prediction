import torch

from src.quantum.qlstm_model import QLSTMModel


model = QLSTMModel()

# New multimodal input:
# 8 samples, 4 timesteps, 791 features
x = torch.randn(8, 4, 791)

disease_output, lesion_output = model(x)

print("Input Shape        :", x.shape)
print("Disease Output     :", disease_output.shape)
print("Lesion Area Output :", lesion_output.shape)

print("\nDisease predictions:")
print(disease_output)

print("\nLesion predictions:")
print(lesion_output)