import torch

from src.quantum.qlstm_cell import QLSTMCell

model = QLSTMCell()

batch_size = 2

x = torch.randn(batch_size, 768)

h = torch.zeros(batch_size, 32)

c = torch.zeros(batch_size, 32)

new_h, new_c = model(x, (h, c))

print("Input :", x.shape)
print("Hidden:", new_h.shape)
print("Cell  :", new_c.shape)