import torch

from src.models.baseline_models import (
    RNNModel,
    LSTMModel,
    GRUModel
)

X = torch.randn(2, 4, 768)

models = [
    RNNModel(),
    LSTMModel(),
    GRUModel()
]

for model in models:
    output = model(X)
    print(
        model.__class__.__name__,
        "Input:", X.shape,
        "Output:", output.shape
    )