import torch

from src.quantum.lstm_model import LSTMModel
from src.baselines.gru_model import GRUModel
from src.quantum.qlstm_model import QLSTMModel


INPUT_SIZE = 791
HIDDEN_SIZE = 32


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_all_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


models = {
    "Classical LSTM": LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    ),

    "Classical GRU": GRUModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    ),

    "QLSTM": QLSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    )
}


print("=" * 70)
print("MODEL PARAMETER COUNT")
print("=" * 70)

for name, model in models.items():

    trainable = count_parameters(model)
    total = count_all_parameters(model)

    print(f"\n{name}")
    print("-" * 70)
    print(f"Trainable parameters : {trainable:,}")
    print(f"Total parameters     : {total:,}")


print("\n" + "=" * 70)
print("DETAILED QLSTM PARAMETERS")
print("=" * 70)

qlstm = models["QLSTM"]

for name, parameter in qlstm.named_parameters():

    print(
        f"{name:<50} "
        f"{parameter.numel():>10,}"
    )