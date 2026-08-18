import torch
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=768,
        hidden_size=32,
        num_layers=1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction