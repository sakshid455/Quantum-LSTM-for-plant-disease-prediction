import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=791,
        hidden_size=32,
        num_layers=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.disease_head = nn.Linear(
            hidden_size,
            1
        )

        self.lesion_head = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # x shape:
        # (batch, sequence_length, features)

        output, (h_n, c_n) = self.lstm(x)

        # Last hidden state
        h = h_n[-1]

        disease_output = self.disease_head(h)

        lesion_output = self.lesion_head(h)

        return disease_output, lesion_output