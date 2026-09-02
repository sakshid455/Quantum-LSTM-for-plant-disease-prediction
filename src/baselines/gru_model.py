import torch
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=791,
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

        # Disease severity head
        self.disease_head = nn.Linear(
            hidden_size,
            1
        )

        # Lesion area head
        self.lesion_head = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        # x:
        # [batch, sequence_length, 791]

        output, hidden = self.gru(x)

        # Last temporal step
        last_hidden = output[:, -1, :]

        disease_output = self.disease_head(
            last_hidden
        )

        lesion_output = self.lesion_head(
            last_hidden
        )

        return (
            disease_output,
            lesion_output
        )