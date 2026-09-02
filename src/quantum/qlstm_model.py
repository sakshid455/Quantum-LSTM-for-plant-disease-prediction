import torch
import torch.nn as nn

from .qlstm_cell import QLSTMCell


class QLSTMModel(nn.Module):

    def __init__(
        self,
        input_size=791,
        hidden_size=32
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = QLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # Disease progression prediction
        self.disease_head = nn.Linear(
            hidden_size,
            1
        )

        # Lesion area prediction
        self.lesion_head = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        batch_size = x.size(0)
        seq_len = x.size(1)

        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        c = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        for t in range(seq_len):

            h, c = self.cell(
                x[:, t, :],
                (h, c)
            )

        disease_output = self.disease_head(h)

        lesion_output = self.lesion_head(h)

        return disease_output, lesion_output