import torch
import torch.nn as nn
from .quantum_layer import QuantumLayer


class QLSTMCell(nn.Module):

    def __init__(self, input_size=768, hidden_size=32):
        super().__init__()

        self.hidden_size = hidden_size

        # Forget Gate
        self.f_encoder = nn.Linear(input_size + hidden_size, 4)
        self.f_quantum = QuantumLayer()
        self.f_decoder = nn.Linear(4, hidden_size)

        # Input Gate
        self.i_encoder = nn.Linear(input_size + hidden_size, 4)
        self.i_quantum = QuantumLayer()
        self.i_decoder = nn.Linear(4, hidden_size)

        # Output Gate
        self.o_encoder = nn.Linear(input_size + hidden_size, 4)
        self.o_quantum = QuantumLayer()
        self.o_decoder = nn.Linear(4, hidden_size)

        # Candidate Gate
        self.g_encoder = nn.Linear(input_size + hidden_size, 4)
        self.g_quantum = QuantumLayer()
        self.g_decoder = nn.Linear(4, hidden_size)

    def forward(self, x, hidden):

        h_prev, c_prev = hidden

        combined = torch.cat([x, h_prev], dim=1)

        f = torch.sigmoid(
            self.f_decoder(
                self.f_quantum(
                    self.f_encoder(combined)
                )
            )
        )

        i = torch.sigmoid(
            self.i_decoder(
                self.i_quantum(
                    self.i_encoder(combined)
                )
            )
        )

        o = torch.sigmoid(
            self.o_decoder(
                self.o_quantum(
                    self.o_encoder(combined)
                )
            )
        )

        g = torch.tanh(
            self.g_decoder(
                self.g_quantum(
                    self.g_encoder(combined)
                )
            )
        )

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c