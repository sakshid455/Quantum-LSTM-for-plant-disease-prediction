import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=32):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)

        # Last timestep
        output = output[:, -1, :]

        return self.fc(output)


class LSTMModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)

        output = output[:, -1, :]

        return self.fc(output)


class GRUModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=32):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)

        output = output[:, -1, :]

        return self.fc(output)