import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)