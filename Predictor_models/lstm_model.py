import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 32, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32)
        c0 = torch.zeros(1, x.size(0), 32)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
