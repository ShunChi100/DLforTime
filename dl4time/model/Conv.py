import torch.nn as nn
import torch.nn.functional as F

class Conv1D(nn.Module):
    def __init__(self, in_channels, output_dim, dorp_prob=0.2):
        super().__init__()
        # Hidden dimensions
        self.in_channels = in_channels
        self.dorp_prob = dorp_prob

        self.conv1 = nn.Conv1d(self.in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 7, padding=3)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 16, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dorp_prob)

        # Readout layer
        self.fc = nn.Linear(112, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = F.relu(self.conv4(out))
        out = self.fc(self.dropout(out.view(-1, 112))) 
        return out