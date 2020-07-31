import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, in_channels, output_dim):
        super().__init__()
        # Hidden dimensions
        self.in_channels = in_channels

        self.conv1 = nn.Conv1d(self.in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(64, 16, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)

        # Readout layer
        self.fc = nn.Linear(112, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = self.fc(self.dropout(out.view(-1, 112))) 
        return out