"""DECENT-plus: deep model for read-level maternal contamination probability."""

import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output


class DISMIR_deep(nn.Module):
    """CNN + self-attention + LSTM for sequence + methylation input (N, 5, 132)."""

    def __init__(self, n_classes):
        super(DISMIR_deep, self).__init__()
        self.n_classes = n_classes

        self.cov1 = nn.Sequential(
            nn.Conv1d(5, 100, 10, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.2),
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=100, num_heads=4, dropout=0.2)
        self.layer_norm = nn.LayerNorm(100)
        self.custom_lstm = CustomLSTM(100, 132)
        self.cov2 = nn.Sequential(
            nn.Conv1d(264, 100, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.2),
        )
        self.flatten = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        if n_classes == 2:
            self.output = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        else:
            self.output = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cov1(x)
        x = x.permute(2, 0, 1)
        x_residual = x
        x, _ = self.self_attention(x, x, x)
        x = x + x_residual
        x = self.layer_norm(x)
        x = self.custom_lstm(x)
        x = self.cov2(x.permute(1, 2, 0))
        x = self.flatten(x)
        x = self.output(x)
        return x
