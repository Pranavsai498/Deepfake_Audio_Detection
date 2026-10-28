import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfAttention(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.projection(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class HybridCNNLSTMDetector(nn.Module):

    def __init__(self, in_channels: int=120, cnn_channels: int=128, lstm_hidden: int=64, lstm_layers: int=2, num_classes: int=2, dropout: float=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.conv1 = nn.Conv1d(in_channels, cnn_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.attention = TemporalSelfAttention(lstm_hidden * 2)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden * 2, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = self.dropout(h)
        h_seq = h.transpose(1, 2)
        lstm_out, _ = self.lstm(h_seq)
        context = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))
if __name__ == '__main__':
    model = HybridCNNLSTMDetector()
    print(f'Hybrid CNN-BiLSTM initialized successfully.')
    print(f'Total Trainable Parameters: {model.count_parameters():,}')
    dummy_input = torch.randn(2, 120, 300)
    out = model(dummy_input)
    print(f'Forward pass output shape: {out.shape}')
