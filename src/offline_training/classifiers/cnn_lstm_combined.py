import torch
import torch.nn as nn


def _build_cnn(n_channels, cnn_dim, in_channels=2):
    """CNN: `in_channels` → cnn_dim, spatial 24→12→6. Each reference becomes a 6×6 grid of
    `cnn_dim` tokens. `in_channels` defaults to 2 (foreground + background similarities); set
    to 1 for single-channel ablations."""

    ch1, ch2 = n_channels, n_channels * 2
    return nn.Sequential(
        # Stage 1: 24 → 12
        nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch1),
        nn.GELU(),
        nn.Conv2d(ch1, ch1, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch1),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Stage 2: 12 → 6
        nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Stage 3: 6 → 3
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        nn.BatchNorm2d(ch2),
        nn.GELU(),
        nn.MaxPool2d(2),

        # Projection to cnn_dim (no spatial pooling).
        nn.Conv2d(ch2, cnn_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(cnn_dim),
        nn.GELU())


def _build_head(cnn_dim, mlp_hidden, dropout):
    """4-layer MLP head producing the calibrated IoU scalar."""

    return nn.Sequential(
        nn.Linear(cnn_dim, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, 1))


class CNNLSTMCombined(nn.Module):
    """CNN + LSTM for IoU calibration. Per-reference CNN → global-pooled (B, R, cnn_dim) sequence
    → LSTM in slot order → final hidden state → MLP → scalar IoU. Simpler than the transformer
    variant: one spatial vector per slot, sequential aggregation, no positional embeddings."""

    def __init__(self, n_references=7, n_channels=128, cnn_dim=256, mlp_hidden=512,
                 dropout=0.15, lstm_hidden=256, n_lstm_layers=1):
        super().__init__()
        self.n_references = n_references
        self.cnn_dim = cnn_dim

        self.cnn = _build_cnn(n_channels=n_channels, cnn_dim=cnn_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=cnn_dim, hidden_size=lstm_hidden, num_layers=n_lstm_layers,
                            batch_first=True, dropout=dropout)
        self.head = _build_head(lstm_hidden, mlp_hidden, dropout)

    def forward(self, x):
        """`x`: `(B, n_input_channels, H, W, 2)`. Returns IoU `(B, 1)`."""

        x = x[:, :self.n_references]
        batch_size, n_references, height, width, _ = x.shape

        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size * n_references, 2, height, width)
        x = self.global_pool(self.cnn(x)).flatten(start_dim=1)
        x = x.view(batch_size, n_references, self.cnn_dim)

        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).float()
