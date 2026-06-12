# ============================================================
# model.py
# DB6 EMG / ACC / Fusion Transformer
#
# Input:
#   x: [batch, window_size, input_dim]
#
# Example:
#   EMG    [batch, 40, 16]
#   ACC    [batch, 40, 42]
#   Fusion [batch, 40, 58]
#
# Output:
#   logits: [batch, num_classes]
# ============================================================

import math
import torch
import torch.nn as nn


# ============================================================
# 1. Sinusoidal Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ============================================================
# 2. Pure Transformer Classifier
# ============================================================

class SensorTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        window_size=40,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.window_size = window_size
        self.d_model = d_model

        # ----------------------------------------------------
        # Input embedding
        # 把每个时间点的传感器向量映射到 d_model 维
        # ----------------------------------------------------
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------
        # Fixed sinusoidal positional encoding
        # ----------------------------------------------------
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=window_size
        )

        self.input_norm = nn.LayerNorm(d_model)

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        self.feature_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, window_size, input_dim]

        x = self.input_embedding(x)
        # [batch, 40, d_model]

        x = self.pos_encoding(x)
        x = self.input_norm(x)

        x = self.transformer(x)
        # [batch, 40, d_model]

        x = x.mean(dim=1)
        # [batch, d_model]

        x = self.feature_norm(x)

        logits = self.classifier(x)
        # [batch, num_classes]

        return logits


# ============================================================
# 3. Quick Test
# ============================================================

if __name__ == "__main__":
    batch_size = 8
    window_size = 40
    num_classes = 7

    for input_dim in [16, 42, 58]:
        x = torch.randn(batch_size, window_size, input_dim)

        model = SensorTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            window_size=window_size
        )

        y = model(x)

        print("Input dim:", input_dim)
        print("Input shape:", x.shape)
        print("Output shape:", y.shape)
        print("-" * 50)