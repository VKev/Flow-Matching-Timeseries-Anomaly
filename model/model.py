import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Simple transformer for sequence reconstruction (no time/context inputs).
    """

    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer("pos_enc", self._build_sin_cos(max_len, d_model), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation=F.silu,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def _build_sin_cos(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x):
        """
        x: (B, T, C) sequence to reconstruct.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, C), got {x.shape}")

        T = x.size(1)
        pos = self.pos_enc[:, :T, :]
        h = self.input_proj(x) + pos  # (B,T,D)
        out = self.transformer(h)  # (B,T,D)
        return self.output_proj(out)  # (B,T,C)

