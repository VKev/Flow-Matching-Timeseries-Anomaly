import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Positional embedding for scalar time inputs.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("Time embedding dimension must be at least 2.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (batch,) or scalar tensor in [0, 1].

        Returns:
            Tensor of shape (batch, dim) containing sinusoidal embeddings.
        """
        if t.dim() == 0:
            t = t[None]
        t = t.view(-1, 1)

        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(
                start=0,
                end=half_dim,
                device=t.device,
                dtype=t.dtype,
            )
            * (-math.log(10000.0) / max(half_dim, 1))
        )
        args = t * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if emb.shape[-1] < self.dim:
            pad = torch.zeros(
                emb.shape[0],
                self.dim - emb.shape[-1],
                device=emb.device,
                dtype=emb.dtype,
            )
            emb = torch.cat([emb, pad], dim=-1)

        return emb


class FlowMatchingTransformer(nn.Module):
    """
    Transformer-based velocity model for flow matching on windowed time series.
    Each time step is treated as a token and conditioned on the continuous time
    through a sinusoidal embedding.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_len: int = 512,
        latent_dim: int | None = None,
    ):
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be at least 2 to build embeddings.")

        self.d_model = d_model
        self.latent_dim = latent_dim or input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer(
            "pos_enc", self._build_pos_enc(max_len, d_model), persistent=False
        )

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation=F.silu,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Velocity in input space for ODE consistency
        self.output_proj = nn.Linear(d_model, input_dim)
        # Optional latent projection head
        self.latent_proj = None
        if self.latent_dim != input_dim:
            self.latent_proj = nn.Linear(d_model, self.latent_dim)

    def _build_pos_enc(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, input_dim) or (batch, seq_len) for univariate.
            t: shape (batch,) or scalar tensor in [0, 1].

        Returns:
            Predicted velocity with shape matching x.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, C), got {x.shape}")

        batch_size, seq_len, _ = x.shape

        t = t.to(device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.dim() == 1 and t.numel() == 1 and batch_size > 1:
            t = t.expand(batch_size)
        elif t.dim() != 1 or t.shape[0] != batch_size:
            raise ValueError(f"Time input shape mismatch: got {t.shape}, batch={batch_size}")

        pos = self.pos_enc[:, :seq_len, :].to(x.device)
        tokens = self.input_proj(x) + pos

        t_emb = self.time_mlp(self.time_embed(t))
        tokens = tokens + t_emb[:, None, :]

        hidden = self.transformer(tokens)
        velocity = self.output_proj(hidden)
        return velocity

    def latent(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get latent projection (if configured); otherwise returns velocity projection.
        """
        hidden = self._encode_only(x, t)
        if self.latent_proj is not None:
            return self.latent_proj(hidden)
        return self.output_proj(hidden)

    def _encode_only(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, C), got {x.shape}")

        batch_size, seq_len, _ = x.shape

        t = t.to(device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.dim() == 1 and t.numel() == 1 and batch_size > 1:
            t = t.expand(batch_size)
        elif t.dim() != 1 or t.shape[0] != batch_size:
            raise ValueError(f"Time input shape mismatch: got {t.shape}, batch={batch_size}")

        pos = self.pos_enc[:, :seq_len, :].to(x.device)
        tokens = self.input_proj(x) + pos

        t_emb = self.time_mlp(self.time_embed(t))
        tokens = tokens + t_emb[:, None, :]

        return self.transformer(tokens)


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        B, T, D = x.shape
        pe = self.pe[:, :T].to(x.device, x.dtype)
        return self.dropout(x + pe)