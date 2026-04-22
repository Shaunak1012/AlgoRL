"""Causal Transformer features extractor for SB3 PPO (MVP Step 6).

Input obs shape : (batch, T, F)  — emitted by MultiAssetTradingEnv with window_size=T.
Output          : (batch, d_model) — last-timestep token, consumed by SB3's default
                  MLP actor/critic heads.

Design fixed in DECISIONS.md (2026-04-20 — Transformer encoder architecture):
    d_model=64, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.1,
    learned positional embedding, upper-triangular causal mask.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        assert len(observation_space.shape) == 2, (
            f"expected 2D obs (T, F), got {observation_space.shape}"
        )
        T, F = observation_space.shape
        super().__init__(observation_space, features_dim=d_model)

        self.T = T
        self.F = F
        self.d_model = d_model

        self.input_proj = nn.Linear(F, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Causal mask: mask[i, j] = True blocks position i from attending to j (j > i).
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, F)
        x = self.input_proj(obs) + self.pos_embed       # (B, T, d_model)
        x = self.encoder(x, mask=self.causal_mask)      # (B, T, d_model)
        return self.out_norm(x[:, -1, :])               # (B, d_model)


if __name__ == "__main__":
    import numpy as np

    T, F = 60, 65
    space = gym.spaces.Box(low=-10, high=10, shape=(T, F), dtype=np.float32)
    m = TransformerFeaturesExtractor(space)
    x = torch.randn(4, T, F)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[transformer_extractor] out={tuple(y.shape)}  params={n_params:,}  OK")
