"""Unit tests for TransformerFeaturesExtractor (MVP Step 6)."""
import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.transformer_extractor import TransformerFeaturesExtractor


T, F = 60, 65


@pytest.fixture
def space():
    return gym.spaces.Box(low=-10, high=10, shape=(T, F), dtype=np.float32)


def test_output_shape(space):
    m = TransformerFeaturesExtractor(space)
    out = m(torch.randn(8, T, F))
    assert out.shape == (8, m.features_dim)


def test_features_dim_matches_d_model(space):
    m = TransformerFeaturesExtractor(space, d_model=32)
    assert m.features_dim == 32
    assert m(torch.randn(2, T, F)).shape == (2, 32)


def test_gradient_flow(space):
    m = TransformerFeaturesExtractor(space)
    x = torch.randn(4, T, F, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_causal_mask_last_token_depends_on_all(space):
    """Last-token output should depend on every input position (via causal attention)."""
    m = TransformerFeaturesExtractor(space)
    m.eval()
    x = torch.randn(1, T, F)
    with torch.no_grad():
        y_base = m(x).clone()
    for t in range(T):
        x_perturb = x.clone()
        x_perturb[0, t] += 5.0
        with torch.no_grad():
            y = m(x_perturb)
        assert not torch.allclose(y, y_base, atol=1e-4), (
            f"last-token output unchanged when position {t} was perturbed"
        )


def test_causal_mask_blocks_future_leakage(space):
    """
    Position t's output must NOT depend on positions > t.
    Check at t=T//2: perturbing positions t+1..T should not change output[:, t].
    We expose this by running the encoder manually and reading the middle token.
    """
    m = TransformerFeaturesExtractor(space)
    m.eval()
    x = torch.randn(1, T, F)
    t = T // 2

    def middle_token(inp):
        z = m.input_proj(inp) + m.pos_embed
        z = m.encoder(z, mask=m.causal_mask)
        return z[:, t, :].clone()

    with torch.no_grad():
        y_base = middle_token(x)
        x2 = x.clone()
        x2[0, t + 1 :] += 10.0
        y_future = middle_token(x2)
    assert torch.allclose(y_base, y_future, atol=1e-6), "future leakage detected"


def test_deterministic_eval(space):
    m = TransformerFeaturesExtractor(space)
    m.eval()
    x = torch.randn(3, T, F)
    with torch.no_grad():
        a = m(x); b = m(x)
    assert torch.allclose(a, b)
