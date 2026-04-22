"""Unit tests for SingleAssetTradingEnv."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.trading_env import SingleAssetTradingEnv

RNG = np.random.default_rng(42)
N = 500


def _make_df(n=N, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.002, 0.002, n)),
        "high":   close * (1 + rng.uniform(0.001, 0.005, n)),
        "low":    close * (1 - rng.uniform(0.001, 0.005, n)),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


@pytest.fixture
def env():
    return SingleAssetTradingEnv(_make_df(), {"window_size": 1, "half_spread": 0.0001})


def test_obs_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 12)


def test_obs_dtype(env):
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32


def test_episode_completes_no_nan(env):
    env.reset(seed=1)
    rewards = []
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(r)
        assert not np.isnan(r), "NaN reward"
        if terminated or truncated:
            break
    assert len(rewards) > 0


def test_no_future_leakage(env):
    """obs at step t must be identical whether or not data at t+1 is changed."""
    obs_before, _ = env.reset(seed=0)
    t1 = env._step + 1
    env._prices[t1, 3] *= 999
    obs_after, _ = env.reset(seed=0)
    env._prices[t1, 3] /= 999
    assert np.allclose(obs_before, obs_after)


def test_window_size(env):
    env2 = SingleAssetTradingEnv(_make_df(), {"window_size": 5})
    obs, _ = env2.reset(seed=0)
    assert obs.shape == (5, 12)


def test_indicators_no_nan(env):
    obs, _ = env.reset(seed=0)
    assert not np.any(np.isnan(obs)), "NaN in initial obs (indicator warmup not handled)"


def test_indicators_disabled():
    df = _make_df(300)
    env = SingleAssetTradingEnv(df, {"window_size": 1, "use_indicators": False})
    obs, _ = env.reset(seed=0)
    # indicator columns should be zero when disabled
    assert np.all(obs[0, 5:10] == 0.0)


def test_norm_stats_transfer():
    train_df = _make_df(300, seed=1)
    val_df = _make_df(100, seed=2)
    train_env = SingleAssetTradingEnv(train_df, {"window_size": 1})
    stats = train_env.get_norm_stats()
    val_env = SingleAssetTradingEnv(val_df, {"window_size": 1}, norm_stats=stats)
    obs, _ = val_env.reset(seed=0)
    assert obs.shape == (1, 12)
    assert not np.any(np.isnan(obs))


def test_spread_cost_on_position_change(env):
    env.reset(seed=0)
    env._position = -1.0  # force short
    _, r, _, _, info = env.step(2)  # switch to long
    assert info["spread_cost"] > 0


def test_no_spread_cost_when_position_unchanged(env):
    env.reset(seed=0)
    env._position = 1.0  # force long
    _, r, _, _, info = env.step(2)  # stay long
    assert info["spread_cost"] == 0.0
