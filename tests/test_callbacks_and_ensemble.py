"""Tests for ValEvalCallback and ensemble rollout logic (MVP Step 7)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_asset_env import MultiAssetTradingEnv, TICKERS
from scripts.callbacks import ValEvalCallback
from scripts.ensemble_eval import _rollout


def _make_dfs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dfs = {}
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    for t in TICKERS:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        dfs[t] = pd.DataFrame({
            "open":   close * (1 + rng.uniform(-0.002, 0.002, n)),
            "high":   close * (1 + rng.uniform(0.001, 0.005, n)),
            "low":    close * (1 - rng.uniform(0.001, 0.005, n)),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=idx)
    return dfs


class _StubModel:
    """Mimics the bits of a SB3 PPO model that _rollout / ValEvalCallback need."""

    def __init__(self, const_action):
        self.const_action = np.asarray(const_action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.const_action, None


def test_valevalcallback_evaluate_returns_expected_keys(tmp_path):
    dfs = _make_dfs()
    env_cfg = {"tickers": TICKERS, "window_size": 1, "half_spread": 0.0001,
               "use_indicators": True, "use_cvar": False}
    cb = ValEvalCallback(
        val_env_factory=lambda: MultiAssetTradingEnv(dfs, env_cfg),
        log_path=str(tmp_path / "val.csv"),
        best_ckpt_path=str(tmp_path / "best"),
        eval_freq=1000,
    )
    cb.model = _StubModel(np.zeros(5))  # zero positions → zero PnL, zero std
    stats = cb._evaluate()
    assert set(stats.keys()) == {"sharpe", "log_return", "max_dd"}
    assert stats["log_return"] == pytest.approx(0.0, abs=1e-10)
    assert stats["max_dd"] == pytest.approx(0.0, abs=1e-10)


def test_ensemble_rollout_mean_vs_uncertainty_shrinks_position():
    dfs = _make_dfs()
    env_cfg = {"tickers": TICKERS, "window_size": 1, "half_spread": 0.0001,
               "use_indicators": True, "use_cvar": False}

    # Disagreeing heads: +1 and -1 across assets → std=1.0 per asset, mean=0
    models_disagree = [_StubModel(np.ones(5)), _StubModel(-np.ones(5))]
    env = MultiAssetTradingEnv(dfs, env_cfg)
    r_mean = _rollout(models_disagree, env, scale=5.0, mode="mean")

    env = MultiAssetTradingEnv(dfs, env_cfg)
    r_uncert = _rollout(models_disagree, env, scale=5.0, mode="uncertainty")

    # Mean action is 0 either way here; both should give ~0 log return.
    assert abs(r_mean["log_return"]) < 1e-6
    assert abs(r_uncert["log_return"]) < 1e-6
    # Sanity on mean_head_std reporting
    assert r_mean["mean_head_std"] == pytest.approx(1.0, rel=1e-2)


def test_ensemble_rollout_agreeing_heads_no_shrinkage_effect():
    dfs = _make_dfs()
    env_cfg = {"tickers": TICKERS, "window_size": 1, "half_spread": 0.0001,
               "use_indicators": True, "use_cvar": False}
    models_agree = [_StubModel(0.5 * np.ones(5)) for _ in range(3)]

    env = MultiAssetTradingEnv(dfs, env_cfg)
    r_mean = _rollout(models_agree, env, scale=5.0, mode="mean")
    env = MultiAssetTradingEnv(dfs, env_cfg)
    r_uncert = _rollout(models_agree, env, scale=5.0, mode="uncertainty")

    # With zero inter-head std, uncertainty-scaled == mean.
    assert r_mean["log_return"] == pytest.approx(r_uncert["log_return"], abs=1e-10)
    assert r_mean["mean_head_std"] == pytest.approx(0.0, abs=1e-8)
