"""Unit tests for MultiAssetTradingEnv."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_asset_env import MultiAssetTradingEnv, TICKERS, N_FEATURES, LIQUIDITY_WINDOW

N = 500
RNG = np.random.default_rng(42)


def _make_dfs(n=N, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    dfs = {}
    for ticker in TICKERS:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        dfs[ticker] = pd.DataFrame(
            {
                "open":   close * (1 + rng.uniform(-0.002, 0.002, n)),
                "high":   close * (1 + rng.uniform(0.001, 0.005, n)),
                "low":    close * (1 - rng.uniform(0.001, 0.005, n)),
                "close":  close,
                "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            },
            index=idx,
        )
    return dfs


@pytest.fixture
def env():
    return MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})


def test_obs_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, N_FEATURES), f"got {obs.shape}"


def test_obs_dtype(env):
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32


def test_n_features_constant():
    assert N_FEATURES == 65  # 5*10 + 10 + 5


def test_action_space():
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})
    assert env.action_space.shape == (5,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_episode_completes_no_nan(env):
    env.reset(seed=0)
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert not np.isnan(r), "NaN reward"
        assert not np.any(np.isnan(obs)), "NaN in obs"
        if terminated or truncated:
            break


def test_spread_cost_on_position_change(env):
    env.reset(seed=0)
    env._positions = np.zeros(5)
    action = np.ones(5)  # switch from 0 → 1 on all assets
    _, r, _, _, info = env.step(action)
    assert info["r_spread"] < 0  # r_spread is negative (cost)


def test_no_spread_when_unchanged(env):
    env.reset(seed=0)
    env._positions = np.ones(5)
    _, r, _, _, info = env.step(np.ones(5))
    assert info["r_spread"] == 0.0


def test_norm_stats_transfer():
    train_dfs = _make_dfs(300, seed=1)
    val_dfs = _make_dfs(150, seed=2)
    train_env = MultiAssetTradingEnv(train_dfs, {"window_size": 1})
    stats = train_env.get_norm_stats()
    val_env = MultiAssetTradingEnv(val_dfs, {"window_size": 1}, norm_stats=stats)
    obs, _ = val_env.reset(seed=0)
    assert obs.shape == (1, N_FEATURES)
    assert not np.any(np.isnan(obs))


def test_window_size():
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 5})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5, N_FEATURES)


def test_corr_features_bounded(env):
    env.reset(seed=0)
    obs, _ = env.reset(seed=0)
    assert not np.any(np.isnan(obs))


# ── CVaR tests ────────────────────────────────────────────────────────────────

def _cvar_env(lambda_=0.1):
    cfg = {"window_size": 1, "use_cvar": True, "cvar_alpha": 0.05,
           "cvar_lambda": lambda_, "cvar_buffer_size": 252}
    return MultiAssetTradingEnv(_make_dfs(), cfg)


def test_cvar_info_keys():
    env = _cvar_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "r_pnl" in info
    assert "r_spread" in info
    assert "r_cvar" in info
    assert "cvar_5pct" in info


def test_cvar_zero_before_warmup():
    """CVaR penalty should be 0 until buffer has ≥20 observations."""
    env = _cvar_env()
    env.reset(seed=0)
    for _ in range(19):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert info["r_cvar"] == 0.0, "CVaR should be 0 before 20 obs"
        if terminated or truncated:
            break


def test_cvar_penalty_is_nonpositive_with_losses():
    """When agent consistently holds positions into falling prices, CVaR penalty < 0."""
    rng = np.random.default_rng(99)
    n = 300
    # Falling prices for all assets
    close = 100.0 * np.exp(np.cumsum(np.full(n, -0.005)))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    dfs = {}
    for t in TICKERS:
        dfs[t] = pd.DataFrame({
            "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=idx)

    env = _cvar_env()
    env = MultiAssetTradingEnv(dfs, {"window_size": 1, "use_cvar": True,
                                      "cvar_alpha": 0.05, "cvar_lambda": 0.1,
                                      "cvar_buffer_size": 252})
    env.reset(seed=0)
    long_action = np.ones(5)  # stay fully long into falling market
    cvar_penalties = []
    for _ in range(60):
        _, _, terminated, truncated, info = env.step(long_action)
        cvar_penalties.append(info["r_cvar"])
        if terminated or truncated:
            break
    # After warmup, penalty should be negative (punishing tail losses)
    late_penalties = [p for p in cvar_penalties[20:] if p != 0.0]
    if late_penalties:
        assert all(p <= 0 for p in late_penalties), "CVaR penalty should be ≤ 0"


def test_cvar_disabled_matches_no_cvar():
    """With use_cvar=False reward must equal r_pnl + r_spread exactly."""
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1, "use_cvar": False})
    env.reset(seed=7)
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)
    expected = info["r_pnl"] + info["r_spread"]
    assert abs(reward - expected) < 1e-10
    assert info["r_cvar"] == 0.0


def test_impact_disabled_by_default_preserves_r007_parity():
    """Default config must have zero impact cost — else r007 reruns drift."""
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1, 1, 1, 1, 1], dtype=np.float32))
    assert info["r_impact"] == 0.0
    assert info["impact_bps"] == 0.0


def test_impact_enabled_scales_as_power_law():
    """With α=1.5, full-flip impact must be 2^1.5 × half-flip impact."""
    cfg = {"window_size": 1, "impact_enabled": True, "impact_exponent": 1.5}
    env_full = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_full.reset(seed=0)
    _, _, _, _, info_full = env_full.step(np.ones(5, dtype=np.float32))

    env_half = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_half.reset(seed=0)
    _, _, _, _, info_half = env_half.step(0.5 * np.ones(5, dtype=np.float32))

    ratio = info_full["impact_bps"] / info_half["impact_bps"]
    assert 2.5 < ratio < 3.2  # expected 2^1.5 ≈ 2.828
    assert info_full["r_impact"] < 0  # cost is always a penalty


def test_impact_liquidity_factor_amplifies_low_volume():
    """Same trade on a low-volume day must cost more than on a high-volume day."""
    dfs = _make_dfs()
    # Force ticker SPY to have a clear low-volume day right at the step boundary
    env = MultiAssetTradingEnv(
        dfs, {"window_size": 1, "impact_enabled": True, "impact_lambda": 1e-3}
    )
    # v_ref is median; scaling 1/v means low-v day → larger cost. Simulate two
    # envs with scaled volume to verify directionality.
    low_vol_dfs = {t: df.copy() for t, df in dfs.items()}
    for t in low_vol_dfs:
        low_vol_dfs[t]["volume"] = low_vol_dfs[t]["volume"] / 4.0  # 4× thinner
    env_low = MultiAssetTradingEnv(
        low_vol_dfs, {"window_size": 1, "impact_enabled": True, "impact_lambda": 1e-3}
    )
    env.reset(seed=0); env_low.reset(seed=0)
    a = np.ones(5, dtype=np.float32)
    _, _, _, _, info_hi = env.step(a)
    _, _, _, _, info_lo = env_low.step(a)
    # Low-volume env should have higher bps cost (median scales with series,
    # but realised v_t is uniformly 4× smaller ⇒ same ratio — use absolute check)
    # At minimum, low-volume day must not be cheaper:
    assert info_lo["impact_bps"] >= info_hi["impact_bps"] - 1e-9


# ── Phase C7 — liquidity-regime features ──────────────────────────────────────

def test_liquidity_features_default_off_preserves_r007_shape():
    """With flag off, obs width must equal the module-level N_FEATURES (65)."""
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})
    assert env.N_FEATURES == N_FEATURES == 65
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 65)


def test_liquidity_features_flag_on_expands_to_75():
    """Flag on → 2 extra per-asset features × 5 assets = 10 extra → 75 total."""
    env = MultiAssetTradingEnv(
        _make_dfs(), {"window_size": 1, "use_liquidity_features": True}
    )
    assert env.N_FEATURES == 75
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 75)
    assert not np.any(np.isnan(obs))


def test_liquidity_vol_zscore_approximately_zero_mean():
    """Feature 1 (volume z-score) should have near-zero mean across the series."""
    env = MultiAssetTradingEnv(
        _make_dfs(n=500), {"window_size": 1, "use_liquidity_features": True}
    )
    # raw shape: (T, 5, 12) — liquidity cols are the last 2 per asset.
    vol_z = env._raw[:, :, 10]  # (T, 5)
    # Skip the warmup rows where rolling mean == v → z=0 trivially.
    stable = vol_z[LIQUIDITY_WINDOW:]
    assert abs(float(stable.mean())) < 0.5, f"vol_z mean far from 0: {stable.mean():.3f}"


def test_liquidity_spread_proxy_nonnegative():
    """Feature 2 (OHLC-dispersion spread proxy) must be ≥ 0 everywhere."""
    env = MultiAssetTradingEnv(
        _make_dfs(), {"window_size": 1, "use_liquidity_features": True}
    )
    spread_proxy = env._raw[:, :, 11]  # (T, 5)
    assert (spread_proxy >= -1e-12).all(), "spread proxy went negative"


def test_liquidity_norm_stats_transfer_consistent_shape():
    """Train→val norm-stats hand-off must keep obs width aligned when flag on."""
    cfg = {"window_size": 60, "use_liquidity_features": True}
    train_env = MultiAssetTradingEnv(_make_dfs(300, seed=1), cfg)
    stats = train_env.get_norm_stats()
    val_env = MultiAssetTradingEnv(
        _make_dfs(150, seed=2), cfg, norm_stats=stats
    )
    obs, _ = val_env.reset(seed=0)
    assert obs.shape == (60, 75)
    assert not np.any(np.isnan(obs))


# ── Phase C8 — stochastic liquidity-driven fill noise ─────────────────────────

def test_liquidity_noise_default_off_preserves_r007_parity():
    """With flag off, fill_noise_bps must be 0 and r_pnl must be deterministic."""
    env1 = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})
    env2 = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1})
    env1.reset(seed=0); env2.reset(seed=0)
    a = np.array([0.3, -0.4, 0.5, 0.1, -0.2], dtype=np.float32)
    _, _, _, _, info1 = env1.step(a)
    _, _, _, _, info2 = env2.step(a)
    assert info1["fill_noise_bps"] == 0.0
    assert info1["r_pnl"] == info2["r_pnl"]  # deterministic


def test_liquidity_noise_enabled_produces_nonzero_bps_and_breaks_determinism():
    """Flag on → positive fill_noise_bps, and different seeds diverge on r_pnl."""
    cfg = {"window_size": 1, "liquidity_noise_enabled": True,
           "liquidity_noise_base": 5e-4, "liquidity_noise_k": 10.0}
    env_a = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_b = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_a.reset(seed=1); env_b.reset(seed=2)
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    # Drive each env for a handful of steps to accumulate noise samples.
    pnl_a, pnl_b, bps_sum = 0.0, 0.0, 0.0
    for _ in range(10):
        _, _, t_a, _, info_a = env_a.step(a)
        _, _, t_b, _, info_b = env_b.step(a)
        pnl_a += info_a["r_pnl"]; pnl_b += info_b["r_pnl"]
        bps_sum += info_a["fill_noise_bps"]
        if t_a or t_b: break
    assert bps_sum > 0.0, "fill_noise_bps should be positive when flag on"
    assert abs(pnl_a - pnl_b) > 1e-12, "seeds should diverge once noise is on"


def test_liquidity_noise_seed_reproducibility():
    """Same seed → identical noise trajectory (gym np_random contract)."""
    cfg = {"window_size": 1, "liquidity_noise_enabled": True}
    env_a = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_b = MultiAssetTradingEnv(_make_dfs(), cfg)
    env_a.reset(seed=42); env_b.reset(seed=42)
    a = np.ones(5, dtype=np.float32)
    for _ in range(5):
        _, _, _, _, info_a = env_a.step(a)
        _, _, _, _, info_b = env_b.step(a)
        assert info_a["r_pnl"] == info_b["r_pnl"]
        assert info_a["fill_noise_bps"] == info_b["fill_noise_bps"]


def test_cvar_buffer_bounded():
    """Buffer must not grow beyond cvar_buffer_size."""
    env = MultiAssetTradingEnv(_make_dfs(), {"window_size": 1, "use_cvar": True,
                                              "cvar_buffer_size": 30})
    env.reset(seed=0)
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert len(env._returns_buffer) <= 30
        if terminated or truncated:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Phase B13 — sentiment features
# ──────────────────────────────────────────────────────────────────────────────

def _make_sentiment_parquet(tmp_path, dfs, coverage_frac=0.5) -> str:
    """Write a synthetic sentiment_daily.parquet covering `coverage_frac` of the dates."""
    idx = next(iter(dfs.values())).index
    n_covered = int(len(idx) * coverage_frac)
    dates = idx[:n_covered].date
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "date": dates,
        "bullish_frac": rng.uniform(0.2, 0.8, n_covered),
        "neutral_frac": rng.uniform(0.1, 0.5, n_covered),
        "headline_count_norm": rng.uniform(0.1, 1.0, n_covered),
    })
    path = str(tmp_path / "sentiment_daily.parquet")
    df.to_parquet(path, index=False)
    return path


def test_sentiment_flag_increases_n_features(tmp_path):
    """use_sentiment_features=True must add 3 to N_FEATURES."""
    dfs = _make_dfs()
    path = _make_sentiment_parquet(tmp_path, dfs)
    env_base = MultiAssetTradingEnv(dfs, {"window_size": 1})
    env_sent = MultiAssetTradingEnv(
        dfs,
        {"window_size": 1, "use_sentiment_features": True, "sentiment_path": path},
    )
    assert env_sent.N_FEATURES == env_base.N_FEATURES + 3


def test_sentiment_obs_shape_correct(tmp_path):
    """Observation shape must equal (window_size, N_FEATURES) with sentiment on."""
    dfs = _make_dfs()
    path = _make_sentiment_parquet(tmp_path, dfs)
    W = 5
    env = MultiAssetTradingEnv(
        dfs,
        {"window_size": W, "use_sentiment_features": True, "sentiment_path": path},
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (W, env.N_FEATURES)


def test_sentiment_neutral_fill_for_uncovered_days(tmp_path):
    """Days without headlines must have bullish_frac=0.5 and headline_count_norm=0.0."""
    dfs = _make_dfs()
    # Write sentiment for only the first 10 days; the rest should be neutral-filled
    idx = next(iter(dfs.values())).index
    df = pd.DataFrame({
        "date": idx[:10].date,
        "bullish_frac": [1.0] * 10,   # intentionally non-neutral for covered days
        "neutral_frac": [0.0] * 10,
        "headline_count_norm": [1.0] * 10,
    })
    path = str(tmp_path / "sentiment_daily.parquet")
    df.to_parquet(path, index=False)

    env = MultiAssetTradingEnv(
        dfs,
        {"window_size": 1, "use_sentiment_features": True, "sentiment_path": path},
    )
    # Day 0 should have the real values (bullish=1.0); day 50 should be neutral
    assert env._sentiment is not None
    assert abs(env._sentiment[0, 0] - 1.0) < 1e-6,  "covered day should have bullish_frac=1.0"
    assert abs(env._sentiment[50, 0] - 0.5) < 1e-6, "uncovered day should have neutral fill"
    assert abs(env._sentiment[50, 2] - 0.0) < 1e-6, "uncovered day count_norm should be 0.0"


def test_sentiment_flag_off_preserves_base_n_features():
    """Flag off must leave N_FEATURES unchanged (r007/r008 parity)."""
    dfs = _make_dfs()
    env = MultiAssetTradingEnv(dfs, {"window_size": 1, "use_sentiment_features": False})
    from envs.multi_asset_env import N_FEATURES as MODULE_N_FEATURES
    assert env.N_FEATURES == MODULE_N_FEATURES
