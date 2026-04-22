"""Single-asset trading environment (Gymnasium-compatible).

State:  window_size × 12 float32 array
        Price features:  [log_ret, high_ratio, low_ratio, open_ratio, vol_ratio]
        Indicators:      [rsi, macd, macd_signal, bb_pct, bb_width]
        Live context:    [position, upnl]
Action: Discrete(3) — 0=short, 1=flat, 2=long  →  position ∈ {-1, 0, +1}
Reward: position × log_return − |Δposition| × half_spread
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False


class SingleAssetTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    POSITIONS = np.array([-1.0, 0.0, 1.0])

    # 5 price features + 5 indicator features + 2 live context = 12
    N_PRICE_FEATURES = 5
    N_INDICATOR_FEATURES = 5
    N_LIVE_FEATURES = 2
    N_FEATURES = N_PRICE_FEATURES + N_INDICATOR_FEATURES + N_LIVE_FEATURES  # 12

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        norm_stats: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.window_size: int = config.get("window_size", 1)
        self.half_spread: float = config.get("half_spread", 0.0001)
        self.initial_cash: float = config.get("initial_cash", 100_000.0)
        self.use_indicators: bool = config.get("use_indicators", True)

        self._prices = (
            df[["open", "high", "low", "close", "volume"]]
            .reset_index(drop=True)
            .values.astype(np.float64)
        )
        self._n = len(self._prices)

        # Raw feature matrix: N_PRICE_FEATURES + N_INDICATOR_FEATURES columns
        # (position and upnl are appended live in _obs())
        self._raw = self._build_raw_features()
        self._n_raw = self._raw.shape[1]  # 10

        if norm_stats is None:
            norm_stats = {
                "mean": self._raw.mean(axis=0),
                "std": self._raw.std(axis=0) + 1e-8,
            }
        self._mean: np.ndarray = norm_stats["mean"]
        self._std: np.ndarray = norm_stats["std"]
        self._norm = (self._raw - self._mean) / self._std

        obs_shape = (self.window_size, self.N_FEATURES)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._step: int = 0
        self._position: float = 0.0
        self._entry_price: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_norm_stats(self) -> dict:
        """Return stats computed on this split (call on train env only)."""
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._step = self.window_size
        self._position = 0.0
        self._entry_price = self._prices[self._step, 3]
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        new_pos = float(self.POSITIONS[action])
        prev_close = self._prices[self._step, 3]

        self._step += 1
        cur_close = self._prices[self._step, 3]
        log_ret = float(np.log(cur_close / prev_close))

        spread_cost = abs(new_pos - self._position) * self.half_spread
        reward = self._position * log_ret - spread_cost

        if new_pos != self._position:
            self._entry_price = cur_close
        self._position = new_pos

        terminated = self._step >= self._n - 1
        info = {
            "log_ret": log_ret,
            "spread_cost": spread_cost,
            "position": self._position,
            "close": cur_close,
        }
        return self._obs(), float(reward), terminated, False, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_raw_features(self) -> np.ndarray:
        o, h, l, c, v = (self._prices[:, i] for i in range(5))
        close_s = pd.Series(c)
        high_s = pd.Series(h)
        low_s = pd.Series(l)

        # ── Price features (5) ───────────────────────────────────────────
        log_ret = np.empty(self._n)
        log_ret[0] = 0.0
        log_ret[1:] = np.log(c[1:] / c[:-1])

        vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values
        vol_ratio = v / (vol_ma + 1e-8)

        price_feats = np.stack(
            [log_ret, np.log(h / c), np.log(l / c), np.log(o / c), vol_ratio],
            axis=1,
        )

        # ── Technical indicators (5) ─────────────────────────────────────
        if self.use_indicators and _TA_AVAILABLE:
            indicator_feats = self._compute_indicators(close_s, high_s, low_s)
        else:
            indicator_feats = np.zeros((self._n, self.N_INDICATOR_FEATURES))

        return np.concatenate([price_feats, indicator_feats], axis=1)

    def _compute_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
    ) -> np.ndarray:
        """Compute 5 indicator features; NaNs filled with neutral values."""
        # RSI(14) — neutral = 50, normalised to [-1, 1] range
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        rsi_norm = (rsi.fillna(50.0) - 50.0) / 50.0  # → [-1, 1]

        # MACD line and signal line (12, 26, 9)
        macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd().fillna(0.0)
        macd_signal = macd_obj.macd_signal().fillna(0.0)

        # Bollinger Bands (20, 2σ)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_pct = bb.bollinger_pband().fillna(0.5)   # %B: 0=lower, 1=upper
        bb_width = bb.bollinger_wband().fillna(0.0)  # (upper-lower)/middle

        return np.stack(
            [
                rsi_norm.values,
                macd_line.values,
                macd_signal.values,
                bb_pct.values,
                bb_width.values,
            ],
            axis=1,
        )

    def _obs(self) -> np.ndarray:
        start = max(0, self._step - self.window_size + 1)
        norm_slice = self._norm[start : self._step + 1]  # (≤W, 10)

        cur_close = self._prices[self._step, 3]
        upnl = (
            self._position * (cur_close - self._entry_price) / (self._entry_price + 1e-8)
            if self._entry_price > 0
            else 0.0
        )

        n = len(norm_slice)
        pos_col = np.full((n, 1), self._position, dtype=np.float32)
        upnl_col = np.full((n, 1), upnl, dtype=np.float32)
        obs = np.concatenate([norm_slice.astype(np.float32), pos_col, upnl_col], axis=1)

        if n < self.window_size:
            pad = np.zeros((self.window_size - n, self.N_FEATURES), dtype=np.float32)
            obs = np.concatenate([pad, obs], axis=0)

        return obs


# ------------------------------------------------------------------
# Sanity check (runs in <10 s)
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.002, 0.002, n)),
            "high": close * (1 + rng.uniform(0.001, 0.005, n)),
            "low": close * (1 - rng.uniform(0.001, 0.005, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )
    env = SingleAssetTradingEnv(df, {"window_size": 1, "half_spread": 0.0001})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 12), f"Bad obs shape: {obs.shape}"
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"[trading_env] sanity OK — steps={env._step}, obs_shape={obs.shape}, total_reward={total_reward:.4f}")
