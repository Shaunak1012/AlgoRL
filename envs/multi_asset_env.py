"""Multi-asset trading environment (Gymnasium-compatible).

State per timestep (window_size × F):
    Per-asset features  : 10 (or 12) features × 5 assets  = 50 (or 60)
                          [log_ret, high_ratio, low_ratio, open_ratio, vol_ratio,
                           rsi, macd, macd_signal, bb_pct, bb_width,
                           (vol_zscore, spread_proxy)  ← Phase C7, flag-gated]  × each asset
    Correlation (upper) : 10  (rolling-20d upper triangle of 5×5 return corr matrix)
    Current positions   :  5  (one weight per asset, ∈ [−1, 1])
    Sentiment (Phase B) :  3  (bullish_frac, neutral_frac, headline_count_norm)
                              appended when use_sentiment_features=True; gap-filled with
                              (0.5, 0.5, 0.0) on days without headline coverage.

F = 65 (base) | 75 (+ C7 liquidity) | 68 (+ B sentiment) | 78 (+ C7 + B)

Action: Box(−1, 1, shape=(5,)) — continuous target weight per asset
Reward: r_pnl + r_spread + r_impact + r_cvar
    r_pnl    = Σ_i position_i × log_ret_i
    r_spread = −Σ_i |Δposition_i| × half_spread
    r_impact = −Σ_i λ_kyle × |Δposition_i|^α × (v_ref_i / v_t,i)  [0 if impact_enabled=False]
    r_cvar   = λ × CVaR_α(rolling returns buffer)                 [0 if use_cvar=False]

Stochastic execution noise (Phase C8 — flag `liquidity_noise_enabled`, default off):
    fill_close_t,i = close_t,i × (1 + ε_t,i),   ε ~ N(0, σ_t,i)
    σ_t,i = σ_base × (1 + k × spread_proxy_t,i)
    — illiquid days (wide OHLC dispersion) → wider fill noise.
    log_rets computed against the noisy fill, so realised PnL is dispersed
    around the frictionless close by the inferred liquidity regime.

Market-impact model (Phase A — Kyle's-λ power law):
    Temporary impact per asset, modulated by realised-vs-reference daily volume so
    low-liquidity days (v_t < v_ref) amplify the cost. Dimensionless (positions are
    weights ∈ [−1, 1]). Default λ_kyle=1e-3, α=1.5 → ~10 bps on a full-position flip
    at median liquidity; ~30 bps when realised volume is 1/3 of the reference.

CVaR_α = E[return | return ≤ VaR_α]  — expected return in worst α-fraction of steps.
Negative when the agent has tail losses; adding λ·CVaR_α to reward penalises drawdowns.
"""
from __future__ import annotations

from pathlib import Path
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

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
N_ASSETS = 5
N_PRICE_FEATURES = 5
N_INDICATOR_FEATURES = 5
N_ASSET_FEATURES = N_PRICE_FEATURES + N_INDICATOR_FEATURES   # 10 raw (before adding live cols)
N_LIQUIDITY_FEATURES = 2   # Phase C7: volume z-score + OHLC-dispersion spread proxy
N_SENTIMENT_FEATURES = 3   # Phase B: bullish_frac, neutral_frac, headline_count_norm
N_CORR_FEATURES = N_ASSETS * (N_ASSETS - 1) // 2            # 10
CORR_WINDOW = 20
LIQUIDITY_WINDOW = 20

# Full obs features per timestep (base, all flags off):
#   10 raw per asset × 5  +  10 corr  +  5 positions              = 65
# With use_liquidity_features=True:
#   12 raw per asset × 5  +  10 corr  +  5 positions              = 75
# With use_sentiment_features=True (+ 3 global):
#   base + 3 sentiment                                             = 68
# With both:
#   75 + 3                                                         = 78
N_FEATURES = N_ASSETS * N_ASSET_FEATURES + N_CORR_FEATURES + N_ASSETS  # 65 (base)


class MultiAssetTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    N_FEATURES = N_FEATURES

    def __init__(
        self,
        dfs: dict[str, pd.DataFrame],
        config: dict,
        norm_stats: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.tickers: list[str] = config.get("tickers", TICKERS)
        self.window_size: int = config.get("window_size", 1)
        self.half_spread: float = config.get("half_spread", 0.0001)
        self.use_indicators: bool = config.get("use_indicators", True)

        # Market impact (Kyle's-λ power law; flag-gated so r007 replay is bit-identical)
        self.impact_enabled: bool = config.get("impact_enabled", False)
        self.impact_lambda: float = config.get("impact_lambda", 1e-3)
        self.impact_exponent: float = config.get("impact_exponent", 1.5)

        # Liquidity-regime features (Phase C7; flag-gated — default False preserves r007 parity)
        self.use_liquidity_features: bool = config.get("use_liquidity_features", False)
        self.n_asset_features: int = N_ASSET_FEATURES + (
            N_LIQUIDITY_FEATURES if self.use_liquidity_features else 0
        )

        # Sentiment features (Phase B; flag-gated — default False preserves r007/r008 parity)
        # 3 global features appended per timestep: bullish_frac, neutral_frac, headline_count_norm
        # Appended AFTER normalised price/corr features and positions (not normalised themselves
        # since all three are already in [0, 1]).
        self.use_sentiment_features: bool = config.get("use_sentiment_features", False)
        self._sentiment_path: str = config.get(
            "sentiment_path",
            str(Path(__file__).parent.parent / "data" / "sentiment_daily.parquet"),
        )
        self._sentiment: np.ndarray | None = None

        self.N_FEATURES: int = (
            N_ASSETS * self.n_asset_features
            + N_CORR_FEATURES
            + N_ASSETS
            + (N_SENTIMENT_FEATURES if self.use_sentiment_features else 0)
        )

        # Stochastic execution noise (Phase C8; flag-gated — default False preserves r007 parity)
        self.liquidity_noise_enabled: bool = config.get("liquidity_noise_enabled", False)
        self.liquidity_noise_base: float = config.get("liquidity_noise_base", 5e-4)  # 5 bps
        self.liquidity_noise_k: float = config.get("liquidity_noise_k", 10.0)

        # CVaR regularisation
        self.use_cvar: bool = config.get("use_cvar", False)
        self.cvar_alpha: float = config.get("cvar_alpha", 0.05)
        self.cvar_lambda: float = config.get("cvar_lambda", 0.1)
        self.cvar_buffer_size: int = config.get("cvar_buffer_size", 252)
        self._returns_buffer: list[float] = []

        # Align all DataFrames to a common date index (inner join)
        # Capture dates before reset_index so sentiment can be aligned to trading days.
        aligned, self._dates = _align(dfs, self.tickers)
        self._n = len(next(iter(aligned.values())))

        # Reference (median) daily volume per asset — used by the Kyle's-λ impact model.
        # Computed on the full supplied history; for val/test frames this biases
        # toward the OOS regime, which is the appropriate "ambient liquidity" proxy.
        self._vol_ref = np.array([
            float(np.median(dfs[t]["volume"].values) + 1e-8) for t in self.tickers
        ])

        # prices[t, asset, ohlcv]
        self._prices = np.stack(
            [
                aligned[t][["open", "high", "low", "close", "volume"]].values
                for t in self.tickers
            ],
            axis=1,
        ).astype(np.float64)   # shape (T, N_ASSETS, 5)

        # Per-asset spread proxy (T, N_ASSETS) — reused by C8 noise regardless of
        # whether C7 observation features are on. Cheap precompute.
        self._spread_proxy = self._build_spread_proxy()

        # Sentiment array (T, 3) — loaded after _dates is available from _align.
        if self.use_sentiment_features:
            self._sentiment = self._load_sentiment(self._sentiment_path)

        # raw_feats[t, asset, feature]  — (T, 5, n_asset_features)  n=10 or 12
        self._raw = self._build_raw_features()

        # corr_feats[t, 10]  — precomputed rolling correlation upper triangle
        self._corr = self._build_corr_features()

        # Normalisation (fit on train split; passed in for val/test)
        # Flatten to (T, 5*n_asset_features + 10) for mean/std computation
        flat_raw = self._raw.reshape(self._n, -1)           # (T, 50 or 60)
        full_mat = np.concatenate([flat_raw, self._corr], axis=1)  # (T, 60 or 70)

        if norm_stats is None:
            norm_stats = {
                "mean": full_mat.mean(axis=0),
                "std": full_mat.std(axis=0) + 1e-8,
            }
        self._mean: np.ndarray = norm_stats["mean"]
        self._std: np.ndarray = norm_stats["std"]
        self._norm = (full_mat - self._mean) / self._std    # (T, 60)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.window_size, self.N_FEATURES),
            dtype=np.float32,
        )
        # Continuous weights ∈ [−1, 1] per asset
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ASSETS,), dtype=np.float32
        )

        self._step: int = 0
        self._positions: np.ndarray = np.zeros(N_ASSETS)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_norm_stats(self) -> dict:
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._step = max(self.window_size, CORR_WINDOW)
        self._positions = np.zeros(N_ASSETS)
        self._returns_buffer = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        prev_positions = self._positions.copy()
        prev_closes = self._prices[self._step, :, 3]

        self._step += 1
        cur_closes = self._prices[self._step, :, 3]

        # Stochastic liquidity-driven fill noise (Phase C8; flag-gated).
        if self.liquidity_noise_enabled:
            spread_t = self._spread_proxy[self._step]           # (N_ASSETS,)
            sigma_t = self.liquidity_noise_base * (1.0 + self.liquidity_noise_k * spread_t)
            eps = self.np_random.normal(0.0, 1.0, size=N_ASSETS) * sigma_t
            fill_closes = cur_closes * (1.0 + eps)
            fill_noise_bps = float(np.mean(np.abs(eps)) * 1e4)
        else:
            fill_closes = cur_closes
            fill_noise_bps = 0.0

        log_rets = np.log(fill_closes / prev_closes)

        r_pnl = float(np.sum(prev_positions * log_rets))
        trade_size = np.abs(action - prev_positions)
        spread_costs = trade_size * self.half_spread
        r_spread = -float(spread_costs.sum())

        # Kyle's-λ market impact (Phase A): temporary cost ∝ |Δpos|^α, amplified
        # on low-volume days (v_ref / v_t > 1) and dampened on high-volume days.
        if self.impact_enabled:
            v_t = self._prices[self._step, :, 4]  # realised daily volume at fill time
            liq_factor = self._vol_ref / (v_t + 1e-8)
            impact_per_asset = (
                self.impact_lambda * np.power(trade_size, self.impact_exponent) * liq_factor
            )
            r_impact = -float(impact_per_asset.sum())
            impact_bps = float(impact_per_asset.sum() * 1e4)
        else:
            r_impact = 0.0
            impact_bps = 0.0

        # CVaR penalty (computed on net step returns before CVaR term)
        r_net = r_pnl + r_spread + r_impact
        if self.use_cvar:
            self._returns_buffer.append(r_net)
            if len(self._returns_buffer) > self.cvar_buffer_size:
                self._returns_buffer.pop(0)
            cvar = self._compute_cvar()
            r_cvar = self.cvar_lambda * cvar  # negative → penalises tail losses
        else:
            cvar = 0.0
            r_cvar = 0.0

        reward = r_net + r_cvar
        self._positions = action
        terminated = self._step >= self._n - 1

        info = {
            "r_pnl": r_pnl,
            "r_spread": r_spread,
            "r_impact": r_impact,
            "impact_bps": impact_bps,
            "fill_noise_bps": fill_noise_bps,
            "r_cvar": r_cvar,
            "cvar_5pct": cvar,
            "log_rets": log_rets.tolist(),
            "positions": self._positions.tolist(),
            "closes": cur_closes.tolist(),
            "trade_size": trade_size.tolist(),
        }
        return self._obs(), reward, terminated, False, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # CVaR
    # ------------------------------------------------------------------

    def _compute_cvar(self) -> float:
        """CVaR_α: expected step return in the worst α-fraction of recent steps.

        Returns a negative float when the agent has tail losses.
        Returns 0.0 until the buffer has at least 20 observations.
        """
        if len(self._returns_buffer) < 20:
            return 0.0
        rets = np.array(self._returns_buffer)
        threshold = np.quantile(rets, self.cvar_alpha)
        tail = rets[rets <= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(threshold)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_raw_features(self) -> np.ndarray:
        """Return (T, N_ASSETS, n_asset_features) raw feature array.

        n_asset_features is 10 (base) or 12 (when use_liquidity_features=True).
        """
        feats = []
        for a in range(N_ASSETS):
            o = self._prices[:, a, 0]
            h = self._prices[:, a, 1]
            l = self._prices[:, a, 2]
            c = self._prices[:, a, 3]
            v = self._prices[:, a, 4]

            log_ret = np.empty(self._n)
            log_ret[0] = 0.0
            log_ret[1:] = np.log(c[1:] / c[:-1])

            vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values
            vol_ratio = v / (vol_ma + 1e-8)

            price_f = np.stack(
                [log_ret, np.log(h / c), np.log(l / c), np.log(o / c), vol_ratio],
                axis=1,
            )

            if self.use_indicators and _TA_AVAILABLE:
                ind_f = self._compute_indicators(
                    pd.Series(c), pd.Series(h), pd.Series(l)
                )
            else:
                ind_f = np.zeros((self._n, N_INDICATOR_FEATURES))

            cols = [price_f, ind_f]

            if self.use_liquidity_features:
                liq_f = self._compute_liquidity_features(h, l, c, v)
                cols.append(liq_f)

            feats.append(np.concatenate(cols, axis=1))

        return np.stack(feats, axis=1)  # (T, N_ASSETS, n_asset_features)

    def _load_sentiment(self, path: str) -> np.ndarray:
        """Load sentiment_daily.parquet and align to env's trading-day index (self._dates).

        Returns (T, 3) float32 array: [bullish_frac, neutral_frac, headline_count_norm].
        Trading days without headline coverage receive neutral fill (0.5, 0.5, 0.0).
        """
        neutral_row = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        out = np.tile(neutral_row, (self._n, 1))

        try:
            sent_df = pd.read_parquet(path)
            sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.date
            sent_df = sent_df.set_index("date")

            for i, d in enumerate(self._dates):
                d_key = d.date() if hasattr(d, "date") else d
                if d_key in sent_df.index:
                    row = sent_df.loc[d_key, ["bullish_frac", "neutral_frac", "headline_count_norm"]]
                    out[i] = row.values.astype(np.float32)
        except Exception as exc:
            print(f"  ⚠ Sentiment load failed ({exc}); using neutral fill for all days.")

        covered = int((out[:, 2] > 0).sum())
        print(f"  Sentiment: {covered}/{self._n} trading days have real headline coverage "
              f"({covered/self._n*100:.1f}%).")
        return out

    def _build_spread_proxy(self) -> np.ndarray:
        """Per-asset smoothed (H-L)/C, shape (T, N_ASSETS). Used by C8 noise."""
        out = np.zeros((self._n, N_ASSETS), dtype=np.float64)
        for a in range(N_ASSETS):
            h = self._prices[:, a, 1]
            l = self._prices[:, a, 2]
            c = self._prices[:, a, 3]
            raw = (h - l) / (np.abs(c) + 1e-8)
            out[:, a] = (
                pd.Series(raw).rolling(LIQUIDITY_WINDOW, min_periods=1).mean().values
            )
        return out

    def _compute_liquidity_features(self, h, l, c, v) -> np.ndarray:
        """Phase C7: 2 per-asset features capturing liquidity-regime state.

        Feature 1 — volume z-score over LIQUIDITY_WINDOW (20 d):
            z_t = (v_t - μ20) / (σ20 + 1e-8), clipped to [-5, 5].
            Unusually thin days (z < 0) proxy hidden-liquidity withdrawal;
            unusually heavy days (z > 0) proxy forced unwinds / information events.

        Feature 2 — OHLC-dispersion effective-spread proxy, smoothed:
            raw_t = (h_t - l_t) / c_t  (non-negative by construction)
            out_t = rolling_mean_20(raw_t)
            Wide intraday ranges = wide effective spreads = illiquid regime.

        Both features use min_periods=1 and include v[t] / range[t] themselves —
        consistent with the existing vol_ratio pattern, since OHLCV for bar t is
        public at close(t) before action at t+1 is executed. No future leakage.
        """
        v_series = pd.Series(v)
        mean20 = v_series.rolling(LIQUIDITY_WINDOW, min_periods=1).mean().values
        std20 = v_series.rolling(LIQUIDITY_WINDOW, min_periods=1).std().fillna(0.0).values
        vol_z = (v - mean20) / (std20 + 1e-8)
        vol_z = np.clip(vol_z, -5.0, 5.0)

        raw_spread = (h - l) / (np.abs(c) + 1e-8)
        spread_proxy = (
            pd.Series(raw_spread).rolling(LIQUIDITY_WINDOW, min_periods=1).mean().values
        )

        return np.stack([vol_z, spread_proxy], axis=1)  # (T, 2)

    def _compute_indicators(self, close, high, low) -> np.ndarray:
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        rsi_norm = (rsi.fillna(50.0) - 50.0) / 50.0

        macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd().fillna(0.0)
        macd_sig = macd_obj.macd_signal().fillna(0.0)

        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_pct = bb.bollinger_pband().fillna(0.5)
        bb_wid = bb.bollinger_wband().fillna(0.0)

        return np.stack(
            [rsi_norm.values, macd_line.values, macd_sig.values, bb_pct.values, bb_wid.values],
            axis=1,
        )

    def _build_corr_features(self) -> np.ndarray:
        """Rolling 20-day return correlation upper triangle → (T, 10)."""
        closes = self._prices[:, :, 3]                        # (T, 5)
        log_rets = pd.DataFrame(np.log(closes[1:] / closes[:-1]))
        log_rets = pd.concat([pd.DataFrame(np.zeros((1, N_ASSETS))), log_rets], ignore_index=True)

        corr_feats = np.zeros((self._n, N_CORR_FEATURES))
        triu_idx = [(i, j) for i in range(N_ASSETS) for j in range(i + 1, N_ASSETS)]

        for t in range(CORR_WINDOW, self._n):
            window = log_rets.iloc[t - CORR_WINDOW : t]
            corr = window.corr().values
            corr_feats[t] = [corr[i, j] for i, j in triu_idx]

        return corr_feats  # NaN-free: rows 0..CORR_WINDOW-1 are zero

    def _obs(self) -> np.ndarray:
        start = max(0, self._step - self.window_size + 1)
        norm_slice = self._norm[start : self._step + 1]       # (≤W, norm_dim)

        n = len(norm_slice)
        pos_tile = np.tile(self._positions.astype(np.float32), (n, 1))  # (n, 5)
        parts = [norm_slice.astype(np.float32), pos_tile]

        if self.use_sentiment_features and self._sentiment is not None:
            sent_slice = self._sentiment[start : self._step + 1]         # (≤W, 3)
            parts.append(sent_slice)

        obs = np.concatenate(parts, axis=1)   # (n, N_FEATURES)

        if n < self.window_size:
            pad = np.zeros((self.window_size - n, obs.shape[1]), dtype=np.float32)
            obs = np.concatenate([pad, obs], axis=0)

        return obs


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _align(
    dfs: dict[str, pd.DataFrame], tickers: list[str]
) -> tuple[dict[str, pd.DataFrame], list]:
    """Inner-join all DataFrames on their date index.

    Returns (aligned_dfs, dates) where dates is the sorted common date list
    (used by Phase B sentiment alignment).
    """
    common = None
    for t in tickers:
        idx = dfs[t].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    dates = list(common)
    return {t: dfs[t].loc[common].reset_index(drop=True) for t in tickers}, dates


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300

    dfs = {}
    for ticker in TICKERS:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
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

    env = MultiAssetTradingEnv(dfs, {"window_size": 1})
    print(f"N_FEATURES computed: {env.N_FEATURES}")
    obs, _ = env.reset(seed=0)
    print(f"obs.shape = {obs.shape}")

    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"[multi_asset_env] sanity OK — steps={env._step}, total_reward={total_reward:.4f}")
