"""MVP Step 8: backtest the r007 uncertainty ensemble on the test split and
compare against realistic benchmarks.

Outputs (under <run_dir>/):
  - tearsheet.html       : QuantStats HTML report, strategy vs SPY buy-and-hold
  - backtest_metrics.csv : side-by-side metrics for strategy + all benchmarks
  - backtest_returns.csv : daily simple-return series for each strategy/benchmark

Benchmarks:
  - strategy       : 5-head uncertainty ensemble, scale=5
  - buy_and_hold   : 100% SPY from first test day
  - equal_weight   : daily-rebalanced 20/20/20/20/20 across the 5 ETFs
  - sixty_forty    : daily-rebalanced 60% SPY / 40% TLT
  - ma_crossover   : 50/200 SMA on SPY; long SPY when SMA50>SMA200 else flat (classic trend rule)

Note: vectorbt skipped for the MVP (descope per CLAUDE.md §3.4) — the env already
produces the per-step returns we need, and QuantStats handles everything else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from envs.multi_asset_env import MultiAssetTradingEnv, TICKERS
from envs.transformer_extractor import TransformerFeaturesExtractor  # noqa: F401
from scripts.download_data import download_multi, split_multi


def _strategy_returns(
    models, env, scale: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run uncertainty-sized ensemble on env.

    Returns:
        simple_returns : daily simple returns (for QuantStats / metrics)
        head_stds      : per-step mean head std (uncertainty signal)
        exec_stats     : execution statistics dict with keys:
                           total_impact_bps   — sum of per-step impact_bps
                           mean_impact_bps    — per-day average
                           total_spread_bps   — sum of |r_spread| × 1e4
                           mean_turnover      — mean sum(|Δpos|) per step
                           impl_shortfall_bps — total_impact_bps + total_spread_bps
    """
    obs, _ = env.reset(seed=0)
    log_rets, stds = [], []
    impact_bps_list, spread_bps_list, turnover_list = [], [], []

    while True:
        acts = np.stack(
            [m.predict(obs, deterministic=True)[0] for m in models], axis=0
        )
        mean = acts.mean(axis=0)
        std = acts.std(axis=0)
        action = mean * (1.0 / (1.0 + scale * std.mean()))
        obs, _, terminated, truncated, info = env.step(action)
        log_rets.append(info["r_pnl"] + info["r_spread"])
        stds.append(float(std.mean()))
        impact_bps_list.append(info.get("impact_bps", 0.0))
        spread_bps_list.append(abs(info["r_spread"]) * 1e4)
        turnover_list.append(float(np.sum(np.abs(info.get("trade_size", [0.0])))))
        if terminated or truncated:
            break

    log_rets = np.array(log_rets)
    simple = np.expm1(log_rets)

    total_impact = float(np.sum(impact_bps_list))
    total_spread = float(np.sum(spread_bps_list))
    exec_stats = {
        "total_impact_bps":   total_impact,
        "mean_impact_bps":    total_impact / max(len(impact_bps_list), 1),
        "total_spread_bps":   total_spread,
        "mean_spread_bps":    total_spread / max(len(spread_bps_list), 1),
        "mean_turnover":      float(np.mean(turnover_list)),
        "impl_shortfall_bps": total_impact + total_spread,
    }
    return simple, np.array(stds), exec_stats


def _benchmark_returns(test_dfs: dict, weights: dict) -> np.ndarray:
    """Daily-rebalanced portfolio of {ticker: weight} on test_dfs closes."""
    closes = pd.DataFrame({t: test_dfs[t]["close"].values for t in test_dfs})
    rets = closes.pct_change().fillna(0.0)
    w = np.array([weights.get(t, 0.0) for t in rets.columns])
    port = (rets.values * w).sum(axis=1)
    return port[1:]  # drop first (NaN-pctchange-filled-0) row to match strategy length


def _ma_crossover_returns(
    full_spy: pd.DataFrame,
    test_index: pd.Index,
    fast: int = 50,
    slow: int = 200,
) -> np.ndarray:
    """Classic 50/200-SMA trend rule on SPY.

    Signal at t is computed from closes up to and including t-1 (no lookahead),
    then applied to the simple return on day t. Long SPY (weight 1) when
    SMA_fast(t-1) > SMA_slow(t-1), else flat (weight 0, sitting in cash).

    Uses the full-history closes (including pre-test period) so that the
    first test-day signal already has a fully-warmed 200-day window.
    Returned array is aligned to `test_index`.
    """
    closes = full_spy["close"].copy()
    sma_f = closes.rolling(fast).mean()
    sma_s = closes.rolling(slow).mean()
    signal = (sma_f > sma_s).astype(float).shift(1).fillna(0.0)  # lag 1 → no lookahead
    rets = closes.pct_change().fillna(0.0)
    strat = (signal * rets).reindex(test_index).fillna(0.0).values
    return strat


def _twap_returns(
    test_dfs: dict,
    weights: dict,
    ramp_days: int = 10,
) -> np.ndarray:
    """TWAP execution baseline: linear ramp into target weights over `ramp_days`.

    Models the implementation drag of spreading entry over multiple days instead of
    instant rebalancing. Actual position on day t:
        pos_t = min(t / ramp_days, 1.0) × target_weight_per_asset

    After ramp_days the portfolio is fully invested and held constant (no further
    rebalancing). Uses closing prices for daily PnL. The spread cost is NOT included
    here — this is purely a return-impact comparison.

    Returned array has the same length as _benchmark_returns output.
    """
    closes = pd.DataFrame({t: test_dfs[t]["close"].values for t in test_dfs})
    rets = closes.pct_change().fillna(0.0).values           # (T, N_ASSETS)
    target_w = np.array([weights.get(t, 0.0) for t in closes.columns])

    # Iterate over tradeable days (same indexing as _benchmark_returns which drops row 0).
    # Day 0 (output index 0): scale=0 — just entered, no position yet.
    # Day t: scale = min(t / ramp_days, 1.0) — gradually build to full exposure.
    # Day ramp_days+: scale=1.0 — identical to instant equal-weight from here on.
    T_out = len(rets) - 1
    port = np.zeros(T_out)
    for t in range(T_out):
        scale = min(t / max(ramp_days, 1), 1.0)
        port[t] = float((rets[t + 1] * (scale * target_w)).sum())

    return port


def _vwap_proxy_returns(test_dfs: dict, weights: dict) -> np.ndarray:
    """VWAP-proxy benchmark: execute at OHLC4 = (O + H + L + C) / 4.

    Approximates the average fill price for an order spread uniformly across
    the trading session. At daily-bar resolution, OHLC4 is the closest proxy
    available without tick data.

    Return on day t = (OHLC4_t / OHLC4_{t-1}) - 1, weighted by `weights`.
    """
    ohlc4 = pd.DataFrame(
        {
            t: (
                test_dfs[t]["open"].values
                + test_dfs[t]["high"].values
                + test_dfs[t]["low"].values
                + test_dfs[t]["close"].values
            ) / 4.0
            for t in test_dfs
        }
    )
    rets = ohlc4.pct_change().fillna(0.0)
    w = np.array([weights.get(t, 0.0) for t in rets.columns])
    port = (rets.values * w).sum(axis=1)
    return port[1:]   # drop row 0 for length parity


def _align_lengths(*arrs: np.ndarray) -> list[np.ndarray]:
    n = min(len(a) for a in arrs)
    return [a[-n:] for a in arrs]


def _metrics(returns: np.ndarray, periods: int = 252) -> dict:
    """Hand-rolled metrics so we don't need quantstats for the side-by-side table."""
    r = np.asarray(returns)
    total_return = float(np.prod(1 + r) - 1)
    ann_return = float((1 + total_return) ** (periods / len(r)) - 1) if len(r) else 0.0
    vol = float(r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0
    sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods))
    downside = r[r < 0]
    sortino = float(
        r.mean() / (downside.std() + 1e-12) * np.sqrt(periods)
    ) if len(downside) else float("inf")
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    max_dd = float(dd.max()) if len(dd) else 0.0
    calmar = float(ann_return / max_dd) if max_dd > 0 else float("inf")
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_days": len(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/ensemble_r007",
                    help="dir with per-seed subdirs containing model_best_val.zip")
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--ckpt-name", default="model_best_val.zip")
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--no-tearsheet", action="store_true",
                    help="skip QuantStats HTML generation (metrics CSV still produced)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    paths = sorted(glob.glob(os.path.join(args.run_dir, "*", args.ckpt_name)))
    if not paths:
        raise FileNotFoundError(f"no {args.ckpt_name} under {args.run_dir}/*/")
    print(f"[backtest] loading {len(paths)} ensemble members")
    models = [PPO.load(p, device="cpu") for p in paths]

    ma = cfg["multi_asset"]; d = cfg["data"]
    dfs = download_multi(ma["tickers"], d["train_start"], d["test_end"], d["cache_dir"])
    train_dfs, _, test_dfs = split_multi(dfs, d)

    env_cfg = {k: ma.get(k) for k in [
        "tickers", "window_size", "half_spread", "use_indicators",
    ]}
    env_cfg["use_cvar"] = False
    # Phase C7: obs width must match training env (else models fail to predict).
    env_cfg["use_liquidity_features"] = ma.get("use_liquidity_features", False)
    # Phase C8: noise is OFF for both backtest replays so the r007↔r008 compare
    # isolates *learned* behaviour under impact, not evaluation stochasticity.
    env_cfg["liquidity_noise_enabled"] = False
    norm_stats = MultiAssetTradingEnv(train_dfs, ma).get_norm_stats()

    # --- Primary env run (impact OFF — matches original r007 numbers) ---
    env = MultiAssetTradingEnv(test_dfs, env_cfg, norm_stats=norm_stats)
    strategy_r, head_stds, _ = _strategy_returns(models, env, args.scale)
    print(f"[backtest] strategy n_days={len(strategy_r)}  "
          f"mean_head_std={head_stds.mean():.3f}")

    # --- Impact-enabled replay (impact ON — for execution stats only) ---
    env_impact_cfg = {**env_cfg, "impact_enabled": True}
    env_impact = MultiAssetTradingEnv(test_dfs, env_impact_cfg, norm_stats=norm_stats)
    _, _, exec_stats = _strategy_returns(models, env_impact, args.scale)
    print(
        f"[backtest] execution stats — "
        f"impl_shortfall={exec_stats['impl_shortfall_bps']:.1f} bps  "
        f"mean_impact={exec_stats['mean_impact_bps']:.2f} bps/day  "
        f"mean_spread={exec_stats['mean_spread_bps']:.2f} bps/day  "
        f"mean_turnover={exec_stats['mean_turnover']:.4f}"
    )
    out_exec = os.path.join(args.run_dir, "execution_stats.csv")
    pd.DataFrame([exec_stats]).to_csv(out_exec, index=False)
    print(f"[backtest] wrote {out_exec}")

    # Benchmarks on raw test_dfs — all aligned to strategy length
    ew_weights = {t: 0.2 for t in TICKERS}
    bh   = _benchmark_returns(test_dfs, {"SPY": 1.0})
    ew   = _benchmark_returns(test_dfs, ew_weights)
    s64  = _benchmark_returns(test_dfs, {"SPY": 0.6, "TLT": 0.4})
    # MA crossover uses full-history SPY so the 200-day SMA is warm on test-day-1
    mac  = _ma_crossover_returns(dfs["SPY"], test_dfs["SPY"].index)[1:]
    # TWAP: equal-weight target ramped in over 10 days — models implementation drag
    twap = _twap_returns(test_dfs, ew_weights, ramp_days=10)
    # VWAP-proxy: equal-weight at OHLC4 fills vs closing-price fills
    vwap = _vwap_proxy_returns(test_dfs, ew_weights)
    strategy_r, bh, ew, s64, mac, twap, vwap = _align_lengths(
        strategy_r, bh, ew, s64, mac, twap, vwap
    )

    all_rets = {
        "strategy":     strategy_r,
        "buy_and_hold": bh,
        "equal_weight": ew,
        "sixty_forty":  s64,
        "ma_crossover": mac,
        "twap_ew":      twap,
        "vwap_proxy_ew": vwap,
    }

    metrics_rows = []
    for name, r in all_rets.items():
        m = _metrics(r)
        m["name"] = name
        metrics_rows.append(m)
        print(f"[{name:12s}]  ann_ret={m['ann_return']:+.2%}  "
              f"sharpe={m['sharpe']:+.3f}  sortino={m['sortino']:+.3f}  "
              f"max_dd={m['max_drawdown']:.2%}  calmar={m['calmar']:+.2f}")

    out_metrics = os.path.join(args.run_dir, "backtest_metrics.csv")
    pd.DataFrame(metrics_rows).set_index("name").to_csv(out_metrics)
    print(f"[backtest] wrote {out_metrics}")

    # Date-indexed returns for QuantStats tearsheet
    test_idx = test_dfs[TICKERS[0]].index
    idx = test_idx[-len(strategy_r):]
    ret_df = pd.DataFrame({k: v for k, v in all_rets.items()}, index=idx)
    out_ret = os.path.join(args.run_dir, "backtest_returns.csv")
    ret_df.to_csv(out_ret)
    print(f"[backtest] wrote {out_ret}")

    if not args.no_tearsheet:
        try:
            import quantstats as qs
            strategy_series = pd.Series(strategy_r, index=idx, name="Strategy")
            bh_series       = pd.Series(bh,         index=idx, name="SPY_BH")
            out_html = os.path.join(args.run_dir, "tearsheet.html")
            qs.reports.html(
                strategy_series, benchmark=bh_series, output=out_html, title="r007 Ensemble vs SPY B&H"
            )
            print(f"[backtest] wrote {out_html}")
        except Exception as e:
            print(f"[backtest] QuantStats tearsheet failed: {e!r}")


if __name__ == "__main__":
    main()
