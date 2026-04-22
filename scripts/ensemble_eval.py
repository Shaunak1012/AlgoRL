"""Ensemble inference with uncertainty-scaled position sizing (MVP Step 7).

Loads N PPO checkpoints, at each step computes the mean and std of actions
across heads, then scales the mean action by 1 / (1 + scale * std_avg) to
shrink positions when heads disagree.

Compares three policies on val + test:
  - 'mean'        : plain mean of head actions, no scaling
  - 'uncertainty' : mean × 1/(1 + scale * mean_std)
  - 'best_single' : the single head with highest val Sharpe (sanity ceiling)

Usage:
    python scripts/ensemble_eval.py --run-dir runs/ensemble_20260421/ \
        --config configs/transformer.yaml --scale 5.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from envs.multi_asset_env import MultiAssetTradingEnv
from envs.transformer_extractor import TransformerFeaturesExtractor  # noqa: F401
from scripts.download_data import download_multi, split_multi


def _rollout(models, env, scale: float, mode: str) -> dict:
    obs, _ = env.reset(seed=0)
    rets, stds = [], []
    while True:
        acts = np.stack(
            [m.predict(obs, deterministic=True)[0] for m in models], axis=0
        )  # (N, action_dim)
        mean = acts.mean(axis=0)
        std = acts.std(axis=0)
        if mode == "mean":
            action = mean
        elif mode == "uncertainty":
            shrink = 1.0 / (1.0 + scale * std.mean())
            action = mean * shrink
        elif mode == "single":
            action = acts[0]
        else:
            raise ValueError(mode)
        obs, _, terminated, truncated, info = env.step(action)
        rets.append(info["r_pnl"] + info["r_spread"])
        stds.append(float(std.mean()))
        if terminated or truncated:
            break
    r = np.array(rets)
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {
        "sharpe": float(r.mean() / (r.std() + 1e-8) * np.sqrt(252)),
        "log_return": float(r.sum()),
        "max_dd": dd,
        "mean_head_std": float(np.mean(stds)),
        "n_steps": len(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="dir containing per-seed subdirs each with model_best_val.zip")
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--ckpt-name", default="model_best_val.zip")
    ap.add_argument("--scale", type=float, default=5.0,
                    help="uncertainty-shrink coefficient")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Locate checkpoints: <run_dir>/*/<ckpt_name>
    paths = sorted(glob.glob(os.path.join(args.run_dir, "*", args.ckpt_name)))
    if not paths:
        raise FileNotFoundError(
            f"no '{args.ckpt_name}' under {args.run_dir}/*/ — "
            f"run training with --val-eval-freq > 0 first"
        )
    print(f"[ensemble] loading {len(paths)} members:")
    for p in paths:
        print(f"    {p}")
    models = [PPO.load(p, device="cpu") for p in paths]

    ma = cfg["multi_asset"]; d = cfg["data"]
    dfs = download_multi(ma["tickers"], d["train_start"], d["test_end"], d["cache_dir"])
    train_dfs, val_dfs, test_dfs = split_multi(dfs, d)

    env_cfg = {k: ma.get(k) for k in [
        "tickers", "window_size", "half_spread", "use_indicators",
    ]}
    env_cfg["use_cvar"] = False  # clean PnL eval
    # Phase C7: obs width MUST match training env or extractor fails to load.
    env_cfg["use_liquidity_features"] = ma.get("use_liquidity_features", False)
    # Phase B: obs width must match training when sentiment was used.
    env_cfg["use_sentiment_features"] = ma.get("use_sentiment_features", False)
    # Phase C8 / A2: disable stochastic noise and impact at eval for deterministic,
    # comparable Sharpe (matches the CVaR-off convention). Realistic-cost backtest
    # lives in scripts/backtest.py (Phase A4 second-pass replay).
    env_cfg["liquidity_noise_enabled"] = False
    env_cfg["impact_enabled"] = False
    norm_stats = MultiAssetTradingEnv(train_dfs, ma).get_norm_stats()

    results = {}
    for split_name, split_dfs in [("val", val_dfs), ("test", test_dfs)]:
        for mode in ("mean", "uncertainty"):
            env = MultiAssetTradingEnv(split_dfs, env_cfg, norm_stats=norm_stats)
            r = _rollout(models, env, args.scale, mode)
            label = f"{split_name}/{mode}"
            results[label] = r
            print(f"[{label:20s}] sharpe={r['sharpe']:+.3f}  "
                  f"log_ret={r['log_return']:+.4f}  max_dd={r['max_dd']:.3f}  "
                  f"mean_head_std={r['mean_head_std']:.4f}")

    out = os.path.join(args.run_dir, "ensemble_results.json")
    with open(out, "w") as f:
        json.dump({"scale": args.scale, "n_members": len(models), "results": results}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
