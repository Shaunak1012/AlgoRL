"""MVP Step 9: ablation study — rebuild the results table from existing checkpoints.

Each row removes one component of the full system and measures test-set impact.
No retraining required; all checkpoints already exist.

Rows (ordered by cumulative contribution):
  A. No multi-asset / no CVaR / MLP     → r003 model_final (MLP, multi-asset, λ=0)
  B. + CVaR (λ=0.01)                    → r005 model_final (MLP + CVaR)
  C. + Transformer, no early stop       → r006 model_final (Transformer, overfit)
  D. + early stopping                   → r006 model_50000_steps
  E. + 5-seed ensemble (mean)           → r007 ensemble 'mean' policy
  F. + uncertainty sizing (FULL SYSTEM) → r007 ensemble 'uncertainty' policy

Usage:
    python scripts/run_ablations.py --out runs/ablation_table.csv
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
from scripts.backtest import _metrics
from scripts.download_data import download_multi, split_multi


def _rollout(models, env, mode: str, scale: float) -> np.ndarray:
    """Return simple daily returns for the chosen aggregation mode."""
    obs, _ = env.reset(seed=0)
    log_rets = []
    while True:
        acts = np.stack(
            [m.predict(obs, deterministic=True)[0] for m in models], axis=0
        )
        mean = acts.mean(axis=0)
        std = acts.std(axis=0)
        if mode == "single":
            action = acts[0]
        elif mode == "mean":
            action = mean
        elif mode == "uncertainty":
            action = mean * (1.0 / (1.0 + scale * std.mean()))
        else:
            raise ValueError(mode)
        obs, _, terminated, truncated, info = env.step(action)
        log_rets.append(info["r_pnl"] + info["r_spread"])
        if terminated or truncated:
            break
    return np.expm1(np.array(log_rets))


def _env_for(run_dir: str, test_dfs: dict, train_dfs: dict) -> MultiAssetTradingEnv:
    """Build a test env with the same window_size / indicators as the run's config."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        c = yaml.safe_load(f)
    ma = c.get("multi_asset", c.get("env", {}))
    env_cfg = {
        "tickers":        ma.get("tickers", TICKERS),
        "window_size":    ma.get("window_size", 1),
        "half_spread":    ma.get("half_spread", 0.0001),
        "use_indicators": ma.get("use_indicators", True),
        "use_cvar":       False,  # strip CVaR for clean PnL eval (same as backtest.py)
    }
    # Normalisation stats fit on THIS run's train env so obs match what the model saw.
    norm_stats = MultiAssetTradingEnv(train_dfs, {**env_cfg, **ma}).get_norm_stats()
    return MultiAssetTradingEnv(test_dfs, env_cfg, norm_stats=norm_stats)


def _align(*arrs):
    n = min(len(a) for a in arrs)
    return [a[-n:] for a in arrs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ablation_table.csv")
    ap.add_argument("--scale", type=float, default=5.0)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Load test data once
    with open("configs/transformer.yaml") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    dfs = download_multi(TICKERS, d["train_start"], d["test_end"], d["cache_dir"])
    train_dfs, _, test_dfs = split_multi(dfs, d)

    ablations = [
        # label, run_dirs, checkpoint_name, mode
        ("A: MLP multi-asset, no CVaR",
         ["runs/20260420_115238_ppo_multi_asset"], "model_final.zip", "single"),
        ("B: + CVaR (λ=0.01)",
         ["runs/20260420_115843_ppo_multi_cvar_l001"], "model_final.zip", "single"),
        ("C: + Transformer, no early stop",
         ["runs/20260420_163618_r006_transformer"], "model_final.zip", "single"),
        ("D: + early stopping (50k)",
         ["runs/20260420_163618_r006_transformer"], "model_50000_steps.zip", "single"),
        ("E: + 5-seed ensemble (mean)",
         sorted(glob.glob("runs/ensemble_r007/*_seed*")), "model_best_val.zip", "mean"),
        ("F: + uncertainty sizing (FULL)",
         sorted(glob.glob("runs/ensemble_r007/*_seed*")), "model_best_val.zip", "uncertainty"),
    ]

    rows = []
    returns_by_row = {}
    for label, run_dirs, ckpt, mode in ablations:
        paths = [os.path.join(d, ckpt) for d in run_dirs]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[skip] {label} — missing: {missing}")
            continue

        # Use the FIRST run_dir's config to build the env (all ensemble members share config)
        env = _env_for(run_dirs[0], test_dfs, train_dfs)
        models = [PPO.load(p, device="cpu") for p in paths]
        rets = _rollout(models, env, mode, args.scale)

        m = _metrics(rets)
        m["label"] = label
        m["n_members"] = len(models)
        m["mode"] = mode
        rows.append(m)
        returns_by_row[label] = rets
        print(f"[{label:40s}] n={len(rets)}  ann_ret={m['ann_return']:+.2%}  "
              f"sharpe={m['sharpe']:+.3f}  max_dd={m['max_drawdown']:.2%}  "
              f"calmar={m['calmar']:+.2f}")

    # Tabulate
    df = pd.DataFrame(rows).set_index("label")[
        ["n_members", "mode", "ann_return", "sharpe", "sortino", "max_drawdown", "calmar", "n_days"]
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print(f"\nwrote {args.out}\n")
    print(df.to_string())

    # Markdown snippet for the paper
    md_path = args.out.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Ablation table (test split 2022–2024)\n\n")
        f.write(df.to_markdown(floatfmt=".4f"))
        f.write("\n")
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
