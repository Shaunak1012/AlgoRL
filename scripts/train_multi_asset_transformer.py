"""MVP Step 6: PPO with causal Transformer encoder on multi-asset env.

Usage:
    python scripts/train_multi_asset_transformer.py --config configs/transformer.yaml [--tag r006_transformer]

Differences vs train_multi_asset.py:
  - window_size=60 (3D obs)
  - policy_kwargs wires TransformerFeaturesExtractor
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.multi_asset_env import MultiAssetTradingEnv
from envs.transformer_extractor import TransformerFeaturesExtractor
from scripts.callbacks import ValEvalCallback
from scripts.download_data import download_multi, split_multi
from scripts.train_multi_asset import LearningCurveCallback, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--tag", default="r006_transformer")
    parser.add_argument("--steps", type=int, default=None, help="override total_steps (smoke tests)")
    parser.add_argument("--seed", type=int, default=None, help="override training seed (for ensemble sweeps)")
    parser.add_argument("--val-eval-freq", type=int, default=10_000,
                        help="val-Sharpe eval period; disable with 0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    seed = cfg["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["training"]["runs_dir"], f"{timestamp}_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    print(f"[train] run dir: {run_dir}")

    ma_cfg = cfg["multi_asset"]
    data_cfg = cfg["data"]

    dfs = download_multi(ma_cfg["tickers"], data_cfg["train_start"], data_cfg["test_end"], data_cfg["cache_dir"])
    train_dfs, val_dfs, test_dfs = split_multi(dfs, data_cfg)

    env_cfg = {
        "tickers":         ma_cfg["tickers"],
        "window_size":     ma_cfg["window_size"],
        "half_spread":     ma_cfg["half_spread"],
        "use_indicators":  ma_cfg["use_indicators"],
        "use_cvar":        ma_cfg.get("use_cvar", False),
        "cvar_alpha":      ma_cfg.get("cvar_alpha", 0.05),
        "cvar_lambda":     ma_cfg.get("cvar_lambda", 0.01),
        "cvar_buffer_size":ma_cfg.get("cvar_buffer_size", 252),
        # Phase A2 — market impact (default off)
        "impact_enabled":      ma_cfg.get("impact_enabled", False),
        "impact_lambda":       ma_cfg.get("impact_lambda", 1e-3),
        "impact_exponent":     ma_cfg.get("impact_exponent", 1.5),
        # Phase C7/C8 — liquidity-aware env (default off for r007 parity)
        "use_liquidity_features":  ma_cfg.get("use_liquidity_features", False),
        "liquidity_noise_enabled": ma_cfg.get("liquidity_noise_enabled", False),
        "liquidity_noise_base":    ma_cfg.get("liquidity_noise_base", 5e-4),
        "liquidity_noise_k":       ma_cfg.get("liquidity_noise_k", 10.0),
        # Phase B — FinBERT sentiment features (default off for r007/r008 parity)
        "use_sentiment_features":  ma_cfg.get("use_sentiment_features", False),
    }
    train_env_raw = MultiAssetTradingEnv(train_dfs, env_cfg)
    norm_stats = train_env_raw.get_norm_stats()

    print(f"[train] obs_space={train_env_raw.observation_space.shape}  "
          f"action_space={train_env_raw.action_space.shape}")

    def make_env():
        return Monitor(MultiAssetTradingEnv(train_dfs, env_cfg, norm_stats=norm_stats))

    train_env = DummyVecEnv([make_env])

    ppo_cfg = cfg["ppo"]
    tcfg = cfg["transformer"]
    policy_kwargs = {
        "net_arch": ppo_cfg["net_arch"],
        "features_extractor_class": TransformerFeaturesExtractor,
        "features_extractor_kwargs": {
            "d_model":         tcfg["d_model"],
            "n_layers":        tcfg["n_layers"],
            "n_heads":         tcfg["n_heads"],
            "dim_feedforward": tcfg["dim_feedforward"],
            "dropout":         tcfg["dropout"],
        },
    }

    model = PPO(
        policy=ppo_cfg["policy"],
        env=train_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] policy params: {n_params:,}  device={model.device}")
    print(f"[train] starting PPO — {cfg['training']['total_steps']:,} steps  seed={seed}")

    callbacks = [
        LearningCurveCallback(os.path.join(run_dir, "learning_curve.csv")),
        CheckpointCallback(cfg["training"]["checkpoint_every"], run_dir, "model"),
    ]
    if args.val_eval_freq > 0:
        def val_env_factory():
            return MultiAssetTradingEnv(val_dfs, {**env_cfg, "use_cvar": False}, norm_stats=norm_stats)
        callbacks.append(
            ValEvalCallback(
                val_env_factory=val_env_factory,
                log_path=os.path.join(run_dir, "val_curve.csv"),
                best_ckpt_path=os.path.join(run_dir, "model_best_val"),
                eval_freq=args.val_eval_freq,
                seed=seed,
            )
        )

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["training"]["total_steps"],
        callback=callbacks,
    )
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s  ({cfg['training']['total_steps']/elapsed:.0f} steps/s)")
    model.save(os.path.join(run_dir, "model_final"))

    def _eval_and_print(m, label_prefix):
        for name, split_dfs in [("val", val_dfs), ("test", test_dfs)]:
            s = evaluate(m, split_dfs, env_cfg, norm_stats, seed)
            print(f"[{label_prefix} {name}]  total_log_ret={s['total_log_return']:+.4f}  "
                  f"sharpe≈{s['sharpe_approx']:.2f}  max_dd={s['max_drawdown']:.4f}")

    _eval_and_print(model, "final")

    best_path = os.path.join(run_dir, "model_best_val.zip")
    if args.val_eval_freq > 0 and os.path.exists(best_path):
        best_model = PPO.load(best_path, device=model.device)
        _eval_and_print(best_model, "best_val")


if __name__ == "__main__":
    main()
