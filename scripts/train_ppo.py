"""MVP Step 2: PPO baseline with MLP policy on single-asset SPY.

Usage:
    python scripts/train_ppo.py --config configs/base.yaml [--tag my_run]

Outputs (in runs/<timestamp>_<tag>/):
    config.yaml       — snapshot of config used
    learning_curve.csv — ep_rew_mean logged every eval_freq steps
    model_final.zip   — final SB3 checkpoint
    model_<step>.zip  — intermediate checkpoints
"""
import argparse
import csv
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.trading_env import SingleAssetTradingEnv
from scripts.download_data import download_ticker, split_df


# ── Callback: log ep_rew_mean to CSV ─────────────────────────────────────────

class LearningCurveCallback(BaseCallback):
    def __init__(self, log_path: str, eval_freq: int = 1000):
        super().__init__()
        self.log_path = log_path
        self.eval_freq = eval_freq
        self._last_log = 0
        self._file = None
        self._writer = None

    def _on_training_start(self):
        self._file = open(self.log_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "ep_rew_mean", "ep_len_mean"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.eval_freq:
            return True
        self._last_log = self.num_timesteps

        ep_info_buf = self.model.ep_info_buffer
        if len(ep_info_buf) == 0:
            return True
        ep_rew_mean = float(np.mean([ep["r"] for ep in ep_info_buf]))
        ep_len_mean = float(np.mean([ep["l"] for ep in ep_info_buf]))
        self._writer.writerow([self.num_timesteps, ep_rew_mean, ep_len_mean])
        self._file.flush()
        print(f"  step={self.num_timesteps:7d}  ep_rew_mean={ep_rew_mean:+.4f}  ep_len_mean={ep_len_mean:.0f}")
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tag", default="ppo_mlp_baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Seeding ──────────────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Run directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["training"]["runs_dir"], f"{timestamp}_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    print(f"[train] run dir: {run_dir}")

    # ── Data ─────────────────────────────────────────────────────────────
    ticker = cfg["env"]["ticker"]
    df = download_ticker(
        ticker,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["test_end"],
        cache_dir=cfg["data"]["cache_dir"],
    )
    train_df, val_df, _ = split_df(df, cfg["data"])

    # ── Envs ─────────────────────────────────────────────────────────────
    env_cfg = cfg["env"]
    train_env_raw = SingleAssetTradingEnv(train_df, env_cfg)
    norm_stats = train_env_raw.get_norm_stats()

    def make_train_env():
        e = SingleAssetTradingEnv(train_df, env_cfg, norm_stats=norm_stats)
        return Monitor(e)

    train_env = DummyVecEnv([make_train_env])

    # ── PPO ──────────────────────────────────────────────────────────────
    ppo_cfg = cfg["ppo"]
    policy_kwargs = {"net_arch": ppo_cfg["net_arch"]}

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
        tensorboard_log=None,
        seed=seed,
        verbose=0,
    )

    callbacks = [
        LearningCurveCallback(
            log_path=os.path.join(run_dir, "learning_curve.csv"),
            eval_freq=2000,
        ),
        CheckpointCallback(
            save_freq=cfg["training"]["checkpoint_every"],
            save_path=run_dir,
            name_prefix="model",
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[train] starting PPO — {cfg['training']['total_steps']:,} steps  seed={seed}")
    t0 = time.time()
    model.learn(total_timesteps=cfg["training"]["total_steps"], callback=callbacks)
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s  ({cfg['training']['total_steps']/elapsed:.0f} steps/s)")

    model.save(os.path.join(run_dir, "model_final"))
    print(f"[train] model saved → {run_dir}/model_final.zip")
    print(f"[train] learning curve → {run_dir}/learning_curve.csv")

    # ── Quick val rollout ─────────────────────────────────────────────────
    val_env = SingleAssetTradingEnv(val_df, env_cfg, norm_stats=norm_stats)
    obs, _ = val_env.reset(seed=seed)
    val_rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = val_env.step(int(action))
        val_rewards.append(reward)
        if terminated or truncated:
            break
    val_total = float(np.sum(val_rewards))
    val_sharpe = float(np.mean(val_rewards) / (np.std(val_rewards) + 1e-8) * np.sqrt(252))
    print(f"[val]   total_log_return={val_total:+.4f}  annualized_sharpe≈{val_sharpe:.2f}")

    return run_dir


if __name__ == "__main__":
    main()
