"""MVP Step 4: PPO on multi-asset env (5 ETFs, continuous portfolio weights).

Usage:
    python scripts/train_multi_asset.py --config configs/base.yaml [--tag my_run]
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

from envs.multi_asset_env import MultiAssetTradingEnv
from scripts.download_data import download_multi, split_multi


class LearningCurveCallback(BaseCallback):
    def __init__(self, log_path: str, eval_freq: int = 2000):
        super().__init__()
        self.log_path = log_path
        self.eval_freq = eval_freq
        self._last = 0
        self._f = self._w = None
        # reward component accumulators (reset each eval window)
        self._r_pnl: list[float] = []
        self._r_spread: list[float] = []
        self._r_cvar: list[float] = []

    def _on_training_start(self):
        self._f = open(self.log_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestep", "ep_rew_mean", "ep_len_mean",
                          "r_pnl_mean", "r_spread_mean", "r_cvar_mean", "cvar_5pct_mean"])

    def _on_step(self) -> bool:
        # Accumulate reward components from infos
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "r_pnl" in info:
                    self._r_pnl.append(info["r_pnl"])
                    self._r_spread.append(info["r_spread"])
                    self._r_cvar.append(info["r_cvar"])

        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf:
            return True

        rew = float(np.mean([e["r"] for e in buf]))
        llen = float(np.mean([e["l"] for e in buf]))
        r_pnl_m   = float(np.mean(self._r_pnl))   if self._r_pnl   else 0.0
        r_spr_m    = float(np.mean(self._r_spread)) if self._r_spread else 0.0
        r_cvar_m   = float(np.mean(self._r_cvar))  if self._r_cvar  else 0.0

        # Reward component scale check (CLAUDE.md §6.3)
        total_mag = abs(r_pnl_m) + abs(r_spr_m) + abs(r_cvar_m) + 1e-8
        if abs(r_cvar_m) / total_mag > 0.8:
            print(f"  [WARN] CVaR penalty dominates reward ({abs(r_cvar_m)/total_mag:.0%}) — consider reducing cvar_lambda")

        self._w.writerow([self.num_timesteps, rew, llen, r_pnl_m, r_spr_m, r_cvar_m, r_cvar_m])
        self._f.flush()
        self._r_pnl.clear(); self._r_spread.clear(); self._r_cvar.clear()

        print(f"  step={self.num_timesteps:7d}  ep_rew={rew:+.4f}  "
              f"pnl={r_pnl_m:+.5f}  spread={r_spr_m:+.5f}  cvar={r_cvar_m:+.5f}")
        return True

    def _on_training_end(self):
        if self._f:
            self._f.close()


def evaluate(model, env_dfs: dict, env_cfg: dict, norm_stats: dict, seed: int) -> dict:
    """Evaluate on pure PnL (no CVaR penalty) for fair cross-run comparison."""
    eval_cfg = {**env_cfg, "use_cvar": False}  # strip CVaR for clean metrics
    env = MultiAssetTradingEnv(env_dfs, eval_cfg, norm_stats=norm_stats)
    obs, _ = env.reset(seed=seed)
    pnl_returns = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        pnl_returns.append(info["r_pnl"] + info["r_spread"])  # net PnL only
        if terminated or truncated:
            break
    r_arr = np.array(pnl_returns)
    cum = np.cumsum(r_arr)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {
        "total_log_return": float(r_arr.sum()),
        "sharpe_approx": float(r_arr.mean() / (r_arr.std() + 1e-8) * np.sqrt(252)),
        "max_drawdown": dd,
        "n_steps": len(pnl_returns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tag", default="ppo_multi_asset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["training"]["runs_dir"], f"{timestamp}_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    print(f"[train] run dir: {run_dir}")

    # ── Data ─────────────────────────────────────────────────────────────
    ma_cfg = cfg["multi_asset"]
    tickers = ma_cfg["tickers"]
    data_cfg = cfg["data"]

    dfs = download_multi(tickers, data_cfg["train_start"], data_cfg["test_end"], data_cfg["cache_dir"])
    train_dfs, val_dfs, test_dfs = split_multi(dfs, data_cfg)

    # ── Env ──────────────────────────────────────────────────────────────
    env_cfg = {
        "tickers":         tickers,
        "window_size":     ma_cfg["window_size"],
        "half_spread":     ma_cfg["half_spread"],
        "use_indicators":  ma_cfg["use_indicators"],
        "use_cvar":        ma_cfg.get("use_cvar", False),
        "cvar_alpha":      ma_cfg.get("cvar_alpha", 0.05),
        "cvar_lambda":     ma_cfg.get("cvar_lambda", 0.1),
        "cvar_buffer_size":ma_cfg.get("cvar_buffer_size", 252),
    }
    train_env_raw = MultiAssetTradingEnv(train_dfs, env_cfg)
    norm_stats = train_env_raw.get_norm_stats()

    print(f"[train] obs_space={train_env_raw.observation_space.shape}  "
          f"action_space={train_env_raw.action_space.shape}  "
          f"N_FEATURES={train_env_raw.N_FEATURES}")

    def make_env():
        e = MultiAssetTradingEnv(train_dfs, env_cfg, norm_stats=norm_stats)
        return Monitor(e)

    train_env = DummyVecEnv([make_env])

    # ── PPO ──────────────────────────────────────────────────────────────
    ppo_cfg = cfg["ppo"]
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
        policy_kwargs={"net_arch": ppo_cfg["net_arch"]},
        tensorboard_log=None,
        seed=seed,
        verbose=0,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[train] starting PPO — {cfg['training']['total_steps']:,} steps  seed={seed}")
    t0 = time.time()
    model.learn(
        total_timesteps=cfg["training"]["total_steps"],
        callback=[
            LearningCurveCallback(os.path.join(run_dir, "learning_curve.csv")),
            CheckpointCallback(cfg["training"]["checkpoint_every"], run_dir, "model"),
        ],
    )
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s  ({cfg['training']['total_steps']/elapsed:.0f} steps/s)")
    model.save(os.path.join(run_dir, "model_final"))

    # ── Evaluate ──────────────────────────────────────────────────────────
    for name, split_dfs in [("val", val_dfs), ("test", test_dfs)]:
        s = evaluate(model, split_dfs, env_cfg, norm_stats, seed)
        print(f"[{name}]  total_log_ret={s['total_log_return']:+.4f}  "
              f"sharpe≈{s['sharpe_approx']:.2f}  max_dd={s['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
