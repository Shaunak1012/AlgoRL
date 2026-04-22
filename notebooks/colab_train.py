# =============================================================================
# COLAB TRAINING SCRIPT — algo-trading-rl
# =============================================================================
# Instructions:
#   1. Zip your local project folder (the whole algo-trading-rl directory).
#   2. Open a new Colab notebook, add a code cell, paste this entire file.
#   3. Run each cell block in order (delimited by # ── CELL ──).
#   4. Cell 2 will prompt you to upload the zip — pick algo-trading-rl.zip.
#   5. At the end, download `colab_output.zip` from the Colab file browser.
#   6. Drop colab_output.zip into your local workspace root.
#
# Output zip contains:
#   runs/<run_id>/model_final.zip
#   runs/<run_id>/learning_curve.csv
#   runs/<run_id>/config.yaml
#   runs/<run_id>/val_stats.json
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "gymnasium", "stable-baselines3", "yfinance", "ta", "pyyaml", "-q"],
    check=True,
)
print("Dependencies installed.")


# ── CELL 2: Upload and extract project zip ───────────────────────────────────
import os, zipfile
from google.colab import files

print("Select your algo-trading-rl.zip file when the upload dialog appears...")
uploaded = files.upload()  # prompts file picker in Colab

zip_name = next(iter(uploaded))          # filename of whatever was uploaded
with zipfile.ZipFile(zip_name, "r") as z:
    z.extractall(".")

# Handle both "algo-trading-rl/" and flat extraction
if os.path.isdir("algo-trading-rl"):
    os.chdir("algo-trading-rl")
elif os.path.isdir("algo-trading-rl-main"):  # GitHub zip naming
    os.chdir("algo-trading-rl-main")
# else: files extracted flat into cwd — already correct

import sys
sys.path.insert(0, os.getcwd())
print(f"Working directory: {os.getcwd()}")


# ── CELL 3: Config ───────────────────────────────────────────────────────────
import os, sys, json, random, time, csv, shutil
from datetime import datetime

import numpy as np
import torch
import yaml

CONFIG = {
    "env": {
        "ticker": "SPY",
        "window_size": 1,
        "half_spread": 0.0001,
        "initial_cash": 100_000.0,
        "use_indicators": True,
    },
    "data": {
        "train_start": "2018-01-01",
        "train_end":   "2020-12-31",
        "val_start":   "2021-01-01",
        "val_end":     "2021-12-31",
        "test_start":  "2022-01-01",
        "test_end":    "2024-12-31",
        "cache_dir":   "data/",
    },
    "training": {
        "seed": 42,
        "total_steps": 200_000,   # increase from 100k — Colab has headroom
        "checkpoint_every": 50_000,
        "runs_dir": "runs/",
        "tag": "ppo_mlp_indicators",
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
    },
}

SEED = CONFIG["training"]["seed"]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("Config OK")


# ── CELL 4: Data ─────────────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd

def download_spy(cfg):
    os.makedirs(cfg["data"]["cache_dir"], exist_ok=True)
    path = os.path.join(cfg["data"]["cache_dir"], "SPY.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"Loaded SPY from cache ({len(df)} rows)")
        return df
    raw = yf.download("SPY", start=cfg["data"]["train_start"],
                      end=cfg["data"]["test_end"], auto_adjust=True, progress=False)
    df = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df = df.dropna()
    df.to_parquet(path)
    print(f"Downloaded SPY: {len(df)} rows")
    return df

def split(df, cfg):
    d = cfg["data"]
    return (df.loc[d["train_start"]:d["train_end"]],
            df.loc[d["val_start"]:d["val_end"]],
            df.loc[d["test_start"]:d["test_end"]])

df = download_spy(CONFIG)
train_df, val_df, test_df = split(df, CONFIG)
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


# ── CELL 5: Environment ───────────────────────────────────────────────────────
# Make sure the envs/ folder is on the path
sys.path.insert(0, os.getcwd())
from envs.trading_env import SingleAssetTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

env_cfg = CONFIG["env"]
train_env_raw = SingleAssetTradingEnv(train_df, env_cfg)
norm_stats = train_env_raw.get_norm_stats()
print(f"obs_space={train_env_raw.observation_space.shape}  n_features={train_env_raw.N_FEATURES}")


# ── CELL 6: Callbacks ─────────────────────────────────────────────────────────
class LearningCurveCallback(BaseCallback):
    def __init__(self, log_path, eval_freq=2000):
        super().__init__()
        self.log_path = log_path
        self.eval_freq = eval_freq
        self._last = 0
        self._f = self._w = None

    def _on_training_start(self):
        self._f = open(self.log_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["timestep","ep_rew_mean","ep_len_mean"])

    def _on_step(self):
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        buf = self.model.ep_info_buffer
        if not buf: return True
        rew = float(np.mean([e["r"] for e in buf]))
        llen = float(np.mean([e["l"] for e in buf]))
        self._w.writerow([self.num_timesteps, rew, llen])
        self._f.flush()
        print(f"  step={self.num_timesteps:7d}  ep_rew_mean={rew:+.4f}")
        return True

    def _on_training_end(self):
        if self._f: self._f.close()


# ── CELL 7: Train ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tag = CONFIG["training"]["tag"]
run_dir = os.path.join(CONFIG["training"]["runs_dir"], f"{timestamp}_{tag}")
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "config.yaml"), "w") as f:
    yaml.dump(CONFIG, f)

def make_env():
    e = SingleAssetTradingEnv(train_df, env_cfg, norm_stats=norm_stats)
    return Monitor(e)

train_env = DummyVecEnv([make_env])
ppo_cfg = CONFIG["ppo"]

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
    seed=SEED,
    verbose=0,
)

t0 = time.time()
model.learn(
    total_timesteps=CONFIG["training"]["total_steps"],
    callback=[
        LearningCurveCallback(os.path.join(run_dir, "learning_curve.csv")),
        CheckpointCallback(CONFIG["training"]["checkpoint_every"], run_dir, "model"),
    ],
)
elapsed = time.time() - t0
print(f"\nTraining done: {elapsed:.0f}s  ({CONFIG['training']['total_steps']/elapsed:.0f} steps/s)")
model.save(os.path.join(run_dir, "model_final"))


# ── CELL 8: Val + test evaluation ────────────────────────────────────────────
def evaluate(env_df, label):
    e = SingleAssetTradingEnv(env_df, env_cfg, norm_stats=norm_stats)
    obs, _ = e.reset(seed=SEED)
    rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = e.step(int(action))
        rewards.append(r)
        if terminated or truncated:
            break
    r_arr = np.array(rewards)
    stats = {
        "total_log_return": float(r_arr.sum()),
        "sharpe_approx": float(r_arr.mean() / (r_arr.std() + 1e-8) * np.sqrt(252)),
        "max_drawdown": float(_max_drawdown(r_arr)),
        "n_steps": len(rewards),
    }
    print(f"[{label}] total_log_ret={stats['total_log_return']:+.4f}  "
          f"sharpe≈{stats['sharpe_approx']:.2f}  "
          f"max_dd={stats['max_drawdown']:.4f}")
    return stats

def _max_drawdown(log_rets):
    cum = np.cumsum(log_rets)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())

val_stats  = evaluate(val_df,  "val")
test_stats = evaluate(test_df, "test")

summary = {
    "run_id": os.path.basename(run_dir),
    "elapsed_s": round(elapsed, 1),
    "steps_per_s": round(CONFIG["training"]["total_steps"] / elapsed, 0),
    "total_steps": CONFIG["training"]["total_steps"],
    "val":  val_stats,
    "test": test_stats,
}
with open(os.path.join(run_dir, "val_stats.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {run_dir}/val_stats.json")


# ── CELL 9: Package output zip ───────────────────────────────────────────────
# Zip up the run directory → colab_output.zip
zip_path = "colab_output"
shutil.make_archive(zip_path, "zip", CONFIG["training"]["runs_dir"])
print(f"\nDownload: colab_output.zip  ({os.path.getsize(zip_path+'.zip')/1024:.0f} KB)")
print("Drop it in your local workspace root — Claude Code will read val_stats.json and learning_curve.csv.")
