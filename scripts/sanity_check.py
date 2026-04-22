"""MVP Step 1 sanity check: random agent runs one full episode without crashing.

Usage:
    python scripts/sanity_check.py [--config configs/base.yaml]

Passes if:
    1. Data downloads (or loads from cache) successfully.
    2. Env resets without error and returns correct obs shape.
    3. Random agent runs to episode end; no NaN rewards.
    4. Norm stats from train env apply cleanly to val env.
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_data import download_ticker, split_df
from envs.trading_env import SingleAssetTradingEnv


def run_episode(env: SingleAssetTradingEnv, seed: int = 0) -> dict:
    obs, _ = env.reset(seed=seed)
    rewards, positions = [], []
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.isnan(reward), f"NaN reward at step {steps}"
        assert obs.shape == env.observation_space.shape, f"Bad obs shape {obs.shape}"
        rewards.append(reward)
        positions.append(info["position"])
        steps += 1
        if terminated or truncated:
            break
    return {
        "steps": steps,
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "reward_nan": int(np.any(np.isnan(rewards))),
        "unique_positions": sorted(set(positions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── 1. Data ──────────────────────────────────────────────────────────
    t0 = time.time()
    ticker = cfg["env"]["ticker"]
    df = download_ticker(
        ticker,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["test_end"],
        cache_dir=cfg["data"]["cache_dir"],
    )
    train_df, val_df, test_df = split_df(df, cfg["data"])
    print(f"[1/4] Data OK  ({time.time()-t0:.1f}s) — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── 2. Env construction & norm stats ─────────────────────────────────
    env_cfg = cfg["env"]
    train_env = SingleAssetTradingEnv(train_df, env_cfg)
    norm_stats = train_env.get_norm_stats()

    val_env = SingleAssetTradingEnv(val_df, env_cfg, norm_stats=norm_stats)
    test_env = SingleAssetTradingEnv(test_df, env_cfg, norm_stats=norm_stats)
    print(f"[2/4] Env OK   — obs_space={train_env.observation_space.shape}  action_space={train_env.action_space.n}")

    # ── 3. Random agent on each split ────────────────────────────────────
    rng = np.random.default_rng(cfg["training"]["seed"])
    for name, env in [("train", train_env), ("val", val_env), ("test", test_env)]:
        stats = run_episode(env, seed=int(rng.integers(1000)))
        print(
            f"[3/4] {name:5s} episode — steps={stats['steps']:4d}  "
            f"total_reward={stats['total_reward']:+.4f}  "
            f"NaN={stats['reward_nan']}  "
            f"positions={stats['unique_positions']}"
        )
        assert stats["reward_nan"] == 0, "NaN reward detected — env is broken"

    # ── 4. No-future-leakage check ───────────────────────────────────────
    # State at step t must not change when we alter data at t+1
    obs_t, _ = train_env.reset(seed=0)
    obs_before = obs_t.copy()
    train_env._prices[train_env._step + 1, 3] *= 999  # corrupt t+1
    obs_after, _ = train_env.reset(seed=0)  # reset restores state to t=window_size
    # The obs at reset (t=window_size) should be unaffected by corruption at t+1
    # We verify indirectly: norm features at step=window_size use only indices ≤ window_size
    assert np.allclose(obs_before, obs_after), "Future leakage detected!"
    train_env._prices[train_env._step + 1, 3] /= 999  # restore
    print("[4/4] No-future-leakage check OK")

    print("\n✓ All checks passed. MVP Step 1 complete.")


if __name__ == "__main__":
    main()
