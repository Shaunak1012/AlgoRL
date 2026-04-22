"""Shared SB3 callbacks for multi-asset trading runs (MVP Step 7+).

ValEvalCallback
    Every `eval_freq` training steps, runs a deterministic policy rollout on a
    held-out env and logs val Sharpe + max_dd + log-return to a CSV. Saves the
    best-val-Sharpe checkpoint to `<run_dir>/model_best_val.zip`. This exists
    because r006 final-checkpoint overfit (val Sharpe -1.09) while the 50k
    checkpoint was actually +0.012 — we want automatic best-val selection from
    r007 onward. See ISSUES.md #1 and DECISIONS.md 2026-04-21.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ValEvalCallback(BaseCallback):
    def __init__(
        self,
        val_env_factory,
        log_path: str,
        best_ckpt_path: str,
        eval_freq: int = 10_000,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.val_env_factory = val_env_factory
        self.log_path = log_path
        self.best_ckpt_path = best_ckpt_path
        self.eval_freq = eval_freq
        self.seed = seed
        self._last_eval = 0
        self._best_sharpe: float = -float("inf")
        self._f = self._w = None

    def _on_training_start(self) -> None:
        self._f = open(self.log_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(
            ["timestep", "val_sharpe", "val_log_return", "val_max_dd", "is_best"]
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        stats = self._evaluate()
        is_best = stats["sharpe"] > self._best_sharpe
        if is_best:
            self._best_sharpe = stats["sharpe"]
            self.model.save(self.best_ckpt_path)

        self._w.writerow(
            [self.num_timesteps, stats["sharpe"], stats["log_return"],
             stats["max_dd"], int(is_best)]
        )
        self._f.flush()
        marker = "  <- best" if is_best else ""
        print(f"  [val] step={self.num_timesteps:7d}  sharpe={stats['sharpe']:+.3f}"
              f"  log_ret={stats['log_return']:+.4f}  max_dd={stats['max_dd']:.3f}{marker}")
        return True

    def _on_training_end(self) -> None:
        if self._f:
            self._f.close()
        print(f"  [val] best val Sharpe over training: {self._best_sharpe:+.3f}  "
              f"(saved to {self.best_ckpt_path})")

    def _evaluate(self) -> dict:
        env = self.val_env_factory()
        obs, _ = env.reset(seed=self.seed)
        rets = []
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            rets.append(info["r_pnl"] + info["r_spread"])
            if terminated or truncated:
                break
        r = np.array(rets)
        cum = np.cumsum(r)
        dd = float((np.maximum.accumulate(cum) - cum).max())
        return {
            "sharpe": float(r.mean() / (r.std() + 1e-8) * np.sqrt(252)),
            "log_return": float(r.sum()),
            "max_dd": dd,
        }
