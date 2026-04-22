"""Plot learning curve from a training run's learning_curve.csv.

Usage:
    python scripts/plot_learning_curve.py --run runs/<timestamp>_tag
    python scripts/plot_learning_curve.py --run runs/<timestamp>_tag --save
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot(run_dir: str, save: bool = False):
    csv_path = os.path.join(run_dir, "learning_curve.csv")
    if not os.path.exists(csv_path):
        print(f"[plot] no learning_curve.csv found in {run_dir}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[plot] learning_curve.csv is empty")
        return

    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps = df["timestep"].values
    rew = df["ep_rew_mean"].values

    ax1.plot(steps, rew, alpha=0.3, color="steelblue", linewidth=0.8, label="raw")
    ax1.plot(steps, smooth(rew, 10), color="steelblue", linewidth=2, label="smoothed (10)")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("ep_rew_mean (log-return)")
    ax1.set_title(f"PPO Learning Curve — {os.path.basename(run_dir)}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, df["ep_len_mean"].values, color="coral", linewidth=1.5)
    ax2.set_ylabel("ep_len_mean (steps)")
    ax2.set_xlabel("timestep")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        out = os.path.join(run_dir, "learning_curve.png")
        plt.savefig(out, dpi=150)
        print(f"[plot] saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to run directory")
    parser.add_argument("--save", action="store_true", help="save PNG instead of showing")
    args = parser.parse_args()
    plot(args.run, save=args.save)
