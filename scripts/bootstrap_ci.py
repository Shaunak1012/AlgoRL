"""
Bootstrap confidence intervals for backtest metrics.
Circular block bootstrap (block=21 trading days) preserves return autocorrelation.

Usage:
    python scripts/bootstrap_ci.py
Output:
    runs/bootstrap_ci.csv   — 95% CI table for all strategies
    (also prints a paper-ready summary)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RUNS   = Path(__file__).parent.parent / "runs"
R007   = RUNS / "ensemble_r007"
BLOCK  = 21       # ~1 trading month
N_BOOT = 2000
ALPHA  = 0.05
ANN    = 252
RNG    = np.random.default_rng(42)


def metrics(r: np.ndarray) -> dict:
    n     = len(r)
    cum   = (1 + r).prod() - 1
    ann_r = (1 + cum) ** (ANN / n) - 1
    ann_v = r.std() * np.sqrt(ANN)
    sh    = ann_r / ann_v if ann_v > 1e-10 else 0.0
    eq    = (1 + r).cumprod()
    mdd   = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"ann_return": ann_r, "sharpe": sh, "max_dd": mdd}


def block_bootstrap(r: np.ndarray) -> np.ndarray:
    n      = len(r)
    n_blk  = int(np.ceil(n / BLOCK))
    starts = RNG.integers(0, n, size=n_blk)
    idx    = np.concatenate([np.arange(s, s + BLOCK) % n for s in starts])
    return r[idx[:n]]


def ci(samples: np.ndarray) -> tuple:
    lo = float(np.percentile(samples, 100 * ALPHA / 2))
    hi = float(np.percentile(samples, 100 * (1 - ALPHA / 2)))
    return lo, hi


def main():
    ret = pd.read_csv(R007 / "backtest_returns.csv", index_col="Date")
    rows = []

    for col in ret.columns:
        r      = ret[col].values
        point  = metrics(r)
        boot   = [metrics(block_bootstrap(r)) for _ in range(N_BOOT)]

        sh_lo,  sh_hi  = ci(np.array([b["sharpe"]     for b in boot]))
        ret_lo, ret_hi = ci(np.array([b["ann_return"]  for b in boot]))
        dd_lo,  dd_hi  = ci(np.array([b["max_dd"]      for b in boot]))

        rows.append({
            "strategy":      col,
            "sharpe":        round(point["sharpe"],     3),
            "sharpe_ci_lo":  round(sh_lo,  3),
            "sharpe_ci_hi":  round(sh_hi,  3),
            "ann_return":    round(point["ann_return"], 4),
            "ret_ci_lo":     round(ret_lo, 4),
            "ret_ci_hi":     round(ret_hi, 4),
            "max_dd":        round(point["max_dd"],     4),
            "dd_ci_lo":      round(dd_lo,  4),
            "dd_ci_hi":      round(dd_hi,  4),
        })

    df = pd.DataFrame(rows).set_index("strategy")
    out = RUNS / "bootstrap_ci.csv"
    df.to_csv(out)
    print(f"Saved → {out}\n")

    # Paper-ready summary
    LABELS = {
        "strategy":      "Ensemble PPO (ours)",
        "buy_and_hold":  "SPY Buy & Hold",
        "equal_weight":  "Equal Weight",
        "sixty_forty":   "60/40",
        "ma_crossover":  "MA Crossover",
        "twap_ew":       "TWAP",
        "vwap_proxy_ew": "VWAP-proxy",
    }
    print(f"{'Strategy':<26} {'Sharpe':>8}  {'95% CI':^18}  {'Ann Ret':>8}  {'Max DD':>8}")
    print("-" * 74)
    for col, row in df.iterrows():
        label = LABELS.get(col, col)
        ci_str = f"[{row.sharpe_ci_lo:+.2f}, {row.sharpe_ci_hi:+.2f}]"
        print(f"{label:<26} {row.sharpe:>+8.3f}  {ci_str:^18}  "
              f"{row.ann_return:>+7.1%}  {abs(row.max_dd):>7.1%}")


if __name__ == "__main__":
    main()
