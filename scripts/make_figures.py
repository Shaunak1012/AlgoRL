"""
scripts/make_figures.py
-----------------------
Generate 8 publication-quality figures for the presentation and IEEE paper.
Reads only CSV / JSON artefacts in runs/ — no model loading, no torch.

Output:  figures/fig{01..08}_{name}.{pdf,png}
         PDF for LaTeX inclusion, PNG for slides.

Run:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RUNS  = ROOT / "runs" / "ensemble_r007"
R008  = ROOT / "runs" / "ensemble_r008_liquidity"
FIGS  = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

RETURNS_CSV  = RUNS / "backtest_returns.csv"
METRICS_CSV  = RUNS / "backtest_metrics.csv"
ABLATION_CSV = ROOT / "runs" / "ablation_table.csv"
ENSEMBLE_JSON = RUNS / "ensemble_results.json"

SEED_DIRS = sorted(glob.glob(str(RUNS / "2*")))

# ──────────────────────────────────────────────────────────────────────────────
# Style — clean, IEEE-compatible, serif axis labels, 300 dpi
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "strategy":     "#7C9EFF",   # periwinkle-blue
    "buy_and_hold": "#FFB347",   # amber
    "equal_weight": "#95E1A4",   # mint
    "sixty_forty":  "#FF6B6B",   # coral
}
LABEL_MAP = {
    "strategy":     "Ensemble PPO (ours)",
    "buy_and_hold": "SPY Buy & Hold",
    "equal_weight": "Equal Weight",
    "sixty_forty":  "60/40",
}
SEED_COLORS = ["#7C9EFF", "#FFB347", "#95E1A4", "#FF6B6B", "#C084FC"]
SEEDS       = [42, 101, 202, 303, 404]

RC = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.85,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.5,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}
plt.rcParams.update(RC)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = FIGS / f"{stem}.{ext}"
        fig.savefig(p)
    print(f"  ✓  {stem}.{{pdf,png}}")
    plt.close(fig)


def load_returns() -> pd.DataFrame:
    df = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    return df


def equity(returns: pd.DataFrame, init: float = 1.0) -> pd.DataFrame:
    return (1 + returns).cumprod() * init


def drawdown(returns: pd.Series) -> pd.Series:
    eq = (1 + returns).cumprod()
    return eq / eq.cummax() - 1


def rolling_sharpe(returns: pd.Series, window: int = 30, ann: int = 252) -> pd.Series:
    mu = returns.rolling(window).mean() * ann
    sd = returns.rolling(window).std() * (ann ** 0.5)
    return (mu / sd).replace([np.inf, -np.inf], np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 01 — Equity curves
# ──────────────────────────────────────────────────────────────────────────────
def fig01_equity_curves() -> None:
    ret = load_returns()
    eq  = equity(ret)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    # Shaded area under strategy
    ax.fill_between(eq.index, eq["strategy"], 1.0,
                    color=PALETTE["strategy"], alpha=0.12)

    for col in ["strategy", "buy_and_hold", "equal_weight", "sixty_forty"]:
        lw    = 2.4 if col == "strategy" else 1.3
        ls    = "-"  if col == "strategy" else "--"
        alpha = 1.0  if col == "strategy" else 0.80
        ax.plot(eq.index, eq[col],
                label=LABEL_MAP[col],
                color=PALETTE[col], lw=lw, ls=ls, alpha=alpha)

    # Annotate max-DD trough on strategy
    dd    = drawdown(ret["strategy"])
    trough_date = dd.idxmin()
    trough_val  = eq.loc[trough_date, "strategy"]
    ax.annotate(
        f"Max DD\n−{abs(dd.min()):.1%}",
        xy=(trough_date, trough_val),
        xytext=(trough_date, trough_val - 0.04),
        fontsize=7.5, color=PALETTE["strategy"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["strategy"], lw=0.9),
        ha="center",
    )

    ax.set_ylabel("Portfolio value (normalised to 1.0)")
    ax.set_title("Cumulative Returns — Test Set 2022–2024")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig01_equity_curves")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 02 — Underwater drawdown
# ──────────────────────────────────────────────────────────────────────────────
def fig02_underwater_drawdown() -> None:
    ret = load_returns()

    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    for col in ["strategy", "buy_and_hold", "equal_weight", "sixty_forty"]:
        dd   = drawdown(ret[col]) * 100.0   # percent
        lw   = 2.2 if col == "strategy" else 1.2
        alpha = 1.0 if col == "strategy" else 0.75
        ax.plot(dd.index, dd,
                label=LABEL_MAP[col],
                color=PALETTE[col], lw=lw, alpha=alpha)
        if col == "strategy":
            ax.fill_between(dd.index, dd, 0, color=PALETTE[col], alpha=0.10)

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater Equity — Test Set 2022–2024")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax.legend(loc="lower right", ncol=2)
    ax.invert_yaxis()   # drawdown negative — flip so troughs go down
    fig.tight_layout()
    save(fig, "fig02_underwater_drawdown")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 03 — Ablation study (dual panel: Sharpe + Max DD)
# ──────────────────────────────────────────────────────────────────────────────
def fig03_ablation() -> None:
    abl = pd.read_csv(ABLATION_CSV)
    # Short labels
    abl["short"] = abl["label"].apply(
        lambda s: s.split(":")[0] + ": " + s.split(":", 1)[1].strip()[:34]
        if ":" in s else s[:38]
    )
    labels = abl["short"].tolist()
    n      = len(labels)
    xs     = np.arange(n)
    w      = 0.55

    highlight = n - 1   # last row = FULL system

    sharpe_colors = [PALETTE["strategy"] if i == highlight else "#ADBDE3" for i in range(n)]
    dd_colors     = [PALETTE["strategy"] if i == highlight else "#F4A0A0" for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # Sharpe
    bars = ax1.bar(xs, abl["sharpe"], width=w, color=sharpe_colors, edgecolor="none")
    for bar, v in zip(bars, abl["sharpe"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.04,
                 f"{v:+.2f}", ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold" if bar.get_facecolor() == tuple(matplotlib.colors.to_rgba(PALETTE["strategy"])) else "normal")
    ax1.axhline(0, color="#888", lw=0.8, ls="--")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe — Component Ablation")

    # Max DD (show as positive percentage)
    dd_vals = abl["max_drawdown"] * 100.0
    bars2 = ax2.bar(xs, dd_vals, width=w, color=dd_colors, edgecolor="none")
    for bar, v in zip(bars2, dd_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Max Drawdown — Component Ablation")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Legend patch for highlighted bar
    patch = mpatches.Patch(color=PALETTE["strategy"], label="Full system (F)")
    fig.legend(handles=[patch], loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout()
    save(fig, "fig03_ablation")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 04 — Per-seed validation curves
# ──────────────────────────────────────────────────────────────────────────────
def fig04_per_seed_val() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    best_step, best_sharpe, best_seed_idx = None, -99, -1

    for i, (sdir, seed) in enumerate(zip(SEED_DIRS, SEEDS)):
        vpath = Path(sdir) / "val_curve.csv"
        if not vpath.exists():
            continue
        df = pd.read_csv(vpath)
        ax.plot(df["timestep"] / 1000, df["val_sharpe"],
                color=SEED_COLORS[i], lw=1.5, alpha=0.85,
                label=f"Seed {seed}")

        # Best checkpoint star
        best_idx = df["val_sharpe"].idxmax()
        bx = df.loc[best_idx, "timestep"] / 1000
        by = df.loc[best_idx, "val_sharpe"]
        ax.plot(bx, by, marker="*", ms=9, color=SEED_COLORS[i], zorder=5)

        if by > best_sharpe:
            best_sharpe, best_step, best_seed_idx = by, bx, i

    ax.axhline(0, color="#888", lw=0.7, ls="--")
    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Validation Sharpe Ratio")
    ax.set_title("Per-Seed Validation Sharpe — 5 Seeds × 100k Steps\n★ = best-val checkpoint saved")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig04_per_seed_val")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 05 — Ensemble: mean vs uncertainty (val + test bar chart)
# ──────────────────────────────────────────────────────────────────────────────
def fig05_ensemble_comparison() -> None:
    with open(ENSEMBLE_JSON) as f:
        ens = json.load(f)["results"]

    labels  = ["Val / Mean", "Val / Uncertainty", "Test / Mean", "Test / Uncertainty"]
    sharpes = [
        ens["val/mean"]["sharpe"],
        ens["val/uncertainty"]["sharpe"],
        ens["test/mean"]["sharpe"],
        ens["test/uncertainty"]["sharpe"],
    ]
    max_dds = [
        ens["val/mean"]["max_dd"] * 100,
        ens["val/uncertainty"]["max_dd"] * 100,
        ens["test/mean"]["max_dd"] * 100,
        ens["test/uncertainty"]["max_dd"] * 100,
    ]

    xs = np.arange(4)
    w  = 0.38

    bar_colors_sh = ["#ADBDE3", PALETTE["strategy"], "#F4A0A0", PALETTE["strategy"]]
    bar_colors_dd = ["#ADBDE3", PALETTE["strategy"], "#F4A0A0", PALETTE["strategy"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    bars1 = ax1.bar(xs, sharpes, width=w, color=bar_colors_sh, edgecolor="none")
    for bar, v in zip(bars1, sharpes):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.04 if v >= 0 else -0.10),
                 f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold")
    ax1.axhline(0, color="#888", lw=0.7, ls="--")
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Ensemble: Sharpe — Mean vs Uncertainty")

    bars2 = ax2.bar(xs, max_dds, width=w, color=bar_colors_dd, edgecolor="none")
    for bar, v in zip(bars2, max_dds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Ensemble: Max DD — Mean vs Uncertainty")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    fig.tight_layout()
    save(fig, "fig05_ensemble_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 06 — ETF correlation heatmap (test set daily returns)
# ──────────────────────────────────────────────────────────────────────────────
def fig06_correlation_heatmap() -> None:
    # We have daily returns for strategy + benchmarks from backtest_returns.
    # For ETF-vs-ETF correlation we need per-asset returns.
    # Approximate: use yfinance cached parquets if present, else derive from
    # backtest_returns columns (only 4, not 5 ETFs) and note the limitation.
    data_dir = ROOT / "data"
    parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []

    if parquet_files:
        frames = {}
        ASSETS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        for pf in parquet_files:
            ticker = pf.stem.upper()
            if ticker in ASSETS:
                df = pd.read_parquet(pf)
                # Adjusted close column may vary
                close_col = next(
                    (c for c in df.columns if "close" in c.lower() or "adj" in c.lower()),
                    df.columns[-1]
                )
                frames[ticker] = df[close_col]
        if len(frames) >= 4:
            prices = pd.DataFrame(frames).dropna()
            rets   = prices.pct_change().dropna()
            # Filter to test period
            rets   = rets.loc["2022":"2024"]
            corr   = rets.corr()
        else:
            corr = None
    else:
        corr = None

    if corr is None:
        # Fall back: use backtest return columns as a 4-asset proxy
        ret  = load_returns()
        corr = ret.rename(columns=LABEL_MAP).corr()

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ticks = list(range(len(corr.columns)))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.55 else "#333")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson ρ", fontsize=9)
    ax.set_title("Asset Correlation Matrix — Test Period 2022–2024")
    fig.tight_layout()
    save(fig, "fig06_correlation_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 07 — Monthly returns heatmap (strategy only)
# ──────────────────────────────────────────────────────────────────────────────
def fig07_monthly_returns() -> None:
    ret    = load_returns()["strategy"]
    monthly = (ret + 1).resample("ME").prod() - 1

    months_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    years = sorted(monthly.index.year.unique())

    grid = pd.DataFrame(index=years, columns=range(1, 13), dtype=float)
    for ts, val in monthly.items():
        grid.loc[ts.year, ts.month] = val

    grid.columns = [months_map[m] for m in grid.columns]

    fig, ax = plt.subplots(figsize=(8.0, 2.0 + 0.55 * len(years)))

    vmax = 0.06
    im   = ax.imshow(grid.values.astype(float), cmap="RdYlGn",
                     vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(12)); ax.set_xticklabels(grid.columns, fontsize=8.5)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years, fontsize=8.5)

    for i, yr in enumerate(years):
        for j, mo in enumerate(range(1, 13)):
            v = grid.loc[yr, list(months_map.values())[j - 1]]
            if pd.notna(v):
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.04 else "#222")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Monthly Return", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.set_title("Monthly Returns Heatmap — Ensemble PPO Strategy")
    fig.tight_layout()
    save(fig, "fig07_monthly_returns")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 08 — Rolling Sharpe (30-day) — strategy vs SPY
# ──────────────────────────────────────────────────────────────────────────────
def fig08_rolling_sharpe() -> None:
    ret = load_returns()
    rs_strat = rolling_sharpe(ret["strategy"],    window=30)
    rs_spy   = rolling_sharpe(ret["buy_and_hold"], window=30)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    ax.fill_between(rs_strat.index, rs_strat, 0,
                    where=(rs_strat >= 0),
                    color=PALETTE["strategy"], alpha=0.18, label="_nolegend_")
    ax.fill_between(rs_strat.index, rs_strat, 0,
                    where=(rs_strat < 0),
                    color="#FF6B6B", alpha=0.18, label="_nolegend_")

    ax.plot(rs_strat.index, rs_strat,
            color=PALETTE["strategy"], lw=1.8, label="Ensemble PPO (ours)")
    ax.plot(rs_spy.index, rs_spy,
            color=PALETTE["buy_and_hold"], lw=1.3, ls="--", alpha=0.8,
            label="SPY Buy & Hold")

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_ylabel("30-day Rolling Sharpe")
    ax.set_title("Rolling Sharpe Ratio — Test Set 2022–2024")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig08_rolling_sharpe")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 09 — Liquidity stress: r007 vs r008 drawdown + execution cost comparison
# ──────────────────────────────────────────────────────────────────────────────
def fig09_liquidity_stress() -> None:
    r007_ret_path = RUNS / "backtest_returns.csv"
    r008_ret_path = R008 / "backtest_returns.csv"
    r007_exec_path = RUNS / "execution_stats.csv"
    r008_exec_path = R008 / "execution_stats.csv"

    if not r008_ret_path.exists() or not r008_exec_path.exists():
        print("  ⚠  fig09 skipped — r008 backtest artefacts not found.")
        return

    ret7 = pd.read_csv(r007_ret_path, index_col=0, parse_dates=True)["strategy"]
    ret8 = pd.read_csv(r008_ret_path, index_col=0, parse_dates=True)["strategy"]

    dd7 = drawdown(ret7) * 100.0
    dd8 = drawdown(ret8) * 100.0

    exec7 = pd.read_csv(r007_exec_path).iloc[0]
    exec8 = pd.read_csv(r008_exec_path).iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # ── Panel 1: underwater drawdown overlay ──────────────────────────────────
    ax1.plot(dd7.index, dd7, color="#FFB347", lw=1.8, label="r007 (baseline)")
    ax1.fill_between(dd7.index, dd7, 0, color="#FFB347", alpha=0.10)
    ax1.plot(dd8.index, dd8, color="#7C9EFF", lw=2.2, label="r008 (liquidity-aware)")
    ax1.fill_between(dd8.index, dd8, 0, color="#7C9EFF", alpha=0.12)
    ax1.axhline(0, color="#888", lw=0.7, ls="--")
    ax1.set_ylabel("Drawdown (%)")
    ax1.set_title("Drawdown Under Liquidity-Aware Training\nr007 vs r008 — Test Set 2022–2024")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax1.legend(loc="lower right")
    ax1.invert_yaxis()

    # ── Panel 2: execution cost comparison bar chart ───────────────────────────
    metrics_labels = ["Impl. Shortfall\n(bps)", "Total Impact\n(bps)", "Mean Turnover\n(daily)"]
    r007_vals = [
        exec7["impl_shortfall_bps"],
        exec7["total_impact_bps"],
        exec7["mean_turnover"] * 1000,   # scale to same order as bps
    ]
    r008_vals = [
        exec8["impl_shortfall_bps"],
        exec8["total_impact_bps"],
        exec8["mean_turnover"] * 1000,
    ]
    # turnover label needs its own axis scale — keep it as ×1000 for visual clarity
    xs = np.arange(3)
    w  = 0.35

    bars7 = ax2.bar(xs - w / 2, r007_vals, width=w, color="#FFB347",
                    label="r007 (baseline)", edgecolor="none")
    bars8 = ax2.bar(xs + w / 2, r008_vals, width=w, color="#7C9EFF",
                    label="r008 (liquidity-aware)", edgecolor="none")

    for bar, v in zip(bars7, r007_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#FFB347")
    for bar, v in zip(bars8, r008_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#7C9EFF")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(metrics_labels, fontsize=8.5)
    ax2.set_ylabel("bps  (Turnover × 1000 for scale)")
    ax2.set_title("Execution Cost — r007 vs r008\n(Impact bps · Shortfall · Turnover)")

    # Annotate % reduction arrows
    reductions = [(r007_vals[i] - r008_vals[i]) / r007_vals[i] * 100 for i in range(3)]
    for i, pct in enumerate(reductions):
        ymax = max(r007_vals[i], r008_vals[i])
        ax2.annotate(
            f"−{pct:.0f}%",
            xy=(xs[i], ymax + 20),
            ha="center", fontsize=8, color="#95E1A4", fontweight="bold",
        )

    ax2.legend(loc="upper right", fontsize=8)

    note = "(Turnover column multiplied ×1000 for joint display)"
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#666")

    fig.tight_layout()
    save(fig, "fig09_liquidity_stress")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures → figures/\n")
    fig01_equity_curves()
    fig02_underwater_drawdown()
    fig03_ablation()
    fig04_per_seed_val()
    fig05_ensemble_comparison()
    fig06_correlation_heatmap()
    fig07_monthly_returns()
    fig08_rolling_sharpe()
    fig09_liquidity_stress()
    print("\nAll figures saved.")
