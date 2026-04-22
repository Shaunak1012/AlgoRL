# AlgoRL

**Ensemble Transformer-PPO with CVaR Regularisation and Uncertainty-Scaled Position Sizing for Multi-Asset Algorithmic Trading under Market Impact**

A deep reinforcement learning system for risk-controlled portfolio allocation across 5 US ETFs (SPY, QQQ, IWM, TLT, GLD), trained on 2018–2020 data and evaluated on a held-out 2022–2024 test split.

---

## Key Results (2022–2024 OOS, 691 trading days)

| Strategy | Sharpe | 95% CI | Max DD |
|---|---|---|---|
| **Ensemble PPO (ours)** | **+0.27** | [−1.05, +1.59] | **8.6%** |
| SPY Buy & Hold | +0.70 | [−0.44, +2.16] | 21.3% |
| 60/40 SPY/TLT | +0.25 | [−1.06, +1.70] | 22.0% |
| MA Crossover 50/200 | +1.49 | [+0.39, +2.82] | 10.0% |

The strategy underperforms passive equity on return and Sharpe but cuts maximum drawdown by **59% vs SPY** and **~5× vs naive ensemble mean**. Framed honestly as a drawdown-control primitive, not an alpha generator.

---

## Architecture

- **Env:** `MultiAssetTradingEnv` — 5-ETF Gymnasium environment with Kyle's-λ market impact, stochastic fill noise, and volume-inferred liquidity-regime features
- **Encoder:** Causal Transformer (d=64, 2 layers, 4 heads, T=60 window)
- **Policy:** 5-seed PPO ensemble with per-seed val-Sharpe early stopping
- **Aggregation:** Uncertainty-scaled position sizing `a = μ / (1 + β·σ̄)`
- **Reward:** ΔPnL − spread − Kyle's-λ impact − λ·CVaR₅%

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8504
```

Open **http://localhost:8504** — dashboard loads pre-computed results, no retraining needed.

See [QUICKSTART.md](QUICKSTART.md) for full instructions including retraining and the Groq explain-trade panel.

---

## Project Structure

```
├── app.py                        # 8-tab Streamlit dashboard — Overview, Risk, Assets,
│                                 #   Ablation, Training, Simulate, Liquidity, Execution.
│                                 #   Loads pre-computed CSVs; no retraining needed to run.
│
├── requirements.txt              # All Python dependencies (PyTorch, SB3, Streamlit, etc.)
├── QUICKSTART.md                 # One-page run guide — dashboard, retraining, paper build
├── DECISIONS.md                  # Append-only log of every architectural choice + rationale
├── EXPERIMENTS.md                # Training run registry — config, metrics, checkpoint path
│
├── configs/
│   ├── transformer.yaml          # Full hyperparameter config for the Transformer-PPO run:
│   │                             #   d_model=64, 2 layers, 4 heads, window T=60, lr=3e-4
│   └── base.yaml                 # Single-asset SPY baseline config (MLP policy, no CVaR)
│
├── envs/
│   ├── multi_asset_env.py        # MultiAssetTradingEnv: 5-ETF Gymnasium environment with
│   │                             #   Kyle's-λ impact, stochastic fill noise, liquidity-regime
│   │                             #   features (vol z-score + OHLC dispersion), CVaR reward
│   ├── trading_env.py            # Single-asset SPY environment used for early ablation runs
│   └── transformer_extractor.py  # Causal Transformer as SB3 BaseFeaturesExtractor:
│                                 #   pre-LN, learned positional embedding, upper-tri causal mask
│
├── scripts/
│   ├── train_multi_asset_transformer.py  # Main training entry point — launches PPO with
│   │                                     #   Transformer extractor, ValEvalCallback, checkpoints
│   ├── train_multi_asset.py      # Earlier MLP-policy multi-asset trainer (ablation rows A–B)
│   ├── train_ppo.py              # Single-asset PPO trainer (ablation baseline)
│   ├── ensemble_eval.py          # Loads 5 best-val checkpoints, runs uncertainty-scaled
│   │                             #   aggregation a=μ/(1+β·σ̄), writes ensemble_results.json
│   ├── backtest.py               # Full backtest pipeline: strategy + 6 baselines (SPY B&H,
│   │                             #   EW, 60/40, MA crossover 50/200, TWAP, VWAP-proxy).
│   │                             #   Outputs backtest_returns.csv, backtest_metrics.csv
│   ├── bootstrap_ci.py           # Circular block bootstrap (N=2000, block=21 days) —
│   │                             #   computes 95% CIs on Sharpe, Ann. Return, Max DD
│   ├── make_figures.py           # Regenerates all 9 publication figures (PDF + PNG)
│   ├── fetch_sentiment.py        # Alpha Vantage + NewsAPI headline fetch → FinBERT inference
│   │                             #   pipeline → sentiment_daily.parquet (r009 prep)
│   ├── run_ablations.py          # Runs all 7 ablation configurations sequentially
│   ├── callbacks.py              # ValEvalCallback: evaluates on 2021 val set every 10k steps,
│   │                             #   saves best-val checkpoint, logs Sharpe/DD/return
│   ├── download_data.py          # Downloads 2018–2024 adjusted-close OHLCV via yfinance
│   ├── plot_learning_curve.py    # Plots episode reward + val Sharpe curves per seed
│   └── sanity_check.py           # Runs a random agent for 1 episode — verifies env integrity
│
├── notebooks/
│   ├── colab_train_ensemble.ipynb      # Colab notebook to train the r007 5-seed ensemble
│   ├── colab_train_r008.ipynb          # Colab notebook for liquidity-aware r008 ensemble
│   ├── colab_train_r009.ipynb          # Sentiment-augmented r009 (ready; blocked on news data)
│   └── colab_train_multi_transformer.ipynb  # Single-seed Transformer training (development)
│
├── tests/
│   ├── test_multi_asset_env.py         # Env step/reset, reward components, no-lookahead check
│   ├── test_transformer_extractor.py   # Shape, causal mask, gradient flow, deterministic eval
│   ├── test_backtest.py                # Backtest output schema and metric sanity checks
│   ├── test_callbacks_and_ensemble.py  # ValEvalCallback + ensemble aggregation unit tests
│   └── test_trading_env.py             # Single-asset env smoke tests
│
├── data/
│   ├── SPY.parquet               # S&P 500 ETF — adj. close OHLCV, 2018–2024 (1,762 rows)
│   ├── QQQ.parquet               # Nasdaq-100 ETF
│   ├── IWM.parquet               # Russell 2000 small-cap ETF
│   ├── TLT.parquet               # 20+ year Treasury bond ETF
│   ├── GLD.parquet               # Gold ETF
│   └── sentiment_daily.parquet   # FinBERT market-level sentiment scores (bullish/neutral/
│                                 #   bearish fraction per day, gap-filled, 4.8% coverage)
│
├── figures/
│   ├── fig01_equity_curves.png         # Equity curves: r007/r008 vs. all 6 baselines
│   ├── fig02_underwater_drawdown.png   # Underwater chart — drawdown over time
│   ├── fig03_ablation.png              # Sharpe + Max DD bar chart across ablation rows A–G
│   ├── fig04_per_seed_val.png          # Per-seed val Sharpe curves — early stopping visualised
│   ├── fig05_ensemble_comparison.png   # Mean vs. uncertainty-scaled aggregation comparison
│   ├── fig06_correlation_heatmap.png   # Rolling 60-day cross-asset correlation heatmap
│   ├── fig07_monthly_returns.png       # Monthly returns heatmap (strategy, 2022–2024)
│   ├── fig08_rolling_sharpe.png        # 30-day rolling Sharpe — strategy vs. SPY
│   ├── fig09_liquidity_stress.png      # r007 vs. r008 execution metrics under impact replay
│   └── architecture.png               # System architecture diagram (rendered from .mmd)
│
├── runs/
│   ├── ensemble_r007/            # Frictionless 5-seed ensemble — primary results
│   │   ├── backtest_returns.csv  #   Daily returns for strategy + 6 baselines (691 rows)
│   │   ├── backtest_metrics.csv  #   Ann. return, Sharpe, Sortino, Max DD, Calmar per strategy
│   │   ├── ensemble_results.json #   Aggregation scale, n_members, per-seed val metrics
│   │   ├── execution_stats.csv   #   Kyle's-λ impact bps, spread bps, turnover, impl. shortfall
│   │   ├── tearsheet.html        #   QuantStats full tearsheet (interactive HTML)
│   │   └── 2026042*_r007_seed*/  #   Per-seed dirs: config.yaml, learning_curve.csv, val_curve.csv
│   ├── ensemble_r008_liquidity/  # Liquidity-aware ensemble — same structure as r007
│   ├── ablation_table.csv        # 7-row ablation: rows A–G, all metrics
│   └── bootstrap_ci.csv          # Bootstrap 95% CIs: Sharpe, Ann. Return, Max DD per strategy
│
└── paper/
    ├── main.tex                  # 8-page IEEE conference paper (IEEEtran two-column)
    ├── references.bib            # 24 BibTeX entries — all cited in main.tex
    └── figures/                  # Symlink → ../figures (paper pulls architecture.png from here)
```

---

## Reproducing Results

**Backtest (uses saved checkpoints):**
```bash
python scripts/backtest.py --run-dir runs/ensemble_r007
```

**Bootstrap CIs:**
```bash
python scripts/bootstrap_ci.py
```

**Retrain (GPU recommended — use Colab):**
- Open `notebooks/colab_train_ensemble.ipynb` in Google Colab
- For liquidity-aware variant: `notebooks/colab_train_r008.ipynb`

**Regenerate figures:**
```bash
python scripts/make_figures.py
```

---

## Ablation

| Row | Configuration | Sharpe | Max DD |
|---|---|---|---|
| A | MLP, no CVaR | −0.56 | 72.1% |
| B | + CVaR (λ=0.01) | −0.48 | 62.3% |
| C | + Transformer, no early stop | −1.09 | 78.4% |
| D | + early stopping | +0.74 | 56.3% |
| E | + 5-seed ensemble (mean) | −0.20 | 44.2% |
| F | **+ uncertainty sizing (full)** | **+0.27** | **8.6%** |
| G | + liquidity-aware env (r008) | −0.27 | 13.7% |

Uncertainty-scaled aggregation (row F→E) is the only configuration with single-digit max drawdown.

---

## Paper

The 8-page IEEE-format paper is in `paper/`. Build with:
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Requires a LaTeX distribution (TinyTeX or TeX Live).

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=       # For explain-trade panel (Simulate tab)
ALPHA_VANTAGE_KEY=  # For sentiment backfill (optional)
```

---

## License

MIT
