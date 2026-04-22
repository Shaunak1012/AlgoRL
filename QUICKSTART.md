# Quick Start

## Prerequisites

Python 3.11+ required. Install dependencies once:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py --server.port 8504
```

Then open **http://localhost:8504** in your browser.

The dashboard is fully self-contained — it reads pre-computed CSVs from `runs/` so no retraining is needed.

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| 📊 Overview | Equity curves, headline metrics vs SPY / TWAP / VWAP / MA baselines |
| ⚠️ Risk & Returns | Drawdown timeline, rolling Sharpe, correlation heatmap |
| 🌐 Assets | Per-asset price history, volume, and allocation weights |
| 🔬 Ablation | Component ablation table (rows A–G) and per-seed variance |
| 🧠 Training | Learning curves and reward breakdown by run |
| 📅 Simulate | Step-through replay with Groq explain-trade panel (needs `GROQ_API_KEY`) |
| 💧 Liquidity | Liquidity-stress figure, r007 vs r008 execution metrics |
| ⚙️ Execution | Backtest tearsheet (run `scripts/backtest.py` to populate) |

## Optional: Groq explain-trade panel

Add your key to `.env` in the project root:

```
GROQ_API_KEY=your_key_here
```

## Optional: Retrain models

```bash
# Single-asset PPO baseline (CPU, ~minutes)
python scripts/train.py --config configs/base.yaml

# Multi-asset Transformer ensemble (GPU recommended — use Colab)
python scripts/train_multi_asset_transformer.py --config configs/transformer.yaml
```

## Paper

```
paper/main.pdf   — 8-page IEEE conference paper
paper/main.docx  — Word version
```
