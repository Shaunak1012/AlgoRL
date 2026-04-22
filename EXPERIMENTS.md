# Experiment Log

| Run ID | Date | Config | Steps | Best Sharpe | Max DD | CVaR 5% | Notes |
|--------|------|--------|-------|-------------|--------|---------|-------|
| r001 | 2026-04-20 | base.yaml / ppo_mlp_baseline | 100k | val≈0.17 | — | — | MVP Step 2. OHLCV only. ep_rew_mean: -0.013→+1.04. Val log_return=+0.018. |
| r002 | 2026-04-20 | base.yaml / ppo_mlp_indicators | 100k | val≈2.42 | — | — | MVP Step 3. +RSI, MACD, BB. ep_rew_mean: +0.15→+1.25. Val log_return=+0.309. Sharpe 2.42 vs 0.17 baseline. |
| r003 | 2026-04-20 | base.yaml / ppo_multi_asset | 100k | val≈-1.31 | 0.40 | — | MVP Step 4. 5 ETFs, continuous action. Train -0.12→+4.18. Val negative — pre-CVaR baseline. |
| r004 | 2026-04-20 | base.yaml / ppo_multi_cvar λ=0.1 | 100k | val≈-1.89 | 0.93 | -0.004/step | CVaR dominated reward (88%) — discarded. λ too large. |
| r005 | 2026-04-20 | base.yaml / ppo_multi_cvar λ=0.01 | 100k | val≈-0.48 | 0.35 | -0.0004/step | MVP Step 5. CVaR at 10% reward share. Val Sharpe -0.48 vs -1.31 (r003). Max DD 0.35 vs 0.40. CVaR is helping. |
| r006 | 2026-04-20 | transformer.yaml / r006_transformer | 100k | train=+2.99 / val=**-1.09** / test=-0.20 | val 0.53 / test 1.26 | -0.00066/step (~9% share) | MVP Step 6. Causal Transformer encoder (d_model=64, 2L, 4H, T=60). ep_rew_mean -1.00→+3.00 monotone on train. **Severe overfitting** at final step: train Sharpe +2.99 but val regressed −0.48 (r005)→−1.09. Test max_dd blew out to 1.26. Colab GPU, no errors. Model: model_final.zip. See ISSUES.md #1. |
| r006-ckpt50k | 2026-04-20 | same as r006 / checkpoint at 50k | 50k | val=**+0.012** / test=+0.495 | val 0.270 / test 0.829 | same | **Early-stopping winner.** First positive val Sharpe in multi-asset setting. Beats r005 (val -0.48, val_dd 0.35) on every val metric. Confirms ISSUES.md #1 overfitting hypothesis without retraining. Use this checkpoint, not model_final, for downstream (Step 7 ensemble, Step 9 ablation). File: model_50000_steps.zip. |
| r007-seed42  | 2026-04-20 | transformer.yaml seed=42  | 100k | val best=+1.368 @ step 80k  | 0.175 | ~9% share | Single member, ValEvalCallback → model_best_val.zip @ 80k. |
| r007-seed101 | 2026-04-20 | transformer.yaml seed=101 | 100k | val best=+0.683 @ step 20k  | 0.217 | ~9% share | Single member. Best-val hit very early (20k). |
| r007-seed202 | 2026-04-20 | transformer.yaml seed=202 | 100k | val best=+0.912 @ step 60k  | 0.278 | ~9% share | Single member. |
| r007-seed303 | 2026-04-20 | transformer.yaml seed=303 | 100k | val best=+1.634 @ step 100k | 0.258 | ~9% share | Single member. Best seed; val still rising at 100k (room for longer training). |
| r007-seed404 | 2026-04-20 | transformer.yaml seed=404 | 100k | val best=+0.587 @ step 90k  | 0.190 | ~9% share | Single member. Weakest seed. |
| r007-ensemble-mean        | 2026-04-20 | 5-head mean          | — | val=+1.484 / test=-0.338 | val 0.104 / test 0.583 | — | Plain mean over 5 best-val heads. Already jumps val Sharpe from 0.012 (r006-ckpt50k) → +1.48 (120× single-head). |
| **r007-ensemble-uncertainty** | **2026-04-20** | **5-head μ · 1/(1+5·σ̄)** | — | **val=+1.977 / test=+0.242** | **val 0.014 / test 0.090** | — | **Headline result.** Uncertainty-scaled sizing: val Sharpe +1.98, val max_dd **0.014** (20× lower than single-head ckpt50k's 0.27). Test Sharpe flipped −0.34→+0.24 vs mean-ensemble; test max_dd 0.09 vs 0.58. mean_head_std≈0.80 → uncertainty signal is active, not degenerate. |
| r008-seed42  | 2026-04-22 | transformer.yaml + C7+C8 flags on, seed=42  | 100k | val best=+1.234 @ step 50k  | 0.202 | ~9% share | Liquidity-aware env (obs width 75 incl. vol z-score + spread proxy; fill noise σ_base=5bps k=10). |
| r008-seed101 | 2026-04-22 | transformer.yaml + C7+C8, seed=101 | 100k | val best=+1.877 @ step 30k  | 0.158 | ~9% share | Strongest val seed; best-val very early (30k). |
| r008-seed202 | 2026-04-22 | transformer.yaml + C7+C8, seed=202 | 100k | val best=+1.230 @ step 40k  | 0.163 | ~9% share | |
| r008-seed303 | 2026-04-22 | transformer.yaml + C7+C8, seed=303 | 100k | val best=+0.213 @ step 90k  | 0.307 | ~9% share | Weakest seed this run (r007 seed 303 was strongest — seed lottery flipped). |
| r008-seed404 | 2026-04-22 | transformer.yaml + C7+C8, seed=404 | 100k | val best=+1.355 @ step 80k  | 0.241 | ~9% share | |
| r008-ensemble-mean        | 2026-04-22 | 5-head mean | — | val=+1.213 / test=-0.447 | val 0.104 / test 0.602 | — | Plain mean. |
| **r008-ensemble-uncertainty** | **2026-04-22** | **5-head μ · 1/(1+5·σ̄)** | — | **val=+1.768 / test=-0.274** | **val 0.024 / test 0.137** | — | **Phase-C result.** Val metrics held up (Sharpe 1.77, dd 0.024 — both close to r007). **Test Sharpe regressed +0.24 → −0.27** and test max_dd 0.09 → 0.14. Per-seed mean val Sharpe actually *higher* than r007 (1.18 vs 1.04), so the added state features + noise appear to let individual heads overfit the 2021 val regime in a way that does not transfer to 2022–2024. Uncertainty-sizing mechanism still functions (mean_head_std≈0.70, val_dd tiny). See ISSUES.md #3. |

## Step 8 Benchmark Comparison (test split 2022–2024, 691 days)
| Strategy                         | Ann Return | Sharpe | Max DD | Calmar |
|----------------------------------|-----------:|-------:|-------:|-------:|
| r007 uncertainty ensemble (ours) | **+1.27%** | +0.268 | **8.64%** | +0.147 |
| SPY buy-and-hold                 | +11.14%    | +0.703 | 21.28% | +0.524 |
| Equal-weight 5-ETF               | +6.36%     | +0.510 | 21.31% | +0.298 |
| 60/40 SPY/TLT                    | +2.47%     | +0.251 | 21.99% | +0.112 |

**Strategy wins on drawdown only** (59% lower than SPY, 61% lower than 60/40). Loses on return, Sharpe, Calmar to SPY buy-and-hold. Defensible paper framing = "drawdown-control strategy, not alpha strategy." Artefacts: `runs/ensemble_r007/backtest_metrics.csv`, `backtest_returns.csv`, `tearsheet.html`.

## Step 9 Ablation Table (test split 2022–2024, all rows same eval harness)
Artefacts: `runs/ablation_table.csv`, `runs/ablation_table.md`.

| Row | Config | Ann Ret | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|---:|
| A | MLP multi-asset, no CVaR (r003)  | −3.19% | +0.167 | 50.91% | −0.06 |
| B | + CVaR λ=0.01 (r005)             | −1.35% | +0.180 | 51.39% | −0.03 |
| C | + Transformer, no early stop (r006-final) | −10.84% | +0.093 | 71.76% | −0.15 |
| D | + early stop (r006 @ 50k)         | +29.02% | +0.744 | 56.33% | +0.52 |
| E | + 5-seed ensemble mean (r007)     | −8.80% | −0.202 | 44.20% | −0.20 |
| **F** | **+ uncertainty sizing (FULL, r007)** | **+1.27%** | **+0.268** | **8.64%** | **+0.15** |

### Per-seed r007 (each head alone on test):
| Seed | val Sharpe | test Sharpe | test ann | test max_dd |
|---|---:|---:|---:|---:|
| 42  | +1.37 | −0.350 | −30.4% | 71.1% |
| 101 | +0.68 | **+0.741** | **+26.0%** | 42.0% |
| 202 | +0.91 | −1.000 | −54.7% | 90.4% |
| 303 | +1.63 | +0.019 | −15.2% | 66.2% |
| 404 | +0.59 | +0.094 | −2.4%  | 31.3% |

Best-val seed ≠ best-test seed. Per-seed test-Sharpe range: [−1.00, +0.74]. Ensemble uncertainty is the only configuration that does not depend on a lucky seed pick.

## r007 vs r008 impact-adjusted side-by-side (2022–2024 test, 691 days, `impact_enabled=True` replay)
Artefacts: `runs/ensemble_r007/execution_stats.csv`, `runs/ensemble_r008_liquidity/execution_stats.csv`.

| Metric | r007 | r008 | Δ |
|---|---:|---:|---:|
| Implementation shortfall (bps, total 691d) | 719.1 | **616.6** | **−14.3%** |
| Mean daily impact cost (bps) | 0.79 | **0.67** | **−15.2%** |
| Mean daily spread cost (bps) | 0.25 | **0.22** | **−12.0%** |
| Mean daily turnover | 0.2478 | **0.2203** | **−11.1%** |
| Raw Sharpe (no impact) | **+0.268** | −0.231 | −0.50 |
| Max DD | **8.64%** | 12.82% | +4.2 pp |

**Interpretation:** liquidity-aware training produces the predicted microstructure behaviour — the agent trades less and pays less impact. Raw Sharpe regresses because 100k training steps haven't compensated reduced gross turnover with higher per-trade quality. Both are honest paper-worthy findings: r007 is the cost-blind baseline, r008 is the cost-aware variant whose execution-quality claim is empirically validated.
