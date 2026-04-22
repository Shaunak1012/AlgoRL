# Architectural Decisions

## 2026-04-22 — Phase C7/C8 env design (session 13)
**Decision:**
- **C7 liquidity features** (flag `use_liquidity_features`, default False):
  - Feature 1: 20-day rolling volume z-score, clipped to ±5. Unusually thin days ⇒ hidden-liquidity withdrawal proxy.
  - Feature 2: 20-day smoothed `(H−L)/C` OHLC-dispersion spread proxy. Non-negative by construction.
  - Width: base 65 → 75 with flag on. Instance attr `self.N_FEATURES` dynamic; module-level `N_FEATURES=65` kept as the base constant so existing tests and r007 replay stay bit-identical.
- **C8 stochastic fill noise** (flag `liquidity_noise_enabled`, default False):
  - `fill_close = close × (1 + ε)`, `ε ~ N(0, σ_t)`, `σ_t = σ_base × (1 + k · spread_proxy_t)`. Defaults `σ_base=5bps`, `k=10`.
  - Noise applied to mark-to-market close so PnL disperses around frictionless close by inferred regime (simpler and cleaner than applying only to the traded delta).
  - Uses gym `self.np_random` → seed-reproducible.
**Rationale:** Adds a training-time liquidity signal to the agent state (C7) AND a stochastic friction that penalises aggressive trades in illiquid regimes (C8). Together they deliver on the synopsis claim "hidden liquidity and market impact" at daily-bar resolution, without overclaiming a tick-level dark-pool sim (per the honesty clause).
**Alternatives considered:**
- Apply noise only to the traded portion (|Δpos|) — rejected; harder to motivate (noise exists on M2M too, not just on fills) and misses the main story that *holding* in illiquid periods is itself uncertain.
- Use Kyle's-λ-scaled noise instead of spread-proxy-scaled — rejected; we already have Kyle's-λ as a deterministic cost (A2), so keeping noise's scale driver distinct (spread proxy) makes the ablation cleaner.
- Make `N_FEATURES` fully dynamic at module level — rejected; breaks `test_n_features_constant` and changes the contract of the module. Instance-level dynamic + module-level base constant keeps both worlds.
**Reversibility:** easy — both flags default False; r007 replay is bit-identical.
**Paper relevance:** Section III (MDP — two new state features when flag on; stochastic transition in reward/observation coupling), Section IV (Methodology — the microstructure proxy construction), Section VI (Ablation row G will isolate the Phase-C contribution on top of r007).

## 2026-04-22 — Synopsis-alignment scope expansion (4-phase Phase A→C→B→D)
**Decision:** Extend MVP from "risk-controlled ensemble RL" to the full synopsis-promised scope: "Deep RL for Algorithmic Trading under Hidden Liquidity and Market Impact." Four phases, executed A → C → B → D:
- **Phase A (no API keys):** add MA-crossover / TWAP / VWAP-proxy baselines, Kyle's-λ market-impact cost in env reward, impact metrics in backtest, Groq explain panel stub on Simulate tab, branding refresh.
- **Phase C (priority, no API keys):** volume-based liquidity-regime features + stochastic execution noise in env; retrain 5-seed ensemble → run ID `r008_liquidity`; new Liquidity dashboard tab + liquidity-stress figure.
- **Phase B (needs GROQ_API_KEY, NEWSAPI_KEY, GNEWS_KEY in .env):** NewsAPI+GNews headline backfill 2018–2024, FinBERT sentiment pipeline, 3 sentiment features appended to state → run ID `r009_liquidity_sentiment`.
- **Phase D:** regenerate figures (rows G+H), polish UI (Execution + Liquidity tabs), set up paper/ LaTeX (IEEEtran), draft all sections.
**Rationale:** Current build does not deliver on synopsis title (hidden liquidity, market impact, execution microstructure). User explicitly chose to close the gap while preserving existing r007 results. Phase C bumped ahead of B because liquidity awareness is the core synopsis claim; sentiment is additive. All work is additive — existing ensemble + ablation story stays intact as the "no-liquidity / no-sentiment" baseline rows.
**Alternatives considered:**
- Retitle paper to match current build (rejected — user wants synopsis delivery).
- Build tick-level dark-pool simulator (rejected — out of scope for daily-bar data; would require L2 feed we don't have).
- Rewrite env from scratch (rejected — r007 is a validated baseline, don't throw away 5-seed ensemble).
**Reversibility:** medium — env changes are flag-gated (`impact_enabled`, `liquidity_noise_enabled`, `use_sentiment`); each Phase can be disabled via config. Retraining cost ~6 GPU-hrs per new run ID on Colab T4.
**Dark-pool honesty clause:** Paper will frame hidden liquidity as "an unobservable component of execution uncertainty inferred via public-tape microstructure proxies (volume z-score + OHLC-dispersion spread proxy)." No claim of tick-level dark-pool simulation.
**Paper relevance:** reshapes Sections I (motivation), III (MDP — new state features + stochastic fill), IV (methodology — impact model + liquidity regime + sentiment), V (new baselines), VI (rows G, H in ablation), VII (limitations — proxy vs tick-level).

## 2026-04-21 — Ensemble design + uncertainty-scaled position sizing (MVP Step 7)
**Decision:**
- Train N=5 independent PPO+Transformer heads (seeds 42, 101, 202, 303, 404), each with `ValEvalCallback` early-stopping on val Sharpe (eval every 10k steps; save `model_best_val.zip`).
- At inference, stack head actions → `(N, 5)`. Define `μ = mean`, `σ = std` across heads.
- Action = μ × 1/(1 + scale · mean(σ)) with `scale=5.0` default — shrinks portfolio size when heads disagree. No per-asset scaling (uses scalar mean σ) to keep the uncertainty signal simple and low-variance.
- Report both `mean` (no shrinkage) and `uncertainty` (shrunk) policies so the sizing contribution is isolatable.
**Rationale:** r006 proved a single head overfits; ensembling should both (a) reduce variance of final val metrics and (b) provide a legitimate disagreement-based risk signal to justify the "uncertainty-scaled" contribution claim in the paper. Per-asset shrinkage was considered but adds 5 degrees of freedom in the inference path for little expected gain — keep MVP simple.
**Alternatives considered:** bootstrap DQN-style prior networks (heavier plumbing), MC-Dropout on a single head (fewer flops but doesn't give real model-level disagreement), weight averaging across heads (loses the uncertainty signal).
**Reversibility:** easy — ensemble code is in `scripts/ensemble_eval.py`, callback in `scripts/callbacks.py`; can revert to single-head by evaluating one checkpoint.
**Paper relevance:** Section IV (Methodology) — this IS the uncertainty-quantification contribution; and Section VI — expect ensemble row to be the headline result.

## 2026-04-20 — Transformer encoder architecture (MVP Step 6)
**Decision:** Replace MLP with a causal Transformer encoder implemented as an SB3 `BaseFeaturesExtractor` (NOT a custom `ActorCriticPolicy` subclass). Config:
- window_size T=60, d_model=64, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.1
- Learned positional embedding (T, d_model); input projection Linear(F=65 → d_model)
- Causal mask (upper-triangular −inf) on self-attention
- Output = last-timestep token embedding → fed to SB3's default MLP actor/critic heads
**Rationale:** The env already emits (T, F) 3D observations (Box). SB3's `MlpPolicy` accepts 2D obs via `features_extractor_class` — no policy subclass needed, avoiding a large surface area of fragile SB3 internals. Learned positional encoding is simpler and typically equal-or-better than sinusoidal at T=60. Causal mask is required so the policy at position t cannot attend to future pad tokens during early-episode warmup.
**Alternatives considered:**
- Full custom `ActorCriticPolicy` subclass — rejected as unnecessary complexity.
- GRU/LSTM encoder via `sb3-contrib RecurrentPPO` — rejected; Transformer is the paper's novel contribution.
- Sinusoidal positional encoding — deferred; learned embedding for a fixed T=60 window is standard and removes one hyperparameter.
- Larger d_model (128/256) — deferred until r006 baseline lands; 64 keeps parameter count low for fair compare vs MLP (net_arch=[128,128]).
**Reversibility:** easy — extractor is a drop-in; reverting to MLP is a one-line config change.
**Paper relevance:** goes in Section IV (Methodology) — this is the architectural contribution.

## 2026-04-20 — Streamlit dashboard added to roadmap
**Decision:** Add Streamlit interactive dashboard as MVP Step 10 (between ablation and paper), shifting paper to Step 11.
**Rationale:** User wants an interactive UI for demo; QuantStats tearsheet alone is sufficient for the paper but not for live exploration. Streamlit is the fastest path to a polished demo.
**Alternatives considered:** Dash (more complex), Gradio (less suited to time-series finance), raw HTML (too much work).
**Reversibility:** easy — it's additive, doesn't affect training pipeline.
**Paper relevance:** not paper-relevant; demo only.

## 2026-04-20 — Colab training workflow
**Decision:** All training runs that take >2 min will be executed on Google Colab (user's MacBook is the dev machine; RTX 5080 rig used via Colab). Results returned as a zip file (model checkpoint + learning_curve.csv + config snapshot) and dropped into the workspace. Claude Code infers results from the zip and updates EXPERIMENTS.md.
**Rationale:** MacBook is too slow for Transformer + ensemble training (Steps 6–7 onward). Colab keeps iteration fast.
**Alternatives considered:** SSH to remote box (no persistent setup), cloud VM (more overhead).
**Reversibility:** easy — local training still works for fast experiments.
**Paper relevance:** goes in Section V (Experimental Setup) — hardware footnote.

## 2026-04-20 — Initial stack selection
**Decision:** Python 3.11, PyTorch 2.x, Gymnasium, Stable-Baselines3 (DQN baseline only), VectorBT + QuantStats for backtesting.
**Rationale:** Matches CLAUDE.md Section 6.1 requirements. Gymnasium is the maintained fork of OpenAI Gym. SB3 provides a clean PPO reference implementation.
**Alternatives considered:** RLlib (too heavy), CleanRL (less ecosystem), raw PyTorch PPO (more control but slower to prototype).
**Reversibility:** medium
**Paper relevance:** goes in Section V (Experimental Setup)
