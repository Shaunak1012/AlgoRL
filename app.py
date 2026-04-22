"""Streamlit dashboard — Deep RL for Algorithmic Trading under
Market Impact and Inferred Hidden Liquidity.

CSV-driven, no torch/SB3 imports. Interactive charts via Plotly.
Run with: /opt/homebrew/bin/streamlit run app.py --server.port 8504
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── load .env if present (local dev) ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional; keys can be set in environment directly

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoRL | Deep RL Trading · Market Impact & Hidden Liquidity",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], [class*="st-"] { font-family: 'Inter', sans-serif !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(124, 58, 237, 0.35);
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(124, 58, 237, 0.7);
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* Tabs */
[data-baseweb="tab-list"] { gap: 6px; background: transparent !important; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0e1117 100%);
    border-right: 1px solid rgba(124,58,237,0.2);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 3px; }

/* Hide broken Material Icons ligature text throughout */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
span.material-icons, span.material-icons-sharp,
span.material-icons-outlined, span.material-icons-round { display: none !important; }


.hero-title {
    background: linear-gradient(135deg, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 6px;
    line-height: 1.6;
}
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(124,58,237,0.3);
}
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.5) 50%, transparent 100%);
    margin: 20px 0 18px 0;
}
.claim-box {
    background: rgba(124, 58, 237, 0.08);
    border-left: 3px solid #7C3AED;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: #e2e8f0;
    line-height: 1.5;
}
.info-card {
    background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    color: #cbd5e0;
    line-height: 1.6;
}
.badge {
    display: inline-block;
    background: rgba(124,58,237,0.18);
    color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 5px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "strategy":      "#FF6B6B",
    "buy_and_hold":  "#4ECDC4",
    "equal_weight":  "#FFE66D",
    "sixty_forty":   "#95E1A4",
    "ma_crossover":  "#a78bfa",
    "twap_ew":       "#fb923c",
    "vwap_proxy_ew": "#38bdf8",
    "accent":        "#7C3AED",
    "accent2":       "#EC4899",
    "bg":            "#0e1117",
    "bg2":           "#16213e",
    "grid":          "rgba(255,255,255,0.05)",
    "text":          "#e2e8f0",
}
LABELS = {
    "strategy":      "Ensemble PPO (ours)",
    "buy_and_hold":  "SPY Buy & Hold",
    "equal_weight":  "Equal Weight",
    "sixty_forty":   "60/40",
    "ma_crossover":  "MA Crossover 50/200",
    "twap_ew":       "TWAP (equal-weight)",
    "vwap_proxy_ew": "VWAP-proxy (equal-weight)",
}
SEED_COLORS = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1A4", "#a78bfa"]

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["bg"],
    font=dict(family="Inter, sans-serif", color=C["text"], size=12),
    margin=dict(l=50, r=20, t=36, b=40),
    legend=dict(
        bgcolor="rgba(22,33,62,0.85)",
        bordercolor="rgba(124,58,237,0.35)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#7C3AED", font_size=12),
)


def pstyle(fig: go.Figure, *, height: int = 360, grid: bool = True) -> go.Figure:
    """Apply shared Plotly styling."""
    fig.update_layout(**_BASE_LAYOUT, height=height)
    if grid:
        fig.update_xaxes(gridcolor=C["grid"], showgrid=True, zeroline=False)
        fig.update_yaxes(gridcolor=C["grid"], showgrid=True, zeroline=False)
    return fig


def sh(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def div():
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


def _groq_explain(
    strat_m: dict,
    bh_m: dict,
    window_label: str,
    cost_mult: float,
) -> str:
    """Call Groq Llama-3.3-70B to narrate the selected simulation window.

    Returns the explanation string, or an error message if the key is absent.
    Results are cached per (window_label, cost_mult) to avoid repeat API calls.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return (
            "⚠️ **GROQ_API_KEY not set.** Add it to your `.env` file and restart the app.\n\n"
            "`GROQ_API_KEY=gsk_...`"
        )
    try:
        from groq import Groq
    except ImportError:
        return "⚠️ `groq` package not installed. Run `pip install groq`."

    client = Groq(api_key=groq_key)
    prompt = f"""You are a quantitative analyst presenting results to a university professor.
Explain the performance of an AI-driven ensemble RL trading strategy in plain language.

Time window: {window_label}
Transaction cost multiplier: {cost_mult:.2f}× baseline (1.0 = model default)

Strategy metrics:
  Annualised Return : {strat_m['Ann. Return']:+.1%}
  Sharpe Ratio      : {strat_m['Sharpe']:+.2f}
  Sortino Ratio     : {strat_m['Sortino']:+.2f}
  Max Drawdown      : {strat_m['Max DD']:.1%}
  Calmar Ratio      : {strat_m['Calmar']:+.2f}
  Final Value       : ${strat_m['Final Value']:,.0f}
  P&L               : ${strat_m['P&L']:+,.0f}

SPY Buy-and-Hold benchmark:
  Annualised Return : {bh_m['Ann. Return']:+.1%}
  Sharpe Ratio      : {bh_m['Sharpe']:+.2f}
  Max Drawdown      : {bh_m['Max DD']:.1%}

Task:
1. Summarise what happened in this window in 2–3 sentences (e.g., market regime, whether
   the strategy protected capital or left gains on the table).
2. Highlight one strength and one weakness versus the SPY benchmark.
3. If transaction costs were stressed (cost_mult > 1), comment on the impact.
4. Keep the response under 200 words. Use plain language, no LaTeX, no bullet symbols."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=320,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ Groq API error: {exc}"


# ── paths ─────────────────────────────────────────────────────────────────────
RUNS     = Path(__file__).parent / "runs"
R007     = RUNS / "ensemble_r007"
R008     = RUNS / "ensemble_r008_liquidity"
SEED_DIRS = sorted(R007.glob("*_r007_seed*"))


# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_returns() -> pd.DataFrame:
    return pd.read_csv(R007 / "backtest_returns.csv", parse_dates=["Date"], index_col="Date")

@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(R007 / "backtest_metrics.csv", index_col="name")

@st.cache_data
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(RUNS / "ablation_table.csv")

@st.cache_data
def load_ensemble_json() -> dict:
    with open(R007 / "ensemble_results.json") as f:
        return json.load(f)

@st.cache_data
def load_val_curves() -> dict[str, pd.DataFrame]:
    out = {}
    for d in SEED_DIRS:
        p = d / "val_curve.csv"
        if p.exists():
            out[f"seed {d.name.split('seed')[-1]}"] = pd.read_csv(p)
    return out

@st.cache_data
def load_learning_curves() -> dict[str, pd.DataFrame]:
    out = {}
    for d in SEED_DIRS:
        p = d / "learning_curve.csv"
        if p.exists():
            out[f"seed {d.name.split('seed')[-1]}"] = pd.read_csv(p)
    return out

@st.cache_data
def load_r008_returns() -> pd.DataFrame | None:
    p = R008 / "backtest_returns.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["Date"], index_col="Date")
    return None

@st.cache_data
def load_r008_metrics() -> pd.DataFrame | None:
    p = R008 / "backtest_metrics.csv"
    if p.exists():
        return pd.read_csv(p, index_col="name")
    return None

@st.cache_data
def load_r008_exec() -> pd.DataFrame | None:
    p = R008 / "execution_stats.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_r008_ensemble() -> dict | None:
    p = R008 / "ensemble_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# ── derived ───────────────────────────────────────────────────────────────────
@st.cache_data
def get_equity() -> pd.DataFrame:
    return (1 + load_returns()).cumprod()

@st.cache_data
def get_drawdown() -> pd.DataFrame:
    eq = get_equity()
    return eq / eq.cummax() - 1

@st.cache_data
def get_rolling_sharpe(w: int = 30) -> pd.DataFrame:
    r = load_returns()
    return (
        r.rolling(w)
         .apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 1e-10 else 0.0, raw=True)
         .dropna()
    )

@st.cache_data
def get_monthly_returns() -> pd.DataFrame:
    return load_returns().resample("ME").apply(lambda x: (1 + x).prod() - 1)


# ── raw asset data (OHLCV parquets) ───────────────────────────────────────────
ASSETS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
ASSET_COLORS = {
    "SPY": "#4ECDC4",  # teal
    "QQQ": "#A78BFA",  # purple
    "IWM": "#FFE66D",  # yellow
    "GLD": "#F59E0B",  # amber (gold)
    "TLT": "#FF6B6B",  # coral (bond)
}
ASSET_NAMES = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "IWM": "Russell 2000 ETF",
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury ETF",
}

@st.cache_data
def load_asset_prices() -> pd.DataFrame:
    """Load close prices for all 5 ETFs over the test window (2022–2024)."""
    closes = {}
    data_dir = Path(__file__).parent / "data"
    for sym in ASSETS:
        df = pd.read_parquet(data_dir / f"{sym}.parquet")
        closes[sym] = df["close"]
    out = pd.DataFrame(closes)
    # align to test period only
    ret = load_returns()
    return out.loc[ret.index.min():ret.index.max()].dropna()

@st.cache_data
def load_asset_returns() -> pd.DataFrame:
    return load_asset_prices().pct_change().dropna()

@st.cache_data
def detect_spy_regimes() -> pd.Series:
    """Classify each test-set day into bull/bear/sideways based on 60-day SPY return."""
    spy = load_asset_prices()["SPY"]
    trailing = spy.pct_change(60)
    regime = pd.Series("sideways", index=spy.index)
    regime[trailing > 0.06] = "bull"
    regime[trailing < -0.06] = "bear"
    return regime


# ── metric help text ──────────────────────────────────────────────────────────
HELP = {
    "sharpe":  "Sharpe Ratio: ann. excess return ÷ ann. volatility. >1 is good, >2 excellent.",
    "max_dd":  "Max Drawdown: largest peak-to-trough loss during the period.",
    "ann_ret": "Annualised Return: geometric mean return scaled to a one-year horizon.",
    "sortino": "Sortino Ratio: like Sharpe but only penalises downside volatility.",
    "calmar":  "Calmar Ratio: ann. return ÷ |max drawdown|. Risk-adjusted return.",
    "vol":     "Annualised Volatility: std of daily returns × √252.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="hero-title" style="font-size:1.5rem; margin-bottom:4px; line-height:1.3">
        AlgoRL
    </div>
    <div class="hero-subtitle" style="font-size:0.78rem; line-height:1.5">
        Market Impact &amp; Inferred<br>Hidden Liquidity
    </div>
    """, unsafe_allow_html=True)

    div()

    st.markdown("""
    <div class="info-card">
        <b style="color:#a78bfa">Architecture</b><br>
        5 × PPO heads · Transformer encoder<br>
        d_model=64 · window T=60 · 2 layers<br>
        Causal attn · Uncertainty sizing<br>
        Kyle's-λ market impact · CVaR reward
    </div>
    <div class="info-card">
        <b style="color:#a78bfa">Dataset</b><br>
        SPY · QQQ · IWM · GLD · TLT<br>
        Train 2018–2020 · Val 2021<br>
        Test 2022–2024 · 691 trading days
    </div>
    <div class="info-card">
        <b style="color:#a78bfa">Key Results</b><br>
        <span style="color:#94a3b8;font-size:0.78rem">r007 (baseline):</span><br>
        ↓ <b>59%</b> max drawdown vs SPY<br>
        ↓ <b>6.5×</b> DD vs naive ensemble<br>
        <span style="color:#94a3b8;font-size:0.78rem;margin-top:6px;display:inline-block">r008 (liquidity-aware):</span><br>
        ↓ <b>14.3%</b> implementation shortfall<br>
        ↓ <b>15.2%</b> mean impact bps/day<br>
        ↓ <b>11.1%</b> turnover
    </div>
    """, unsafe_allow_html=True)

    div()

    st.markdown("""
    <span style="color:#718096;font-size:0.75rem">
        Shaunak A. Rai · Deep RL for Algorithmic Trading · 2026
    </span>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_ov, tab_risk, tab_assets, tab_abl, tab_train, tab_sim, tab_liq, tab_exec = st.tabs([
    "📊  Overview",
    "⚠️  Risk & Returns",
    "🌐  Assets",
    "🔬  Ablation",
    "🧠  Training",
    "📅  Simulate",
    "💧  Liquidity",
    "⚙️  Execution",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    st.markdown("""
    <div style="margin-bottom:22px">
        <div class="hero-title">AlgoRL · Test Set 2022–2024</div>
        <div class="hero-subtitle">
            Market Impact &amp; Inferred Hidden Liquidity &nbsp;|&nbsp;
            5-asset portfolio · SPY · QQQ · IWM · GLD · TLT &nbsp;|&nbsp;
            691 days · Uncertainty-scaled ensemble · Kyle's-λ impact
        </div>
    </div>
    """, unsafe_allow_html=True)

    returns = load_returns()
    metrics = load_metrics()
    strat   = metrics.loc["strategy"]
    spybh   = metrics.loc["buy_and_hold"]

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe Ratio",    f"{strat['sharpe']:.2f}",
              delta=f"{strat['sharpe'] - spybh['sharpe']:+.2f} vs SPY",
              help=HELP["sharpe"])
    c2.metric("Max Drawdown",    f"−{strat['max_drawdown']:.1%}",
              delta=f"{(strat['max_drawdown'] - spybh['max_drawdown'])*100:+.1f}pp",
              delta_color="inverse", help=HELP["max_dd"])
    c3.metric("Ann. Return",     f"{strat['ann_return']:+.1%}",
              delta=f"{(strat['ann_return'] - spybh['ann_return'])*100:+.1f}pp vs SPY",
              help=HELP["ann_ret"])
    c4.metric("Sortino",         f"{strat['sortino']:.2f}",
              delta=f"{strat['sortino'] - spybh['sortino']:+.2f} vs SPY",
              help=HELP["sortino"])
    c5.metric("Calmar",          f"{strat['calmar']:.2f}",
              delta=f"{strat['calmar'] - spybh['calmar']:+.2f} vs SPY",
              help=HELP["calmar"])
    c6.metric("Ann. Volatility", f"{strat['ann_vol']:.1%}",
              delta=f"{(strat['ann_vol'] - spybh['ann_vol'])*100:+.1f}pp",
              delta_color="inverse", help=HELP["vol"])

    div()

    # ── Equity curve ──────────────────────────────────────────────────────────
    eq_head_l, eq_head_r = st.columns([3, 1])
    with eq_head_l:
        sh("CUMULATIVE EQUITY CURVES — INTERACTIVE")
    with eq_head_r:
        show_regimes = st.toggle("Show market regimes", value=False,
                                  help="Shade bull/bear/sideways zones based on SPY 60-day return")
    equity = get_equity()
    dd_s   = get_drawdown()

    worst_end   = dd_s["strategy"].idxmin()
    worst_start = equity["strategy"][:worst_end].idxmax()

    fig_eq = go.Figure()

    # Market regime shading
    if show_regimes:
        try:
            regimes = detect_spy_regimes()
            # reindex to equity index
            regimes = regimes.reindex(equity.index, method="ffill").fillna("sideways")
            regime_colors = {
                "bull":     "rgba(149, 225, 164, 0.10)",
                "bear":     "rgba(255, 107, 107, 0.10)",
                "sideways": "rgba(255, 230, 109, 0.05)",
            }
            # compute contiguous regime segments
            rvals = regimes.values
            idxs  = regimes.index
            start = 0
            for i in range(1, len(rvals)):
                if rvals[i] != rvals[start]:
                    fig_eq.add_vrect(
                        x0=idxs[start], x1=idxs[i],
                        fillcolor=regime_colors[rvals[start]],
                        line_width=0, layer="below",
                    )
                    start = i
            fig_eq.add_vrect(
                x0=idxs[start], x1=idxs[-1],
                fillcolor=regime_colors[rvals[start]],
                line_width=0, layer="below",
            )
        except Exception:
            pass

    for col in equity.columns:
        lw = 2.5 if col == "strategy" else 1.4
        dash = "solid" if col == "strategy" else "dot"
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity[col],
            name=LABELS[col],
            line=dict(color=C[col], width=lw, dash=dash),
            hovertemplate=(
                f"<b>{LABELS[col]}</b><br>"
                "%{x|%b %d %Y}<br>"
                "Value: %{y:.4f}<extra></extra>"
            ),
        ))
    fig_eq.add_vrect(
        x0=worst_start, x1=worst_end,
        fillcolor=C["strategy"], opacity=0.10, line_width=0,
        annotation_text="Max DD", annotation_position="top left",
        annotation_font_color=C["strategy"], annotation_font_size=11,
    )
    pstyle(fig_eq, height=390)
    fig_eq.update_layout(yaxis_title="Portfolio Value (normalised to 1.0)")
    st.plotly_chart(fig_eq, width="stretch")

    div()

    col_heat, col_table = st.columns([1.5, 1])

    with col_heat:
        sh("MONTHLY RETURNS HEATMAP — ENSEMBLE PPO")
        monthly = get_monthly_returns()["strategy"]
        mdf = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        })
        pivot = mdf.pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
        text_vals = pivot.map(
            lambda v: f"{v*100:+.1f}%" if not pd.isna(v) else ""
        )
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            text=text_vals.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            colorscale="RdYlGn",
            zmid=0, zmin=-8, zmax=8,
            colorbar=dict(title="Ret %", ticksuffix="%", len=0.8),
            hoverongaps=False,
            hovertemplate="<b>%{x} %{y}</b><br>Return: %{z:.2f}%<extra></extra>",
        ))
        pstyle(fig_hm, height=230, grid=False)
        fig_hm.update_layout(margin=dict(l=50, r=60, t=10, b=30))
        st.plotly_chart(fig_hm, width="stretch")

    with col_table:
        sh("BENCHMARK COMPARISON")
        disp = metrics[["ann_return","ann_vol","sharpe","sortino","max_drawdown","calmar"]].copy()
        disp.index = [LABELS.get(i, i) for i in disp.index]
        disp.columns = ["Ann Ret", "Vol", "Sharpe", "Sortino", "Max DD", "Calmar"]
        st.dataframe(
            disp.style.format({
                "Ann Ret": "{:+.1%}", "Vol":     "{:.1%}",
                "Sharpe":  "{:.2f}",  "Sortino": "{:.2f}",
                "Max DD":  "{:.1%}",  "Calmar":  "{:.2f}",
            })
            .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-0.5, vmax=1.0)
            .background_gradient(subset=["Max DD"], cmap="RdYlGn_r", vmin=0.0, vmax=0.6)
            .background_gradient(subset=["Vol"],    cmap="RdYlGn_r", vmin=0.0, vmax=0.25),
            width="stretch", height=250,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · RISK & RETURNS
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="hero-title">Risk & Returns Analytics</div>
        <div class="hero-subtitle">
            Drawdown decomposition · Return distributions · Rolling Sharpe · VaR/CVaR · Volatility
        </div>
    </div>
    """, unsafe_allow_html=True)

    returns = load_returns()

    # ── Underwater chart ──────────────────────────────────────────────────────
    sh("UNDERWATER CHART — DRAWDOWN OVER TIME")
    dd = get_drawdown()
    fig_dd = go.Figure()
    for col in dd.columns:
        is_strat = col == "strategy"
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd[col] * 100,
            name=LABELS[col],
            fill="tozeroy" if is_strat else "none",
            fillcolor="rgba(255,107,107,0.12)",
            line=dict(color=C[col], width=2.5 if is_strat else 1.2),
            hovertemplate=(
                f"<b>{LABELS[col]}</b><br>"
                "%{x|%b %d %Y}<br>"
                "DD: %{y:.2f}%<extra></extra>"
            ),
        ))
    for lvl, lbl in [(-5,"−5%"), (-10,"−10%"), (-20,"−20%"), (-40,"−40%")]:
        fig_dd.add_hline(
            y=lvl, line_dash="dot",
            line_color="rgba(255,255,255,0.12)",
            annotation_text=lbl, annotation_font_size=9,
            annotation_font_color="rgba(255,255,255,0.4)",
        )
    pstyle(fig_dd, height=320)
    fig_dd.update_layout(yaxis_title="Drawdown (%)", yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_dd, width="stretch")

    div()
    col_rs, col_dist = st.columns(2)

    with col_rs:
        # ── Rolling Sharpe ────────────────────────────────────────────────────
        sh("ROLLING 30-DAY SHARPE RATIO")
        rsh = get_rolling_sharpe(30)
        fig_rs = go.Figure()
        for col in rsh.columns:
            fig_rs.add_trace(go.Scatter(
                x=rsh.index, y=rsh[col],
                name=LABELS[col],
                line=dict(color=C[col], width=2 if col == "strategy" else 1.2),
                hovertemplate=f"<b>{LABELS[col]}</b><br>%{{y:.2f}}<extra></extra>",
            ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
        pstyle(fig_rs, height=300)
        fig_rs.update_layout(yaxis_title="Rolling Sharpe (annualised)")
        st.plotly_chart(fig_rs, width="stretch")

    with col_dist:
        # ── Return distribution ────────────────────────────────────────────────
        sh("DAILY RETURN DISTRIBUTIONS")
        fig_dist = go.Figure()
        for col in returns.columns:
            fig_dist.add_trace(go.Histogram(
                x=returns[col] * 100,
                name=LABELS[col],
                opacity=0.55,
                marker_color=C[col],
                xbins=dict(size=0.12),
                hovertemplate=f"<b>{LABELS[col]}</b><br>Bin: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>",
            ))
        pstyle(fig_dist, height=300, grid=False)
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            xaxis=dict(range=[-3.5, 3.5]),
        )
        st.plotly_chart(fig_dist, width="stretch")

    div()
    col_var, col_vol = st.columns(2)

    with col_var:
        # ── VaR / CVaR ────────────────────────────────────────────────────────
        sh("VALUE AT RISK SUMMARY")
        rows_var = []
        for csv_col, display_name in LABELS.items():
            r = returns[csv_col]
            row = {"Strategy": display_name}
            for cl in [0.90, 0.95, 0.99]:
                var_val  = r.quantile(1 - cl)
                cvar_val = r[r <= var_val].mean()
                row[f"VaR {int(cl*100)}%"]  = f"{var_val*100:.2f}%"
                row[f"CVaR {int(cl*100)}%"] = f"{cvar_val*100:.2f}%"
            rows_var.append(row)
        st.dataframe(
            pd.DataFrame(rows_var).set_index("Strategy"),
            width="stretch",
        )
        st.caption(
            "**VaR**: worst-case daily return at confidence level.  "
            "**CVaR**: expected loss given VaR is breached."
        )

    with col_vol:
        # ── Rolling volatility ────────────────────────────────────────────────
        sh("ROLLING 30-DAY ANNUALISED VOLATILITY")
        rvol = returns.rolling(30).std().dropna() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        for col in rvol.columns:
            fig_vol.add_trace(go.Scatter(
                x=rvol.index, y=rvol[col],
                name=LABELS[col],
                line=dict(color=C[col], width=2 if col == "strategy" else 1.2),
                hovertemplate=f"<b>{LABELS[col]}</b><br>%{{y:.1f}}%<extra></extra>",
            ))
        pstyle(fig_vol, height=280)
        fig_vol.update_layout(
            yaxis_title="Ann. Volatility (%)",
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_vol, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · ASSETS  (investable universe — SPY · QQQ · IWM · GLD · TLT)
# ─────────────────────────────────────────────────────────────────────────────
with tab_assets:
    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="hero-title">The Investable Universe</div>
        <div class="hero-subtitle">
            5 liquid ETFs spanning equities, small-cap, gold, and long-duration bonds ·
            Correlation · Per-asset performance · Rolling cross-correlation
        </div>
    </div>
    """, unsafe_allow_html=True)

    asset_ret = load_asset_returns()
    asset_px  = load_asset_prices()

    # ── Asset cards strip ─────────────────────────────────────────────────────
    cols_a = st.columns(5)
    for sym, col in zip(ASSETS, cols_a):
        r = asset_ret[sym]
        ann_r = (1 + r.mean()) ** 252 - 1
        sh_r  = r.mean() / r.std() * np.sqrt(252) if r.std() > 1e-10 else 0.0
        col.markdown(f"""
        <div class="info-card" style="text-align:center;border-color:{ASSET_COLORS[sym]}">
            <div style="color:{ASSET_COLORS[sym]};font-weight:700;font-size:1.4rem">{sym}</div>
            <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:8px">{ASSET_NAMES[sym]}</div>
            <div style="font-size:0.8rem;color:#e2e8f0">
                Ann Ret <b>{ann_r:+.1%}</b><br>
                Sharpe <b>{sh_r:.2f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    div()

    col_px, col_corr = st.columns([1.4, 1])

    with col_px:
        # ── Normalised price curves ───────────────────────────────────────────
        sh("NORMALISED PRICE PERFORMANCE — TEST WINDOW")
        px_norm = asset_px / asset_px.iloc[0]
        fig_px = go.Figure()
        for sym in ASSETS:
            fig_px.add_trace(go.Scatter(
                x=px_norm.index, y=px_norm[sym],
                name=sym, line=dict(color=ASSET_COLORS[sym], width=1.8),
                hovertemplate=f"<b>{sym}</b> — {ASSET_NAMES[sym]}<br>%{{x|%b %d %Y}}<br>Value: %{{y:.3f}}<extra></extra>",
            ))
        pstyle(fig_px, height=360)
        fig_px.update_layout(yaxis_title="Normalised Price (start = 1.0)")
        st.plotly_chart(fig_px, width="stretch")

    with col_corr:
        # ── Correlation heatmap ───────────────────────────────────────────────
        sh("RETURN CORRELATION MATRIX")
        corr = asset_ret.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            text=corr.values, texttemplate="%{text:.2f}",
            textfont=dict(size=13, color="white"),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="ρ", len=0.8),
            hovertemplate="<b>%{y} × %{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        ))
        pstyle(fig_corr, height=360, grid=False)
        fig_corr.update_layout(margin=dict(l=50, r=60, t=10, b=30))
        st.plotly_chart(fig_corr, width="stretch")

    div()

    col_rc, col_tbl = st.columns([1.4, 1])

    with col_rc:
        # ── Rolling correlation with SPY ──────────────────────────────────────
        sh("ROLLING 60-DAY CORRELATION WITH SPY")
        rc = pd.DataFrame({
            sym: asset_ret[sym].rolling(60).corr(asset_ret["SPY"])
            for sym in ASSETS if sym != "SPY"
        }).dropna()
        fig_rc = go.Figure()
        for sym in rc.columns:
            fig_rc.add_trace(go.Scatter(
                x=rc.index, y=rc[sym],
                name=sym, line=dict(color=ASSET_COLORS[sym], width=1.6),
                hovertemplate=f"<b>{sym} ↔ SPY</b><br>ρ = %{{y:.3f}}<extra></extra>",
            ))
        fig_rc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
        pstyle(fig_rc, height=320)
        fig_rc.update_layout(yaxis_title="ρ (60-day rolling)",
                             yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig_rc, width="stretch")

    with col_tbl:
        # ── Per-asset stats ───────────────────────────────────────────────────
        sh("PER-ASSET TEST STATS")
        rows_a = []
        for sym in ASSETS:
            r = asset_ret[sym]
            ann_r = (1 + r.mean()) ** 252 - 1
            vol   = r.std() * np.sqrt(252)
            sh_r  = ann_r / vol if vol > 1e-10 else 0.0
            eq    = (1 + r).cumprod()
            dd    = (eq / eq.cummax() - 1).min()
            rows_a.append({
                "Asset":   sym,
                "Ann Ret": ann_r,
                "Vol":     vol,
                "Sharpe":  sh_r,
                "Max DD":  dd,
            })
        stats_df = pd.DataFrame(rows_a).set_index("Asset")
        st.dataframe(
            stats_df.style.format({
                "Ann Ret": "{:+.1%}", "Vol": "{:.1%}",
                "Sharpe":  "{:.2f}",  "Max DD": "{:.1%}",
            })
            .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-0.5, vmax=1.5)
            .background_gradient(subset=["Max DD"], cmap="RdYlGn_r", vmin=-0.6, vmax=0),
            width="stretch", height=270,
        )
        st.caption(
            "**Paper context:** the ensemble allocates across these 5 with continuous weights. "
            "Low correlation between equities and TLT/GLD provides natural diversification."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · ABLATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_abl:
    abl = load_ablation()

    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="hero-title">Component Ablation Study</div>
        <div class="hero-subtitle">
            Each row adds one component to the prior. Row F is the full system and the only
            drawdown-monotone configuration across the sequence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="claim-box">
        🎯 <b>Row F (full system):</b> Sharpe 0.268, Max DD <b>8.6%</b> — uniquely combines positive Sharpe <i>and</i> single-digit drawdown.
    </div>
    <div class="claim-box">
        ⚠️ <b>Row E (ensemble mean, no shrinkage):</b> Sharpe <b>−0.20</b>. Disagreeing heads cancel each other; only transaction spread cost survives. This is exactly WHY uncertainty sizing is needed.
    </div>
    <div class="claim-box">
        🎰 <b>Row D (seed-42 early-stopped):</b> Sharpe 0.744 — but other seeds with identical recipe lose 30–55%. Never deploy a cherry-picked checkpoint.
    </div>
    """, unsafe_allow_html=True)

    div()
    col_tbl, col_bars = st.columns([1, 1.4])

    with col_tbl:
        sh("ABLATION TABLE")
        disp_abl = abl[["label","n_members","mode",
                         "ann_return","sharpe","max_drawdown","calmar"]].copy()
        disp_abl.columns = ["Config","N","Mode","Ann Ret","Sharpe","Max DD","Calmar"]
        st.dataframe(
            disp_abl.style.format({
                "Ann Ret": "{:+.1%}", "Sharpe": "{:.3f}",
                "Max DD":  "{:.1%}",  "Calmar": "{:.2f}",
            })
            .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-0.5, vmax=1.0)
            .background_gradient(subset=["Max DD"], cmap="RdYlGn_r", vmin=0.0, vmax=0.8),
            width="stretch", hide_index=True, height=310,
        )

    with col_bars:
        sh("SHARPE & MAX DRAWDOWN BY CONFIG")
        labels_short = ["A\nMLP", "B\n+CVaR", "C\n+Trans.\n(no ES)", "D\n+EarlyStop",
                        "E\n+Ensemble\n(mean)", "F\nFull\n(ours)"]
        sh_vals = abl["sharpe"].tolist()
        dd_vals = (abl["max_drawdown"] * 100).tolist()

        sh_colors = [C["accent2"] if v < 0 else C["buy_and_hold"] for v in sh_vals]
        sh_colors[-1] = C["accent"]
        dd_colors = [C["strategy"] if v > 50 else C["sixty_forty"] for v in dd_vals]
        dd_colors[-1] = C["accent"]

        fig_abl = make_subplots(rows=1, cols=2,
                                subplot_titles=["Sharpe Ratio", "Max Drawdown (%)"],
                                horizontal_spacing=0.12)
        fig_abl.add_trace(go.Bar(
            x=labels_short, y=sh_vals,
            marker_color=sh_colors,
            text=[f"{v:.3f}" for v in sh_vals],
            textposition="outside", showlegend=False,
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig_abl.add_trace(go.Bar(
            x=labels_short, y=dd_vals,
            marker_color=dd_colors,
            text=[f"{v:.0f}%" for v in dd_vals],
            textposition="outside", showlegend=False,
            hovertemplate="<b>%{x}</b><br>Max DD: %{y:.1f}%<extra></extra>",
        ), row=1, col=2)
        fig_abl.add_hline(y=0, row=1, col=1,
                          line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_abl.update_layout(
            **{k: v for k, v in _BASE_LAYOUT.items()
               if k not in ("hovermode",)},
            height=370, showlegend=False,
            hovermode="closest",
        )
        fig_abl.update_xaxes(tickfont=dict(size=8), gridcolor=C["grid"])
        fig_abl.update_yaxes(gridcolor=C["grid"])
        fig_abl.update_annotations(font_size=11, font_color="#a78bfa")
        st.plotly_chart(fig_abl, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    val_curves   = load_val_curves()
    learn_curves = load_learning_curves()
    ens          = load_ensemble_json()

    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="hero-title">Training Dynamics & Ensemble</div>
        <div class="hero-subtitle">
            Learning curves · Reward decomposition · Val vs test rank inversion · Ensemble aggregation
        </div>
    </div>
    """, unsafe_allow_html=True)
    div()

    col_vc, col_lc = st.columns(2)

    with col_vc:
        sh("VALIDATION SHARPE — 5 SEEDS  (★ = best checkpoint)")
        fig_vc = go.Figure()
        for (name, df), clr in zip(val_curves.items(), SEED_COLORS):
            fig_vc.add_trace(go.Scatter(
                x=df["timestep"], y=df["val_sharpe"],
                name=name, line=dict(color=clr, width=1.8),
                hovertemplate=f"<b>{name}</b><br>Step: %{{x:,}}<br>Val Sharpe: %{{y:.3f}}<extra></extra>",
            ))
            best_idx = df["val_sharpe"].idxmax()
            best     = df.loc[best_idx]
            fig_vc.add_trace(go.Scatter(
                x=[best["timestep"]], y=[best["val_sharpe"]],
                mode="markers",
                marker=dict(color=clr, size=13, symbol="star",
                            line=dict(color="white", width=0.5)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{name} — best ckpt</b><br>"
                    f"Step: {int(best['timestep']):,}<br>"
                    f"Val Sharpe: {best['val_sharpe']:.3f}<extra></extra>"
                ),
            ))
        fig_vc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        pstyle(fig_vc, height=310)
        fig_vc.update_layout(
            xaxis_title="Training Step",
            yaxis_title="Validation Sharpe",
        )
        st.plotly_chart(fig_vc, width="stretch")

    with col_lc:
        sh("EPISODE REWARD — ALL SEEDS (5-step smoothed)")
        fig_lc = go.Figure()
        for (name, df), clr in zip(learn_curves.items(), SEED_COLORS):
            sm = df["ep_rew_mean"].rolling(5, min_periods=1).mean()
            fig_lc.add_trace(go.Scatter(
                x=df["timestep"], y=sm,
                name=name, line=dict(color=clr, width=1.8),
                hovertemplate=f"<b>{name}</b><br>Step: %{{x:,}}<br>Reward: %{{y:.3f}}<extra></extra>",
            ))
        fig_lc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        pstyle(fig_lc, height=310)
        fig_lc.update_layout(
            xaxis_title="Training Step",
            yaxis_title="Episode Reward (smoothed)",
        )
        st.plotly_chart(fig_lc, width="stretch")

    div()
    col_rew, col_rank = st.columns(2)

    with col_rew:
        sh("REWARD DECOMPOSITION — SEED 42  (PnL vs. Spread vs. CVaR)")
        df0 = list(learn_curves.values())[0]
        fig_rew = go.Figure()
        comps = {
            "r_pnl_mean":    ("PnL Return",   C["buy_and_hold"]),
            "r_spread_mean": ("Spread Cost",  C["strategy"]),
            "r_cvar_mean":   ("CVaR Penalty", C["accent2"]),
        }
        for col, (lbl, clr) in comps.items():
            sm = df0[col].rolling(5, min_periods=1).mean()
            fig_rew.add_trace(go.Scatter(
                x=df0["timestep"], y=sm,
                name=lbl, line=dict(color=clr, width=1.8),
                hovertemplate=f"<b>{lbl}</b><br>%{{y:.5f}}<extra></extra>",
            ))
        fig_rew.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        pstyle(fig_rew, height=300)
        fig_rew.update_layout(
            xaxis_title="Training Step",
            yaxis_title="Reward Component (smoothed)",
        )
        st.plotly_chart(fig_rew, width="stretch")

    with col_rank:
        sh("VAL → TEST RANK INVERSION  (parallel lines)")
        seed_order   = [42, 101, 202, 303, 404]
        test_sharpes = {42: 0.47, 101: 0.74, 202: -0.30, 303: 0.15, 404: -1.00}
        best_vals    = {int(n.split()[-1]): df["val_sharpe"].max()
                        for n, df in val_curves.items()}

        fig_rank = go.Figure()
        for s, clr in zip(seed_order, SEED_COLORS):
            vy, ty = best_vals.get(s, 0), test_sharpes.get(s, 0)
            fig_rank.add_trace(go.Scatter(
                x=["Val Sharpe", "Test Sharpe"], y=[vy, ty],
                mode="lines+markers+text",
                name=f"seed {s}",
                line=dict(color=clr, width=2),
                marker=dict(color=clr, size=9),
                text=[f"{vy:.2f}", f"{ty:.2f}"],
                textposition=["middle left", "middle right"],
                textfont=dict(color=clr, size=10),
                hovertemplate=f"<b>seed {s}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        fig_rank.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        pstyle(fig_rank, height=300)
        fig_rank.update_layout(
            xaxis=dict(
                tickvals=["Val Sharpe", "Test Sharpe"],
                tickfont=dict(size=13),
                range=[-0.4, 1.4],
            ),
            yaxis_title="Sharpe Ratio",
        )
        st.plotly_chart(fig_rank, width="stretch")
        st.caption(
            "Crossing lines = val rank ≠ test rank. Best-val seed (303, val 1.63) "
            "ranks **3rd** on test. Best-test seed (101) ranks **4th** on val. "
            "Justifies ensembling over checkpoint selection."
        )

    div()
    sh("ENSEMBLE AGGREGATION — MEAN vs. UNCERTAINTY POLICY")
    splits = ["val", "test"]
    modes  = ["mean", "uncertainty"]
    cats   = ["Val / Mean", "Val / Uncertainty", "Test / Mean", "Test / Uncertainty"]
    sharpes_e = [ens["results"][f"{s}/{m}"]["sharpe"] for s in splits for m in modes]
    dds_e     = [ens["results"][f"{s}/{m}"]["max_dd"] * 100 for s in splits for m in modes]
    ens_clrs  = [C["sixty_forty"], C["accent"], C["equal_weight"], C["accent2"]]

    fig_ens = make_subplots(rows=1, cols=2,
                            subplot_titles=["Sharpe Ratio", "Max Drawdown (%)"],
                            horizontal_spacing=0.12)
    for r, c, vals in [
        (1, 1, sharpes_e),
        (1, 2, dds_e),
    ]:
        fmt = [f"{v:.2f}" for v in vals] if c == 1 else [f"{v:.1f}%" for v in vals]
        fig_ens.add_trace(go.Bar(
            x=cats, y=vals,
            marker_color=ens_clrs,
            text=fmt, textposition="outside",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
        ), row=r, col=c)
    fig_ens.add_hline(y=0, row=1, col=1,
                      line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_ens.update_layout(
        **{k: v for k, v in _BASE_LAYOUT.items() if k != "hovermode"},
        height=300, showlegend=False, hovermode="closest",
    )
    fig_ens.update_xaxes(tickfont=dict(size=9), gridcolor=C["grid"])
    fig_ens.update_yaxes(gridcolor=C["grid"])
    fig_ens.update_annotations(font_size=11, font_color="#a78bfa")
    st.plotly_chart(fig_ens, width="stretch")
    st.caption(
        "Uncertainty sizing flips test Sharpe from **−0.34 → +0.24** and "
        "cuts max DD **6.5×** (58.3% → 9.0%) vs the naive ensemble mean."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · SIMULATE
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.markdown("""
    <div style="margin-bottom:16px">
        <div class="hero-title">Interactive Simulation</div>
        <div class="hero-subtitle">
            Select any sub-window of the 2022–2024 test set · Set initial capital ·
            All metrics recomputed live
        </div>
    </div>
    """, unsafe_allow_html=True)

    returns_all = load_returns()
    dmin = returns_all.index.min().to_pydatetime()
    dmax = returns_all.index.max().to_pydatetime()

    col_s, col_e, col_cap = st.columns([1, 1, 1])
    start_d  = col_s.date_input("Start date",       value=dmin, min_value=dmin, max_value=dmax)
    end_d    = col_e.date_input("End date",          value=dmax, min_value=dmin, max_value=dmax)
    init_cap = col_cap.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=10_000,
    )

    col_fric, col_anim = st.columns([2, 1])
    cost_mult = col_fric.slider(
        "Transaction cost stress (× baseline spread)",
        min_value=0.5, max_value=5.0, value=1.0, step=0.25,
        help=(
            "Multiplies the baseline spread/slippage drag applied to the "
            "strategy's daily returns. 1.0× = model default (5 bps/day). "
            "5.0× = extreme-friction stress test."
        ),
    )
    animate = col_anim.toggle(
        "Animated replay", value=False,
        help="Play the equity curve day-by-day with a cinematic animation.",
    )

    if start_d >= end_d:
        st.error("Start date must be before end date.")
        st.stop()

    window = returns_all.loc[str(start_d):str(end_d)].copy()
    # Apply cost stress: extra drag only on the strategy column.
    # Baseline is already baked in; slider adds (mult - 1) * 5bps/day.
    BASE_COST_DAILY = 0.0005  # 5 bps/day
    extra_drag = (cost_mult - 1.0) * BASE_COST_DAILY
    if abs(extra_drag) > 1e-12:
        window["strategy"] = window["strategy"] - extra_drag
    n = len(window)
    if n < 20:
        st.warning("Window too short (< 20 trading days).")
        st.stop()

    ann = 252

    def win_metrics(col: str) -> dict:
        r     = window[col]
        cum   = (1 + r).prod() - 1
        ann_r = (1 + cum) ** (ann / n) - 1
        ann_v = r.std() * np.sqrt(ann)
        sh_r  = ann_r / ann_v if ann_v > 1e-10 else 0.0
        neg   = r[r < 0]
        sort  = ann_r / (neg.std() * np.sqrt(ann)) if len(neg) > 1 else 0.0
        eq    = (1 + r).cumprod()
        max_d = (eq / eq.cummax() - 1).min()
        cal   = ann_r / abs(max_d) if max_d < -1e-10 else 0.0
        fval  = init_cap * (1 + cum)
        return {
            "Ann. Return": ann_r, "Sharpe": sh_r, "Sortino": sort,
            "Max DD": max_d, "Calmar": cal,
            "Final Value": fval, "P&L": fval - init_cap,
        }

    all_m  = {col: win_metrics(col) for col in window.columns}
    strat_m = all_m["strategy"]

    div()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe",      f"{strat_m['Sharpe']:+.2f}")
    c2.metric("Ann. Return", f"{strat_m['Ann. Return']:+.1%}")
    c3.metric("Max DD",      f"−{abs(strat_m['Max DD']):.1%}")
    c4.metric("Final Value", f"${strat_m['Final Value']:,.0f}",
              delta=f"${strat_m['P&L']:+,.0f}")
    c5.metric("Trading Days", f"{n}")

    div()
    sh("EQUITY CURVES — SELECTED WINDOW")
    eq_w = (1 + window).cumprod() * init_cap
    fig_sim = go.Figure()
    for col in eq_w.columns:
        fig_sim.add_trace(go.Scatter(
            x=eq_w.index, y=eq_w[col],
            name=LABELS[col],
            line=dict(
                color=C[col],
                width=2.5 if col == "strategy" else 1.4,
                dash="solid" if col == "strategy" else "dot",
            ),
            hovertemplate=(
                f"<b>{LABELS[col]}</b><br>"
                "%{x|%b %d %Y}<br>"
                "$%{y:,.0f}<extra></extra>"
            ),
        ))
    pstyle(fig_sim, height=380)
    fig_sim.update_layout(
        yaxis_title="Portfolio Value ($)",
        yaxis=dict(tickprefix="$"),
    )
    st.plotly_chart(fig_sim, width="stretch")

    if abs(cost_mult - 1.0) > 1e-6:
        sign = "↑" if cost_mult > 1 else "↓"
        st.caption(
            f"Friction stress active: **{cost_mult:.2f}× baseline spread** "
            f"({sign} {abs(cost_mult-1)*5:.1f} bps/day extra drag on strategy). "
            f"Sharpe at 1.0×: reload to compare."
        )

    if animate:
        div()
        sh("ANIMATED REPLAY — STRATEGY EQUITY GROWTH")
        eq_strat = eq_w["strategy"]
        # Stride so we produce ≤ 80 frames for snappy playback.
        stride = max(1, len(eq_strat) // 80)
        idx = list(range(0, len(eq_strat), stride))
        if idx[-1] != len(eq_strat) - 1:
            idx.append(len(eq_strat) - 1)

        frames = [
            go.Frame(
                data=[go.Scatter(
                    x=eq_strat.index[: i + 1],
                    y=eq_strat.iloc[: i + 1].values,
                    mode="lines",
                    line=dict(color=C["strategy"], width=3),
                    fill="tozeroy",
                    fillcolor="rgba(124,158,255,0.12)",
                )],
                name=str(i),
            )
            for i in idx
        ]

        first_i = idx[0]
        fig_anim = go.Figure(
            data=[go.Scatter(
                x=eq_strat.index[: first_i + 1],
                y=eq_strat.iloc[: first_i + 1].values,
                mode="lines",
                line=dict(color=C["strategy"], width=3),
                fill="tozeroy",
                fillcolor="rgba(124,158,255,0.12)",
                name="Ensemble PPO (ours)",
            )],
            frames=frames,
        )
        y_min = float(eq_strat.min()) * 0.98
        y_max = float(eq_strat.max()) * 1.02
        fig_anim.update_layout(
            xaxis=dict(range=[eq_strat.index[0], eq_strat.index[-1]]),
            yaxis=dict(range=[y_min, y_max], tickprefix="$", title="Portfolio Value ($)"),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.02, y=1.15, xanchor="left", yanchor="top",
                showactive=False,
                bgcolor="rgba(124,158,255,0.15)",
                bordercolor="rgba(124,158,255,0.5)",
                font=dict(color="#f4f5f8"),
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, {"frame": {"duration": 40, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 0}}]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}]),
                ],
            )],
        )
        pstyle(fig_anim, height=420)
        st.plotly_chart(fig_anim, width="stretch")
        st.caption("Click ▶ Play to watch the strategy equity grow day by day.")

    sh("FULL METRICS TABLE — ALL STRATEGIES")
    rows_sim = []
    for col in window.columns:
        m = all_m[col]
        rows_sim.append({
            "Strategy":    LABELS[col],
            "Ann. Return": m["Ann. Return"],
            "Sharpe":      m["Sharpe"],
            "Sortino":     m["Sortino"],
            "Max DD":      m["Max DD"],
            "Calmar":      m["Calmar"],
            "Final Value": m["Final Value"],
            "P&L":         m["P&L"],
        })
    sim_df = pd.DataFrame(rows_sim).set_index("Strategy")
    st.dataframe(
        sim_df.style.format({
            "Ann. Return": "{:+.1%}", "Sharpe":  "{:.2f}",
            "Sortino":     "{:.2f}",  "Max DD":  "{:.1%}",
            "Calmar":      "{:.2f}",  "Final Value": "${:,.0f}",
            "P&L":         "${:+,.0f}",
        })
        .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-1, vmax=1.5)
        .background_gradient(subset=["Max DD"], cmap="RdYlGn_r", vmin=-0.6, vmax=0),
        width="stretch",
    )

    # ── Groq AI Trade Explanation ─────────────────────────────────────────────
    div()
    sh("🤖  AI TRADE ANALYST — POWERED BY GROQ (LLAMA-3.3-70B)")
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.82rem;margin-top:-8px;margin-bottom:12px">'
        "Ask the LLM to narrate the strategy's behaviour in the selected window. "
        "Cached per window/cost setting — no repeat charges."
        "</p>",
        unsafe_allow_html=True,
    )

    groq_col1, groq_col2 = st.columns([3, 1])
    with groq_col2:
        run_groq = st.button(
            "✨ Explain This Window",
            use_container_width=True,
            help="Calls Groq Llama-3.3-70B to narrate strategy performance. ~2s.",
        )

    if run_groq or st.session_state.get("groq_result_key") == f"{start_d}_{end_d}_{cost_mult}":
        window_label = f"{start_d} → {end_d} ({n} trading days)"
        bh_m = all_m.get("buy_and_hold", {})
        cache_key = f"{start_d}_{end_d}_{cost_mult}"

        if st.session_state.get("groq_result_key") != cache_key:
            with st.spinner("Calling Groq API…"):
                explanation = _groq_explain(strat_m, bh_m, window_label, cost_mult)
            st.session_state["groq_result_key"] = cache_key
            st.session_state["groq_result_text"] = explanation
        else:
            explanation = st.session_state.get("groq_result_text", "")

        st.markdown(
            f'<div class="info-card" style="border-color:rgba(167,139,250,0.45);'
            f'background:linear-gradient(135deg,#1a1a2e,#1e1b4b);'
            f'padding:18px 22px;border-radius:12px;line-height:1.7;">'
            f'{explanation}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIQUIDITY  (Phase C10)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_liq:
    st.markdown("""
    <div style="margin-bottom:22px">
        <div class="hero-title">Liquidity Regime Analysis</div>
        <div class="hero-subtitle">
            Volume z-score · OHLC-dispersion spread proxy · Stochastic fill noise ·
            r007 (baseline) vs. r008 (liquidity-aware) — impact-adjusted comparison
        </div>
    </div>
    """, unsafe_allow_html=True)

    r8_ret  = load_r008_returns()
    r8_met  = load_r008_metrics()
    r8_exec = load_r008_exec()
    r8_ens  = load_r008_ensemble()
    r7_exec_path = R007 / "execution_stats.csv"
    r7_exec = pd.read_csv(r7_exec_path) if r7_exec_path.exists() else None
    r7_met  = load_metrics()
    r7_ret  = load_returns()

    if r8_ret is None:
        st.warning("r008 backtest data not found. Run `python scripts/backtest.py` with the r008 run dir.")
    else:
        st.markdown("""
        <div class="claim-box">
            🔬 <b>Phase C contribution (r008):</b>
            Liquidity-aware training (volume z-score + OHLC-dispersion features + stochastic fill noise)
            reduces implementation shortfall by <b>−14.3%</b>, mean impact by <b>−15.2%</b>, and turnover by
            <b>−11.1%</b> vs. the r007 baseline — empirical evidence of lighter trading in illiquid regimes.
        </div>
        """, unsafe_allow_html=True)

        div()

        # ── KPI comparison strip ──────────────────────────────────────────────
        sh("R007 vs. R008 — IMPACT-ADJUSTED EXECUTION METRICS")
        if r7_exec is not None and r8_exec is not None:
            r7e, r8e = r7_exec.iloc[0], r8_exec.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            shortfall_delta = (r8e["impl_shortfall_bps"] - r7e["impl_shortfall_bps"]) / r7e["impl_shortfall_bps"] * 100
            impact_delta    = (r8e["total_impact_bps"] - r7e["total_impact_bps"]) / r7e["total_impact_bps"] * 100
            turnover_delta  = (r8e["mean_turnover"] - r7e["mean_turnover"]) / r7e["mean_turnover"] * 100
            c1.metric("Impl. Shortfall (r007)", f"{r7e['impl_shortfall_bps']:.0f} bps",
                      delta=f"{shortfall_delta:+.1f}% r008 change",
                      delta_color="inverse",
                      help="Implementation shortfall = total_impact + total_spread. Lower is better.")
            c2.metric("Total Impact (r007)", f"{r7e['total_impact_bps']:.0f} bps",
                      delta=f"{impact_delta:+.1f}% r008 change",
                      delta_color="inverse",
                      help="Cumulative Kyle's-λ impact cost. Lower is better.")
            c3.metric("Mean Turnover (r007)", f"{r7e['mean_turnover']:.4f}",
                      delta=f"{turnover_delta:+.1f}% r008 change",
                      delta_color="inverse")
            # Val Sharpe comparison (uncertainty mode)
            r7_val_sh = load_ensemble_json()["results"]["val/uncertainty"]["sharpe"]
            r8_val_sh = r8_ens["results"]["val/uncertainty"]["sharpe"] if r8_ens else None
            if r8_val_sh is not None:
                c4.metric("Val Sharpe uncertainty (r007)", f"{r7_val_sh:.3f}",
                          delta=f"{r8_val_sh:.3f} (r008)",
                          delta_color="normal")

        div()

        # ── Drawdown overlay ──────────────────────────────────────────────────
        sh("DRAWDOWN COMPARISON — r007 vs. r008 (TEST SET 2022–2024)")
        dd7 = get_drawdown()["strategy"]
        eq8 = (1 + r8_ret["strategy"]).cumprod()
        dd8 = eq8 / eq8.cummax() - 1

        fig_liq_dd = go.Figure()
        fig_liq_dd.add_trace(go.Scatter(
            x=dd7.index, y=dd7 * 100,
            name="r007 — baseline",
            fill="tozeroy", fillcolor="rgba(255,179,71,0.10)",
            line=dict(color="#FFB347", width=2.0),
            hovertemplate="<b>r007</b><br>%{x|%b %d %Y}<br>DD: %{y:.2f}%<extra></extra>",
        ))
        fig_liq_dd.add_trace(go.Scatter(
            x=dd8.index, y=dd8 * 100,
            name="r008 — liquidity-aware",
            fill="tozeroy", fillcolor="rgba(124,158,255,0.12)",
            line=dict(color="#7C9EFF", width=2.4),
            hovertemplate="<b>r008</b><br>%{x|%b %d %Y}<br>DD: %{y:.2f}%<extra></extra>",
        ))
        for lvl, lbl in [(-5, "−5%"), (-10, "−10%"), (-20, "−20%")]:
            fig_liq_dd.add_hline(
                y=lvl, line_dash="dot",
                line_color="rgba(255,255,255,0.12)",
                annotation_text=lbl, annotation_font_size=9,
                annotation_font_color="rgba(255,255,255,0.35)",
            )
        pstyle(fig_liq_dd, height=340)
        fig_liq_dd.update_layout(
            yaxis_title="Drawdown (%)",
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_liq_dd, width="stretch")

        div()
        col_exec, col_ens = st.columns(2)

        with col_exec:
            # ── Execution cost bar chart ──────────────────────────────────────
            sh("EXECUTION COST — r007 vs. r008")
            if r7_exec is not None and r8_exec is not None:
                exec_cats  = ["Impl. Shortfall (bps)", "Total Impact (bps)", "Total Spread (bps)"]
                exec_r7    = [r7e["impl_shortfall_bps"], r7e["total_impact_bps"], r7e["total_spread_bps"]]
                exec_r8    = [r8e["impl_shortfall_bps"], r8e["total_impact_bps"], r8e["total_spread_bps"]]
                fig_exec = go.Figure()
                fig_exec.add_trace(go.Bar(
                    name="r007 (baseline)", x=exec_cats, y=exec_r7,
                    marker_color="#FFB347",
                    text=[f"{v:.0f}" for v in exec_r7], textposition="outside",
                    hovertemplate="<b>r007</b> %{x}<br>%{y:.1f} bps<extra></extra>",
                ))
                fig_exec.add_trace(go.Bar(
                    name="r008 (liquidity-aware)", x=exec_cats, y=exec_r8,
                    marker_color="#7C9EFF",
                    text=[f"{v:.0f}" for v in exec_r8], textposition="outside",
                    hovertemplate="<b>r008</b> %{x}<br>%{y:.1f} bps<extra></extra>",
                ))
                pstyle(fig_exec, height=320)
                fig_exec.update_layout(
                    barmode="group",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_exec.update_yaxes(title_text="bps")
                st.plotly_chart(fig_exec, width="stretch")
                st.caption(
                    f"r008 vs r007: shortfall **{shortfall_delta:+.1f}%** · "
                    f"impact **{impact_delta:+.1f}%** · "
                    f"turnover **{turnover_delta:+.1f}%**. "
                    "Lighter trading in illiquid regimes = Phase C's empirical signature."
                )

        with col_ens:
            # ── Ensemble Sharpe comparison (val/test, mean/uncertainty) ────────
            sh("ENSEMBLE METRICS — r007 vs. r008")
            if r8_ens:
                ens7 = load_ensemble_json()["results"]
                ens8 = r8_ens["results"]
                rows_ens = []
                for split in ["val", "test"]:
                    for mode in ["mean", "uncertainty"]:
                        key = f"{split}/{mode}"
                        rows_ens.append({
                            "Split/Mode": key,
                            "r007 Sharpe": ens7[key]["sharpe"],
                            "r008 Sharpe": ens8[key]["sharpe"],
                            "r007 MaxDD":  ens7[key]["max_dd"],
                            "r008 MaxDD":  ens8[key]["max_dd"],
                        })
                ens_df = pd.DataFrame(rows_ens).set_index("Split/Mode")
                st.dataframe(
                    ens_df.style.format({
                        "r007 Sharpe": "{:+.3f}", "r008 Sharpe": "{:+.3f}",
                        "r007 MaxDD": "{:.1%}", "r008 MaxDD": "{:.1%}",
                    })
                    .background_gradient(subset=["r007 Sharpe", "r008 Sharpe"],
                                         cmap="RdYlGn", vmin=-0.5, vmax=2.0)
                    .background_gradient(subset=["r007 MaxDD", "r008 MaxDD"],
                                         cmap="RdYlGn_r", vmin=0.0, vmax=0.65),
                    width="stretch", height=230,
                )
                st.caption(
                    "r008 test Sharpe regression (−0.274 vs r007 +0.242) reflects the 100k-step "
                    "training budget — the extra liquidity features add state complexity without "
                    "more training steps. Execution cost reduction is the validated contribution."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — EXECUTION  (Phase D16 — full r007/r008 cost decomposition + baselines)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_exec:
    st.markdown("""
    <div style="margin-bottom:22px">
        <div class="hero-title">Execution Cost Analysis</div>
        <div class="hero-subtitle">
            Kyle's-λ market impact · Implementation shortfall · Spread-cost decomposition ·
            r007 ↔ r008 vs TWAP / VWAP-proxy / MA-crossover baselines
        </div>
    </div>
    """, unsafe_allow_html=True)

    r7_exec_path = R007 / "execution_stats.csv"
    r8_exec_path = R008 / "execution_stats.csv"

    if not r7_exec_path.exists():
        st.info(
            "🚧 **Run `python scripts/backtest.py --run-dir runs/ensemble_r007` to populate this tab.** "
            "The backtest produces `execution_stats.csv` with Kyle's-λ impact metrics "
            "(implementation shortfall, mean impact bps/day, turnover).",
            icon="⚙️",
        )
    else:
        # ── Methodology callout ───────────────────────────────────────────────
        st.markdown("""
        <div class="claim-box">
            🔬 <b>How execution cost is measured.</b>
            For every traded position change Δw, we charge a Kyle's-λ market-impact cost
            <code>λ·(|Δw|/ADV)<sup>α</sup></code> with λ=1×10⁻³, α=1.5, scaled by a per-day
            liquidity factor <code>v_ref / v_t</code>. A constant 5 bps round-trip spread
            cost is added on top. Implementation shortfall = total impact + total spread,
            measured in basis points relative to the frictionless mark-to-market path.
        </div>
        """, unsafe_allow_html=True)

        div()

        # ── KPI strip — r007 baseline vs r008 liquidity-aware ─────────────────
        sh("IMPLEMENTATION SHORTFALL — R007 BASELINE vs. R008 LIQUIDITY-AWARE")
        r7e = pd.read_csv(r7_exec_path).iloc[0]
        r8e = (pd.read_csv(r8_exec_path).iloc[0]
               if r8_exec_path.exists() else None)

        c1, c2, c3, c4 = st.columns(4)
        if r8e is not None:
            sf_d = (r8e["impl_shortfall_bps"] - r7e["impl_shortfall_bps"]) / r7e["impl_shortfall_bps"] * 100
            ip_d = (r8e["total_impact_bps"]   - r7e["total_impact_bps"])   / r7e["total_impact_bps"]   * 100
            sp_d = (r8e["total_spread_bps"]   - r7e["total_spread_bps"])   / r7e["total_spread_bps"]   * 100
            to_d = (r8e["mean_turnover"]      - r7e["mean_turnover"])      / r7e["mean_turnover"]      * 100
            c1.metric("Impl. Shortfall", f"{r7e['impl_shortfall_bps']:.0f} bps",
                      delta=f"{sf_d:+.1f}% on r008", delta_color="inverse",
                      help="Total cost paid relative to frictionless execution. Lower is better.")
            c2.metric("Total Impact",   f"{r7e['total_impact_bps']:.0f} bps",
                      delta=f"{ip_d:+.1f}% on r008", delta_color="inverse",
                      help="Cumulative Kyle's-λ market-impact cost.")
            c3.metric("Total Spread",   f"{r7e['total_spread_bps']:.0f} bps",
                      delta=f"{sp_d:+.1f}% on r008", delta_color="inverse",
                      help="Cumulative half-spread cost (5 bps per round-trip).")
            c4.metric("Mean Turnover",  f"{r7e['mean_turnover']:.3f}",
                      delta=f"{to_d:+.1f}% on r008", delta_color="inverse",
                      help="Average daily |Δw|. Lower turnover = lighter trading.")
        else:
            c1.metric("Impl. Shortfall", f"{r7e['impl_shortfall_bps']:.0f} bps")
            c2.metric("Total Impact",    f"{r7e['total_impact_bps']:.0f} bps")
            c3.metric("Total Spread",    f"{r7e['total_spread_bps']:.0f} bps")
            c4.metric("Mean Turnover",   f"{r7e['mean_turnover']:.3f}")

        div()

        # ── Stacked decomposition: shortfall = impact + spread ────────────────
        col_decomp, col_strats = st.columns([1, 1])

        with col_decomp:
            sh("SHORTFALL DECOMPOSITION (bps)")
            if r8e is not None:
                fig_decomp = go.Figure()
                fig_decomp.add_trace(go.Bar(
                    name="Market Impact",
                    x=["r007 (baseline)", "r008 (liquidity-aware)"],
                    y=[r7e["total_impact_bps"], r8e["total_impact_bps"]],
                    marker_color="#FF6B6B",
                    text=[f"{r7e['total_impact_bps']:.0f}", f"{r8e['total_impact_bps']:.0f}"],
                    textposition="inside",
                    hovertemplate="<b>Impact</b><br>%{x}<br>%{y:.1f} bps<extra></extra>",
                ))
                fig_decomp.add_trace(go.Bar(
                    name="Spread Cost",
                    x=["r007 (baseline)", "r008 (liquidity-aware)"],
                    y=[r7e["total_spread_bps"], r8e["total_spread_bps"]],
                    marker_color="#FFB347",
                    text=[f"{r7e['total_spread_bps']:.0f}", f"{r8e['total_spread_bps']:.0f}"],
                    textposition="inside",
                    hovertemplate="<b>Spread</b><br>%{x}<br>%{y:.1f} bps<extra></extra>",
                ))
                pstyle(fig_decomp, height=340)
                fig_decomp.update_layout(
                    barmode="stack",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_decomp.update_yaxes(title_text="bps")
                st.plotly_chart(fig_decomp, width="stretch")
                st.caption(
                    f"Liquidity-aware training cuts **total impact by {abs(ip_d):.1f}%** "
                    f"and **spread by {abs(sp_d):.1f}%** — the agent learned to route trades "
                    "through deeper-liquidity windows."
                )

        with col_strats:
            sh("BENCHMARK COMPARISON — STRATEGY vs. RULE-BASED EXECUTION")
            r7_met = pd.read_csv(R007 / "backtest_metrics.csv").set_index("name")
            display_rows = ["strategy", "ma_crossover", "twap_ew", "vwap_proxy_ew", "equal_weight"]
            display_rows = [r for r in display_rows if r in r7_met.index]
            cmp = r7_met.loc[display_rows, ["ann_return", "sharpe", "max_drawdown", "calmar"]].copy()
            cmp.index = cmp.index.map({
                "strategy":      "RL Ensemble (r007)",
                "ma_crossover":  "MA Crossover 50/200",
                "twap_ew":       "TWAP equal-weight",
                "vwap_proxy_ew": "VWAP-proxy equal-weight",
                "equal_weight":  "Equal-weight buy-hold",
            })
            cmp.columns = ["Ann. Return", "Sharpe", "Max DD", "Calmar"]
            st.dataframe(
                cmp.style.format({
                    "Ann. Return": "{:+.2%}", "Sharpe": "{:+.3f}",
                    "Max DD": "{:.1%}",      "Calmar": "{:+.3f}",
                })
                .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-0.5, vmax=1.5)
                .background_gradient(subset=["Max DD"], cmap="RdYlGn_r", vmin=0.05, vmax=0.30)
                .background_gradient(subset=["Calmar"], cmap="RdYlGn", vmin=-0.3, vmax=1.6),
                width="stretch", height=240,
            )
            st.caption(
                "RL ensemble's edge is **drawdown control** (8.6% vs ≥21% for buy-hold benchmarks), "
                "not raw return. MA-crossover dominates Sharpe in this 2022–2024 regime — "
                "a known limitation discussed in the paper."
            )
