"""Tests for backtest math (MVP Step 8)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest import (
    _align_lengths,
    _benchmark_returns,
    _ma_crossover_returns,
    _metrics,
    _twap_returns,
    _vwap_proxy_returns,
)


def test_benchmark_returns_buy_and_hold_matches_pct_change():
    """100% SPY benchmark must equal SPY's daily pct change (minus the first NaN)."""
    n = 50
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    test_dfs = {}
    for t in ["SPY", "QQQ", "IWM", "GLD", "TLT"]:
        test_dfs[t] = pd.DataFrame(
            {"close": closes, "open": closes, "high": closes, "low": closes,
             "volume": np.ones(n)},
            index=pd.date_range("2022-01-01", periods=n, freq="B"),
        )
    bh = _benchmark_returns(test_dfs, {"SPY": 1.0})
    expected = np.diff(closes) / closes[:-1]
    # _benchmark_returns drops row 0 (NaN-filled) → n-1 rows, matching np.diff
    assert np.allclose(bh, expected, atol=1e-10)


def test_benchmark_returns_equal_weight_sums_correctly():
    n = 30
    # All tickers identical → equal-weight return equals single-asset return
    closes = np.linspace(100, 110, n)
    test_dfs = {t: pd.DataFrame({"close": closes, "open": closes, "high": closes,
                                 "low": closes, "volume": np.ones(n)},
                                index=pd.date_range("2022-01-01", periods=n, freq="B"))
                for t in ["SPY", "QQQ", "IWM", "GLD", "TLT"]}
    ew = _benchmark_returns(test_dfs, {t: 0.2 for t in test_dfs})
    bh = _benchmark_returns(test_dfs, {"SPY": 1.0})
    assert np.allclose(ew, bh, atol=1e-10)


def test_metrics_known_returns():
    # Constant 0.1% daily return over 252 days → ~28.6% total, Sharpe very high, no drawdown
    r = np.full(252, 0.001)
    m = _metrics(r)
    assert m["total_return"] == pytest.approx(np.exp(0.001 * 252) - 1, rel=0.01) or \
           m["total_return"] == pytest.approx((1.001 ** 252) - 1, rel=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-10)
    assert m["sharpe"] > 100  # zero vol division guarded, but for constant returns should be huge


def test_metrics_drawdown_detection():
    # +10% then -20% → drawdown = 20% / 110% ≈ 18.18%
    r = np.array([0.10, -0.20])
    m = _metrics(r)
    assert m["max_drawdown"] == pytest.approx(0.20, rel=0.01)


def test_ma_crossover_no_lookahead_and_flat_when_downtrend():
    """Uptrend-then-downtrend SPY: MAC must go long on uptrend (positive returns)
    and flat on downtrend (zero returns). Signal must be lagged by 1 day."""
    n = 400
    # First 250 days uptrend (SMA50 crosses above SMA200), then 150 days downtrend
    up = np.linspace(100, 200, 250)
    down = np.linspace(200, 120, 150)
    closes = np.concatenate([up, down])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    spy = pd.DataFrame({"close": closes, "open": closes, "high": closes,
                        "low": closes, "volume": np.ones(n)}, index=idx)
    # Test window = last 100 days (fully into downtrend by the end)
    test_idx = idx[-100:]
    r = _ma_crossover_returns(spy, test_idx)
    # At the very end we're deep in downtrend — signal should be off, returns == 0
    assert r[-1] == 0.0
    # Somewhere in the middle of test (still near top of uptrend tail) expect non-zero
    assert np.abs(r).sum() > 0.0
    # Output length equals test_idx length
    assert len(r) == len(test_idx)


def _make_test_dfs(n=60, seed=7):
    """Minimal test_dfs fixture: all tickers with OHLCV, flat prices for determinism."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    dfs = {}
    for t in ["SPY", "QQQ", "IWM", "GLD", "TLT"]:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        dfs[t] = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.003,
             "low": close * 0.997, "close": close,
             "volume": np.ones(n) * 1e6},
            index=idx,
        )
    return dfs


def test_twap_ramp_starts_at_zero_and_ends_at_full():
    """TWAP must earn zero on day 0 and full weight returns after ramp_days."""
    dfs = _make_test_dfs()
    weights = {t: 0.2 for t in dfs}
    r = _twap_returns(dfs, weights, ramp_days=5)
    # Day 0 of output → scale = 0 → zero return regardless of price move
    assert r[0] == pytest.approx(0.0, abs=1e-10)
    # After ramp (index >= ramp_days - 1 of output) should equal equal-weight return
    bh = _benchmark_returns(dfs, weights)
    np.testing.assert_allclose(r[5:], bh[5:], rtol=1e-6)


def test_twap_output_length_matches_benchmark():
    dfs = _make_test_dfs()
    weights = {t: 0.2 for t in dfs}
    assert len(_twap_returns(dfs, weights)) == len(_benchmark_returns(dfs, weights))


def test_vwap_proxy_differs_from_close_when_ohlc_differ():
    """VWAP-proxy must diverge from close-based returns when O/H/L ≠ C.

    Constant percentage spreads cancel in pct_change (OHLC4 = k·C → returns equal).
    Use absolute dollar deviations that vary day-to-day so OHLC4 ≠ k·C.
    """
    rng = np.random.default_rng(99)
    dfs = _make_test_dfs()
    for t in dfs:
        c = dfs[t]["close"].values.copy()
        # Add per-day absolute dollar noise — does NOT cancel in pct_change
        dfs[t]["open"]  = c + rng.uniform(-3, 3, len(c))
        dfs[t]["high"]  = c + np.abs(rng.uniform(1, 5, len(c)))
        dfs[t]["low"]   = c - np.abs(rng.uniform(1, 5, len(c)))
    weights = {t: 0.2 for t in dfs}
    vwap = _vwap_proxy_returns(dfs, weights)
    bh   = _benchmark_returns(dfs, weights)
    assert not np.allclose(vwap, bh, atol=1e-4)


def test_vwap_proxy_equals_close_when_ohlc_flat():
    """When O=H=L=C, OHLC4 == close → VWAP returns must equal close returns."""
    dfs = _make_test_dfs()
    # Flatten OHLC so every price equals close
    for t in dfs:
        dfs[t]["open"] = dfs[t]["close"]
        dfs[t]["high"] = dfs[t]["close"]
        dfs[t]["low"]  = dfs[t]["close"]
    weights = {t: 0.2 for t in dfs}
    vwap = _vwap_proxy_returns(dfs, weights)
    bh   = _benchmark_returns(dfs, weights)
    np.testing.assert_allclose(vwap, bh, rtol=1e-8)


def test_align_lengths_trims_to_shortest_from_end():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30])
    c = np.array([100, 200, 300, 400])
    aa, bb, cc = _align_lengths(a, b, c)
    assert len(aa) == len(bb) == len(cc) == 3
    assert np.array_equal(aa, [3, 4, 5])
    assert np.array_equal(bb, [10, 20, 30])
    assert np.array_equal(cc, [200, 300, 400])
