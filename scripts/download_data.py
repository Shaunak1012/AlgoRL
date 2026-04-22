"""Download and cache OHLCV data via yfinance (single and multi-asset).

Usage:
    python scripts/download_data.py --config configs/base.yaml
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_ticker(ticker: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, f"{ticker}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"[download] loaded {ticker} from cache ({len(df)} rows)")
        return df

    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    print(f"[download] fetched {ticker}: {len(df)} rows → {path}")
    return df


def download_multi(tickers: list[str], start: str, end: str, cache_dir: str) -> dict[str, pd.DataFrame]:
    return {t: download_ticker(t, start, end, cache_dir) for t in tickers}


def split_df(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[cfg["train_start"] : cfg["train_end"]]
    val   = df.loc[cfg["val_start"]   : cfg["val_end"]]
    test  = df.loc[cfg["test_start"]  : cfg["test_end"]]
    return train, val, test


def split_multi(
    dfs: dict[str, pd.DataFrame], cfg: dict
) -> tuple[dict, dict, dict]:
    """Split each ticker DataFrame into train/val/test."""
    train_dfs, val_dfs, test_dfs = {}, {}, {}
    for ticker, df in dfs.items():
        tr, va, te = split_df(df, cfg)
        train_dfs[ticker] = tr
        val_dfs[ticker]   = va
        test_dfs[ticker]  = te
    return train_dfs, val_dfs, test_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Single-asset
    ticker = cfg["env"]["ticker"]
    df = download_ticker(ticker, data_cfg["train_start"], data_cfg["test_end"], data_cfg["cache_dir"])
    train, val, test = split_df(df, data_cfg)
    print(f"  {ticker} — train:{len(train)}  val:{len(val)}  test:{len(test)}")

    # Multi-asset
    tickers = cfg["multi_asset"]["tickers"]
    dfs = download_multi(tickers, data_cfg["train_start"], data_cfg["test_end"], data_cfg["cache_dir"])
    train_dfs, val_dfs, test_dfs = split_multi(dfs, data_cfg)
    for t in tickers:
        print(f"  {t} — train:{len(train_dfs[t])}  val:{len(val_dfs[t])}  test:{len(test_dfs[t])}")
