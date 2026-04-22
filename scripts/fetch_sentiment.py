"""
scripts/fetch_sentiment.py  (Phase B11 + B12)
---------------------------------------------
B11  Fetch daily market-level headlines from Alpha Vantage NEWS_SENTIMENT + NewsAPI.
     Output: data/headlines_raw.parquet
             columns: date (date), source (str), title (str), description (str)

B12  Run ProsusAI/finbert on headline titles and aggregate per trading day.
     Output: data/sentiment_daily.parquet
             columns: date (date), bullish_frac, neutral_frac, headline_count_norm
             Gap-filled with (0.5, 0.5, 0.0) for trading days without coverage.

Usage:
    python scripts/fetch_sentiment.py --fetch                   # B11 only
    python scripts/fetch_sentiment.py --infer                   # B12 only (needs headlines_raw)
    python scripts/fetch_sentiment.py --all                     # B11 + B12
    python scripts/fetch_sentiment.py --all --dry-run           # smoke-test, no API calls
    python scripts/fetch_sentiment.py --fetch --from-date 2023-01-01 --to-date 2023-12-31

Rate limits:
    Alpha Vantage: 25 req/day (free tier) → 84 monthly batches for 2018-2024 across ~4 days.
                   Script caches fetched months so repeat runs skip already-covered periods.
    NewsAPI:       25 req/day, last-month only → recency top-up only.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADLINES_PATH  = DATA_DIR / "headlines_raw.parquet"
SENTIMENT_PATH  = DATA_DIR / "sentiment_daily.parquet"

# Alpha Vantage topic filter (returns broad financial_markets coverage)
AV_TOPICS = "financial_markets"

# Headline count normalisation cap (per trading day)
HEADLINE_COUNT_CAP = 50

# Training date range
TRAIN_START = date(2018, 1, 1)
TRAIN_END   = date(2024, 12, 31)

# Neutral fill values for days with no coverage
NEUTRAL = {"bullish_frac": 0.5, "neutral_frac": 0.5, "headline_count_norm": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# B11 — Headline fetching
# ─────────────────────────────────────────────────────────────────────────────

def _av_month(year: int, month: int, api_key: str) -> list[dict]:
    """Fetch up to 50 headlines from Alpha Vantage for one calendar month.

    AV free tier: 25 req/day, 50 articles/request max.
    AV has verified historical coverage back to at least 2018-01.
    """
    from_dt = date(year, month, 1)
    if month == 12:
        to_dt = date(year, 12, 31)
    else:
        to_dt = date(year, month + 1, 1) - timedelta(days=1)

    params = {
        "function":  "NEWS_SENTIMENT",
        "topics":    AV_TOPICS,
        "time_from": from_dt.strftime("%Y%m%dT0000"),
        "time_to":   to_dt.strftime("%Y%m%dT2359"),
        "limit":     50,
        "apikey":    api_key,
    }
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=20,
        )
        if resp.status_code == 429:
            print(f"    AV rate-limited ({year}-{month:02d}), sleeping 65s…")
            time.sleep(65)
            resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
        data = resp.json()
        if "Information" in data:
            # Premium-only or daily limit hit
            print(f"    AV quota/premium notice: {data['Information'][:120]}")
            return []
        articles = data.get("feed", [])
        rows = []
        for a in articles:
            pub = a.get("time_published", "")[:8]  # YYYYMMDD
            if len(pub) == 8:
                pub = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
            rows.append({
                "date":        pub,
                "source":      "alphavantage",
                "title":       a.get("title", "") or "",
                "description": a.get("summary", "") or "",
            })
        return rows
    except Exception as exc:
        print(f"    AV error {year}-{month:02d}: {exc}")
        return []


def _newsapi_recent(api_key: str, max_results: int = 100) -> list[dict]:
    """Fetch recent headlines from NewsAPI (free tier: last 30 days only)."""
    params = {
        "q":        "stock market OR S&P500 OR Federal Reserve",
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": min(max_results, 100),
        "apiKey":   api_key,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if resp.status_code != 200:
            print(f"    NewsAPI {resp.status_code}: {resp.text[:120]}")
            return []
        articles = resp.json().get("articles", [])
        return [
            {
                "date":        (a.get("publishedAt") or "")[:10],
                "source":      "newsapi",
                "title":       a.get("title", "") or "",
                "description": a.get("description", "") or "",
            }
            for a in articles
        ]
    except Exception as exc:
        print(f"    NewsAPI error: {exc}")
        return []


def fetch_headlines(
    from_date: date,
    to_date: date,
    dry_run: bool = False,
    max_requests: int | None = None,
) -> pd.DataFrame:
    """Fetch headlines and append to HEADLINES_PATH.

    Incremental: already-fetched month-year combos are skipped so you can
    run this daily until the full 2018-2024 range is covered.

    Multi-key support: reads ALPHA_VANTAGE_KEY, ALPHA_VANTAGE_KEY_2, …
    25-call quota per key; auto-rotates to the next when one is exhausted.

    Args:
        max_requests: AV API calls to make this session. If None, uses
                      25 × number-of-keys (full daily quota across all keys).
    """
    av_keys = [
        v for v in (
            os.environ.get("ALPHA_VANTAGE_KEY", ""),
            os.environ.get("ALPHA_VANTAGE_KEY_2", ""),
            os.environ.get("ALPHA_VANTAGE_KEY_3", ""),
            os.environ.get("ALPHA_VANTAGE_KEY_4", ""),
        ) if v
    ]
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")

    if max_requests is None:
        max_requests = 25 * max(len(av_keys), 1)

    if not av_keys and not newsapi_key:
        print("⚠ No API keys found (ALPHA_VANTAGE_KEY[_N], NEWSAPI_KEY). Aborting fetch.")
        return pd.DataFrame()

    # Load existing cache and track which months are already done
    if HEADLINES_PATH.exists():
        existing = pd.read_parquet(HEADLINES_PATH)
        existing["_date"] = pd.to_datetime(existing["date"], errors="coerce")
        av_done = set(
            existing[existing["source"] == "alphavantage"]["_date"]
            .dt.to_period("M")
            .dropna()
            .astype(str)
            .unique()
        )
        existing = existing.drop(columns=["_date"])
    else:
        existing = pd.DataFrame()
        av_done = set()

    all_rows: list[dict] = []
    requests_made = 0

    # Alpha Vantage: monthly batches, skip already-fetched months
    if av_keys:
        cur = date(from_date.year, from_date.month, 1)
        total_months = (
            (to_date.year - from_date.year) * 12
            + (to_date.month - from_date.month)
            + 1
        )
        pending = []
        tmp = cur
        while tmp <= to_date:
            key_str = f"{tmp.year}-{tmp.month:02d}"
            if key_str not in av_done:
                pending.append((tmp.year, tmp.month))
            if tmp.month == 12:
                tmp = date(tmp.year + 1, 1, 1)
            else:
                tmp = date(tmp.year, tmp.month + 1, 1)

        print(
            f"AV: {len(pending)} months to fetch "
            f"({total_months - len(pending)} already cached). "
            f"Have {len(av_keys)} key(s); making up to {max_requests} calls this session."
        )

        key_idx = 0
        key_calls = 0  # calls made on the current key
        PER_KEY_QUOTA = 25

        for yr, mo in pending:
            if requests_made >= max_requests:
                remaining = len(pending) - (pending.index((yr, mo)))
                print(
                    f"  ↳ Session cap reached ({max_requests} calls). "
                    f"{remaining} months remain."
                )
                break

            month_str = f"{yr}-{mo:02d}"
            if dry_run:
                print(f"  [dry-run] would fetch AV {month_str} (key #{key_idx + 1})")
                requests_made += 1
                continue

            rows = _av_month(yr, mo, av_keys[key_idx])
            requests_made += 1
            key_calls += 1

            if not rows:
                # Current key likely exhausted — try next key once.
                if key_idx + 1 < len(av_keys):
                    key_idx += 1
                    key_calls = 0
                    print(f"  ↳ Key #{key_idx} exhausted on {month_str}; rotating to key #{key_idx + 1}.")
                    rows = _av_month(yr, mo, av_keys[key_idx])
                    requests_made += 1
                    key_calls += 1
                if not rows:
                    print(f"  AV returned 0 articles for {month_str} on all remaining keys. Stopping.")
                    break

            all_rows.extend(rows)

            # Proactive key rotation when current key hits 25 calls
            if key_calls >= PER_KEY_QUOTA and key_idx + 1 < len(av_keys):
                key_idx += 1
                key_calls = 0
                print(f"  ↳ Key #{key_idx} hit {PER_KEY_QUOTA}-call quota; rotating to key #{key_idx + 1}.")

            if requests_made % 5 == 0:
                print(
                    f"  … {requests_made} AV calls made (key #{key_idx + 1}, "
                    f"{key_calls}/{PER_KEY_QUOTA}), {len(all_rows)} new articles so far"
                )
            time.sleep(1.5)   # ~40 req/min, comfortably under AV limit

    # NewsAPI: recency top-up (last 30 days only — no historical)
    if newsapi_key and not dry_run:
        print("NewsAPI: fetching recent headlines (last 30 days)…")
        rows = _newsapi_recent(newsapi_key, max_results=100)
        all_rows.extend(rows)
        print(f"  {len(rows)} articles from NewsAPI.")

    if dry_run:
        print("[dry-run] No real fetches performed.")
        return existing

    if not all_rows:
        print("No new articles fetched this session.")
        return existing

    new_df = pd.DataFrame(all_rows)
    new_df = new_df[new_df["title"].str.strip().ne("")].copy()
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce").dt.date
    new_df = new_df.dropna(subset=["date"])

    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        if not existing.empty else new_df
    )
    combined = combined.drop_duplicates(subset=["date", "source", "title"])
    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_parquet(HEADLINES_PATH, index=False)
    n_av = (combined["source"] == "alphavantage").sum()
    n_months_done = combined[combined["source"] == "alphavantage"]["date"].apply(
        lambda d: f"{d.year}-{d.month:02d}" if hasattr(d, "year") else ""
    ).nunique()
    print(
        f"✓ Cache: {len(combined)} total headlines "
        f"({n_av} from AV across {n_months_done} months) → {HEADLINES_PATH}"
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# B12 — FinBERT inference
# ─────────────────────────────────────────────────────────────────────────────

def run_finbert(
    batch_size: int = 64,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run ProsusAI/finbert on headlines_raw.parquet → sentiment_daily.parquet.

    Returns the per-trading-day sentiment DataFrame.
    """
    if not HEADLINES_PATH.exists():
        print("⚠ headlines_raw.parquet not found. Run --fetch first.")
        return pd.DataFrame()

    headlines = pd.read_parquet(HEADLINES_PATH)
    print(f"Loaded {len(headlines)} headlines for FinBERT inference…")

    if dry_run:
        print("[dry-run] Skipping FinBERT inference.")
        return _build_sentiment_from_scores(
            headlines.assign(label="neutral", score=1.0)
        )

    # Load FinBERT
    try:
        from transformers import pipeline
    except ImportError:
        print("⚠ transformers not installed. Run `pip install transformers`.")
        return pd.DataFrame()

    device = _get_device()
    print(f"Loading ProsusAI/finbert on device={device}…")
    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512,
    )

    titles = headlines["title"].fillna("").tolist()
    labels, scores = [], []

    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        # Filter empty strings (pipeline would error)
        safe = [t if t.strip() else "neutral market" for t in batch]
        results = classifier(safe, batch_size=min(batch_size, len(safe)))
        for r in results:
            labels.append(r["label"].lower())   # positive / negative / neutral
            scores.append(r["score"])

        if (i // batch_size) % 10 == 0:
            print(f"  … FinBERT {i + len(batch)}/{len(titles)} headlines processed")

    headlines = headlines.copy()
    headlines["label"] = labels
    headlines["score"] = scores

    return _build_sentiment_from_scores(headlines)


def _build_sentiment_from_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-day sentiment and gap-fill trading days."""
    # Map FinBERT labels → bullish / neutral / bearish
    df = df.copy()
    df["is_bullish"] = (df["label"] == "positive").astype(float)
    df["is_neutral"] = (df["label"] == "neutral").astype(float)

    daily = (
        df.groupby("date")
          .agg(
              headline_count=("title", "count"),
              bullish_sum=("is_bullish", "sum"),
              neutral_sum=("is_neutral", "sum"),
          )
          .reset_index()
    )
    daily["bullish_frac"]       = daily["bullish_sum"] / daily["headline_count"].clip(lower=1)
    daily["neutral_frac"]       = daily["neutral_sum"] / daily["headline_count"].clip(lower=1)
    daily["headline_count_norm"] = (daily["headline_count"] / HEADLINE_COUNT_CAP).clip(upper=1.0)
    daily = daily[["date", "bullish_frac", "neutral_frac", "headline_count_norm"]]

    # Gap-fill: generate full trading-day calendar and fill missing with neutral
    trading_days = _get_trading_days()
    base = pd.DataFrame({"date": trading_days})
    base["date"] = pd.to_datetime(base["date"]).dt.date
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    merged = base.merge(daily, on="date", how="left")
    merged["bullish_frac"]        = merged["bullish_frac"].fillna(NEUTRAL["bullish_frac"])
    merged["neutral_frac"]        = merged["neutral_frac"].fillna(NEUTRAL["neutral_frac"])
    merged["headline_count_norm"] = merged["headline_count_norm"].fillna(NEUTRAL["headline_count_norm"])

    coverage = (merged["headline_count_norm"] > 0).mean() * 100
    print(f"  Sentiment coverage: {coverage:.1f}% of {len(merged)} trading days have headlines.")
    print(f"  Gap-filled {(merged['headline_count_norm'] == 0).sum()} days with neutral sentinel.")

    merged.to_parquet(SENTIMENT_PATH, index=False)
    print(f"✓ Saved sentiment_daily.parquet → {SENTIMENT_PATH}")
    return merged


def _get_trading_days() -> list[date]:
    """Approximate US trading days 2018–2024 by loading SPY parquet dates."""
    spy_path = DATA_DIR / "SPY.parquet"
    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        return sorted(df.index.date.tolist())
    # Fallback: all business days in range
    idx = pd.bdate_range(TRAIN_START, TRAIN_END)
    return [d.date() for d in idx]


def _get_device() -> int:
    """Return torch device index (-1 = CPU, 0 = first GPU)."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch headlines + run FinBERT sentiment pipeline")
    p.add_argument("--fetch",  action="store_true", help="Run B11: fetch headlines from APIs")
    p.add_argument("--infer",  action="store_true", help="Run B12: FinBERT inference on cached headlines")
    p.add_argument("--all",    action="store_true", help="Run B11 + B12 end-to-end")
    p.add_argument("--from-date", default=TRAIN_START.isoformat(),
                   help="Start date for headline fetch (YYYY-MM-DD)")
    p.add_argument("--to-date",   default=TRAIN_END.isoformat(),
                   help="End date for headline fetch (YYYY-MM-DD)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Smoke-test: print what would be fetched, skip API calls")
    p.add_argument("--batch-size", type=int, default=64,
                   help="FinBERT inference batch size")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from_d = date.fromisoformat(args.from_date)
    to_d   = date.fromisoformat(args.to_date)

    do_fetch = args.fetch or args.all
    do_infer = args.infer or args.all

    if not do_fetch and not do_infer:
        print("Specify --fetch, --infer, or --all.")
        raise SystemExit(1)

    if do_fetch:
        print(f"\n── B11: Fetching headlines {from_d} → {to_d} ──")
        fetch_headlines(from_d, to_d, dry_run=args.dry_run)

    if do_infer:
        print("\n── B12: Running FinBERT inference ──")
        run_finbert(batch_size=args.batch_size, dry_run=args.dry_run)

    print("\nDone.")
