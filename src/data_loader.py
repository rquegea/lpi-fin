"""
OHLCV data loader with local Parquet cache.

Downloads daily OHLCV data from Yahoo Finance via yfinance.
Caches each ticker as data/raw/{ticker}.parquet.
If the cache exists and covers the requested date range, it is used
directly without re-downloading.

Abort conditions (raises RuntimeError, requiring user decision):
  - Fewer than min_tickers_pct * len(universe) tickers download correctly.

Warning conditions (logs and continues):
  - A ticker returns fewer than min_days_per_ticker trading days.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _cache_path(ticker: str, raw_dir: Path) -> Path:
    return raw_dir / f"{ticker}.parquet"


def _load_from_cache(path: Path, start: str, end: str) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if it fully covers [start, end], else None."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    # Cache is valid if it covers up to within 5 calendar days of `end`.
    # This tolerates weekends and holidays (e.g., end="2024-12-31" but last
    # trading day is 2024-12-30).
    import datetime
    end_dt = pd.Timestamp(end)
    tolerance = pd.Timedelta(days=5)
    if df.index.max() >= end_dt - tolerance:
        mask = (df.index >= start) & (df.index <= end)
        sliced = df.loc[mask]
        if not sliced.empty:
            return sliced
    return None


def _download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLCV for a single ticker. Returns None on failure."""
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            logger.warning("No data returned for %s", ticker)
            return None
        # Flatten MultiIndex columns if present (yfinance >= 0.2.38)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"
        # Keep only OHLCV
        raw = raw[["open", "high", "low", "close", "volume"]].copy()
        raw = raw.dropna(how="all")
        return raw
    except Exception as exc:
        logger.warning("Download failed for %s: %s", ticker, exc)
        return None


def load_universe(
    universe: list,
    start: str,
    end: str,
    raw_dir: Path,
    min_days: int = 2000,
    min_tickers_pct: float = 0.833,
) -> dict:
    """
    Download or load from cache OHLCV data for all tickers.

    Parameters
    ----------
    universe : list of ticker strings
    start : str, e.g. "2015-01-01"
    end   : str, e.g. "2024-12-31"
    raw_dir : Path to data/raw/
    min_days : minimum number of trading days required per ticker
    min_tickers_pct : fraction of universe that must succeed

    Returns
    -------
    dict mapping ticker -> DataFrame(open, high, low, close, volume)

    Raises
    ------
    RuntimeError if fewer than min_tickers_pct * len(universe) succeed.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    for ticker in universe:
        cache = _cache_path(ticker, raw_dir)
        df = _load_from_cache(cache, start, end)
        if df is not None:
            logger.info("Loaded %s from cache (%d rows)", ticker, len(df))
        else:
            logger.info("Downloading %s …", ticker)
            df = _download_ticker(ticker, start, end)
            if df is not None:
                df.to_parquet(cache)
                # Re-slice to requested window
                mask = (df.index >= start) & (df.index <= end)
                df = df.loc[mask]

        if df is None or df.empty:
            logger.warning("SKIP %s — no data", ticker)
            failed.append(ticker)
            continue

        if len(df) < min_days:
            logger.warning(
                "SKIP %s — only %d days (< %d required)", ticker, len(df), min_days
            )
            failed.append(ticker)
            continue

        results[ticker] = df

    n_ok = len(results)
    n_required = int(min_tickers_pct * len(universe))
    if n_ok < n_required:
        raise RuntimeError(
            f"Only {n_ok}/{len(universe)} tickers downloaded successfully "
            f"(need at least {n_required}). Failed: {failed}. "
            "Check your internet connection or adjust the universe."
        )

    logger.info(
        "Loaded %d/%d tickers successfully. Failed: %s",
        n_ok,
        len(universe),
        failed if failed else "none",
    )
    return results
