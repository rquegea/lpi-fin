"""
Feature construction for the LPI pipeline.

CRITICAL: Every feature must use ONLY information available at close of day t.
No future data may enter any feature computation. Each function documents
exactly which time window it uses.

The 7 features (vol_ratio was dropped from the synthetic experiment):
  1. iv_level      — IV proxy at t
  2. rv_20         — realized vol over [t-19, t]
  3. iv_rv_spread  — iv_level - rv_20
  4. log_range     — intraday range at t
  5. vol_of_vol    — vol-of-vol using data up to t
  6. log_dvol      — relative dollar volume
  7. momentum      — price momentum

All functions operate on a single-ticker DataFrame with columns:
  open, high, low, close, volume
indexed by date (ascending).

The build_features() function assembles all 7 into a single DataFrame.
"""

import numpy as np
import pandas as pd


_SQRT252 = np.sqrt(252)


def _log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns: log(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def compute_iv_level(df: pd.DataFrame, rv30_window: int, iv_premium: float) -> pd.Series:
    """
    Compute IV proxy at day t.

    Esta feature usa: log_returns en [t-29, t] (rv30_window días).

    IV proxy = annualized realized vol over the past rv30_window days
               multiplied by iv_premium.

    This is the same quantity used as the left-hand side of the target
    (the IV reference). See target.py for why this does NOT create
    look-ahead: the target compares FUTURE realized vol against this
    present-day proxy.

    Parameters
    ----------
    df : DataFrame with column 'close'
    rv30_window : int  (default 30)
    iv_premium : float (default 1.04)

    Returns
    -------
    pd.Series named 'iv_level'
    """
    lr = _log_returns(df["close"])
    rv30 = lr.rolling(rv30_window).std() * _SQRT252
    return (rv30 * iv_premium).rename("iv_level")


def compute_rv_20(df: pd.DataFrame, rv20_window: int) -> pd.Series:
    """
    Realized volatility over the past rv20_window days, annualized.

    Esta feature usa: log_returns en [t-{rv20_window-1}, t].

    Parameters
    ----------
    df : DataFrame with column 'close'
    rv20_window : int (default 20)

    Returns
    -------
    pd.Series named 'rv_20'
    """
    lr = _log_returns(df["close"])
    return (lr.rolling(rv20_window).std() * _SQRT252).rename("rv_20")


def compute_iv_rv_spread(iv_level: pd.Series, rv_20: pd.Series) -> pd.Series:
    """
    Difference between IV proxy and short-term realized vol.

    Esta feature usa: iv_level en t (window [t-29, t]) y rv_20 en t (window [t-19, t]).

    A positive spread means options appear rich relative to recent vol.
    A negative spread means options appear cheap.

    Returns
    -------
    pd.Series named 'iv_rv_spread'
    """
    return (iv_level - rv_20).rename("iv_rv_spread")


def compute_log_range(df: pd.DataFrame) -> pd.Series:
    """
    Intraday log range: log((high - low) / close).

    Esta feature usa: high, low, close en t (solo el día t).

    Captures intraday volatility not visible in close-to-close returns.

    Returns
    -------
    pd.Series named 'log_range'
    """
    return np.log((df["high"] - df["low"]) / df["close"]).rename("log_range")


def compute_vol_of_vol(
    df: pd.DataFrame,
    inner_window: int,
    outer_window: int,
) -> pd.Series:
    """
    Volatility of volatility: rolling std of short-term vol series.

    Esta feature usa: log_returns en [t - (inner_window + outer_window - 2), t].
    Specifically:
      - vol_5d(t) = std of log_returns in [t-4, t]   (inner_window=5)
      - vol_of_vol(t) = std of vol_5d in [t-19, t]   (outer_window=20)

    Both rolling windows end at t inclusive, so no future data is used.

    Parameters
    ----------
    df : DataFrame with column 'close'
    inner_window : int (default 5)
    outer_window : int (default 20)

    Returns
    -------
    pd.Series named 'vol_of_vol'
    """
    lr = _log_returns(df["close"])
    vol_5d = lr.rolling(inner_window).std() * _SQRT252
    return (vol_5d.rolling(outer_window).std()).rename("vol_of_vol")


def compute_log_dvol(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Log ratio of short-term to long-term dollar volume.

    Esta feature usa: close * volume en [t - {long_window-1}, t].
    Specifically:
      - dvol_5d(t)  = mean(close * volume) in [t-4, t]   (short_window=5)
      - dvol_60d(t) = mean(close * volume) in [t-59, t]  (long_window=60)

    log(dvol_5d / dvol_60d) > 0 means recent volume is above the norm.

    Parameters
    ----------
    df : DataFrame with columns 'close', 'volume'
    short_window : int (default 5)
    long_window : int (default 60)

    Returns
    -------
    pd.Series named 'log_dvol'
    """
    dvol = df["close"] * df["volume"]
    dvol_short = dvol.rolling(short_window).mean()
    dvol_long = dvol.rolling(long_window).mean()
    return np.log(dvol_short / dvol_long).rename("log_dvol")


def compute_momentum(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Log price momentum over the past `window` days.

    Esta feature usa: close en t y close en t-{window}.
    Specifically: log(close(t) / close(t-20)) with window=20.

    No future data is used; close(t) is the current day's close.

    Parameters
    ----------
    df : DataFrame with column 'close'
    window : int (default 20)

    Returns
    -------
    pd.Series named 'momentum'
    """
    return np.log(df["close"] / df["close"].shift(window)).rename("momentum")


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construct all 7 LPI features for a single ticker.

    Parameters
    ----------
    df : DataFrame with columns: open, high, low, close, volume
         Indexed by date, ascending.
    cfg : dict with feature window parameters (from config.yaml).

    Returns
    -------
    DataFrame of shape (n, 7) with columns:
      iv_level, rv_20, iv_rv_spread, log_range,
      vol_of_vol, log_dvol, momentum
    Rows with any NaN are retained here; callers should drop NaN
    after combining with the target.
    """
    iv_level = compute_iv_level(df, cfg["rv30_window"], cfg["iv_premium"])
    rv_20 = compute_rv_20(df, cfg["rv20_window"])
    iv_rv_spread = compute_iv_rv_spread(iv_level, rv_20)
    log_range = compute_log_range(df)
    vol_of_vol = compute_vol_of_vol(df, cfg["vol_of_vol_inner"], cfg["vol_of_vol_outer"])
    log_dvol = compute_log_dvol(df, cfg["dvol_short"], cfg["dvol_long"])
    momentum = compute_momentum(df, cfg["momentum_window"])

    return pd.concat(
        [iv_level, rv_20, iv_rv_spread, log_range, vol_of_vol, log_dvol, momentum],
        axis=1,
    )


FEATURE_NAMES = [
    "iv_level",
    "rv_20",
    "iv_rv_spread",
    "log_range",
    "vol_of_vol",
    "log_dvol",
    "momentum",
]
