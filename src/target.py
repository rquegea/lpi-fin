"""
Target construction for the LPI pipeline.

TARGET DEFINITION
-----------------
y(t) = 1  if rv_5_forward(t+1, t+5) > iv_level(t) * threshold
y(t) = 0  otherwise

where:
  rv_5_forward(t+1, t+5) = std(log_returns in [t+1, t+5]) * sqrt(252)
  iv_level(t) = rv_30(t) * iv_premium   (same quantity as the feature)
  threshold = 1.3  (from config: target_threshold)

WHY THIS DOES NOT CREATE LOOK-AHEAD LEAKAGE
---------------------------------------------
It may appear suspicious that iv_level(t) — a feature — also appears in
the target definition. Here is why there is NO mechanical coupling that
leaks signal:

  Example (numerical):
    t = 2023-06-15
    log_returns in [2023-05-05, 2023-06-15] → rv_30 = 0.18 annualized
    iv_premium = 1.04
    iv_level(t) = 0.18 * 1.04 = 0.1872

    TARGET threshold = iv_level(t) * 1.3 = 0.2434

    log_returns in [2023-06-16, 2023-06-22] → rv_5_forward = 0.28 annualized
    0.28 > 0.2434 → y(t) = 1  (market moved more than the IV implied)

  Key point: rv_5_forward uses log_returns in [t+1, t+5] — FUTURE data
  relative to t. The feature iv_level uses log_returns in [t-29, t] —
  PAST data relative to t. They share NO overlapping return observations.

  The only link is that both use the same underlying price series, but
  this is true of any feature/target pair in a returns-based dataset.
  The question the model is asking: "does the current IV proxy correctly
  price future vol, or is it systematically wrong in some identifiable
  cluster of market conditions?" — which is exactly the research question.

FILTERING
---------
Observations where fewer than rv5_window future returns are available
(i.e., near the end of the series) are dropped before returning.
"""

import numpy as np
import pandas as pd

_SQRT252 = np.sqrt(252)


def compute_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Compute binary target for a single ticker.

    Parameters
    ----------
    df : DataFrame with column 'close', indexed by date (ascending).
    cfg : dict with keys: rv30_window, iv_premium, target_threshold,
          rv5_window.

    Returns
    -------
    pd.Series of {0, 1} float, same index as df.
    NaN for rows where the target cannot be computed (last rv5_window rows).
    """
    rv30_window = cfg["rv30_window"]
    iv_premium = cfg["iv_premium"]
    threshold = cfg["target_threshold"]
    rv5_window = cfg["rv5_window"]

    log_ret = np.log(df["close"] / df["close"].shift(1))

    # IV proxy at t (same as feature iv_level)
    rv30 = log_ret.rolling(rv30_window).std() * _SQRT252
    iv_level_t = rv30 * iv_premium

    # Forward realized vol: std of log_returns in [t+1, t+rv5_window]
    # We compute this by shifting the rolling std backwards.
    # rolling(rv5_window).std() at position i uses rows [i-rv5_window+1, i].
    # Shifting by -rv5_window gives us [i+1, i+rv5_window] at position i.
    rv5_forward = (
        log_ret.rolling(rv5_window).std().shift(-rv5_window) * _SQRT252
    )

    target_threshold_series = iv_level_t * threshold
    y = (rv5_forward > target_threshold_series).astype(float)

    # Mark as NaN where forward vol is not available (last rv5_window rows)
    y[rv5_forward.isna()] = np.nan
    # Also NaN where iv_level is not available (first rv30_window-1 rows)
    y[iv_level_t.isna()] = np.nan

    return y.rename("target")
