"""
Target v2: binary IV/RV divergence using CBOE index as IV reference.

TARGET DEFINITION
-----------------
y(t) = 1  if rv_5_forward(t+1, t+5) > iv_cboe(t) * threshold
y(t) = 0  otherwise

where:
  rv_5_forward(t+1, t+5) = std(log_returns in [t+1, t+5]) * sqrt(252)
  iv_cboe(t) = CBOE index value on day t / 100  (decimal, e.g. 0.20)
  threshold = 1.3  (same as previous target)

WHY THIS TARGET IS MORE "HONEST" THAN THE PROXY VERSION
---------------------------------------------------------
In target.py (v1), both the feature iv_level AND the target threshold
were derived from the SAME quantity: rv_30(t) * iv_premium.

Concretely in v1:
  iv_level(t)   = rv_30(t) * 1.04          ← feature
  target(t) = rv_5_fwd > rv_30(t) * 1.04 * 1.3   ← target

This means the target was essentially asking: "will the next 5-day vol
exceed 135% of the last 30-day vol?" — a pure vol-trending question with
no connection to market-implied expectations. The LPI was learning
vol-regime transitions, not IV/RV divergence.

In v2 (this file):
  iv_level_cboe(t) = VIX(t) or VXN(t) / 100     ← from options market
  target(t) = rv_5_fwd > iv_cboe(t) * 1.3        ← target

Now iv_cboe(t) comes from actual options pricing, incorporating:
  - Market participants' expectations of future vol
  - Risk premium / fear premium embedded in options
  - Forward-looking information (not backward-looking like rv_30)

The question the model now answers: "in what cluster of market conditions
does the NEXT 5-day vol SIGNIFICANTLY EXCEED what the options market is
currently pricing?" — which is the genuine volatility risk premium
prediction problem.

Numerical example (v2):
  t = 2023-06-15
  VXN(t) = 18.5  →  iv_cboe(t) = 0.185
  target threshold = 0.185 * 1.3 = 0.2405

  log_returns in [2023-06-16, 2023-06-22]
  → rv_5_forward = 0.28 annualized
  0.28 > 0.2405  →  y(t) = 1  (realized vol exceeded implied)

NO LOOK-AHEAD:
  iv_cboe(t) uses the CBOE index close on day t (past data at time t).
  rv_5_forward uses returns on [t+1, t+5] (future — this is the TARGET,
  not a feature). The features in features_v2.py use only data up to t.
"""

import numpy as np
import pandas as pd

_SQRT252 = np.sqrt(252)


def compute_target_v2(
    df: pd.DataFrame,
    iv_cboe: pd.Series,
    cfg: dict,
) -> pd.Series:
    """
    Compute binary target using CBOE IV as the reference level.

    Parameters
    ----------
    df      : DataFrame with column 'close', indexed by date (ascending).
    iv_cboe : Series of CBOE IV (decimal form), aligned by date to df.
    cfg     : dict with keys: target_threshold, rv5_window.

    Returns
    -------
    pd.Series of {0.0, 1.0, NaN}, same index as df.
    NaN for rows where the target cannot be computed (last rv5_window rows
    or rows where iv_cboe is not available).
    """
    threshold  = cfg["target_threshold"]
    rv5_window = cfg["rv5_window"]

    log_ret = np.log(df["close"] / df["close"].shift(1))

    # Forward realized vol: std of returns in [t+1, t+rv5_window], annualized
    rv5_forward = (
        log_ret.rolling(rv5_window).std().shift(-rv5_window) * _SQRT252
    )

    # CBOE IV aligned to df index
    iv_cboe_aligned = iv_cboe.reindex(df.index)

    target_level = iv_cboe_aligned * threshold
    y = (rv5_forward > target_level).astype(float)

    # NaN where forward vol or CBOE IV is unavailable
    y[rv5_forward.isna()]     = np.nan
    y[iv_cboe_aligned.isna()] = np.nan

    return y.rename("target")
