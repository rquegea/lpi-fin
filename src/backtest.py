"""
Straddle backtest — Fase 2.

Simulates buying at-the-money straddles on tickers selected by the LPI
score (top quintile), holding for hold_days trading days, then closing.

P&L model per trade
-------------------
  cost_iv    = iv_cboe(t) * sqrt(hold_days / 252)
  bid_ask    = bid_ask_cost_pct * cost_iv
  total_cost = cost_iv + bid_ask
  return_5d  = log(close(t + hold_days) / close(t))
  payoff     = |return_5d| - total_cost
  pnl_usd    = payoff * position_value   where
               position_value = position_size_pct * capital_at_open

Capital changes only when positions close (premium plus payoff net out).
The equity curve therefore shows piecewise-constant segments between
closings, with discrete jumps at each settlement.

Known limitations (see config_backtest.yaml for full list):
  - iv_cboe is sector-level (VIX/VXN), not per-ticker — cost is underestimated.
  - Payoff model ignores gamma, theta, early exercise.
  - Execution assumed at close price with no additional slippage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


class StraddleBacktest:
    """
    Day-by-day straddle simulation driven by LPI scores.

    Parameters
    ----------
    scores_df : DataFrame
        MultiIndex (date, ticker), single column 'lpi_score'.
    prices : DataFrame
        Index = sorted trading dates, columns = ticker symbols, values = close prices.
    iv_per_ticker : DataFrame
        Same shape as *prices*; values = annualised implied vol in decimal
        (e.g. VIX = 20 → 0.20).
    config : dict
        Backtest parameters (see config_backtest.yaml).
    """

    def __init__(
        self,
        scores_df: pd.DataFrame,
        prices: pd.DataFrame,
        iv_per_ticker: pd.DataFrame,
        config: dict,
    ) -> None:
        if "lpi_score" not in scores_df.columns:
            raise ValueError("scores_df must have a column named 'lpi_score'.")
        self.scores_df = scores_df
        self.prices = prices
        self.iv = iv_per_ticker
        self.cfg = config

        self._equity_curve: Optional[pd.Series] = None
        self._trades: Optional[pd.DataFrame] = None
        self._ran: bool = False

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full day-by-day simulation."""
        cfg = self.cfg
        initial_capital = float(cfg["initial_capital"])
        position_size_pct = float(cfg["position_size_pct"])
        max_concurrent = int(cfg["max_concurrent_positions"])
        hold_days = int(cfg["hold_days"])
        top_pct = float(cfg["top_quintile_pct"])
        bid_ask_pct = float(cfg["bid_ask_cost_pct"])

        oos_dates: set = set(
            self.scores_df.index.get_level_values("date").unique()
        )
        all_dates: list = sorted(self.prices.index)
        date_to_idx: Dict[Any, int] = {d: i for i, d in enumerate(all_dates)}

        capital: float = initial_capital
        # Each open position is a dict with keys:
        #   open_date, close_date, ticker, open_price, cost_iv, total_cost, position_value
        open_positions: list = []
        trades: list = []
        equity_records: list = []

        oos_start = min(oos_dates) if oos_dates else None

        for date in all_dates:
            if oos_start is not None and date < oos_start:
                continue

            # ---- 1. Close positions that mature today --------------------
            still_open: list = []
            for pos in open_positions:
                if pos["close_date"] == date:
                    close_price = self._safe_price(date, pos["ticker"])
                    if close_price is not None and pos["open_price"] > 0:
                        return_5d = float(
                            np.log(close_price / pos["open_price"])
                        )
                    else:
                        return_5d = 0.0

                    payoff = abs(return_5d) - pos["total_cost"]
                    pnl = payoff * pos["position_value"]
                    capital += pnl

                    trades.append(
                        {
                            "open_date": pos["open_date"],
                            "close_date": date,
                            "ticker": pos["ticker"],
                            "open_price": pos["open_price"],
                            "close_price": close_price if close_price is not None else float("nan"),
                            "return_5d": return_5d,
                            "cost_iv": pos["cost_iv"],
                            "total_cost": pos["total_cost"],
                            "payoff": payoff,
                            "position_value": pos["position_value"],
                            "pnl": pnl,
                            "won": payoff > 0,
                        }
                    )
                else:
                    still_open.append(pos)
            open_positions = still_open

            # ---- 2. Record equity after closings, before openings --------
            equity_records.append({"date": date, "equity": capital})

            # ---- 3. Open new positions on OOS dates ----------------------
            if date not in oos_dates:
                continue

            n_slots = max_concurrent - len(open_positions)
            if n_slots <= 0:
                continue

            day_scores = self._get_day_scores(date)
            if day_scores is None or len(day_scores) == 0:
                continue

            # Determine close date (hold_days trading days ahead)
            cur_idx = date_to_idx.get(date)
            if cur_idx is None or cur_idx + hold_days >= len(all_dates):
                continue
            close_date = all_dates[cur_idx + hold_days]

            # Top-quintile candidates sorted by descending score
            threshold = float(day_scores.quantile(1.0 - top_pct))
            candidates = (
                day_scores[day_scores >= threshold]
                .sort_values(ascending=False)
                .iloc[:n_slots]
            )

            for ticker in candidates.index:
                if len(open_positions) >= max_concurrent:
                    break
                open_price = self._safe_price(date, ticker)
                if open_price is None:
                    continue
                iv_val = self._safe_iv(date, ticker)
                if iv_val is None or iv_val <= 0:
                    continue

                cost_iv = iv_val * np.sqrt(hold_days / 252.0)
                total_cost = cost_iv * (1.0 + bid_ask_pct)
                position_value = position_size_pct * capital

                open_positions.append(
                    {
                        "open_date": date,
                        "close_date": close_date,
                        "ticker": ticker,
                        "open_price": open_price,
                        "cost_iv": cost_iv,
                        "total_cost": total_cost,
                        "position_value": position_value,
                    }
                )

        # ------------------------------------------------------------------
        self._equity_curve = (
            pd.DataFrame(equity_records).set_index("date")["equity"]
        )
        self._trades = (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(
                columns=[
                    "open_date", "close_date", "ticker", "open_price",
                    "close_price", "return_5d", "cost_iv", "total_cost",
                    "payoff", "position_value", "pnl", "won",
                ]
            )
        )
        self._ran = True

    # ------------------------------------------------------------------
    def get_equity_curve(self) -> pd.Series:
        """Return equity curve as pd.Series indexed by trading date."""
        self._assert_ran()
        return self._equity_curve.copy()

    def get_trades(self) -> pd.DataFrame:
        """Return one row per closed trade."""
        self._assert_ran()
        return self._trades.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Compute and return summary performance metrics."""
        self._assert_ran()
        equity = self._equity_curve
        trades = self._trades

        if len(trades) == 0:
            return {"error": "No trades executed"}

        daily_returns = equity.pct_change().dropna()
        rf_daily = float(self.cfg.get("risk_free_rate", 0.0)) / 252.0

        # CAGR
        span_days = (equity.index[-1] - equity.index[0]).days
        n_years = span_days / 365.25
        if n_years > 0 and equity.iloc[0] > 0:
            cagr = float(
                (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1
            )
        else:
            cagr = 0.0

        # Sharpe (annualised, rf configurable, default 0)
        excess = daily_returns - rf_daily
        if excess.std() > 0:
            sharpe = float(excess.mean() / excess.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino (annualised, using downside std of excess returns)
        downside = excess[excess < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(excess.mean() / downside.std() * np.sqrt(252))
        else:
            sortino = float("inf") if excess.mean() > 0 else 0.0

        # Maximum drawdown
        rolling_max = equity.cummax()
        drawdown_series = (equity - rolling_max) / rolling_max
        max_drawdown = float(drawdown_series.min())

        # Trade statistics
        n_trades = len(trades)
        win_rate = float(trades["won"].mean()) if n_trades > 0 else 0.0
        wins = trades.loc[trades["won"], "pnl"]
        losses = trades.loc[~trades["won"], "pnl"]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        return {
            "initial_capital": float(self.cfg["initial_capital"]),
            "final_capital": float(equity.iloc[-1]),
            "total_return_pct": float(
                (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            ),
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "n_trades": n_trades,
            "n_trading_days": len(equity),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_price(self, date, ticker) -> Optional[float]:
        try:
            val = self.prices.loc[date, ticker]
            return float(val) if pd.notna(val) else None
        except KeyError:
            return None

    def _safe_iv(self, date, ticker) -> Optional[float]:
        try:
            val = self.iv.loc[date, ticker]
            return float(val) if pd.notna(val) else None
        except KeyError:
            return None

    def _get_day_scores(self, date) -> Optional[pd.Series]:
        try:
            s = self.scores_df.loc[date]["lpi_score"]
            return s if isinstance(s, pd.Series) else None
        except KeyError:
            return None

    def _assert_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("Call run() before accessing results.")
