"""
Script 05: Economic backtest — Fase 2 (placeholder).

NOT IMPLEMENTED YET.

This script will implement a straddle-based backtest using LPI scores
as a signal, with realistic transaction costs. It will only be
implemented if the LPI shows AUC >= 0.60 on real data (SEÑAL MODERADA
or better in the verdict from 03_run_lpi.py).

To be implemented:
  - Load panel and LPI OOS scores
  - For each day t in the test set, go long a straddle if LPI(t) > threshold
  - P&L based on next-5-day realized vol vs implied vol proxy
  - Apply bid-ask spread (approx 0.1 vol point) and financing costs
  - Report: Sharpe ratio, max drawdown, hit rate, avg P&L per trade
  - Apply Deflated Sharpe Ratio (López de Prado) to adjust for multiple testing
"""

print("Script 05 (economic backtest) is not implemented yet.")
print("Run 03_run_lpi.py first to check if the signal warrants Fase 2.")
