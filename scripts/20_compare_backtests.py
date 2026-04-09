"""
Script 20: Comparación de backtests — original vs honesto.

Usage:
    python3.11 scripts/20_compare_backtests.py

Reads:
    results/tables/backtest_metrics.csv         (backtest original, script 05)
    results/tables/backtest_metrics_honest.csv  (backtest honesto, script 19)

Outputs:
    results/tables/comparison_backtests.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


# Hardcoded original (desde CSV existente, para referencia incluso si el CSV cambia)
ORIGINAL = {
    "position_size_pct": 0.01,
    "bid_ask_cost_pct":  0.08,
    "iv_markup":         1.0,
    "total_return_pct":  31.39,
    "cagr":              0.0341,
    "sharpe":            3.119,
    "sortino":           5.157,
    "max_drawdown":      -0.03971,
    "win_rate":          0.4667,
    "n_trades":          4090,
    "avg_win_usd":       41.15,
    "avg_loss_usd":      -21.63,
}


def fmt_pct(v):
    if v is None:
        return "?"
    return f"{float(v)*100:.2f}%"


def fmt_f(v, decimals=3):
    if v is None:
        return "?"
    return f"{float(v):.{decimals}f}"


def fmt_usd(v):
    if v is None:
        return "?"
    return f"${float(v):,.2f}"


def main():
    tab_dir = ROOT / "results" / "tables"

    honest_path = tab_dir / "backtest_metrics_honest.csv"
    if not honest_path.exists():
        print("Error: backtest_metrics_honest.csv not found. Run 19_backtest_honest.py first.")
        sys.exit(1)

    hon = pd.read_csv(honest_path).iloc[0].to_dict()

    rows = [
        # (label, original_value, honest_value, note)
        ("Position size",     "1%",                  "5%",                                   "—"),
        ("Bid-ask cost",      "8%",                  "15%",                                  "—"),
        ("IV markup",         "1.0× (VIX puro)",     "1.5× (individual equity)",             "—"),
        ("---",               "---",                 "---",                                  "---"),
        ("CAGR",              f"{ORIGINAL['cagr']*100:.2f}%",  fmt_pct(hon.get("cagr")),    _delta_pct(hon.get("cagr"), ORIGINAL["cagr"])),
        ("Sharpe ratio",      fmt_f(ORIGINAL["sharpe"]),       fmt_f(hon.get("sharpe")),    _delta_f(hon.get("sharpe"), ORIGINAL["sharpe"])),
        ("Sortino ratio",     fmt_f(ORIGINAL["sortino"]),      fmt_f(hon.get("sortino")),   _delta_f(hon.get("sortino"), ORIGINAL["sortino"])),
        ("Max drawdown",      fmt_pct(ORIGINAL["max_drawdown"]), fmt_pct(hon.get("max_drawdown")), _delta_pct(hon.get("max_drawdown"), ORIGINAL["max_drawdown"])),
        ("Win rate",          f"{ORIGINAL['win_rate']*100:.1f}%", f"{float(hon.get('win_rate',0))*100:.1f}%", "—"),
        ("N trades",          f"{ORIGINAL['n_trades']:,}",     f"{int(hon.get('n_trades',0)):,}",              "—"),
        ("Avg win (USD)",     fmt_usd(ORIGINAL["avg_win_usd"]), fmt_usd(hon.get("avg_win_usd")), "—"),
        ("Avg loss (USD)",    fmt_usd(ORIGINAL["avg_loss_usd"]), fmt_usd(hon.get("avg_loss_usd")), "—"),
    ]

    # Print table
    print(f"\n{'='*80}")
    print("COMPARACIÓN: Backtest Original  vs  Backtest Honesto")
    print(f"{'='*80}")
    header = f"{'Métrica':22s} | {'Original':>20s} | {'Honesto':>20s} | {'Delta':>12s}"
    print(header)
    print("-" * len(header))
    for label, orig, hon_val, delta in rows:
        if label == "---":
            print("-" * len(header))
            continue
        print(f"{label:22s} | {orig:>20s} | {hon_val:>20s} | {delta:>12s}")
    print(f"{'='*80}")

    # Verdict
    sharpe_hon = float(hon.get("sharpe", 0))
    print(f"\n====== VEREDICTO (Backtest Honesto) ======")
    print(f"Sharpe honesto: {sharpe_hon:.3f}")
    if sharpe_hon < 0:
        verdict = "NO VIABLE — la señal no cubre costes realistas"
    elif sharpe_hon < 0.5:
        verdict = "MARGINAL — cubre costes apenas, no justifica capital"
    elif sharpe_hon < 1.0:
        verdict = "POSITIVO DÉBIL — edge real pero por debajo del umbral institucional"
    elif sharpe_hon < 1.5:
        verdict = "BUENO — nivel profesional, justifica conversación con fondos"
    else:
        verdict = "EXCELENTE — revisar overfitting antes de celebrar"
    print(f"[{verdict}]")
    print()

    # Save CSV
    csv_rows = []
    for label, orig, hon_val, delta in rows:
        if label != "---":
            csv_rows.append({"metric": label, "original": orig, "honest": hon_val, "delta": delta})
    out_path = tab_dir / "comparison_backtests.csv"
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"Guardado en: {out_path}")


def _delta_pct(v, base):
    try:
        d = float(v) - float(base)
        return f"{d*100:+.2f}pp"
    except (TypeError, ValueError):
        return "?"


def _delta_f(v, base):
    try:
        d = float(v) - float(base)
        return f"{d:+.3f}"
    except (TypeError, ValueError):
        return "?"


if __name__ == "__main__":
    main()
