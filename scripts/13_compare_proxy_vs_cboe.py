"""
Script 13: Compare proxy IV (4-feature run) vs CBOE IV (4-feature run).

Prints a side-by-side table and a robustness verdict.

Verdict logic (delta = AUC_cboe - AUC_proxy):
  delta > -0.02             → SEÑAL ROBUSTA — LPI funciona con IV real.
                               Pasar a Fase 2 (backtest económico).
  -0.05 < delta <= -0.02    → SEÑAL DEGRADADA PERO REAL — requiere más
                               trabajo antes de operar.
  -0.10 < delta <= -0.05    → SEÑAL MARGINAL — el proxy inflaba el resultado.
  delta <= -0.10             → ARTEFACTO — resultado anterior era del proxy.
                               Cerrar la línea de investigación.

Hardcoded proxy-IV baseline (from run 06):
  AUC: 0.6406 ± 0.0069, AP: 0.2441, Lift Q5: 1.47, K*: 9
  Shuffle: 0.4997 ± 0.0028, Stability std: 0.0027
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Hardcoded from run 06 (4-feature proxy IV model)
BASELINE_PROXY = {
    "auc_mean":          0.6406,
    "auc_std":           0.0069,
    "ap":                0.2441,
    "lift_q5":           1.47,
    "chi2_p":            0.0,
    "k_star":            9,
    "base_rate":         0.1803,
    "n_obs":             73530,
    "shuffle_auc_mean":  0.4997,
    "shuffle_auc_std":   0.0028,
    "stability_std":     0.0027,
}


def load_cboe_results(root: Path) -> dict:
    results = {}

    main_path = root / "results" / "tables" / "main_results_cboe.csv"
    if not main_path.exists():
        print(f"Error: {main_path} not found. Run 11_run_lpi_cboe.py first.")
        sys.exit(1)
    m = pd.read_csv(main_path)
    results["auc_mean"]   = float(m["auc_mean"].iloc[0])
    results["auc_std"]    = float(m["auc_std"].iloc[0])
    results["ap"]         = float(m["ap"].iloc[0])
    results["lift_q5"]    = float(m["lift_q5"].iloc[0])
    results["chi2_p"]     = float(m["chi2_p"].iloc[0])
    results["k_star"]     = int(m["k_star"].iloc[0])
    results["base_rate"]  = float(m["base_rate"].iloc[0])
    results["n_obs"]      = int(m["n_obs"].iloc[0])

    diag_path = root / "results" / "tables" / "diagnostics_cboe.csv"
    if not diag_path.exists():
        print(f"Error: {diag_path} not found. Run 12_diagnostics_cboe.py first.")
        sys.exit(1)
    d = pd.read_csv(diag_path)
    results["shuffle_auc_mean"] = float(d["shuffle_auc_mean"].iloc[0])
    results["shuffle_auc_std"]  = float(d["shuffle_auc_std"].iloc[0])
    results["stability_std"]    = float(d["stability_std"].iloc[0])

    return results


def main():
    root = Path(__file__).parent.parent

    bp = BASELINE_PROXY
    rc = load_cboe_results(root)

    metrics = [
        ("AUC OOS mean",     "auc_mean",          ".4f"),
        ("AUC OOS std",      "auc_std",            ".4f"),
        ("AP OOS",           "ap",                 ".4f"),
        ("Lift Q5",          "lift_q5",            ".2f"),
        ("Chi² p",           "chi2_p",             None ),
        ("K*",               "k_star",             ".0f"),
        ("Base rate",        "base_rate",          ".3f"),
        ("N observaciones",  "n_obs",              ".0f"),
        ("Shuffle AUC mean", "shuffle_auc_mean",   ".4f"),
        ("Shuffle AUC std",  "shuffle_auc_std",    ".4f"),
        ("Estabilidad std",  "stability_std",      ".4f"),
    ]

    col_w = [22, 15, 15, 10]
    header = (
        f"{'Métrica':<{col_w[0]}} {'Proxy (4feat)':>{col_w[1]}} "
        f"{'CBOE (4feat)':>{col_w[2]}} {'Delta':>{col_w[3]}}"
    )
    sep = "-" * (sum(col_w) + 3)

    rows = []
    lines = [
        "",
        "====== COMPARACIÓN: Proxy IV vs CBOE IV ======",
        "  Proxy: rv_30 × 1.04  (mecánico, backward-looking)",
        "  CBOE:  VIX/VXN      (mercado real, forward-looking)",
        "  Mapping: AAPL/MSFT/GOOGL/... → ^VXN | JPM/V/WMT/... → ^VIX",
        "  Nota: IV compartida por grupo (sector-level, no per-ticker)",
        "",
        header,
        sep,
    ]

    for label, key, fmt_str in metrics:
        v_proxy = bp[key]
        v_cboe  = rc[key]
        if fmt_str is None:
            sp = "~0" if v_proxy == 0.0 else f"{v_proxy:.4f}"
            sc = "~0" if v_cboe  == 0.0 else f"{v_cboe:.4f}"
            delta_str = "n/a"
        else:
            sp = format(v_proxy, fmt_str)
            sc = format(v_cboe,  fmt_str)
            delta = v_cboe - v_proxy
            sign  = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:{fmt_str}}"

        line = (
            f"{label:<{col_w[0]}} {sp:>{col_w[1]}} "
            f"{sc:>{col_w[2]}} {delta_str:>{col_w[3]}}"
        )
        lines.append(line)
        rows.append({"metric": label, "proxy_4feat": v_proxy, "cboe_4feat": v_cboe})

    lines.append(sep)

    # Robustness verdict
    delta_auc = rc["auc_mean"] - bp["auc_mean"]
    if delta_auc > -0.02:
        verdict = (
            "SEÑAL ROBUSTA — El LPI funciona también con IV real (CBOE). "
            "Pasar a Fase 2: backtest económico de straddles."
        )
    elif delta_auc > -0.05:
        verdict = (
            "SEÑAL DEGRADADA PERO REAL — Sigue habiendo señal, pero requiere "
            "más trabajo (IV por ticker, más features) antes de operar."
        )
    elif delta_auc > -0.10:
        verdict = (
            "SEÑAL MARGINAL — El proxy estaba inflando el resultado. "
            "Hay algo pero es muy débil. Investigar antes de continuar."
        )
    else:
        verdict = (
            "ARTEFACTO — El resultado anterior era prácticamente del proxy. "
            "Cerrar la línea de investigación."
        )

    lines += [
        "",
        f"Delta AUC (CBOE - Proxy): {'+' if delta_auc >= 0 else ''}{delta_auc:.4f}",
        "",
        "====== VEREDICTO DE ROBUSTEZ ======",
        f"[{verdict}]",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    out_csv = root / "results" / "tables" / "comparison_proxy_vs_cboe.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nTabla guardada en: {out_csv}")

    log_path = root / "results" / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(report_text + "\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
