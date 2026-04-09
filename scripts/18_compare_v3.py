# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
Script 18: Compare CBOE 4-feature baseline vs v3 single vs v3 ensemble.

Usage:
    python3.11 scripts/18_compare_v3.py

Reads:
  results/tables/main_results_cboe.csv
  results/tables/main_results_v3.csv
  results/tables/main_results_v3_ensemble.csv
  results/tables/diagnostics_cboe.csv        (optional)
  results/tables/diagnostics_v3_ensemble.csv (optional)

Outputs:
  results/tables/comparison_cboe_vs_v3_vs_ensemble.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Hardcoded baseline (from confirmed CBOE run)
BASELINE_CBOE = {
    "auc_mean":     0.6253,
    "auc_std":      0.0112,
    "ap":           0.4203,
    "lift_q5":      1.39,
    "k_star":       4,
    "base_rate":    0.329,
    "shuffle_mean": 0.5005,
    "shuffle_std":  0.0024,
}


def _load_safe(path):
    if path.exists():
        return pd.read_csv(path).iloc[0].to_dict()
    return {}


def _delta_str(v, base):
    if v is None or base is None:
        return "?"
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main():
    root = Path(__file__).parent.parent
    tab_dir = root / "results" / "tables"

    # Load results
    r_v3    = _load_safe(tab_dir / "main_results_v3.csv")
    r_ens   = _load_safe(tab_dir / "main_results_v3_ensemble.csv")
    d_cboe  = _load_safe(tab_dir / "diagnostics_cboe.csv")
    d_ens   = _load_safe(tab_dir / "diagnostics_v3_ensemble.csv")

    def get(d, key, default=None):
        v = d.get(key, default)
        return float(v) if v is not None and v != "" else default

    cboe_shuffle_mean = get(d_cboe, "shuffle_auc_mean", BASELINE_CBOE["shuffle_mean"])
    cboe_shuffle_std  = get(d_cboe, "shuffle_auc_std",  BASELINE_CBOE["shuffle_std"])

    rows = [
        ("AUC OOS mean",    BASELINE_CBOE["auc_mean"],     get(r_v3, "auc_mean"),    get(r_ens, "auc_mean"),    "auc_mean"),
        ("AUC OOS std",     BASELINE_CBOE["auc_std"],      get(r_v3, "auc_std"),     get(r_ens, "auc_std"),     None),
        ("AP OOS",          BASELINE_CBOE["ap"],            get(r_v3, "ap"),          get(r_ens, "ap"),          None),
        ("Lift Q5",         BASELINE_CBOE["lift_q5"],       get(r_v3, "lift_q5"),     get(r_ens, "lift_q5"),     None),
        ("K* (o media K*)", BASELINE_CBOE["k_star"],        get(r_v3, "k_star"),      get(r_ens, "k_star_mean"), None),
        ("Base rate",       BASELINE_CBOE["base_rate"],     get(r_v3, "base_rate"),   get(r_ens, "base_rate"),   None),
        ("Shuffle AUC mean",cboe_shuffle_mean,              None,                     get(d_ens, "shuffle_auc_mean"), None),
        ("Shuffle AUC std", cboe_shuffle_std,               None,                     get(d_ens, "shuffle_auc_std"),  None),
    ]

    auc_v3  = get(r_v3,  "auc_mean")
    auc_ens = get(r_ens, "auc_mean")
    delta_v3  = (auc_v3  - BASELINE_CBOE["auc_mean"]) if auc_v3  is not None else None
    delta_ens = (auc_ens - BASELINE_CBOE["auc_mean"]) if auc_ens is not None else None

    # Print table
    col_w = 14
    print(f"\n{'='*80}")
    print("TABLA COMPARATIVA: CBOE 4f  vs  v3 7f (single)  vs  v3 Ensemble")
    print(f"{'='*80}")
    header = (
        f"{'Métrica':25s} | {'CBOE 4f':>{col_w}s} | {'CBOE 7f':>{col_w}s} | "
        f"{'Ensemble 7f':>{col_w}s} | {'Delta total':>{col_w}s}"
    )
    print(header)
    print("-" * len(header))

    csv_rows = []
    for label, v_cboe, v_v3, v_ens, _ in rows:
        def fmt(v):
            if v is None:
                return "?"
            if isinstance(v, float):
                if abs(v) < 10:
                    return f"{v:.4f}"
                return f"{v:.1f}"
            return str(v)

        delta_str = "?"
        if label == "AUC OOS mean" and delta_ens is not None:
            delta_str = f"{delta_ens:+.4f}"

        line = (
            f"{label:25s} | {fmt(v_cboe):>{col_w}s} | {fmt(v_v3):>{col_w}s} | "
            f"{fmt(v_ens):>{col_w}s} | {delta_str:>{col_w}s}"
        )
        print(line)
        csv_rows.append({
            "metric": label,
            "cboe_4f": v_cboe,
            "v3_7f_single": v_v3,
            "v3_7f_ensemble": v_ens,
            "delta_total": delta_ens if label == "AUC OOS mean" else None,
        })

    print(f"{'='*80}")

    # Verdict
    print("\n====== VEREDICTO FINAL ======")
    if delta_ens is None:
        print("No se puede calcular delta — run 16_run_ensemble_v3.py primero.")
    else:
        print(f"Delta AUC (Ensemble 7f - CBOE 4f): {delta_ens:+.4f}")
        if delta_ens > 0.03:
            verdict = "EXCELENTE — las mejoras superan expectativas (delta > +0.03)"
        elif delta_ens > 0.015:
            verdict = "BUENO — mejora significativa y en línea (+0.015 a +0.03)"
        elif delta_ens > 0.005:
            verdict = "MODESTO — mejora real pero menor (+0.005 a +0.015)"
        elif delta_ens > -0.005:
            verdict = "NEUTRO — las mejoras no aportan, revisar (-0.005 a +0.005)"
        else:
            verdict = "REGRESIÓN — algo mal, investigar (delta < -0.005)"
        print(f"[{verdict}]")

    if delta_v3 is not None:
        print(f"\nDesglose:")
        print(f"  Features nuevas (single):  {delta_v3:+.4f}  (CBOE 7f - CBOE 4f)")
        if delta_ens is not None:
            print(f"  Ensemble (vs single 7f):   {(delta_ens - delta_v3):+.4f}  (Ensemble - single 7f)")
    print()

    # Save CSV
    tab_dir.mkdir(parents=True, exist_ok=True)
    out_path = tab_dir / "comparison_cboe_vs_v3_vs_ensemble.csv"
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
