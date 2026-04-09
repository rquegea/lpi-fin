"""
Script 08: Compare 7-feature baseline vs 4-feature experiment.

Reads saved CSV results from scripts 03/04 (7 features) and 06/07 (4 features)
and prints a side-by-side comparison table with a verdict.

Verdict logic:
  delta = AUC_4feat - AUC_7feat
  > +0.005  → CONFIRMADO — modelo parsimonioso es superior
  ±0.005    → EQUIVALENTE — preferir 4 features por parsimonia (Occam)
  < -0.005  → INESPERADO — los features "malos" aportaban regularización
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# Hardcoded baseline from the 7-feature run (from run_log / memory)
BASELINE_7FEAT = {
    "auc_mean":          0.6342,
    "auc_std":           0.0102,
    "ap":                0.2451,
    "lift_q5":           1.51,
    "chi2_p":            0.0,
    "k_star":            9,
    "shuffle_auc_mean":  0.5007,
    "shuffle_auc_std":   0.0041,
    "stability_std":     0.0017,
}


def load_4feat_results(root: Path) -> dict:
    """Load 4-feature results from saved CSVs."""
    results = {}

    main_path = root / "results" / "tables" / "main_results_4features.csv"
    if not main_path.exists():
        print(f"Error: {main_path} not found. Run 06_run_4features.py first.")
        sys.exit(1)
    main_df = pd.read_csv(main_path)
    results["auc_mean"] = float(main_df["auc_mean"].iloc[0])
    results["auc_std"]  = float(main_df["auc_std"].iloc[0])
    results["ap"]       = float(main_df["ap"].iloc[0])
    results["lift_q5"]  = float(main_df["lift_q5"].iloc[0])
    results["chi2_p"]   = float(main_df["chi2_p"].iloc[0])
    results["k_star"]   = int(main_df["k_star"].iloc[0])

    diag_path = root / "results" / "tables" / "diagnostics_4features.csv"
    if not diag_path.exists():
        print(f"Error: {diag_path} not found. Run 07_diagnostics_4features.py first.")
        sys.exit(1)
    diag_df = pd.read_csv(diag_path)
    results["shuffle_auc_mean"] = float(diag_df["shuffle_auc_mean"].iloc[0])
    results["shuffle_auc_std"]  = float(diag_df["shuffle_auc_std"].iloc[0])
    results["stability_std"]    = float(diag_df["stability_std"].iloc[0])

    return results


def fmt(val, fmt_str=".4f"):
    if val == 0.0:
        return "~0"
    return format(val, fmt_str)


def main():
    root = Path(__file__).parent.parent

    b7 = BASELINE_7FEAT
    r4 = load_4feat_results(root)

    metrics = [
        ("AUC OOS mean",      "auc_mean",         ".4f"),
        ("AUC OOS std",       "auc_std",           ".4f"),
        ("AP OOS",            "ap",                ".4f"),
        ("Lift Q5",           "lift_q5",           ".2f"),
        ("Chi² p",            "chi2_p",            None ),
        ("K*",                "k_star",            ".0f"),
        ("Shuffle AUC mean",  "shuffle_auc_mean",  ".4f"),
        ("Shuffle AUC std",   "shuffle_auc_std",   ".4f"),
        ("Estabilidad std",   "stability_std",     ".4f"),
    ]

    col_w = [22, 12, 12, 10]
    header = (
        f"{'Métrica':<{col_w[0]}} {'7 features':>{col_w[1]}} "
        f"{'4 features':>{col_w[2]}} {'Delta':>{col_w[3]}}"
    )
    sep = "-" * (sum(col_w) + 3)

    rows = []
    table_lines = [
        "",
        "====== COMPARACIÓN: 7 features vs 4 features ======",
        header,
        sep,
    ]

    for label, key, fmt_str in metrics:
        v7 = b7[key]
        v4 = r4[key]
        if fmt_str is None:
            s7 = fmt(v7)
            s4 = fmt(v4)
            delta_str = "n/a"
        else:
            s7 = format(v7, fmt_str)
            s4 = format(v4, fmt_str)
            delta = v4 - v7 if isinstance(v4, float) else v4 - v7
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:{fmt_str}}"

        line = (
            f"{label:<{col_w[0]}} {s7:>{col_w[1]}} "
            f"{s4:>{col_w[2]}} {delta_str:>{col_w[3]}}"
        )
        table_lines.append(line)
        rows.append({
            "metric": label,
            "7_features": v7,
            "4_features": v4,
        })

    table_lines.append(sep)

    # Verdict
    delta_auc = r4["auc_mean"] - b7["auc_mean"]
    if delta_auc > 0.005:
        verdict = (
            "CONFIRMADO — modelo parsimonioso es superior. "
            "Adoptar 4 features como baseline."
        )
    elif delta_auc >= -0.005:
        verdict = (
            "EQUIVALENTE — preferir 4 features por parsimonia (Occam)."
        )
    else:
        verdict = (
            "INESPERADO — los features 'malos' aportaban regularización. "
            "Investigar por qué."
        )

    table_lines += [
        "",
        f"Delta AUC (4feat - 7feat): {'+' if delta_auc >= 0 else ''}{delta_auc:.4f}",
        "",
        f"====== VEREDICTO COMPARATIVO ======",
        f"[{verdict}]",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    report_text = "\n".join(table_lines)
    print(report_text)

    # Save CSV
    out_csv = root / "results" / "tables" / "comparison_7vs4.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nTabla guardada en: {out_csv}")

    # Append to log
    log_path = root / "results" / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(report_text + "\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
