"""
Script 12: Diagnostics for the CBOE IV experiment.

Runs:
  1. Shuffle test (12 permutations)
  2. Multi-seed stability (seeds 42, 7, 123)

Abort conditions:
  - Shuffle test does not collapse to 0.50 ± 0.02
  - Any seed AUC differs from the others by > ±0.03
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics import shuffle_test, stability_test, compute_p_empirical
from src.features_v2 import FEATURE_NAMES_V2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_cboe.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_path = root / "data" / "processed" / "panel_cboe.parquet"
    if not panel_path.exists():
        print("Error: panel_cboe.parquet not found. Run 10_build_panel_cboe.py first.")
        sys.exit(1)

    logger.info("Loading CBOE panel …")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_index(level="date")
    X = panel[FEATURE_NAMES_V2].values.astype(float)
    y = panel["target"].values.astype(float)

    # Load real AUC from script 11
    auc_real = None
    scores_path = root / "data" / "processed" / "scores_oos_cboe.npy"
    if scores_path.exists():
        scores = np.load(scores_path)
        y_oos  = np.load(root / "data" / "processed" / "y_oos_cboe.npy")
        auc_real = float(roc_auc_score(y_oos, scores))
        logger.info("Real AUC CBOE (from 11_run_lpi_cboe.py): %.4f", auc_real)

    # 1. Shuffle test
    logger.info("Running shuffle test (%d perms, CBOE) …", cfg["shuffle_n_permutations"])
    shuf = shuffle_test(X, y, cfg)

    # 2. Stability
    logger.info("Running multi-seed stability (CBOE) …")
    stab = stability_test(X, y, cfg)

    # Abort: seed instability
    aucs = list(stab["auc_by_seed"].values())
    max_dev = max(abs(a - stab["auc_mean"]) for a in aucs)
    if max_dev > 0.03:
        print(
            f"\n⚠  CONDICIÓN DE PARADA: inestabilidad entre semillas.\n"
            f"Desviación máxima: {max_dev:.4f} > 0.03. AUCs: {stab['auc_by_seed']}"
        )
        sys.exit(1)

    p_emp = compute_p_empirical(
        auc_real if auc_real is not None else 0.0,
        shuf["auc_permuted"],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "====== DIAGNÓSTICOS (CBOE IV) ======",
        f"Features: {FEATURE_NAMES_V2}",
        f"IV source: CBOE VIX / VXN",
        "",
        f"Shuffle test: AUC permutado {shuf['auc_mean']:.4f} ± {shuf['auc_std']:.4f}  "
        f"(min={shuf['auc_min']:.4f}, max={shuf['auc_max']:.4f})",
        f"  p empírico: {p_emp:.3f}",
        f"  Leakage detectado: {'SÍ ⚠' if shuf['leakage_detected'] else 'NO ✓'}",
        "",
        "Estabilidad multi-semilla (CBOE):",
    ]
    for seed, auc in stab["auc_by_seed"].items():
        lines.append(f"  seed={seed}: AUC={auc:.4f}")
    lines += [
        f"  std = {stab['auc_std']:.4f}  ({'estable ✓' if stab['is_stable'] else 'inestable ⚠'})",
        "",
        f"Generado: {timestamp}",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    log_path = root / "results" / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(report_text + "\n" + "=" * 60 + "\n")

    # Save for comparison script
    pd.DataFrame([{
        "shuffle_auc_mean": shuf["auc_mean"],
        "shuffle_auc_std":  shuf["auc_std"],
        "stability_std":    stab["auc_std"],
        "leakage_detected": shuf["leakage_detected"],
    }]).to_csv(root / "results" / "tables" / "diagnostics_cboe.csv", index=False)

    if shuf["leakage_detected"]:
        print("\n⚠  STOP: leakage detectado en run CBOE.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
