# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
Script 17: Diagnostics for LPI Ensemble v3.

Runs:
  1. Shuffle test (12 permutations) on the ensemble
  2. Permutation importance (permute each feature, measure AUC drop)

Usage:
    python3.11 scripts/17_diagnostics_v3.py

Reads:
  - data/processed/panel_v3.parquet
  - data/processed/scores_oos_v3_ensemble.npy  (for AUC baseline)

Outputs:
  results/tables/diagnostics_v3_ensemble.csv
  results/tables/permutation_importance_v3.csv

Abort conditions:
  - Shuffle test: |mean(shuffle_auc) - 0.50| > 0.02
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lpi_ensemble import run_lpi_ensemble
from src.features_v2 import FEATURE_NAMES_V3
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BOOTSTRAP_B_DIAG = 12  # reduced for speed in diagnostics


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_v3.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_path = root / "data" / "processed" / "panel_v3.parquet"
    if not panel_path.exists():
        print("Error: panel_v3.parquet not found. Run 14_build_panel_v3.py first.")
        sys.exit(1)

    ensemble_scores_path = root / "data" / "processed" / "scores_oos_v3_ensemble.npy"
    if not ensemble_scores_path.exists():
        print("Error: scores_oos_v3_ensemble.npy not found. Run 16_run_ensemble_v3.py first.")
        sys.exit(1)

    logger.info("Loading v3 panel …")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_index(level="date")

    X = panel[FEATURE_NAMES_V3].values.astype(float)
    y = panel["target"].values.astype(float)

    scores_base = np.load(ensemble_scores_path)
    y_oos_base  = np.load(root / "data" / "processed" / "y_oos_v3_ensemble.npy")
    auc_base    = float(roc_auc_score(y_oos_base, scores_base))
    logger.info("AUC base (ensemble): %.4f", auc_base)

    seeds = cfg["ensemble_seeds"]
    n_models = len(seeds)

    # Config for diagnostics (reduced bootstrap_b)
    cfg_diag = dict(cfg, bootstrap_b=BOOTSTRAP_B_DIAG)
    logger.info("Diagnostics using bootstrap_b=%d", BOOTSTRAP_B_DIAG)

    # -----------------------------------------------------------------------
    # 1. Shuffle test (12 permutations)
    # -----------------------------------------------------------------------
    n_perm = cfg["shuffle_n_permutations"]
    rng = np.random.default_rng(42)
    shuffle_aucs = []

    print(f"\n{'='*60}")
    print(f"SHUFFLE TEST (ensemble, {n_perm} permutaciones)")
    print(f"{'='*60}")

    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        r = run_lpi_ensemble(X, y_shuf, cfg_diag, n_models=n_models, seeds=seeds)
        shuffle_aucs.append(r["auc_mean"])
        logger.info("Permutación %d/%d: AUC shuffle = %.4f", i + 1, n_perm, r["auc_mean"])

    shuffle_mean = float(np.mean(shuffle_aucs))
    shuffle_std  = float(np.std(shuffle_aucs))
    p_empirical  = float(np.mean(np.array(shuffle_aucs) >= auc_base))

    print(f"AUC real (ensemble):   {auc_base:.4f}")
    print(f"Shuffle AUC mean:      {shuffle_mean:.4f} ± {shuffle_std:.4f}")
    print(f"p empírico:            {p_empirical:.4f}")

    tolerance = cfg["shuffle_collapse_tolerance"]
    if abs(shuffle_mean - 0.50) > tolerance:
        print(
            f"\n⚠  CONDICIÓN DE PARADA: shuffle AUC mean = {shuffle_mean:.4f} "
            f"no colapsa a 0.50 ± {tolerance}. Investigar leakage potencial."
        )
        sys.exit(2)
    else:
        print(f"✓ Shuffle test OK: shuffle AUC = {shuffle_mean:.4f} ≈ 0.50 (sin señal en labels aleatorios)")

    diag_results = {
        "auc_real": auc_base,
        "shuffle_auc_mean": shuffle_mean,
        "shuffle_auc_std": shuffle_std,
        "n_permutations": n_perm,
        "p_empirical": p_empirical,
        "shuffle_aucs": str([f"{a:.4f}" for a in shuffle_aucs]),
        "bootstrap_b_used": BOOTSTRAP_B_DIAG,
    }
    tab_dir = root / "results" / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([diag_results]).to_csv(tab_dir / "diagnostics_v3_ensemble.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. Permutation importance
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE (ensemble, 1 permutación por feature)")
    print(f"{'='*60}")

    importance_rows = []
    rng2 = np.random.default_rng(0)

    for j, feat_name in enumerate(FEATURE_NAMES_V3):
        X_perm = X.copy()
        X_perm[:, j] = rng2.permutation(X_perm[:, j])
        r_perm = run_lpi_ensemble(X_perm, y, cfg_diag, n_models=n_models, seeds=seeds)
        auc_perm = r_perm["auc_mean"]
        delta = auc_base - auc_perm
        importance_rows.append({
            "feature": feat_name,
            "auc_base": auc_base,
            "auc_permuted": auc_perm,
            "delta_auc": delta,
        })
        logger.info("Feature '%s': AUC permutado=%.4f, delta=%.4f", feat_name, auc_perm, delta)

    imp_df = pd.DataFrame(importance_rows).sort_values("delta_auc", ascending=False)
    imp_df.to_csv(tab_dir / "permutation_importance_v3.csv", index=False)

    print(f"\n{'Feature':25s}  {'AUC sin feat':12s}  {'Delta AUC':10s}")
    print("-" * 50)
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:23s}  {row['auc_permuted']:.4f}        {row['delta_auc']:+.4f}")

    print(f"\n  AUC base (ensemble): {auc_base:.4f}")
    print(f"\nResultados guardados en: {tab_dir}")


if __name__ == "__main__":
    main()
