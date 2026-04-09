# EXPERIMENTO DESCARTADO (2026-04-09)
# Resultado: v3 single AUC 0.6015, v3 ensemble AUC 0.5785
# Peor que baseline CBOE 4-features (AUC 0.6253).
# Razón probable: maldición de dimensionalidad en GMM al pasar de
# 4 a 7 features. Se mantiene código como registro del intento.
# Modelo final del proyecto: CBOE 4-features. Ver scripts 09-13.

"""
Script 16: Run LPI Ensemble on the v3 panel (7 features, 7 seeds).

Usage:
    python3.11 scripts/16_run_ensemble_v3.py

Reads: data/processed/panel_v3.parquet
Protocol: same as single-seed LPI but averaged over 7 random seeds.
  - Bootstrap BIC: K ∈ {2..15}, B=20 (reduced to 12 if performance guard triggers)
  - PurgedTimeSeriesSplit: 5 folds, embargo 10d
  - GMM full-covariance

Outputs:
  results/tables/main_results_v3_ensemble.csv
  results/figures/*_v3_ensemble.png
  data/processed/scores_oos_v3_ensemble.npy

Abort conditions:
  - AUC ensemble > 0.72
Stop warnings (non-aborting):
  - AUC spread across seeds > 0.05 (unstable ensemble)
  - K* outside [3, 12] for any seed
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lpi_ensemble import run_lpi_ensemble
from src.features_v2 import FEATURE_NAMES_V3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BOOTSTRAP_B_ENSEMBLE = 12  # reduced from 20 for ensemble performance


def main():
    root = Path(__file__).parent.parent
    with open(root / "config_v3.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_path = root / "data" / "processed" / "panel_v3.parquet"
    if not panel_path.exists():
        print("Error: panel_v3.parquet not found. Run 14_build_panel_v3.py first.")
        sys.exit(1)

    logger.info("Loading v3 panel …")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_index(level="date")

    X = panel[FEATURE_NAMES_V3].values.astype(float)
    y = panel["target"].values.astype(float)

    seeds = cfg["ensemble_seeds"]
    n_models = len(seeds)
    logger.info("Panel: %d obs, %d features", len(X), X.shape[1])
    logger.info("Ensemble: %d seeds: %s", n_models, seeds)

    # Performance guard: reduce bootstrap_b for ensemble
    cfg_ensemble = dict(cfg)
    if cfg_ensemble["bootstrap_b"] >= 20:
        cfg_ensemble["bootstrap_b"] = BOOTSTRAP_B_ENSEMBLE
        msg = (
            f"Performance guard: bootstrap_b reduced from {cfg['bootstrap_b']} "
            f"to {BOOTSTRAP_B_ENSEMBLE} for ensemble run. "
            f"Single-model runs (scripts 15/17) use the original bootstrap_b={cfg['bootstrap_b']}."
        )
        logger.warning(msg)
        log_path = root / "results" / "run_log.txt"
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    logger.info("Running LPI Ensemble (bootstrap_b=%d) …", cfg_ensemble["bootstrap_b"])
    t0 = time.time()
    result = run_lpi_ensemble(X, y, cfg_ensemble, n_models=n_models, seeds=seeds)
    elapsed = time.time() - t0
    logger.info("Ensemble completed in %.1f seconds", elapsed)

    # Stop condition: AUC > 0.72
    if result["auc_mean"] > cfg["thresholds"]["suspicious_high"]:
        print(
            f"\n⚠  CONDICIÓN DE PARADA: AUC ensemble = {result['auc_mean']:.4f} > "
            f"{cfg['thresholds']['suspicious_high']}. Investigar antes de proceder."
        )
        sys.exit(1)

    # Warning: unstable ensemble (seed spread > 0.05)
    if result["auc_std"] > 0.05:
        logger.warning(
            "AUC spread entre semillas = %.4f > 0.05 — ensemble inestable. "
            "AUCs individuales: %s",
            result["auc_std"],
            [f"{a:.4f}" for a in result["per_seed_aucs"]],
        )

    # Warning: K* out of range for any seed
    k_star_list = result["k_star_list"]
    for i, k in enumerate(k_star_list):
        if not (3 <= k <= 12):
            logger.warning(
                "K* = %d (semilla %d) fuera de [3, 12]", k, seeds[i]
            )

    # Generate figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    from src.reporting import compute_quintile_stats, compute_chi2_quintiles, _verdict
    from datetime import datetime

    scores   = result["scores_oos"]
    y_oos    = result["y_oos"]
    base_rate = float(y_oos.mean())
    n_obs     = len(y_oos)
    ap        = float(average_precision_score(y_oos, scores))
    chi2_stat, chi2_p = compute_chi2_quintiles(scores, y_oos)
    quintile_df       = compute_quintile_stats(scores, y_oos)
    lift_q5           = float(quintile_df.loc[quintile_df["quintile"] == "Q5", "lift"].values[0])
    verdict_label, verdict_rec = _verdict(result["auc_mean"], cfg["thresholds"])

    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_oos, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="darkgreen",
            label=f"AUC = {result['auc_mean']:.4f} (ensemble, {n_models} seeds)")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"LPI Ensemble v3 ({n_models} seeds) — Curva ROC (OOS)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve_v3_ensemble.png", dpi=120)
    plt.close()

    # Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y_oos == 0], bins=50, alpha=0.5, label="y=0", density=True)
    ax.hist(scores[y_oos == 1], bins=50, alpha=0.5, label="y=1", density=True)
    ax.set_xlabel("LPI Score (averaged)")
    ax.set_title(f"Score Distribution — Ensemble v3 ({n_models} seeds)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution_v3_ensemble.png", dpi=120)
    plt.close()

    # Save results table
    ensemble_results = {
        "timestamp": datetime.now().isoformat(),
        "iv_source": "CBOE (VIX/VXN)",
        "features": str(FEATURE_NAMES_V3),
        "n_features": len(FEATURE_NAMES_V3),
        "n_models": n_models,
        "seeds": str(seeds),
        "bootstrap_b_used": cfg_ensemble["bootstrap_b"],
        "universe_size": panel.index.get_level_values("ticker").nunique(),
        "n_obs": n_obs,
        "base_rate": base_rate,
        "k_star_list": str(k_star_list),
        "k_star_mean": float(np.mean(k_star_list)),
        "auc_mean": result["auc_mean"],
        "auc_std": result["auc_std"],
        "per_seed_aucs": str([f"{a:.4f}" for a in result["per_seed_aucs"]]),
        "ap": ap,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "lift_q5": lift_q5,
        "elapsed_seconds": elapsed,
        "verdict": verdict_label,
    }
    pd.DataFrame([ensemble_results]).to_csv(tab_dir / "main_results_v3_ensemble.csv", index=False)
    quintile_df.to_csv(tab_dir / "quintile_stats_v3_ensemble.csv", index=False)

    np.save(root / "data" / "processed" / "scores_oos_v3_ensemble.npy", scores)
    np.save(root / "data" / "processed" / "y_oos_v3_ensemble.npy", y_oos)

    # Print report
    print(f"\n{'='*60}")
    print(f"====== LPI Ensemble v3 ({n_models} seeds) ======")
    print(f"Features: {FEATURE_NAMES_V3}")
    print(f"Seeds: {seeds}")
    print(f"bootstrap_b usado: {cfg_ensemble['bootstrap_b']}")
    print(f"Tiempo: {elapsed:.1f}s")
    print()
    print(f"AUC ensemble (scores promediados): {result['auc_mean']:.4f}")
    print(f"Spread AUC entre semillas (std):   {result['auc_std']:.4f}")
    print()
    print("AUCs individuales por semilla:")
    for seed, k, auc in zip(seeds, k_star_list, result["per_seed_aucs"]):
        print(f"  seed={seed:5d}  K*={k:2d}  AUC={auc:.4f}")
    print()
    print(f"AP OOS:    {ap:.4f}")
    print(f"Chi² p:    {chi2_p:.2e}")
    print(f"Lift Q5:   {lift_q5:.2f}x")
    print(f"Base rate: {base_rate*100:.1f}%")
    print()
    print("Lift por quintil:")
    for _, row in quintile_df.iterrows():
        print(
            f"  {row['quintile']}: {row['lift']:.2f}x  "
            f"(n={row['n']:,}, rate={row['event_rate']*100:.1f}%)"
        )
    print()
    print(f"====== VEREDICTO ======")
    print(f"[{verdict_label}]  {verdict_rec}")
    print(f"{'='*60}")
    print(f"\nResultados en: {root / 'results'}")


if __name__ == "__main__":
    main()
