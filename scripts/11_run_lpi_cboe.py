"""
Script 11: Run LPI on the CBOE IV panel.

Usage:
    python3.11 scripts/11_run_lpi_cboe.py

Reads: data/processed/panel_cboe.parquet
Protocol: identical to all previous runs
  - Bootstrap BIC: K ∈ {2..15}, B=20
  - PurgedTimeSeriesSplit: 5 folds, embargo 10d
  - GMM full-covariance

Outputs:
  results/tables/main_results_cboe.csv
  results/figures/*_cboe.png
  (does NOT overwrite proxy-IV results)

Abort conditions:
  - K* outside [3, 12]
  - AUC > 0.72
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lpi_core import fit_predict
from src.features_v2 import FEATURE_NAMES_V2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _generate_report_cboe(result, cfg, output_dir, universe_size):
    """Generate metrics, figures and print verdict for the CBOE run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    from sklearn.metrics import average_precision_score, roc_curve
    from sklearn.calibration import calibration_curve
    from src.reporting import (
        compute_quintile_stats, compute_chi2_quintiles, _verdict
    )

    scores   = result["scores_oos"]
    y        = result["y_oos"]
    k_star   = result["k_star"]
    auc_mean = result["auc_mean"]
    auc_std  = result["auc_std"]

    base_rate = float(y.mean())
    n_obs     = len(y)
    ap        = float(average_precision_score(y, scores))
    chi2_stat, chi2_p = compute_chi2_quintiles(scores, y)
    quintile_df       = compute_quintile_stats(scores, y)
    lift_q5           = float(quintile_df.loc[quintile_df["quintile"] == "Q5", "lift"].values[0])
    verdict_label, verdict_rec = _verdict(auc_mean, cfg["thresholds"])

    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {auc_mean:.4f} (CBOE IV)")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LPI (CBOE IV) — Curva ROC (OOS)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve_cboe.png", dpi=120)
    plt.close()

    # Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y == 0], bins=50, alpha=0.5, label="y=0", density=True)
    ax.hist(scores[y == 1], bins=50, alpha=0.5, label="y=1", density=True)
    ax.set_xlabel("LPI Score")
    ax.set_title("Score Distribution — CBOE IV")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution_cboe.png", dpi=120)
    plt.close()

    # Calibration
    fraction_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, fraction_pos, "s-", color="steelblue", label="LPI CBOE")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("Calibration — CBOE IV")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_cboe.png", dpi=120)
    plt.close()

    # Table
    main_results = {
        "timestamp": datetime.now().isoformat(),
        "iv_source": "CBOE (VIX/VXN)",
        "features": str(FEATURE_NAMES_V2),
        "universe_size": universe_size,
        "n_obs": n_obs,
        "base_rate": base_rate,
        "k_star": k_star,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "ap": ap,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "lift_q5": lift_q5,
        "verdict": verdict_label,
    }
    pd.DataFrame([main_results]).to_csv(tab_dir / "main_results_cboe.csv", index=False)
    quintile_df.to_csv(tab_dir / "quintile_stats_cboe.csv", index=False)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "====== LPI (CBOE IV: VIX/VXN) ======",
        "IV source: CBOE VIX (S&P500 tickers) / VXN (Nasdaq tickers)",
        "Limitation: sector-level IV (all tickers in same group share one IV series)",
        f"Universo: {universe_size} tickers, {cfg['fecha_inicio']} a {cfg['fecha_fin']}",
        f"N observaciones: {n_obs:,}",
        f"Base rate: {base_rate*100:.1f}%",
        f"K* (Bootstrap BIC): {k_star}",
        "",
        f"AUC OOS: {auc_mean:.4f} ± {auc_std:.4f}",
        f"AP OOS:  {ap:.4f}",
        f"Chi² p:  {chi2_p:.2e}",
        f"Lift Q5: {lift_q5:.2f}x",
        "",
        "Lift por quintil:",
    ]
    for _, row in quintile_df.iterrows():
        lines.append(
            f"  {row['quintile']}: {row['lift']:.2f}x  "
            f"(n={row['n']:,}, rate={row['event_rate']*100:.1f}%)"
        )
    lines += [
        "",
        "====== VEREDICTO ======",
        f"[{verdict_label}]",
        f"{verdict_rec}",
        "",
        f"Generado: {timestamp}",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    log_path = Path(output_dir) / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(report_text + "\n" + "=" * 60 + "\n")

    return main_results


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

    logger.info("Panel: %d obs, %d features", len(X), X.shape[1])
    logger.info("Base rate: %.3f", y.mean())

    logger.info("Running LPI (CBOE IV) …")
    result = fit_predict(X, y, cfg)
    k_star = result["k_star"]
    logger.info("K* = %d", k_star)

    if not (3 <= k_star <= 12):
        print(f"\n⚠  CONDICIÓN DE PARADA: K* = {k_star} fuera de [3, 12].")
        sys.exit(1)

    if result["auc_mean"] > cfg["thresholds"]["suspicious_high"]:
        print(
            f"\n⚠  CONDICIÓN DE PARADA: AUC = {result['auc_mean']:.4f} > "
            f"{cfg['thresholds']['suspicious_high']}. Investigar antes de proceder."
        )
        sys.exit(1)

    _generate_report_cboe(
        result=result,
        cfg=cfg,
        output_dir=str(root / "results"),
        universe_size=panel.index.get_level_values("ticker").nunique(),
    )

    np.save(root / "data" / "processed" / "scores_oos_cboe.npy", result["scores_oos"])
    np.save(root / "data" / "processed" / "y_oos_cboe.npy", result["y_oos"])
    np.save(root / "data" / "processed" / "k_star_cboe.npy", np.array([result["k_star"]]))

    print(f"\nResultados en: {root / 'results'}")


if __name__ == "__main__":
    main()
