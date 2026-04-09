"""
Script 06: Run LPI with the 4-feature subset (ablation-selected).

Features used: iv_rv_spread, log_range, vol_of_vol, log_dvol
Dropped (negative delta from ablation): iv_level, rv_20, momentum

Reads config_4features.yaml and the existing panel (data/processed/panel.parquet).
Does NOT rebuild the panel — the same observations are used, only the feature
columns are subsetted.

Saves results to:
  results/tables/main_results_4features.csv
  results/figures/*_4features.png
Does NOT overwrite 7-feature results.

Abort conditions:
  - K* outside [3, 12]
  - AUC > 0.72 (suspicious given simpler model)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lpi_core import fit_predict
from src.reporting import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).parent.parent
    cfg_path = root / "config_4features.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    active_features = cfg["active_features"]
    logger.info("Active features: %s", active_features)

    panel_path = root / "data" / "processed" / "panel.parquet"
    if not panel_path.exists():
        print("Error: panel.parquet not found. Run 02_build_panel.py first.")
        sys.exit(1)

    logger.info("Loading panel …")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_index(level="date")

    # Subset to active features only
    missing = [f for f in active_features if f not in panel.columns]
    if missing:
        print(f"Error: features not found in panel: {missing}")
        sys.exit(1)

    X = panel[active_features].values.astype(float)
    y = panel["target"].values.astype(float)

    logger.info(
        "Panel: %d observations, %d features (%s)",
        len(X), X.shape[1], active_features,
    )
    logger.info("Base rate: %.3f", y.mean())

    logger.info("Running LPI (4 features) …")
    result = fit_predict(X, y, cfg)

    k_star = result["k_star"]
    logger.info("K* = %d", k_star)

    if not (3 <= k_star <= 12):
        msg = (
            f"⚠  CONDICIÓN DE PARADA: K* = {k_star} fuera de [3, 12].\n"
            "Revisa los datos antes de continuar."
        )
        print(msg)
        sys.exit(1)

    auc_mean = result["auc_mean"]
    if auc_mean > cfg["thresholds"]["suspicious_high"]:
        msg = (
            f"⚠  CONDICIÓN DE PARADA: AUC = {auc_mean:.4f} > {cfg['thresholds']['suspicious_high']}.\n"
            "Sospechoso con solo 4 features. Ejecuta diagnósticos antes de proceder."
        )
        print(msg)
        sys.exit(1)

    # Generate report with suffixed output paths
    # Temporarily patch output filenames by using a custom output subdir approach
    report = _generate_report_4feat(
        result=result,
        cfg=cfg,
        output_dir=str(root / "results"),
        universe_size=panel.index.get_level_values("ticker").nunique(),
        active_features=active_features,
    )

    # Save scores for diagnostics script
    np.save(root / "data" / "processed" / "scores_oos_4feat.npy", result["scores_oos"])
    np.save(root / "data" / "processed" / "y_oos_4feat.npy", result["y_oos"])
    np.save(root / "data" / "processed" / "k_star_4feat.npy", np.array([result["k_star"]]))

    print(f"\nFiguras: {root / 'results' / 'figures'}/(*_4features.png)")
    print(f"Tabla:   {root / 'results' / 'tables' / 'main_results_4features.csv'}")


def _generate_report_4feat(result, cfg, output_dir, universe_size, active_features):
    """Variant of generate_report that writes to *_4features suffixed files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime
    from scipy.stats import chi2_contingency
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, roc_curve,
    )
    from sklearn.calibration import calibration_curve
    from src.reporting import compute_quintile_stats, compute_chi2_quintiles, _verdict

    scores = result["scores_oos"]
    y = result["y_oos"]
    k_star = result["k_star"]
    auc_mean = result["auc_mean"]
    auc_std = result["auc_std"]

    base_rate = float(y.mean())
    n_obs = len(y)
    ap = float(average_precision_score(y, scores))
    chi2_stat, chi2_p = compute_chi2_quintiles(scores, y)
    quintile_df = compute_quintile_stats(scores, y)
    lift_q5 = float(quintile_df.loc[quintile_df["quintile"] == "Q5", "lift"].values[0])
    verdict_label, verdict_rec = _verdict(auc_mean, cfg["thresholds"])

    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC = {auc_mean:.4f} (4 feat)")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LPI (4 features) — Curva ROC (OOS)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve_4features.png", dpi=120)
    plt.close()

    # Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y == 0], bins=50, alpha=0.5, label="y=0", density=True)
    ax.hist(scores[y == 1], bins=50, alpha=0.5, label="y=1", density=True)
    ax.set_xlabel("LPI Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribución del Score — 4 features")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution_4features.png", dpi=120)
    plt.close()

    # Calibration
    fraction_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, fraction_pos, "s-", color="darkorange", label="LPI 4feat")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Score medio predicho")
    ax.set_ylabel("Fracción positivos observados")
    ax.set_title("Calibration — 4 features")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_4features.png", dpi=120)
    plt.close()

    # Table
    main_results = {
        "timestamp": datetime.now().isoformat(),
        "features": str(active_features),
        "n_features": len(active_features),
        "universe_size": universe_size,
        "fecha_inicio": cfg["fecha_inicio"],
        "fecha_fin": cfg["fecha_fin"],
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
    pd.DataFrame([main_results]).to_csv(tab_dir / "main_results_4features.csv", index=False)
    quintile_df.to_csv(tab_dir / "quintile_stats_4features.csv", index=False)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "====== LPI (4 features: iv_rv_spread, log_range, vol_of_vol, log_dvol) ======",
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
        f"====== VEREDICTO ======",
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


if __name__ == "__main__":
    main()
