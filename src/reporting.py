"""
Reporting: metrics, figures, verdict.

Computes all post-hoc statistics on OOS scores and labels,
generates figures, and prints/saves a timestamped verdict.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _verdict(auc: float, thresholds: dict) -> tuple:
    """Return (label, recommendation) based on AUC and config thresholds."""
    no_signal = thresholds["no_signal"]
    marginal = thresholds["marginal"]
    weak = thresholds["weak"]
    moderate = thresholds["moderate"]

    if auc < no_signal:
        return (
            "SIN SEÑAL",
            "La hipótesis se rechaza. Documentar honestamente y cerrar la línea.",
        )
    elif auc < marginal:
        return (
            "SEÑAL MARGINAL",
            "Estadísticamente puede ser real pero económicamente improbable. "
            "Documentar como inconcluyente.",
        )
    elif auc < weak:
        return (
            "SEÑAL DÉBIL",
            "Justifica iterar features y proxy de IV. No justifica backtest económico todavía.",
        )
    elif auc < moderate:
        return (
            "SEÑAL MODERADA",
            "Justifica pasar a Fase 2 (backtest económico de straddles con costes realistas).",
        )
    else:
        return (
            "SEÑAL FUERTE",
            "Aplicar Deflated Sharpe Ratio (López de Prado) y purged combinatorial CV "
            "antes de creérselo. Riesgo de overfitting alto.",
        )


def compute_quintile_stats(scores: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Compute lift and event rate per quintile (Q1=lowest, Q5=highest).

    Returns DataFrame with columns: quintile, n, event_rate, lift.
    """
    base_rate = y.mean()
    df = pd.DataFrame({"score": scores, "y": y})
    df["quintile"] = pd.qcut(df["score"], q=5, labels=[1, 2, 3, 4, 5])
    rows = []
    for q in [1, 2, 3, 4, 5]:
        sub = df[df["quintile"] == q]
        er = sub["y"].mean()
        lift = er / base_rate if base_rate > 0 else float("nan")
        rows.append({"quintile": f"Q{q}", "n": len(sub), "event_rate": er, "lift": lift})
    return pd.DataFrame(rows)


def compute_chi2_quintiles(scores: np.ndarray, y: np.ndarray) -> tuple:
    """Chi² test of independence between score quintile and label."""
    df = pd.DataFrame({"score": scores, "y": y.astype(int)})
    df["quintile"] = pd.qcut(df["score"], q=5, labels=False)
    contingency = pd.crosstab(df["quintile"], df["y"])
    chi2, p, dof, _ = chi2_contingency(contingency)
    return chi2, p


def generate_report(
    result: dict,
    cfg: dict,
    output_dir: str = "results",
    universe_size: int = 30,
) -> dict:
    """
    Generate all metrics, figures, and the verdict.

    Parameters
    ----------
    result : dict from lpi_core.fit_predict()
    cfg : config dict
    output_dir : root results directory
    universe_size : number of tickers used

    Returns
    -------
    dict with all computed metrics.
    """
    scores = result["scores_oos"]
    y = result["y_oos"]
    k_star = result["k_star"]
    auc_folds = result["auc_folds"]
    auc_mean = result["auc_mean"]
    auc_std = result["auc_std"]

    base_rate = float(y.mean())
    n_obs = len(y)

    # --- Metrics ---
    ap = float(average_precision_score(y, scores))
    chi2_stat, chi2_p = compute_chi2_quintiles(scores, y)
    quintile_df = compute_quintile_stats(scores, y)
    lift_q5 = float(quintile_df.loc[quintile_df["quintile"] == "Q5", "lift"].values[0])

    # --- Verdict ---
    verdict_label, verdict_rec = _verdict(auc_mean, cfg["thresholds"])

    # --- Figures ---
    fig_dir = Path(output_dir) / "figures"
    tab_dir = Path(output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_mean:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LPI — Curva ROC (OOS)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=120)
    plt.close()

    # Score distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[y == 0], bins=50, alpha=0.5, label="y=0", density=True)
    ax.hist(scores[y == 1], bins=50, alpha=0.5, label="y=1", density=True)
    ax.set_xlabel("LPI Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribución del Score por Clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution.png", dpi=120)
    plt.close()

    # Calibration plot
    fraction_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, fraction_pos, "s-", label="LPI")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Score medio predicho")
    ax.set_ylabel("Fracción de positivos observados")
    ax.set_title("Calibration Plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration.png", dpi=120)
    plt.close()

    # --- Tables ---
    main_results = {
        "timestamp": datetime.now().isoformat(),
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
    pd.DataFrame([main_results]).to_csv(tab_dir / "main_results.csv", index=False)
    quintile_df.to_csv(tab_dir / "quintile_stats.csv", index=False)

    # --- Console + log output ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"====== LPI sobre datos reales ======",
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
        lines.append(f"  {row['quintile']}: {row['lift']:.2f}x  (n={row['n']:,}, rate={row['event_rate']*100:.1f}%)")
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

    return {
        **main_results,
        "quintile_stats": quintile_df,
        "verdict_label": verdict_label,
        "verdict_rec": verdict_rec,
        "report_text": report_text,
    }


def print_diagnostics_report(
    shuffle_result: dict,
    ablation_df: pd.DataFrame,
    stability_result: dict,
    auc_real: float,
    output_dir: str = "results",
) -> None:
    """Print and append diagnostics summary to run_log.txt."""
    from src.diagnostics import compute_p_empirical

    p_emp = compute_p_empirical(auc_real, shuffle_result["auc_permuted"])

    # Most critical and most useless features
    most_critical = ablation_df.iloc[0]
    most_useless = ablation_df[ablation_df["delta_auc"] <= 0]
    useless_str = (
        ", ".join(most_useless["feature"].tolist())
        if not most_useless.empty
        else "ninguno (todos contribuyen)"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        "====== DIAGNÓSTICOS ======",
        f"Shuffle test: AUC permutado {shuffle_result['auc_mean']:.4f} ± {shuffle_result['auc_std']:.4f}  "
        f"(min={shuffle_result['auc_min']:.4f}, max={shuffle_result['auc_max']:.4f})",
        f"  p empírico: {p_emp:.3f}",
        f"  Leakage detectado: {'SÍ ⚠' if shuffle_result['leakage_detected'] else 'NO ✓'}",
        "",
        "Ablación (leave-one-feature-out):",
    ]
    for _, row in ablation_df.iterrows():
        marker = " ← más crítico" if row["feature"] == most_critical["feature"] else ""
        lines.append(
            f"  drop {row['feature']:15s}: AUC={row['auc_without']:.4f}  delta={row['delta_auc']:+.4f}{marker}"
        )
    lines += [
        "",
        f"Feature más crítico: {most_critical['feature']} (delta = {most_critical['delta_auc']:+.4f})",
        f"Features inútiles (delta ≤ 0): {useless_str}",
        "",
        "Estabilidad multi-semilla:",
    ]
    for seed, auc in stability_result["auc_by_seed"].items():
        lines.append(f"  seed={seed}: AUC={auc:.4f}")
    lines += [
        f"  std = {stability_result['auc_std']:.4f}  ({'estable ✓' if stability_result['is_stable'] else 'inestable ⚠'})",
        "",
        f"Generado: {timestamp}",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    log_path = Path(output_dir) / "run_log.txt"
    with open(log_path, "a") as f:
        f.write(report_text + "\n" + "=" * 60 + "\n")
