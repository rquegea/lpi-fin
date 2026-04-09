"""
LPI Core — Latent Propensity Index algorithm.

This module is deliberately finance-free. It receives a feature matrix X,
a binary label vector y, and configuration, and returns out-of-sample
LPI scores together with diagnostics.

Algorithm (4 steps):
  1. Standardize features with RobustScaler (median, IQR).
  2. Select K* via Bootstrap BIC over GMM full-covariance:
       K ∈ {k_min..k_max}, B bootstraps, K* = argmin mean(BIC).
  3. For each CV fold:
       a. Fit GMM(K*) on X_train (unsupervised).
       b. Compute per-cluster enrichment: f_k = mean(y_train | cluster==k).
       c. For each test sample x: LPI(x) = Σ_k P(C_k|x) * f_k.
  4. Return scores, K*, per-fold AUC, concatenated OOS scores.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from src.cv import PurgedTimeSeriesSplit


def _bootstrap_bic(X_scaled: np.ndarray, cfg: dict, rng: np.random.Generator) -> int:
    """
    Select K* by minimising Bootstrap BIC.

    For each K in [k_min, k_max], fit B GMMs on random subsamples and
    record the BIC. K* = argmin mean(BIC).

    Parameters
    ----------
    X_scaled : ndarray of shape (n, p)
        Already standardised feature matrix.
    cfg : dict
        Must contain: k_min, k_max, bootstrap_b, bootstrap_subsample,
        gmm_reg_covar, gmm_max_iter, random_state.
    rng : np.random.Generator

    Returns
    -------
    k_star : int
    """
    k_min = cfg["k_min"]
    k_max = cfg["k_max"]
    B = cfg["bootstrap_b"]
    subsample = cfg["bootstrap_subsample"]
    reg = cfg["gmm_reg_covar"]
    max_iter = cfg["gmm_max_iter"]
    n = len(X_scaled)
    sub_n = min(n, subsample)

    mean_bics = {}
    for k in range(k_min, k_max + 1):
        bics = []
        for b in range(B):
            idx = rng.choice(n, size=sub_n, replace=False)
            X_sub = X_scaled[idx]
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=reg,
                    max_iter=max_iter,
                    random_state=int(rng.integers(0, 2**31)),
                )
                gmm.fit(X_sub)
                bics.append(gmm.bic(X_sub))
            except Exception:
                # If GMM fails to converge, skip this bootstrap replicate
                pass
        if bics:
            mean_bics[k] = float(np.mean(bics))

    if not mean_bics:
        return k_min  # fallback

    k_star = min(mean_bics, key=mean_bics.get)
    return k_star


def fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    groups=None,
) -> dict:
    """
    Run LPI end-to-end and return OOS scores + diagnostics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        Binary labels (0/1).
    cfg : dict
        Configuration dict. Required keys:
          n_folds, embargo_days, random_state,
          k_min, k_max, bootstrap_b, bootstrap_subsample,
          gmm_reg_covar, gmm_max_iter.
    groups : ignored (kept for API compatibility)

    Returns
    -------
    dict with keys:
      scores_oos : ndarray — LPI scores for all OOS samples (same order)
      y_oos      : ndarray — corresponding labels
      k_star     : int — selected number of clusters
      auc_folds  : list[float] — per-fold AUC
      auc_mean   : float
      auc_std    : float
      fold_sizes : list[tuple] — (n_train, n_test) per fold
    """
    rng = np.random.default_rng(cfg["random_state"])

    # Step 1: scale the full dataset first (scaler fitted per fold below)
    # NOTE: we scale inside the loop to avoid leakage from test to train.

    # Step 2: Bootstrap BIC on the whole training set
    # We use the full X to choose K* (this is done once, outside folds,
    # which is a common simplification; for strict purism one could do it
    # per fold, but it's expensive and K* is stable in practice).
    scaler_global = RobustScaler()
    X_scaled_global = scaler_global.fit_transform(X)
    k_star = _bootstrap_bic(X_scaled_global, cfg, rng)

    # Step 3: CV loop
    cv = PurgedTimeSeriesSplit(
        n_splits=cfg["n_folds"],
        embargo_days=cfg["embargo_days"],
    )

    scores_oos_list = []
    y_oos_list = []
    auc_folds = []
    fold_sizes = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Scale inside the fold (fit on train only)
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=k_star,
            covariance_type="full",
            reg_covar=cfg["gmm_reg_covar"],
            max_iter=cfg["gmm_max_iter"],
            random_state=int(rng.integers(0, 2**31)),
        )
        gmm.fit(X_tr_sc)

        # Compute per-cluster enrichment on training fold only
        train_cluster_labels = gmm.predict(X_tr_sc)
        f_k = np.zeros(k_star)
        for k in range(k_star):
            mask = train_cluster_labels == k
            if mask.sum() > 0:
                f_k[k] = y_train[mask].mean()
            else:
                f_k[k] = y_train.mean()  # fallback to base rate

        # Compute LPI scores for test set
        # P(C_k | x) via GMM posterior probabilities
        proba = gmm.predict_proba(X_te_sc)  # shape (n_test, k_star)
        scores = proba @ f_k               # shape (n_test,)

        scores_oos_list.append(scores)
        y_oos_list.append(y_test)

        if len(np.unique(y_test)) > 1:
            fold_auc = roc_auc_score(y_test, scores)
        else:
            fold_auc = float("nan")
        auc_folds.append(fold_auc)
        fold_sizes.append((len(train_idx), len(test_idx)))

    scores_oos = np.concatenate(scores_oos_list)
    y_oos = np.concatenate(y_oos_list)

    valid_aucs = [a for a in auc_folds if not np.isnan(a)]
    auc_mean = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    auc_std = float(np.std(valid_aucs)) if len(valid_aucs) > 1 else 0.0

    return {
        "scores_oos": scores_oos,
        "y_oos": y_oos,
        "k_star": k_star,
        "auc_folds": auc_folds,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "fold_sizes": fold_sizes,
    }
