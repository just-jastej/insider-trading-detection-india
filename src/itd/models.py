"""Unsupervised anomaly detectors + rule score -> ensemble risk score.

The detectors are fitted WITHOUT labels. Labels (synthetic ground truth) are used in
evaluate.py to measure how well the alerts recover the injected cases. One caveat: the
ensemble weights (LOF = 0) were chosen after inspecting those results, so ensemble
numbers on the synthetic data are mildly optimistic (see REPORT.md, Limitations).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from sklearn.exceptions import ConvergenceWarning

from .config import ENSEMBLE_WEIGHTS, SEED
from .features import ML_FEATURES

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _pct_rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def prepare_matrix(M: pd.DataFrame):
    X = M[ML_FEATURES].astype(float).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5, 95))
    Z = scaler.fit_transform(X)
    return X, np.clip(Z, -8, 8), scaler


def fit_isolation_forest(Z, seed=SEED):
    m = IsolationForest(n_estimators=300, max_samples=256, contamination="auto", random_state=seed).fit(Z)
    return m, -m.score_samples(Z)


def fit_lof(Z, k=35):
    m = LocalOutlierFactor(n_neighbors=k, novelty=False)
    m.fit(Z)
    return m, -m.negative_outlier_factor_


def fit_autoencoder(Z, base_scores, trim=0.90, seed=SEED):
    """Small MLP auto-encoder (sklearn). Trained on the 'most normal' 90% (trimmed by the
    isolation-forest score) so it does not learn to reconstruct the anomalies it should flag."""
    keep = base_scores <= np.quantile(base_scores, trim)
    d = Z.shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), activation="tanh", max_iter=600, random_state=seed,
                      early_stopping=True, n_iter_no_change=20, learning_rate_init=3e-3)
    ae.fit(Z[keep], Z[keep])
    err = ((ae.predict(Z) - Z) ** 2).mean(axis=1)
    return ae, err


def score_market_trades(F: pd.DataFrame, seed=SEED):
    """Adds model scores to market-mode disclosures; returns (F, artefacts)."""
    F = F.copy()
    M = F[F["is_market"]].copy()
    X, Z, scaler = prepare_matrix(M)
    iso, s_if = fit_isolation_forest(Z, seed)
    lof, s_lof = fit_lof(Z)
    ae, s_ae = fit_autoencoder(Z, s_if, seed=seed)
    M["score_iforest"], M["score_lof"], M["score_autoencoder"] = s_if, s_lof, s_ae
    M["score_rules"] = M["rule_score"].to_numpy()
    comp = {
        "iforest": _pct_rank(s_if), "lof": _pct_rank(s_lof),
        "autoencoder": _pct_rank(s_ae), "rules": _pct_rank(M["score_rules"].to_numpy()),
    }
    M["risk_score"] = sum(ENSEMBLE_WEIGHTS[k] * v for k, v in comp.items())
    M["risk_score"] = _pct_rank(M["risk_score"].to_numpy()) * 100
    M["risk_rank"] = rankdata(-M["risk_score"], method="min").astype(int)
    cols = ["score_iforest", "score_lof", "score_autoencoder", "score_rules", "risk_score", "risk_rank"]
    F = F.merge(M[["disclosure_id"] + cols], on="disclosure_id", how="left")
    art = dict(iso=iso, lof=lof, ae=ae, scaler=scaler, X=X, Z=Z, ids=M["disclosure_id"].to_numpy())
    return F, art
