"""Evaluation against the (synthetic) ground truth + comparison with simple baselines."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import FIG_DIR, OUT_DIR
from .features import ML_FEATURES


def _metrics(y, s, name):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(-s)
    res = dict(model=name, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
    for k in (25, 50, 100):
        res[f"precision@{k}"] = y[order[:k]].mean()
    top5 = order[: max(1, int(0.05 * len(y)))]
    res["recall@top5%"] = y[top5].sum() / max(y.sum(), 1)
    return res


def evaluate(F: pd.DataFrame, labels: pd.DataFrame, seed=42):
    D = F[F["is_market"]].merge(labels, on="disclosure_id")
    y = D["is_suspicious"].to_numpy()
    scores = {
        "Ensemble (final)": D["risk_score"],
        "Isolation Forest": D["score_iforest"],
        "Local Outlier Factor": D["score_lof"],
        "Autoencoder": D["score_autoencoder"],
        "SEBI rules only": D["score_rules"],
        "Baseline: trade size": D["log_value"],
        "Baseline: post-trade CAR": D["car_post20_pos"],
    }
    rows = [_metrics(y, s, n) for n, s in scores.items()]
    # supervised reference ("upper bound") with 5-fold CV - NOT used by the unsupervised pipeline
    X = D[ML_FEATURES].astype(float).fillna(0.0)
    oof = cross_val_predict(GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3),
                            X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    rows.append(_metrics(y, oof, "Supervised GBM (CV, reference)"))
    M = pd.DataFrame(rows).set_index("model")
    by_type = (D.assign(top5=D["risk_score"] >= D["risk_score"].quantile(0.95))
                 .groupby("label_type")["top5"].agg(["sum", "count"]).rename(columns={"sum": "flagged_top5pct", "count": "n"}))
    by_type["share_flagged"] = by_type["flagged_top5pct"] / by_type["n"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    M.round(4).to_csv(OUT_DIR / "metrics.csv")
    by_type.round(4).to_csv(OUT_DIR / "metrics_by_label_type.csv")
    (OUT_DIR / "metrics.json").write_text(json.dumps(M.round(4).to_dict(orient="index"), indent=2))
    _plots(D, y, scores, oof, M)
    return M, by_type


def _plots(D, y, scores, oof, M):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in ["Ensemble (final)", "Isolation Forest", "Local Outlier Factor", "Autoencoder", "SEBI rules only",
              "Baseline: trade size"]:
        fpr, tpr, _ = roc_curve(y, scores[n]); ax[0].plot(fpr, tpr, label=n, lw=2 if n.startswith("Ens") else 1)
        p, r, _ = precision_recall_curve(y, scores[n]); ax[1].plot(r, p, label=n, lw=2 if n.startswith("Ens") else 1)
    ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_pr.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    M[["precision@25", "precision@50", "precision@100"]].drop(index="Supervised GBM (CV, reference)").plot.barh(ax=ax)
    ax.set(title="Precision of top-K alerts (unsupervised methods)", xlabel="precision"); plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_at_k.png", dpi=150); plt.close()
