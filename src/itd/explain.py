"""Explainability: per-alert SHAP attributions for the Isolation Forest + readable reason text."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_DESCRIPTIONS, ML_FEATURES


def shap_for_iforest(iso, Z: np.ndarray) -> np.ndarray:
    """SHAP values oriented so that POSITIVE = pushes the trade towards 'more anomalous'."""
    import shap
    try:
        sv = shap.TreeExplainer(iso).shap_values(Z)
        return -np.asarray(sv)            # shap explains the (higher = more normal) score
    except Exception as exc:              # pragma: no cover - fallback keeps pipeline alive
        print(f"[explain] SHAP failed ({exc}); using robust z-score fallback")
        return np.abs(Z)


def build_shap_frame(art) -> pd.DataFrame:
    sv = shap_for_iforest(art["iso"], art["Z"])
    return pd.DataFrame(sv, columns=ML_FEATURES, index=art["ids"]).rename_axis("disclosure_id")


def top_reasons(shap_row: pd.Series, feat_row: pd.Series, k: int = 3) -> str:
    top = shap_row.sort_values(ascending=False).head(k)
    out = []
    for f, v in top.items():
        if v <= 0:
            continue
        val = feat_row.get(f, np.nan)
        out.append(f"{FEATURE_DESCRIPTIONS.get(f, f)} = {val:.3g}")
    return "; ".join(out)
