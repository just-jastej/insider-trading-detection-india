#!/usr/bin/env python3
"""Robustness check: regenerate the synthetic world with several seeds and report mean +- std of metrics.
Writes outputs/multi_seed_metrics.csv. Does not touch data/ or the main outputs."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from itd.config import OUT_DIR
from itd.evaluate import _metrics
from itd.features import build_features, load_raw
from itd.models import score_market_trades
from itd.network import add_network_features
from itd.rules import apply_rules
from itd.synth import generate

rows = []
for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); generate(seed=seed, out_dir=td, verbose=False)
        F = build_features(load_raw(td)); F, _, _ = add_network_features(F); F = apply_rules(F)
        F, _ = score_market_trades(F, seed=seed)
        D = F[F.is_market].merge(pd.read_csv(td / "ground_truth_labels.csv"), on="disclosure_id")
        y = D.is_suspicious.values
        for name, col in [("Ensemble", "risk_score"), ("Isolation Forest", "score_iforest"), ("Autoencoder", "score_autoencoder"),
                          ("LOF", "score_lof"), ("Rules", "score_rules"), ("Baseline: trade size", "log_value"),
                          ("Baseline: post-trade CAR", "car_post20_pos")]:
            m = _metrics(y, D[col].values, name); m["seed"] = seed; m["n_injected"] = int(y.sum()); rows.append(m)
    print("seed", seed, "done", flush=True)
R = pd.DataFrame(rows)
S = R.groupby("model").agg(["mean", "std"]).drop(columns=["seed", "n_injected"], level=0)
S = S[[c for c in S.columns if c[0] in ("roc_auc", "pr_auc", "precision@50", "recall@top5%")]].round(3)
S.to_csv(OUT_DIR / "multi_seed_metrics.csv"); print(S.to_string())
