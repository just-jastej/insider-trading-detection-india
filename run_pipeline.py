#!/usr/bin/env python3
"""End-to-end pipeline: synthetic data -> features -> network -> rules -> models -> SHAP -> evaluation.

Usage:
    python run_pipeline.py                 # generate data + run everything
    python run_pipeline.py --skip-synth    # re-use existing files in data/ (e.g. real data via the adapter)
    python run_pipeline.py --seed 7 --companies 80
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from itd.config import DATA_DIR, N_COMPANIES, OUT_DIR, SEED
from itd.evaluate import evaluate
from itd.explain import build_shap_frame, top_reasons
from itd.features import ML_FEATURES, build_features, load_raw
from itd.models import score_market_trades
from itd.network import add_network_features
from itd.rules import apply_rules
from itd.synth import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--companies", type=int, default=N_COMPANIES)
    ap.add_argument("--skip-synth", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    if not a.skip_synth:
        generate(seed=a.seed, n_companies=a.companies)
    raw = load_raw()
    print("[1/5] engineering features (event study, SEBI window, peers) ...")
    F = build_features(raw)
    print("[2/5] building co-trading network ...")
    F, G, rings = add_network_features(F)
    print("[3/5] applying SEBI rule screen ...")
    F = apply_rules(F)
    print("[4/5] fitting anomaly detectors + SHAP ...")
    F, art = score_market_trades(F, seed=a.seed)
    S = build_shap_frame(art)
    FM = F.set_index("disclosure_id")
    F["why_flagged"] = [top_reasons(S.loc[d], FM.loc[d]) if d in S.index else "" for d in F["disclosure_id"]]

    F.to_csv(OUT_DIR / "scored_trades.csv", index=False)
    S.to_csv(OUT_DIR / "shap_values.csv")
    rings.to_csv(OUT_DIR / "rings.csv", index=False)
    alerts = F[F["is_market"]].sort_values("risk_rank").head(200)
    alerts[["risk_rank", "disclosure_id", "symbol", "person_name", "category_of_person", "transaction_type",
            "securities_value", "date_to", "risk_score", "rule_reasons", "why_flagged"]].to_csv(OUT_DIR / "alerts_top200.csv", index=False)

    print("[5/5] evaluating against ground truth ...")
    gt = DATA_DIR / "ground_truth_labels.csv"
    if gt.exists():
        labels = pd.read_csv(gt)
        M, by_type = evaluate(F, labels, seed=a.seed)
        pd.set_option("display.width", 200)
        print("\n=== Detection quality (market-mode trades) ===")
        print(M.round(3).to_string())
        print("\n=== Share of each injected pattern inside the top-5% alerts ===")
        print(by_type.to_string())
    else:
        print("No ground-truth labels found - skipping evaluation (expected with real data).")
    print(f"\nRings found by network analysis: {len(rings)}")
    print(f"Done in {time.time() - t0:.1f}s. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
