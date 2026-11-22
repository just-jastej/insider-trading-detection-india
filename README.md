# Insider Trading Detection for Indian Markets (SEBI PIT-style, synthetic data)

An end-to-end B.Tech minor project: generate **synthetic SEBI/BSE-style data**, engineer event-study and SEBI-rule features, detect suspicious
insider trades and tipping rings with unsupervised ML, explain every alert, and review them in a Streamlit dashboard.




## Headline results  (104 injected trades in ~2,880 open-market trades)
| | ROC-AUC | PR-AUC | Precision@50 |
|---|---|---|---|
| **Ensemble (IF + autoencoder + SEBI rules)** | 0.987 | 0.823 | 0.94 |
| Trade-size baseline | 0.734 | 0.067 | 0.02 |
| Post-trade-return baseline | 0.862 | 0.278 | 0.40 |

Over 8 seeds: PR-AUC 0.75 ± 0.04 (use this as the headline). All 3 injected tipping rings are recovered by the network analysis.

## What it does
```
synth.py        fictional companies, BSE-style prices, announcements, SEBI PIT Reg 7(2)-style disclosures
                + injected insider trading (single-insider, tipping rings, closed-window) and hard negatives
features.py     market-model abnormal returns (event study), SEBI window / contra-trade / late-disclosure, size & volume features
network.py      co-trading graph; recurring partners across stocks -> candidate rings
rules.py        transparent SEBI-compliance rule score
models.py       Isolation Forest, LOF, MLP autoencoder -> rank-averaged ensemble risk score
explain.py      SHAP (Isolation Forest) + readable reasons
evaluate.py     ROC/PR, precision@k, per-pattern recall, baselines, supervised reference
adapters.py     load real NSE/BSE downloads (unverified column mapping - see docs/real_data_adapter.md)
app/dashboard.py   overview, alert investigator (price chart + SHAP + rules), ring network, model performance
```

## Project layout
```
run_pipeline.py   REPORT.md   
src/itd/…         app/dashboard.py   tests/   experiments/multi_seed.py   docs/real_data_adapter.md
data/             generated CSVs (public-style files + hidden ground_truth_*)
outputs/          scored_trades.csv, shap_values.csv, alerts_top200.csv, rings.csv, metrics*.csv, figures/
```


## Notes on regulations used
* Disclosure threshold ₹10 lakh and 2-trading-day hops (person → company → exchange): SEBI PIT Reg. 7(2).
* Trading window closed from quarter-end until 48 hours after results; 6-month contra-trade restriction: Code of Conduct, Schedule B.
