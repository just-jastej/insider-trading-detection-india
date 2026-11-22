"""Adapters to run the pipeline on REAL Indian data instead of the synthetic files.

IMPORTANT: column names below were written from memory of the NSE "Insider Trading Disclosures
(PIT)" CSV export and have NOT been verified against a live download. The matcher is deliberately
keyword-based and forgiving; after loading a real file, print `df.columns` and fix any column
that maps wrongly (see docs/real_data_adapter.md).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# target column -> list of keyword groups; the first source column containing ALL words of any group wins
NSE_PIT_MAP = {
    "symbol": [["symbol"]],
    "company_name": [["company"]],
    "person_name": [["name", "person"]],
    "category_of_person": [["category", "person"]],
    "prior_holding_qty": [["no", "security", "prior"], ["no.", "security", "prior"]],
    "prior_holding_pct": [["shareholding", "prior"]],
    "securities_qty": [["no", "securities", "acquired"], ["no.", "securities", "acquired"]],
    "securities_value": [["value", "security"]],
    "transaction_type": [["transaction", "type"], ["acquisition/disposal"]],
    "post_holding_qty": [["no", "security", "post"], ["no.", "security", "post"]],
    "post_holding_pct": [["post", "shareholding"], ["%", "post"]],
    "date_from": [["date", "allotment", "from"]],
    "date_to": [["date", "allotment", "to"]],
    "intimation_date_company": [["date", "intimation"], ["date", "initmation"]],
    "mode_of_acquisition": [["mode"]],
    "exchange": [["exchange"]],
    "broadcast_date": [["broadcast"]],
}

MODE_NORMALISE = {
    r"market purchase": "Market Purchase", r"market sale": "Market Sale", r"esop": "ESOP",
    r"off market": "Off Market", r"inter-?se": "Inter-se Transfer", r"gift": "Gift",
    r"pledge": "Pledge", r"preferential": "Preferential Allotment",
}


def _find(cols, groups):
    for g in groups:
        for c in cols:
            lc = c.lower()
            if all(w in lc for w in g):
                return c
    return None


def from_nse_pit_csv(path, symbol_to_scrip: dict | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = pd.DataFrame()
    for tgt, groups in NSE_PIT_MAP.items():
        src = _find(raw.columns, groups)
        out[tgt] = raw[src] if src else np.nan
    for c in ["securities_qty", "securities_value", "prior_holding_qty", "post_holding_qty",
              "prior_holding_pct", "post_holding_pct"]:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "").replace({"-": np.nan, "nan": np.nan}), errors="coerce")
    for c in ["date_from", "date_to", "intimation_date_company", "broadcast_date"]:
        out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True).dt.normalize()
    out["broadcast_date"] = out["broadcast_date"].fillna(out["intimation_date_company"])
    tt = out["transaction_type"].astype(str).str.lower()
    out["transaction_type"] = np.select([tt.str.contains("acqui|buy|purchase"), tt.str.contains("dispos|sell|sale"),
                                         tt.str.contains("pledge")], ["Acquisition", "Disposal", "Pledge Create"], "Other")
    mode = out["mode_of_acquisition"].astype(str).str.lower()
    norm = pd.Series("Other", index=out.index)
    for pat, name in MODE_NORMALISE.items():
        norm[mode.str.contains(pat, regex=True)] = name
    out["mode_of_acquisition"] = norm
    out["scrip_code"] = out["symbol"].map(symbol_to_scrip or {})
    out["insider_id"] = out["person_name"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True)\
        .map({n: f"INS{i:05d}" for i, n in enumerate(sorted(out["person_name"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).unique()))})
    out["securities_type"] = "Equity Shares"
    out["regulation"] = "7(2)"
    out = out.dropna(subset=["symbol", "date_to", "securities_qty", "securities_value"])
    out.insert(0, "disclosure_id", [f"PIT{i:06d}" for i in range(len(out))])
    return out.reset_index(drop=True)


def from_bse_bhavcopy(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate daily BSE bhavcopy frames (SC_CODE, SC_NAME, OPEN, HIGH, LOW, CLOSE, NO_OF_SHRS, NET_TURNOV, TRADING_DATE)."""
    df = pd.concat(frames, ignore_index=True)
    ren = {"SC_CODE": "scrip_code", "SC_NAME": "name", "OPEN": "open", "HIGH": "high", "LOW": "low",
           "CLOSE": "close", "NO_OF_SHRS": "volume", "NET_TURNOV": "turnover", "TRADING_DATE": "date"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["symbol"] = df["name"].astype(str).str.strip().str.upper().str.replace(r"\W+", "", regex=True).str[:10]
    return df[["date", "scrip_code", "symbol", "name", "open", "high", "low", "close", "volume", "turnover"]]
