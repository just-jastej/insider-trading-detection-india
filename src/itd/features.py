"""Feature engineering: event-study abnormal returns + SEBI-aware behavioural features.

Everything here is computable from PUBLIC data only (disclosures, prices, announcements).
The hidden ground-truth labels are never touched in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_utils import closed_window_matrix
from .config import (COTRADE_WINDOW_DAYS, CONTRA_WINDOW_DAYS, DATA_DIR, ESTIMATION_WINDOW,
                     MARKET_MODES)

DATE_COLS = ["date_from", "date_to", "intimation_date_company", "broadcast_date"]


def load_raw(data_dir=DATA_DIR) -> dict:
    d = {}
    d["prices"] = pd.read_csv(data_dir / "prices.csv", parse_dates=["date"])
    d["index"] = pd.read_csv(data_dir / "index.csv", parse_dates=["date"])
    d["announcements"] = pd.read_csv(data_dir / "announcements.csv", parse_dates=["date"])
    d["disclosures"] = pd.read_csv(data_dir / "pit_disclosures.csv", parse_dates=DATE_COLS)
    return d


def market_model_ar(r: np.ndarray, rm: np.ndarray, t: int, est=ESTIMATION_WINDOW) -> np.ndarray:
    """Abnormal returns AR_s = r_s - (alpha + beta * rm_s), market model fitted on [t-250, t-30)."""
    lo, hi = max(0, t - est[0]), max(2, t - est[1])
    x, y = rm[lo:hi], r[lo:hi]
    vx = x.var()
    beta = ((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 0 else 1.0
    alpha = y.mean() - beta * x.mean()
    return r - (alpha + beta * rm)


def build_features(raw: dict) -> pd.DataFrame:
    prices, index, ann, disc = raw["prices"], raw["index"], raw["announcements"], raw["disclosures"].copy()
    dates = pd.DatetimeIndex(index["date"])
    T = len(dates)
    symbols = sorted(prices["symbol"].unique())
    sym_i = {s: i for i, s in enumerate(symbols)}

    close = prices.pivot(index="date", columns="symbol", values="close")[symbols].reindex(dates)
    volume = prices.pivot(index="date", columns="symbol", values="volume")[symbols].reindex(dates)
    R = close.pct_change().fillna(0.0).to_numpy()
    V = volume.to_numpy(float)
    rm = index["index_return"].to_numpy()
    closed = closed_window_matrix(dates, symbols, ann)

    # trading-day indices
    d_idx = lambda s: dates.searchsorted(pd.DatetimeIndex(s))
    disc["t"] = d_idx(disc["date_to"])
    disc["t_company"] = d_idx(disc["intimation_date_company"])
    disc["t_exchange"] = d_idx(disc["broadcast_date"])
    disc["side"] = disc["transaction_type"].map({"Acquisition": 1, "Disposal": -1}).fillna(0).astype(int)
    disc["is_market"] = disc["mode_of_acquisition"].isin(MARKET_MODES)

    # announcements per symbol (material = anything; we use all for "next event" logic)
    ann = ann.copy()
    ann["t"] = d_idx(ann["date"])
    ann_by_sym = {s: g.sort_values("t")["t"].to_numpy() for s, g in ann.groupby("symbol")}

    feats = []
    for r in disc.itertuples():
        j, t = sym_i[r.symbol], int(r.t)
        ar = market_model_ar(R[:, j], rm, t)
        car_pre10 = ar[max(0, t - 10): t].sum()
        car_post5 = ar[t + 1: t + 6].sum()
        car_post20 = ar[t + 1: t + 21].sum()
        evs = ann_by_sym.get(r.symbol, np.array([], int))
        nxt = evs[(evs > t) & (evs <= t + 30)]
        if len(nxt):
            te = int(nxt[0])
            days_to_event = te - t
            event_ar = float(ar[te])
        else:
            days_to_event, event_ar = 60, 0.0
        adv20 = V[max(0, t - 20): t, j].mean() if t > 0 else V[0, j]
        base60 = V[max(0, t - 60): max(1, t - 1), j].mean()
        qty = float(r.securities_qty)
        feats.append(dict(
            disclosure_id=r.disclosure_id,
            car_pre10=car_pre10, car_post5=car_post5, car_post20=car_post20,
            days_to_next_event=days_to_event, next_event_ar=event_ar,
            adv_participation=min(qty / max(adv20, 1.0), 2.0),
            vol_ratio_day=V[t, j] / max(base60, 1.0),
            vol_ratio_pre5=V[max(0, t - 5): t, j].mean() / max(base60, 1.0) if t > 0 else 1.0,
            in_closed_window=int(closed[t, j]),
            delay_company=int(r.t_company - t),
            delay_exchange=int(r.t_exchange - r.t_company),
            holding_change=min(qty / max(float(r.prior_holding_qty), 1.0), 10.0),
            log_value=np.log(max(float(r.securities_value), 1.0)),
        ))
    F = disc.merge(pd.DataFrame(feats), on="disclosure_id")
    F["next_event_ar_signed"] = F["side"] * F["next_event_ar"]

    # one-sided ("profitable timing") versions: only favourable moves are suspicious
    for c in ("car_post5", "car_post20"):
        F[c + "_signed"] = F["side"] * F[c]
        F[c + "_pos"] = F[c + "_signed"].clip(lower=0)
    F["pre_event_alignment"] = F["next_event_ar_signed"].clip(lower=0)
    F["event_proximity"] = 1.0 / (1.0 + F["days_to_next_event"])
    F["runup_signed"] = F["side"] * F["car_pre10"]            # >0 = traded WITH leaked drift
    F["holding_change_log"] = np.log1p(F["holding_change"])

    M = F[F["is_market"]].copy()
    M = _add_peer_features(M)
    M = _add_insider_history(M)
    F = F.merge(M[["disclosure_id", "contra_trade", "n_cotraders", "value_vs_category_z",
                   "value_vs_own", "insider_trades_90d", "insider_hit_rate"]], on="disclosure_id", how="left")
    F["is_promoter_grp"] = F["category_of_person"].isin(["Promoter", "Promoter Group"]).astype(int)
    F["is_relative_or_connected"] = F["category_of_person"].isin(["Immediate Relative", "Connected Person"]).astype(int)
    F["is_director_kmp"] = F["category_of_person"].isin(["Director", "KMP"]).astype(int)
    F["late_company"] = (F["delay_company"] > 2).astype(int)
    F["late_exchange"] = (F["delay_exchange"] > 2).astype(int)
    return F.drop(columns=["t", "t_company", "t_exchange"])


def _add_peer_features(M: pd.DataFrame) -> pd.DataFrame:
    """Contra-trade flag (SEBI 6-month rule) and same-stock/same-direction co-trading."""
    M = M.sort_values("date_to").copy()
    M["contra_trade"] = 0
    for (_, _), g in M.groupby(["insider_id", "symbol"]):
        if len(g) < 2:
            continue
        for i, row in enumerate(g.itertuples()):
            prev = g.iloc[:i]
            prev = prev[(prev["side"] == -row.side) & ((row.date_to - prev["date_to"]).dt.days <= CONTRA_WINDOW_DAYS)]
            if len(prev):
                M.loc[M["disclosure_id"] == row.disclosure_id, "contra_trade"] = 1
    M["n_cotraders"] = 0
    for (_, side), g in M.groupby(["symbol", "side"]):
        t = g["t"].to_numpy()
        ids = g["insider_id"].to_numpy()
        for k, idx in enumerate(g.index):
            near = np.abs(t - t[k]) <= COTRADE_WINDOW_DAYS
            M.loc[idx, "n_cotraders"] = len(set(ids[near]) - {ids[k]})
    return M


def _add_insider_history(M: pd.DataFrame) -> pd.DataFrame:
    # value relative to category norm, and relative to the insider's own (leave-one-out) history
    cat_stats = M.groupby("category_of_person")["log_value"].agg(["mean", "std"]).fillna(1.0)
    M["value_vs_category_z"] = (M["log_value"] - M["category_of_person"].map(cat_stats["mean"])) / \
        M["category_of_person"].map(cat_stats["std"]).replace(0, 1.0)
    g = M.groupby("insider_id")["log_value"]
    n, s = g.transform("count"), g.transform("sum")
    loo_mean = (s - M["log_value"]) / (n - 1).replace(0, np.nan)
    M["value_vs_own"] = (M["log_value"] - loo_mean).fillna(0.0)
    # trailing 90-day activity count
    M = M.sort_values("date_to")
    cnt = []
    for _, grp in M.groupby("insider_id"):
        d = grp["date_to"].to_numpy()
        for i in range(len(d)):
            cnt.append((grp.index[i], int(((d[i] - d[:i]) / np.timedelta64(1, "D") <= 90).sum())))
    M["insider_trades_90d"] = pd.Series(dict(cnt))
    # leave-one-out "hit rate": share of the insider's OTHER trades that were followed by a favourable move
    hit = (M["car_post20_signed"] > 0.04).astype(float)
    hs = hit.groupby(M["insider_id"]).transform("sum")
    M["insider_hit_rate"] = (hs - hit) / (n - 1 + 2)   # +2 = smoothing toward 0
    return M


ML_FEATURES = [
    "side", "log_value", "value_vs_category_z", "value_vs_own", "holding_change_log", "adv_participation",
    "vol_ratio_day", "vol_ratio_pre5", "runup_signed", "car_post5_pos", "car_post20_pos",
    "pre_event_alignment", "event_proximity", "in_closed_window", "late_company", "late_exchange",
    "contra_trade", "n_cotraders", "insider_trades_90d", "insider_hit_rate",
    "is_promoter_grp", "is_relative_or_connected", "is_director_kmp",
    # network features are appended by network.add_network_features
    "n_cotrade_partners", "repeat_partners", "ring_size",
]

FEATURE_DESCRIPTIONS = {
    "side": "+1 buy / -1 sell",
    "log_value": "log of traded value (INR)",
    "value_vs_category_z": "trade size vs. norm for the insider's category (z-score)",
    "value_vs_own": "trade size vs. the insider's own history (log difference)",
    "holding_change_log": "log(1 + quantity / prior holding)",
    "adv_participation": "quantity as a fraction of 20-day average daily volume",
    "vol_ratio_day": "volume on trade day / 60-day average volume",
    "vol_ratio_pre5": "avg volume in the 5 days before trade / 60-day average",
    "runup_signed": "side x CAR[-10,-1] - trading WITH a pre-trade drift",
    "car_post5_pos": "favourable cumulative abnormal return over next 5 days (one-sided)",
    "car_post20_pos": "favourable cumulative abnormal return over next 20 days (one-sided)",
    "pre_event_alignment": "favourable abnormal return on the next company announcement (<=30 days)",
    "event_proximity": "1 / (1 + trading days to next announcement)",
    "in_closed_window": "trade placed while the SEBI trading window was closed",
    "late_company": "intimation to company later than 2 trading days",
    "late_exchange": "company's broadcast to exchange later than 2 trading days",
    "contra_trade": "opposite-direction trade within 6 months (SEBI contra-trade rule)",
    "n_cotraders": "other insiders trading same stock & direction within +-5 trading days",
    "insider_trades_90d": "insider's number of trades in previous 90 days",
    "insider_hit_rate": "share of insider's OTHER trades followed by a favourable move (leave-one-out)",
    "is_promoter_grp": "Promoter / Promoter Group",
    "is_relative_or_connected": "Immediate Relative / Connected Person",
    "is_director_kmp": "Director / KMP",
    "n_cotrade_partners": "distinct insiders this person has co-traded with (any stock)",
    "repeat_partners": "partners co-traded with in >=2 DIFFERENT stocks",
    "ring_size": "size of the recurring-partner network component",
}
