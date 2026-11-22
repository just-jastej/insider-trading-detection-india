"""Synthetic Indian-market data generator.

Produces four *public-style* files that mimic what an analyst would download from
NSE/BSE/SEBI, plus two *hidden ground-truth* files used only for evaluation:

  data/prices.csv                  BSE bhavcopy-style daily OHLCV per scrip
  data/index.csv                   broad market index (stand-in for Nifty/Sensex)
  data/announcements.csv           BSE corporate-announcement-style feed
  data/pit_disclosures.csv         SEBI PIT Reg 7(2) "Form C"-style insider disclosures
  data/ground_truth_labels.csv     (hidden) which disclosures are injected insider trades
  data/ground_truth_events.csv     (hidden) true event jumps / leaky-event flags

All companies, people and numbers are FICTIONAL. Any resemblance to real entities
is coincidental. The generator injects realistic insider-trading patterns
(pre-announcement trading, trading-window violations, cross-company tipping rings)
inside a much larger body of benign activity, including hard negatives.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_utils import closed_window_matrix
from .config import (DATA_DIR, END, MIN_DISCLOSURE_VALUE, N_COMPANIES,
                     QUARTER_ENDS, SEED, START)

SECTORS = {
    "Information Technology": ["Infotech", "Systems", "Digital", "Softech"],
    "Pharma & Healthcare": ["Pharma", "Lifesciences", "Labs", "Healthcare"],
    "Financial Services": ["Finserv", "Capital", "Credit", "Finance"],
    "FMCG": ["Foods", "Consumer", "Agro", "Beverages"],
    "Automobiles": ["Motors", "Autotech", "Components"],
    "Infrastructure": ["Infra", "Realty", "Constructions"],
    "Energy & Power": ["Power", "Energy", "Renewables"],
    "Metals & Mining": ["Steel", "Metals", "Alloys"],
}
PREFIX = ["Kaveri", "Sahyadri", "Deccan", "Malabar", "Vindhya", "Konark", "Himadri", "Tapti",
          "Nilgiri", "Aravali", "Godavari", "Hampi", "Narmada", "Chambal", "Satpura", "Lohit",
          "Mahanadi", "Sundar", "Trishul", "Vajra", "Garuda", "Meru", "Pallava", "Kakatiya",
          "Zanskar", "Bhagirath", "Shivalik", "Ambar", "Rudra", "Jalan"]
FIRST = ["Rajesh", "Sunita", "Amit", "Priya", "Vikram", "Anjali", "Rohit", "Neha", "Sanjay", "Kavita",
         "Arjun", "Meera", "Deepak", "Pooja", "Manoj", "Ritu", "Suresh", "Divya", "Karan", "Shweta",
         "Nikhil", "Swati", "Harish", "Lakshmi", "Ashok", "Geeta", "Pranav", "Ishita", "Vivek", "Radhika"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Mehta", "Gupta", "Kulkarni", "Joshi",
        "Singh", "Chauhan", "Bose", "Das", "Menon", "Rao", "Agarwal", "Malhotra", "Kapoor", "Bhatia",
        "Shetty", "Pillai", "Desai", "Saxena", "Khanna", "Banerjee", "Chopra", "Tiwari", "Mishra", "Jain"]

# category -> (n_range, holding % of shares range, mean trades per insider, median trade value INR)
CAT_PARAMS = {
    "Promoter":           ((1, 3), (4.0, 30.0), 3.0, 3.0e7),
    "Promoter Group":     ((1, 3), (1.0, 8.0),  3.0, 2.0e7),
    "Director":           ((2, 4), (0.01, 0.4), 4.5, 4.0e6),
    "KMP":                ((1, 3), (0.005, 0.1), 4.5, 3.0e6),
    "Designated Person":  ((3, 6), (0.0008, 0.05), 6.0, 2.5e6),
    "Immediate Relative": ((1, 3), (0.001, 0.08), 3.5, 2.5e6),
}
EVENT_CATS = {  # category: P(positive price reaction)
    "Acquisition": 0.75, "Large Order Win": 0.9, "Fund Raise (QIP/Preferential)": 0.3,
    "Regulatory Approval": 0.85, "Block Deal / Promoter Stake Sale": 0.2,
    "Credit Rating Action": 0.5, "Litigation Outcome": 0.5, "Open Offer / Buyback": 0.85,
}
MODES = ["Market Purchase", "Market Sale", "ESOP", "Off Market", "Inter-se Transfer",
         "Gift", "Pledge", "Preferential Allotment"]
MODE_P = [0.0, 0.0, 0.12, 0.06, 0.04, 0.02, 0.04, 0.02]  # remainder -> market


# --------------------------------------------------------------------------- helpers
def _make_companies(rng, n):
    combos = list(itertools.product(PREFIX, SECTORS.items()))
    rng.shuffle(combos)
    rows, used = [], set()
    for pre, (sector, sufs) in combos:
        suf = rng.choice(sufs)
        sym = (pre[:3] + suf[:3]).upper()
        name = f"{pre} {suf} Ltd"
        if sym in used or any(r["name"] == name for r in rows):
            continue
        used.add(sym)
        rows.append(dict(symbol=sym, name=name, sector=sector))
        if len(rows) == n:
            break
    df = pd.DataFrame(rows)
    df["scrip_code"] = rng.choice(np.arange(500000, 543999), size=len(df), replace=False)
    df["beta"] = rng.uniform(0.6, 1.5, len(df))
    df["alpha"] = rng.normal(0.0002, 0.0002, len(df))
    df["idio_vol"] = rng.uniform(0.011, 0.020, len(df))
    df["shares_out"] = rng.integers(5e7, 5e8, len(df))
    df["adv_frac"] = rng.uniform(0.0006, 0.003, len(df))
    df["p0"] = np.exp(rng.uniform(np.log(60), np.log(3000), len(df)))
    return df


def _index_returns(rng, T):
    vol = np.full(T, 0.0080)
    for _ in range(4):  # a few stress regimes
        s = rng.integers(60, T - 60)
        vol[s : s + rng.integers(15, 40)] *= rng.uniform(1.6, 2.4)
    return rng.normal(0.0004, 1.0, T) * vol


def _make_events(rng, companies, dates):
    T = len(dates)
    rows = []
    for c in companies.itertuples():
        for qe in QUARTER_ENDS:
            qe = pd.Timestamp(qe)
            lag = rng.integers(35, 59) if qe.month == 3 else rng.integers(25, 45)
            t = dates.searchsorted(qe + pd.Timedelta(days=int(lag)))
            if t >= T - 1:
                continue
            jump = float(np.clip(rng.normal(0, 0.035), -0.12, 0.12))
            rows.append(dict(symbol=c.symbol, tidx=int(t), category="Financial Results",
                             subject=f"Outcome of Board Meeting - Financial Results for quarter ended {qe:%d-%b-%Y}",
                             jump=jump))
        for _ in range(rng.poisson(3.2)):
            t = int(rng.integers(130, T - 25))
            cat = str(rng.choice(list(EVENT_CATS)))
            pos = rng.random() < EVENT_CATS[cat]
            mag = rng.uniform(0.01, 0.04) if rng.random() < 0.3 else rng.uniform(0.05, 0.14)
            rows.append(dict(symbol=c.symbol, tidx=t, category=cat,
                             subject=f"Announcement under Reg 30 (LODR): {cat}",
                             jump=mag if pos else -mag))
    ev = pd.DataFrame(rows).sort_values(["symbol", "tidx"]).reset_index(drop=True)
    ev["event_id"] = [f"EV{i:05d}" for i in range(len(ev))]
    ev["date"] = dates[ev["tidx"].values]
    big = ev["jump"].abs() >= 0.05
    ok = (ev["tidx"] >= 125) & (ev["tidx"] <= T - 25)
    ev["is_leaky"] = big & ok & (rng.random(len(ev)) < 0.20)
    ev["leak_sign"] = np.sign(ev["jump"]).astype(int)
    return ev


def _simulate_prices(rng, companies, dates, rm, events):
    T, N = len(dates), len(companies)
    close_mat = np.zeros((T, N))
    frames = []
    for i, c in enumerate(companies.itertuples()):
        eps = rng.standard_t(5, T) * c.idio_vol / np.sqrt(5 / 3)
        r = c.alpha + c.beta * rm + eps
        vmult = np.ones(T)
        for e in events[events.symbol == c.symbol].itertuples():
            t = int(e.tidx)
            r[t] += e.jump
            vmult[t] *= rng.uniform(1.6, 3.0)
            if t + 1 < T:
                r[t + 1] += 0.15 * e.jump
                vmult[t + 1] *= 1.3
            if e.is_leaky and rng.random() < 0.7:  # market impact of leaked trading
                lo = max(1, t - 10)
                r[lo:t] += e.leak_sign * 0.0015
                vmult[lo:t] *= 1.15
        r = np.clip(r, -0.2, 0.2)
        close = c.p0 * np.cumprod(1 + r)
        close_mat[:, i] = close
        prev = np.r_[c.p0, close[:-1]]
        opn = prev * (1 + rng.normal(0, 0.004, T))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, T)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, T)))
        base = c.shares_out * c.adv_frac
        vol = (base * np.exp(rng.normal(0, 0.3, T)) * vmult * (1 + 8 * np.abs(eps))).astype(np.int64)
        frames.append(pd.DataFrame({
            "date": dates, "scrip_code": c.scrip_code, "symbol": c.symbol, "name": c.name,
            "open": opn.round(2), "high": high.round(2), "low": low.round(2), "close": close.round(2),
            "volume": vol, "turnover": (vol * close).round(0)}))
    return pd.concat(frames, ignore_index=True), close_mat


# --------------------------------------------------------------------------- insiders & trades
class _People:
    def __init__(self, rng):
        self.rng, self.n, self.rows, self.names = rng, 0, {}, set()

    def new(self, category, symbol=None, relation_of=None):
        while True:
            nm = f"{self.rng.choice(FIRST)} {chr(65 + self.rng.integers(0, 26))}. {self.rng.choice(LAST)}"
            if nm not in self.names:
                self.names.add(nm)
                break
        self.n += 1
        iid = f"INS{self.n:05d}"
        self.rows[iid] = dict(insider_id=iid, person_name=nm, category=category, home_symbol=symbol,
                              relation_of=relation_of)
        return iid


def _pick_by_weight(rng, items, weights):
    w = np.asarray(weights, float)
    return items[rng.choice(len(items), p=w / w.sum())]


def _market_side_mode(side):
    return "Market Purchase" if side > 0 else "Market Sale"


def _build_trades(rng, companies, dates, close_mat, events, closed):
    T, N = close_mat.shape
    sym_i = {s: i for i, s in enumerate(companies.symbol)}
    people = _People(rng)
    insiders = {}       # symbol -> list of (iid, category)
    init_hold = {}      # (symbol, iid) -> shares
    for c in companies.itertuples():
        lst = []
        for cat, (nr, hr, _, _) in CAT_PARAMS.items():
            for _ in range(rng.integers(nr[0], nr[1] + 1)):
                iid = people.new(cat, c.symbol)
                lst.append((iid, cat))
                init_hold[(c.symbol, iid)] = int(c.shares_out * rng.uniform(*hr) / 100)
        insiders[c.symbol] = lst

    trades = []

    def add(symbol, iid, t, mode, side, qty=None, value=None, label="benign", pattern=None, informed=False):
        trades.append(dict(symbol=symbol, insider_id=iid, tidx=int(t), mode=mode, side=int(side), qty=qty,
                           value=value, label_type=label, pattern_id=pattern, informed=informed))

    # ---- (A) benign background activity
    for c in companies.itertuples():
        j = sym_i[c.symbol]
        for iid, cat in insiders[c.symbol]:
            _, _, rate, med = CAT_PARAMS[cat]
            for _ in range(rng.poisson(rate)):
                t = int(rng.integers(100, T - 5))
                u = rng.random()
                cum, mode = 0.0, None
                for m, p in zip(MODES, MODE_P):
                    cum += p
                    if u < cum:
                        mode = m
                        break
                lab = "benign"
                if mode is None:
                    side = 1 if rng.random() < (0.40 if cat.startswith("Promoter") else 0.45) else -1
                    mode = _market_side_mode(side)
                    if closed[t, j]:  # most people respect the window; a few lapse (compliance breach)
                        if rng.random() < 0.85:
                            for _try in range(30):
                                t = int(rng.integers(100, T - 5))
                                if not closed[t, j]:
                                    break
                        if closed[t, j]:
                            lab = "benign_window_lapse"
                else:
                    side = {"ESOP": 1, "Preferential Allotment": 1, "Pledge": 0}.get(mode, 1 if rng.random() < 0.5 else -1)
                val = float(np.clip(np.exp(rng.normal(np.log(med), 0.8)), MIN_DISCLOSURE_VALUE, 1.5e8))
                add(c.symbol, iid, t, mode, side, value=val,
                    label=lab if mode in ("Market Purchase", "Market Sale") else "benign_non_market")

    # ---- (B) hard negative: promoters "creeping acquisition" - steady monthly buying
    for c in companies.sample(6, random_state=int(rng.integers(1e6))).itertuples():
        j = sym_i[c.symbol]
        prom = [i for i, k in insiders[c.symbol] if k == "Promoter"][0]
        m_idx = pd.Series(np.arange(T), index=dates).groupby([dates.year, dates.month]).first().values
        for t in m_idx[6:]:
            if t < T - 5 and not closed[t, j]:
                add(c.symbol, prom, t, "Market Purchase", 1, value=float(rng.uniform(1.5e6, 4e6)),
                    label="benign_creeping_acquisition")

    # ---- (C) injected insider trading
    leaky = events[events.is_leaky].copy()
    res = leaky[leaky.category == "Financial Results"]
    mat = leaky[leaky.category != "Financial Results"].sample(frac=1, random_state=int(rng.integers(1e6)))
    n_ring_events = min(12, max(0, len(mat) // 3))
    ring_events, single_events = mat.iloc[:n_ring_events], mat.iloc[n_ring_events:]

    def informed_value():
        return float(np.clip(np.exp(rng.normal(np.log(1.2e7), 0.7)), 2e6, 8e7))

    # C1: single-insider pre-announcement trading
    wts = {"Director": 3, "KMP": 3, "Designated Person": 3, "Promoter": 1, "Immediate Relative": 1.5,
           "Promoter Group": 0.7}
    for e in single_events.itertuples():
        side = e.leak_sign
        pool = [(i, k) for i, k in insiders[e.symbol] if k in wts]
        if side < 0:  # selling needs holdings
            pool = [(i, k) for i, k in pool if init_hold[(e.symbol, i)] * close_mat[e.tidx - 3, sym_i[e.symbol]] > 3e6]
        if not pool:
            continue
        iid, _ = _pick_by_weight(rng, pool, [wts[k] for _, k in pool])
        for k in rng.choice(np.arange(2, 16), size=rng.integers(1, 4), replace=False):
            add(e.symbol, iid, e.tidx - int(k), _market_side_mode(side), side, value=informed_value(),
                label="insider_single", pattern=f"SINGLE-{e.event_id}", informed=True)

    # C2: cross-company tipping rings (the same small group trades ahead of several companies' news)
    n_rings = 3
    rings = []
    for r in range(n_rings):
        members = [people.new("Immediate Relative") for _ in range(2)] + [people.new("Connected Person") for _ in range(3)]
        rings.append(members)
    for k, e in enumerate(ring_events.itertuples()):
        r = k % n_rings
        members = rings[r]
        j = sym_i[e.symbol]
        base = e.tidx - int(rng.integers(5, 14))
        n_m = int(rng.integers(3, 6))
        for iid in rng.choice(members, size=n_m, replace=False):
            init_hold.setdefault((e.symbol, iid), int(5e7 / close_mat[max(base, 0), j]))
            add(e.symbol, str(iid), base + int(rng.integers(0, 5)), _market_side_mode(e.leak_sign), e.leak_sign,
                value=informed_value() * 0.6, label="insider_ring", pattern=f"RING-{r + 1}", informed=True)

    # C3: closed-window trading ahead of results
    for e in res.itertuples():
        side = e.leak_sign
        qe = max(pd.Timestamp(q) for q in QUARTER_ENDS if pd.Timestamp(q) < e.date)
        start = dates.searchsorted(qe + pd.Timedelta(days=1))
        lo, hi = max(start, e.tidx - 14), e.tidx - 1
        if lo > hi:
            continue
        pool = [(i, k) for i, k in insiders[e.symbol] if k in wts]
        if side < 0:
            pool = [(i, k) for i, k in pool if init_hold[(e.symbol, i)] * close_mat[e.tidx - 3, sym_i[e.symbol]] > 3e6]
        if not pool:
            continue
        iid, _ = _pick_by_weight(rng, pool, [wts[k] for _, k in pool])
        for t in rng.choice(np.arange(lo, hi + 1), size=min(rng.integers(1, 3), hi - lo + 1), replace=False):
            add(e.symbol, iid, t, _market_side_mode(side), side, value=informed_value(),
                label="insider_window_violation", pattern=f"WIN-{e.event_id}", informed=True)

    return pd.DataFrame(trades), people, insiders, init_hold


def _finalise_disclosures(rng, trades, people, companies, dates, close_mat, init_hold):
    T = len(dates)
    sym_i = {s: i for i, s in enumerate(companies.symbol)}
    comp = companies.set_index("symbol")
    persons = pd.DataFrame(people.rows.values()).set_index("insider_id")
    trades = trades.sort_values(["symbol", "insider_id", "tidx"]).reset_index(drop=True)
    hold = dict(init_hold)
    out = []
    for r in trades.itertuples():
        j = sym_i[r.symbol]
        px = float(close_mat[r.tidx, j] * (1 + rng.normal(0, 0.003)))
        if r.mode == "ESOP":
            px *= rng.uniform(0.3, 0.7)  # exercise price well below market
        qty = int(max(1, r.value / px))
        prior = hold.get((r.symbol, r.insider_id), 0)
        if r.side < 0:
            qty = min(qty, prior)
            if qty * px < MIN_DISCLOSURE_VALUE:
                continue
        elif r.side == 0:
            qty = min(max(qty, 1), prior) if prior > 0 else 0
            if qty == 0:
                continue
        post = prior + r.side * qty if r.side != 0 else prior
        hold[(r.symbol, r.insider_id)] = post
        ttype = {1: "Acquisition", -1: "Disposal"}.get(r.side, "Pledge Create")
        late_p = 0.20 if r.informed else 0.10
        d1 = int(rng.integers(3, 13)) if rng.random() < late_p else int(rng.choice([0, 1, 2], p=[.28, .5, .22]))
        d2 = int(rng.integers(3, 11)) if rng.random() < 0.10 else int(rng.choice([0, 1, 2], p=[.25, .55, .2]))
        t1 = min(r.tidx + d1, T - 1)
        t2 = min(t1 + d2, T - 1)
        shares_out = comp.loc[r.symbol, "shares_out"]
        out.append(dict(
            symbol=r.symbol, company_name=comp.loc[r.symbol, "name"], scrip_code=comp.loc[r.symbol, "scrip_code"],
            regulation="7(2) r/w 6(2)", person_name=persons.loc[r.insider_id, "person_name"], insider_id=r.insider_id,
            category_of_person=persons.loc[r.insider_id, "category"],
            securities_type="Equity Shares", prior_holding_qty=prior, prior_holding_pct=round(100 * prior / shares_out, 4),
            securities_qty=qty, securities_value=round(qty * px, 2), transaction_type=ttype,
            post_holding_qty=post, post_holding_pct=round(100 * post / shares_out, 4),
            date_from=dates[r.tidx], date_to=dates[r.tidx], intimation_date_company=dates[t1],
            mode_of_acquisition=r.mode, exchange=str(rng.choice(["NSE", "BSE", "NSE,BSE"], p=[.6, .3, .1])),
            broadcast_date=dates[t2],
            _label=int(r.informed), _label_type=r.label_type, _pattern=r.pattern_id))
    d = pd.DataFrame(out).sort_values(["date_from", "symbol"]).reset_index(drop=True)
    d.insert(0, "disclosure_id", [f"PIT{i:06d}" for i in range(len(d))])
    labels = d[["disclosure_id", "_label", "_label_type", "_pattern"]].rename(
        columns={"_label": "is_suspicious", "_label_type": "label_type", "_pattern": "pattern_id"})
    d = d.drop(columns=["_label", "_label_type", "_pattern"])
    return d, labels


# --------------------------------------------------------------------------- public entry point
def generate(seed: int = SEED, n_companies: int = N_COMPANIES, out_dir: Path = DATA_DIR, verbose: bool = True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)
    companies = _make_companies(rng, n_companies)
    rm = _index_returns(rng, len(dates))
    events = _make_events(rng, companies, dates)
    prices, close_mat = _simulate_prices(rng, companies, dates, rm, events)

    ann = events[["event_id", "date", "symbol", "category", "subject"]].copy()
    closed = closed_window_matrix(dates, companies.symbol, ann)
    trades, people, _, init_hold = _build_trades(rng, companies, dates, close_mat, events, closed)
    disc, labels = _finalise_disclosures(rng, trades, people, companies, dates, close_mat, init_hold)

    index = pd.DataFrame({"date": dates, "index_close": (18000 * np.cumprod(1 + rm)).round(2), "index_return": rm})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_dir / "prices.csv", index=False)
    index.to_csv(out_dir / "index.csv", index=False)
    ann.sort_values("date").to_csv(out_dir / "announcements.csv", index=False)
    disc.to_csv(out_dir / "pit_disclosures.csv", index=False)
    labels.to_csv(out_dir / "ground_truth_labels.csv", index=False)
    events.drop(columns=["tidx"]).to_csv(out_dir / "ground_truth_events.csv", index=False)
    companies[["symbol", "name", "scrip_code", "sector", "shares_out"]].to_csv(out_dir / "companies.csv", index=False)
    if verbose:
        print(f"[synth] {len(companies)} companies | {len(dates)} trading days | {len(prices):,} price rows")
        print(f"[synth] {len(ann):,} announcements | {len(disc):,} PIT disclosures | "
              f"{int(labels.is_suspicious.sum())} injected insider trades "
              f"({100 * labels.is_suspicious.mean():.2f}% of all disclosures)")
        print(labels.label_type.value_counts().to_string())
    return dict(prices=prices, index=index, announcements=ann, disclosures=disc, labels=labels,
                events=events, companies=companies)


if __name__ == "__main__":
    generate()
