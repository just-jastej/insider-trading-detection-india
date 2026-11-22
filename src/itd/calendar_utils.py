from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUARTER_ENDS


def closed_window_matrix(dates: pd.DatetimeIndex, symbols, announcements: pd.DataFrame) -> np.ndarray:
    """Boolean matrix [T x N]: True where the insider trading window is CLOSED.

    Rule (SEBI PIT Regulations 2015, Schedule B): the window is closed from the end of
    every quarter until 48 hours (here: 2 trading days) after the financial results
    are declared. Only the announcements feed (public information) is used.
    """
    symbols = list(symbols)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    T = len(dates)
    M = np.zeros((T, len(symbols)), dtype=bool)
    res = announcements[announcements["category"] == "Financial Results"]
    for sym, g in res.groupby("symbol"):
        if sym not in sym_idx:
            continue
        j = sym_idx[sym]
        gd = pd.to_datetime(g["date"])
        for qe in QUARTER_ENDS:
            qe = pd.Timestamp(qe)
            start = dates.searchsorted(qe + pd.Timedelta(days=1))
            if start >= T:
                continue
            cand = gd[(gd > qe) & (gd <= qe + pd.Timedelta(days=75))]
            if len(cand) == 0:
                continue
            end = min(dates.searchsorted(cand.min()) + 2, T - 1)
            M[start : end + 1, j] = True
    return M


def shift_trading_days(dates: pd.DatetimeIndex, ts, n: int) -> pd.Timestamp:
    i = dates.searchsorted(pd.Timestamp(ts))
    return dates[min(max(i + n, 0), len(dates) - 1)]
