"""Transparent, rule-based SEBI-compliance screen (the way an exchange surveillance desk starts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    # name: (weight, human-readable description)
    "R1_closed_window":   (3.0, "Market trade while the trading window was closed (SEBI PIT Sch. B)"),
    "R2_pre_announcement": (3.0, "Directionally profitable trade shortly before a price-moving announcement"),
    "R3_contra_trade":    (2.0, "Contra trade (opposite direction) within 6 months"),
    "R4_late_disclosure": (1.0, "Disclosure to company / exchange later than 2 trading days"),
    "R5_high_participation": (1.0, "Trade >10% of 20-day average daily volume"),
    "R6_recurring_partners": (2.0, "Co-trades with the same people across multiple stocks"),
}


def apply_rules(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    F["R1_closed_window"] = (F["in_closed_window"] == 1) & F["is_market"]
    F["R2_pre_announcement"] = (F["days_to_next_event"] <= 15) & (F["next_event_ar_signed"] > 0.04)
    F["R3_contra_trade"] = F["contra_trade"].fillna(0) == 1
    F["R4_late_disclosure"] = (F["late_company"] == 1) | (F["late_exchange"] == 1)
    F["R5_high_participation"] = F["adv_participation"] > 0.10
    F["R6_recurring_partners"] = F["repeat_partners"] >= 1
    total = sum(w for w, _ in RULES.values())
    score = np.zeros(len(F))
    reasons = [[] for _ in range(len(F))]
    for name, (w, desc) in RULES.items():
        hit = F[name].to_numpy()
        score += w * hit
        for i in np.where(hit)[0]:
            reasons[i].append(desc)
    F["rule_score"] = np.where(F["is_market"], score / total, 0.0)
    F["rule_reasons"] = ["; ".join(r) for r in reasons]
    return F
