"""Co-trading network: detect groups of insiders that repeatedly trade the same stock,
in the same direction, within a few days of each other - across different companies.

Nodes  = insiders (persons)
Edges  = pairs of persons with same-stock / same-direction market trades <= 5 trading days apart
Strong = pairs co-trading in >= 2 DIFFERENT stocks (a coincidence is unlikely; tipping rings recur)
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from .config import COTRADE_WINDOW_DAYS


def build_cotrade_graph(F: pd.DataFrame) -> nx.Graph:
    M = F[F["is_market"] & (F["side"] != 0)]
    pair_symbols = defaultdict(set)
    pair_count = defaultdict(int)
    for (sym, side), g in M.groupby(["symbol", "side"]):
        g = g.sort_values("date_to")
        rows = list(zip(g["insider_id"], g["date_to"]))
        for (a, da), (b, db) in itertools.combinations(rows, 2):
            if a != b and abs((db - da).days) <= COTRADE_WINDOW_DAYS * 1.5:
                key = tuple(sorted((a, b)))
                pair_symbols[key].add(sym)
                pair_count[key] += 1
    G = nx.Graph()
    for (a, b), syms in pair_symbols.items():
        G.add_edge(a, b, weight=pair_count[(a, b)], n_symbols=len(syms), symbols=sorted(syms))
    return G


def add_network_features(F: pd.DataFrame):
    """Returns (F_with_network_features, graph, ring_table)."""
    G = build_cotrade_graph(F)
    strong = nx.Graph([(a, b) for a, b, d in G.edges(data=True) if d["n_symbols"] >= 2])
    comp_size = {}
    rings = []
    for k, comp in enumerate(sorted(nx.connected_components(strong), key=len, reverse=True), start=1):
        for n in comp:
            comp_size[n] = len(comp)
        if len(comp) >= 3:
            sub = G.subgraph(comp)
            syms = sorted({s for _, _, d in sub.edges(data=True) for s in d["symbols"]})
            rings.append(dict(ring_id=f"RING-C{k}", n_members=len(comp), members=sorted(comp),
                              n_stocks=len(syms), stocks=syms,
                              n_cotrade_events=int(sum(d["weight"] for _, _, d in sub.edges(data=True)))))
    F = F.copy()
    F["n_cotrade_partners"] = F["insider_id"].map(lambda n: G.degree(n) if n in G else 0)
    F["repeat_partners"] = F["insider_id"].map(lambda n: strong.degree(n) if n in strong else 0)
    F["ring_size"] = F["insider_id"].map(lambda n: comp_size.get(n, 1))
    rings = pd.DataFrame(rings)
    return F, G, rings
