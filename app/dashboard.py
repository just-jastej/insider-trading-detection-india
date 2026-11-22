"""Streamlit surveillance dashboard.   Run:  streamlit run app/dashboard.py"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from itd.calendar_utils import closed_window_matrix
from itd.features import FEATURE_DESCRIPTIONS
from itd.rules import RULES

OUT, DATA = ROOT / "outputs", ROOT / "data"
st.set_page_config(page_title="Insider Trading Surveillance (India)", page_icon="🔎", layout="wide")


@st.cache_data
def load():
    F = pd.read_csv(OUT / "scored_trades.csv", parse_dates=["date_from", "date_to", "intimation_date_company", "broadcast_date"])
    S = pd.read_csv(OUT / "shap_values.csv", index_col=0)
    rings = pd.read_csv(OUT / "rings.csv")
    prices = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
    ann = pd.read_csv(DATA / "announcements.csv", parse_dates=["date"])
    metrics = pd.read_csv(OUT / "metrics.csv", index_col=0) if (OUT / "metrics.csv").exists() else None
    labels = pd.read_csv(DATA / "ground_truth_labels.csv") if (DATA / "ground_truth_labels.csv").exists() else None
    return F, S, rings, prices, ann, metrics, labels


if not (OUT / "scored_trades.csv").exists():
    st.error("No outputs found. Run `python run_pipeline.py` first.")
    st.stop()

F, S, rings, prices, ann, metrics, labels = load()
M = F[F["is_market"]].sort_values("risk_rank")

st.sidebar.title("🔎 Insider Trading Surveillance")
st.sidebar.caption("SEBI PIT Reg. 7(2)-style disclosures · **synthetic data**")

# selected = option_menu(
#         menu_title=None,  # Hides the default title box
#         options=[
#             "Overview & alerts",
#             "Alert investigator",
#             "Insider rings (network)",
#             "Model performance",
#         ],
#         icons=["grid", "search", "diagram-3", "bar-chart"],  # Bootstrap icons
#         default_index=0,
#         styles={
#             "container": {"padding": "0!important", "background-color": "transparent"},
#             "icon": {"color": "#ff4b4b", "font-size": "16px"},
#             "nav-link": {
#                 "font-size": "14px",
#                 "text-align": "left",
#                 "margin": "4px 0px",
#                 "--hover-color": "#262730",
#             },
#             "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
#         }
#     )


page = st.sidebar.radio("View", ["Overview & alerts", "Alert investigator", "Insider rings (network)",
                                 "Model performance", 
                                 #  "About / data dictionary"
                                ])






# st.sidebar.warning("All companies, people and trades are fictional. Do not treat any alert as real-world evidence.")


def fmt_inr(x):
    return f"₹{x / 1e7:,.2f} Cr" if x >= 1e7 else f"₹{x / 1e5:,.1f} L"


# ------------------------------------------------------------------ Overview
if page == "Overview & alerts":
    st.title("Surveillance overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disclosures analysed", f"{len(F):,}")
    c2.metric("Open-market trades scored", f"{len(M):,}")
    thr = c3.slider("Alert threshold (risk percentile)", 80, 99, 95)
    alerts = M[M["risk_score"] >= thr]
    c4.metric("Alerts raised", f"{len(alerts):,}")

    f1, f2, f3 = st.columns(3)
    syms = f1.multiselect("Company", sorted(alerts["symbol"].unique()))
    cats = f2.multiselect("Category of person", sorted(alerts["category_of_person"].unique()))
    side = f3.selectbox("Direction", ["All", "Acquisition", "Disposal"])
    a = alerts
    if syms: a = a[a["symbol"].isin(syms)]
    if cats: a = a[a["category_of_person"].isin(cats)]
    if side != "All": a = a[a["transaction_type"] == side]

    show = a[["risk_rank", "disclosure_id", "date_to", "symbol", "person_name", "category_of_person", "transaction_type",
              "securities_value", "risk_score", "rule_reasons", "why_flagged"]].copy()
    show["securities_value"] = show["securities_value"].map(fmt_inr)
    show["risk_score"] = show["risk_score"].round(1)
    st.dataframe(show, width="stretch", hide_index=True, height=420)
    st.download_button("Download alerts (CSV)", a.to_csv(index=False), "alerts.csv", "text/csv")

    g1, g2 = st.columns(2)
    by_c = alerts.groupby("symbol").size().sort_values(ascending=False).head(15)
    g1.plotly_chart(go.Figure(go.Bar(x=by_c.values, y=by_c.index, orientation="h")).update_layout(
        title="Alerts by company (top 15)", yaxis=dict(autorange="reversed"), height=380), width="stretch")
    mth = alerts.set_index("date_to").resample("MS").size()
    g2.plotly_chart(go.Figure(go.Bar(x=mth.index, y=mth.values)).update_layout(title="Alerts over time", height=380),
                    width="stretch")

# ------------------------------------------------------------------ Investigator
elif page == "Alert investigator":
    st.title("Alert investigator")
    top = M.head(300)
    label = lambda r: f"#{r.risk_rank}  {r.symbol} · {r.person_name} · {r.transaction_type} · {r.date_to:%d-%b-%Y}"
    opts = {label(r): r.disclosure_id for r in top.itertuples()}
    did = opts[st.selectbox("Select an alert (ranked by risk)", list(opts))]
    r = F[F["disclosure_id"] == did].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{r.risk_score:.1f}", f"rank #{int(r.risk_rank)}")
    c2.metric("Trade", f"{r.transaction_type}", fmt_inr(r.securities_value))
    c3.metric("Insider", r.person_name, r.category_of_person)
    c4.metric("Post-trade 20d abnormal return", f"{r.car_post20:+.1%}")

    # price chart with trade, announcements and closed windows
    p = prices[prices["symbol"] == r.symbol].set_index("date")
    t0 = r.date_to
    win = p.loc[t0 - pd.Timedelta(days=120): t0 + pd.Timedelta(days=60)]
    fig = go.Figure(go.Scatter(x=win.index, y=win["close"], mode="lines", name="Close"))
    cm = closed_window_matrix(pd.DatetimeIndex(sorted(prices["date"].unique())), [r.symbol], ann[ann.symbol == r.symbol])[:, 0]
    cal = pd.Series(cm, index=pd.DatetimeIndex(sorted(prices["date"].unique()))).loc[win.index[0]: win.index[-1]]
    start = None
    for d, v in cal.items():
        if v and start is None: start = d
        if (not v or d == cal.index[-1]) and start is not None:
            fig.add_vrect(x0=start, x1=d, fillcolor="red", opacity=0.08, line_width=0)
            start = None
    for e in ann[(ann.symbol == r.symbol) & ann.date.between(win.index[0], win.index[-1])].itertuples():
        fig.add_vline(x=e.date, line_dash="dot", line_color="grey")
        fig.add_annotation(x=e.date, y=win["close"].max(), text=e.category[:18], textangle=-90, showarrow=False, font=dict(size=9))
    fig.add_trace(go.Scatter(x=[t0], y=[p.loc[t0, "close"]], mode="markers", name="Insider trade",
                             marker=dict(size=14, color="green" if r.side > 0 else "red",
                                         symbol="triangle-up" if r.side > 0 else "triangle-down")))
    fig.update_layout(title=f"{r.company_name} - price, announcements (dotted) and closed trading windows (red)", height=420)
    st.plotly_chart(fig, width="stretch")

    l, rt = st.columns(2)
    with l:
        st.subheader("Why was it flagged? (SHAP, Isolation Forest)")
        sv = S.loc[did].sort_values(ascending=False).head(8)
        figb = go.Figure(go.Bar(x=sv.values, y=[FEATURE_DESCRIPTIONS.get(i, i)[:60] for i in sv.index], orientation="h"))
        figb.update_layout(yaxis=dict(autorange="reversed"), height=360, xaxis_title="contribution to anomaly score")
        st.plotly_chart(figb, width="stretch")
    with rt:
        st.subheader("SEBI rule screen")
        for name, (w, desc) in RULES.items():
            st.write(("🔴 " if r[name] else "⚪ ") + desc + f"  *(weight {w:g})*")
        st.subheader("Model scores (percentile within all trades)")
        sc = M[["score_iforest", "score_lof", "score_autoencoder", "score_rules"]].rank(pct=True).loc[M["disclosure_id"] == did]
        st.dataframe((sc * 100).round(1).rename(columns=lambda c: c.replace("score_", "")), hide_index=True)

    st.subheader(f"{r.person_name}: all disclosures")
    hist = F[F["insider_id"] == r.insider_id].sort_values("date_to")[
        ["date_to", "symbol", "mode_of_acquisition", "transaction_type", "securities_qty", "securities_value",
         "car_post20", "risk_score"]]
    st.dataframe(hist, hide_index=True, width="stretch")
    if labels is not None:
        gt = labels[labels.disclosure_id == did].iloc[0]
        # with st.expander("🧪 Ground truth (available only because the data is synthetic)"):
        #    st.write(f"Injected insider trade: **{bool(gt.is_suspicious)}**  ·  type: `{gt.label_type}`  ·  pattern: `{gt.pattern_id}`")

# ------------------------------------------------------------------ Network
elif page == "Insider rings (network)":
    st.title("Co-trading network: tipping rings")
    st.caption("Edges join people who traded the same stock, same direction, within a few days of each other. "
               "People who do this in ≥2 *different* stocks are linked as 'recurring partners'.")
    if rings.empty:
        st.info("No rings of ≥3 recurring partners were found.")
    else:
        rs = rings.copy()
        rs["members_n"] = rs["n_members"]
        st.dataframe(rs[["ring_id", "n_members", "n_stocks", "n_cotrade_events", "stocks"]], hide_index=True, width="stretch")
        pick = st.selectbox("Inspect ring", rings["ring_id"])
        row = rings[rings.ring_id == pick].iloc[0]
        members = ast.literal_eval(row["members"])
        T = F[F["insider_id"].isin(members) & F["is_market"]]
        G = nx.Graph()
        for sym, g in T.groupby("symbol"):
            ids = g["insider_id"].unique()
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    w = G.get_edge_data(a, b, {"weight": 0})["weight"]
                    G.add_edge(a, b, weight=w + 1)
        pos = nx.spring_layout(G, seed=3)
        names = F.drop_duplicates("insider_id").set_index("insider_id")["person_name"]
        ex = [c for a, b in G.edges for c in (pos[a][0], pos[b][0], None)]
        ey = [c for a, b in G.edges for c in (pos[a][1], pos[b][1], None)]
        fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#999"), hoverinfo="none"),
                         go.Scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text",
                                    text=[names[n] for n in G], textposition="top center",
                                    marker=dict(size=[12 + 3 * G.degree(n) for n in G], color="crimson"))])
        fig.update_layout(height=480, showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig, width="stretch")
        st.write("**Trades by ring members**")
        st.dataframe(T.sort_values("date_to")[["date_to", "symbol", "person_name", "category_of_person", "transaction_type",
                                               "securities_value", "next_event_ar", "risk_rank"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ Performance
elif page == "Model performance":
    st.title("Model performance (on synthetic ground truth)")
    st.info("Because the data is synthetic, we KNOW which trades are injected insider trades, so we can measure how well "
            "the *unsupervised* pipeline recovers them. These numbers are an upper-end indication, not a real-world accuracy claim.")
    if metrics is not None:
        st.dataframe(metrics.style.format("{:.3f}"), width="stretch")
    for f in ("roc_pr.png", "precision_at_k.png"):
        if (OUT / "figures" / f).exists():
            st.image(str(OUT / "figures" / f))
    t = OUT / "metrics_by_label_type.csv"
    if t.exists():
        st.subheader("Which injected patterns land in the top-5% of alerts?")
        st.dataframe(pd.read_csv(t), hide_index=True, width="stretch")

# ------------------------------------------------------------------ About
# else:
#     st.title("About & data dictionary")
#     st.markdown("""
# **Pipeline**: SEBI-PIT-style disclosures + BSE-style prices + corporate announcements → event-study features
# (market-model abnormal returns) and SEBI-rule features (closed trading window, contra-trade, late disclosure)
# → co-trading network → Isolation Forest / autoencoder / LOF / rule score → rank-averaged risk score → SHAP explanations.

# **Important**: the data is fully synthetic. Real SEBI/NSE/BSE data can be plugged in via `docs/real_data_adapter.md`.
# """)
#     st.subheader("Model features")
#     st.dataframe(pd.DataFrame({"feature": list(FEATURE_DESCRIPTIONS), "meaning": list(FEATURE_DESCRIPTIONS.values())}),
#                  hide_index=True, width="stretch")
#     st.subheader("SEBI rule screen")
#     st.dataframe(pd.DataFrame([(k, w, d) for k, (w, d) in RULES.items()], columns=["rule", "weight", "description"]),
#                  hide_index=True, width="stretch")
