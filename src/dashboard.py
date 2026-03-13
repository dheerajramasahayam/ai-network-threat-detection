"""
dashboard.py
------------
Streamlit real-time monitoring dashboard for the AI-NIDS.

Features:
  • Live threat probability gauge per inspected flow
  • Rolling attack-rate timeline chart
  • Attack type breakdown (donut chart)
  • Inference speed distribution
  • Full alert log feed with export
  • REST API connection to src/api.py

Run with:
    streamlit run src/dashboard.py
    # or with custom API URL:
    API_URL=http://localhost:8000 streamlit run src/dashboard.py
"""

import os
import time
import random
import datetime
from collections import deque

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-NIDS | Network Threat Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
}
/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(8px);
}
/* Alert rows */
.attack-row { background: rgba(220,38,38,0.15); border-left: 3px solid #dc2626; }
.benign-row { background: rgba(22,163,74,0.10); border-left: 3px solid #16a34a; }
/* Header */
.dashboard-header {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-header { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

ATTACK_TYPES = ["DDoS", "PortScan", "Brute Force", "Bot", "Infiltration", "Web Attack"]

def _init_state():
    defaults = {
        "history":         deque(maxlen=200),
        "total":           0,
        "attacks":         0,
        "benign":          0,
        "attack_types":    {t: 0 for t in ATTACK_TYPES},
        "latencies":       deque(maxlen=100),
        "running":         False,
        "api_url":         os.environ.get("API_URL", "http://localhost:8000"),
        "threshold":       0.5,
        "sim_attack_rate": 0.25,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ──────────────────────────────────────────────────────────────────────────────
# Simulated flow generator (used when API is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_flow(attack_rate: float, threshold: float) -> dict:
    """Generate a synthetic detection result for demo purposes."""
    is_attack = random.random() < attack_rate
    attack_prob = (
        random.gauss(0.82, 0.12) if is_attack else random.gauss(0.08, 0.06)
    )
    attack_prob = max(0.0, min(1.0, attack_prob))
    label = "ATTACK" if attack_prob >= threshold else "BENIGN"
    atk_type = random.choice(ATTACK_TYPES) if label == "ATTACK" else None
    latency = abs(random.gauss(0.45, 0.15))
    src_ip = f"192.168.{random.randint(1,10)}.{random.randint(1,254)}"
    dst_port = random.choice([80, 443, 22, 3389, 8080, 53, random.randint(1024, 65535)])
    return {
        "ts":           datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "label":        label,
        "is_attack":    label == "ATTACK",
        "attack_prob":  round(attack_prob, 4),
        "confidence":   round(attack_prob if label == "ATTACK" else 1 - attack_prob, 4),
        "attack_type":  atk_type,
        "latency_ms":   round(latency, 3),
        "src_ip":       src_ip,
        "dst_port":     dst_port,
        "model":        "random_forest (demo)",
    }

def _try_api_flow(threshold: float) -> dict | None:
    """Call live API; return None if unavailable."""
    try:
        import requests
        rng = np.random.default_rng()
        features = rng.normal(size=41).tolist()
        resp = requests.post(
            f"{st.session_state.api_url}/detect",
            json={"features": features, "threshold": threshold},
            timeout=0.8,
        )
        if resp.status_code == 200:
            d = resp.json()
            d["ts"] = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            d["attack_type"] = random.choice(ATTACK_TYPES) if d["is_attack"] else None
            d["src_ip"] = f"10.0.{random.randint(0,5)}.{random.randint(1,254)}"
            d["dst_port"] = random.randint(1, 65535)
            d["model"] = d.get("model_type", "unknown")
            return d
    except Exception:
        pass
    return None

def _ingest_flow():
    """Ingest one flow (API if available, else simulation) and update state."""
    result = _try_api_flow(st.session_state.threshold) or \
             _simulate_flow(st.session_state.sim_attack_rate, st.session_state.threshold)

    st.session_state.history.appendleft(result)
    st.session_state.total += 1
    if result["is_attack"]:
        st.session_state.attacks += 1
        if result.get("attack_type"):
            st.session_state.attack_types[result["attack_type"]] += 1
    else:
        st.session_state.benign += 1
    st.session_state.latencies.append(result["latency_ms"])

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.threshold = st.slider("Detection threshold", 0.1, 0.9,
                                           float(st.session_state.threshold), 0.05)
    st.session_state.sim_attack_rate = st.slider("Simulated attack rate", 0.0, 1.0,
                                                  float(st.session_state.sim_attack_rate), 0.05)
    refresh_ms = st.selectbox("Refresh interval (ms)", [300, 500, 1000, 2000], index=1)

    col1, col2 = st.columns(2)
    if col1.button("▶ Start", use_container_width=True, type="primary"):
        st.session_state.running = True
    if col2.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False

    st.divider()
    if st.button("🗑 Clear history", use_container_width=True):
        for k in ["history", "total", "attacks", "benign", "latencies"]:
            st.session_state[k] = (deque(maxlen=200) if k in ("history", "latencies")
                                   else 0)
        st.session_state.attack_types = {t: 0 for t in ATTACK_TYPES}
    st.divider()
    st.markdown("### 📖 Docs")
    st.markdown("- [README](../README.md)")
    st.markdown("- [API `/docs`](http://localhost:8000/docs)")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="dashboard-header">🛡️ AI Network Intrusion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time threat monitoring · RandomForest + XGBoost + CNN ensemble</p>',
            unsafe_allow_html=True)

status_col, _ = st.columns([1, 4])
with status_col:
    if st.session_state.running:
        st.success("● MONITORING LIVE", icon="🟢")
    else:
        st.warning("○ PAUSED", icon="🟡")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# KPI metrics row
# ──────────────────────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
total      = st.session_state.total
attacks    = st.session_state.attacks
benign     = st.session_state.benign
attack_pct = round(attacks / total * 100, 1) if total else 0.0
avg_lat    = round(float(np.mean(list(st.session_state.latencies))), 3) if st.session_state.latencies else 0.0

m1.metric("Total Flows",     f"{total:,}")
m2.metric("Attacks Detected",f"{attacks:,}",    delta=None)
m3.metric("Benign Flows",    f"{benign:,}")
m4.metric("Attack Rate",     f"{attack_pct} %",  delta=None)
m5.metric("Avg Latency",     f"{avg_lat} ms")

# ──────────────────────────────────────────────────────────────────────────────
# Charts row
# ──────────────────────────────────────────────────────────────────────────────

chart_left, chart_right = st.columns([2, 1])

with chart_left:
    st.markdown("#### 📈 Attack Probability — Rolling 60 flows")
    history = list(st.session_state.history)
    if history:
        df_h = pd.DataFrame(history[:60][::-1])
        colors = ["#dc2626" if v else "#16a34a" for v in df_h["is_attack"]]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            y=df_h["attack_prob"], mode="lines+markers",
            marker=dict(color=colors, size=6),
            line=dict(color="#38bdf8", width=1.5),
            name="Attack Probability",
        ))
        fig_line.add_hline(y=st.session_state.threshold, line_dash="dash",
                           line_color="#f59e0b", annotation_text="Threshold")
        fig_line.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), yaxis=dict(range=[0, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Start monitoring to see live data.")

with chart_right:
    st.markdown("#### 🍩 Attack Types")
    atk_counts = st.session_state.attack_types
    labels = [k for k, v in atk_counts.items() if v > 0]
    values = [v for v in atk_counts.values() if v > 0]
    if values:
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            marker=dict(colors=px.colors.qualitative.Vivid),
        ))
        fig_pie.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), showlegend=True,
            legend=dict(orientation="v", x=1.0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No attacks yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Gauge — latest flow
# ──────────────────────────────────────────────────────────────────────────────

gauge_col, hist_col = st.columns([1, 2])

with gauge_col:
    st.markdown("#### 🎯 Latest Flow Threat Score")
    latest_prob = history[0]["attack_prob"] if history else 0.0
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_prob * 100,
        number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#38bdf8"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 50],  "color": "rgba(22,163,74,0.3)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.3)"},
                {"range": [75, 100],"color": "rgba(220,38,38,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "value": st.session_state.threshold * 100,
            },
        },
    ))
    fig_gauge.update_layout(
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with hist_col:
    st.markdown("#### ⚡ Inference Latency Distribution (ms)")
    lats = list(st.session_state.latencies)
    if lats:
        fig_hist = px.histogram(
            x=lats, nbins=30,
            color_discrete_sequence=["#818cf8"],
            labels={"x": "Latency (ms)", "count": "Flows"},
        )
        fig_hist.update_layout(
            height=220, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), showlegend=False,
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Waiting for flow data …")

# ──────────────────────────────────────────────────────────────────────────────
# Alert log
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("#### 🔔 Live Alert Feed")

alert_filter = st.radio("Show:", ["All", "Attacks only", "Benign only"],
                        horizontal=True, label_visibility="collapsed")

if history:
    log_rows = []
    for r in history[:100]:
        if alert_filter == "Attacks only" and not r["is_attack"]:
            continue
        if alert_filter == "Benign only" and r["is_attack"]:
            continue
        log_rows.append({
            "Time":        r["ts"],
            "Label":       r["label"],
            "Threat Prob": f"{r['attack_prob']:.4f}",
            "Confidence":  f"{r['confidence']:.4f}",
            "Attack Type": r.get("attack_type") or "—",
            "Src IP":      r.get("src_ip", "—"),
            "Dst Port":    r.get("dst_port", "—"),
            "Latency ms":  r.get("latency_ms", "—"),
        })

    if log_rows:
        df_log = pd.DataFrame(log_rows)
        st.dataframe(
            df_log.style.apply(
                lambda row: ["background-color: rgba(220,38,38,0.15)" if row["Label"] == "ATTACK"
                             else "background-color: rgba(22,163,74,0.08)"] * len(row),
                axis=1,
            ),
            use_container_width=True, height=320,
        )
        # CSV export
        csv = df_log.to_csv(index=False)
        st.download_button("⬇ Export CSV", csv, "nids_alerts.csv", "text/csv")
    else:
        st.info("No matching alerts.")
else:
    st.info("Start monitoring to see the alert feed.")

# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh loop
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.running:
    _ingest_flow()
    time.sleep(refresh_ms / 1000)
    st.rerun()
