"""
streamlit_app.py
────────────────
Real-time dashboard for monitoring Multi-Armed Bandit experiments.

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import sqlite3
import os
import time

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH          = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bandit.db")
REFRESH_INTERVAL = 5   # seconds between auto-refresh

# True best-arm win rate for regret calculation.
# Update this to match TRUE_WIN_RATES in simulate.py.
BEST_ARM_RATE = 0.5

st.set_page_config(
    page_title="Bandit Dashboard",
    page_icon="🎰",
    layout="wide",
)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "arm", "reward", "experiment", "created_at"])
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM rewards ORDER BY id ASC", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "arm", "reward", "experiment", "created_at"])
    conn.close()
    return df


def load_experiments():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM experiments", conn)
        return df["name"].tolist()
    except Exception:
        return []
    finally:
        conn.close()


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("🎰 Multi-Armed Bandit Dashboard")
st.caption(f"Data source: `{DB_PATH}` · Auto-refreshes every {REFRESH_INTERVAL}s")

# Experiment selector
experiments    = load_experiments() or ["default"]
selected_exp   = st.selectbox("Experiment", experiments, index=0)

df = load_data()

if selected_exp != "all":
    if "experiment" in df.columns:
        df = df[df["experiment"] == selected_exp]

if df.empty:
    st.warning("No reward data found yet. Run a simulation or start sending rewards via the API.")
    st.code("python simulations/simulate.py", language="bash")
    st.stop()

# ─── KPI Row ─────────────────────────────────────────────────────────────────

total_pulls  = len(df)
total_reward = df["reward"].sum()
mean_reward  = df["reward"].mean()
best_arm     = df.groupby("arm")["reward"].mean().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pulls",  f"{total_pulls:,}")
col2.metric("Total Reward", f"{total_reward:.0f}")
col3.metric("Mean Reward",  f"{mean_reward:.3f}")
col4.metric("Leading Arm",  f"Arm {best_arm}")

st.divider()

# ─── Per-Arm Stats ────────────────────────────────────────────────────────────

arm_stats = (
    df.groupby("arm")
    .agg(pulls=("reward", "count"), total_reward=("reward", "sum"), mean_reward=("reward", "mean"))
    .reset_index()
)
arm_stats["traffic_share"] = arm_stats["pulls"] / arm_stats["pulls"].sum()
arm_stats["mean_reward"]   = arm_stats["mean_reward"].round(4)
arm_stats["traffic_share"] = arm_stats["traffic_share"].round(4)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Mean Reward per Arm")
    fig = px.bar(
        arm_stats,
        x="arm", y="mean_reward",
        color="arm",
        labels={"arm": "Arm", "mean_reward": "Mean Reward"},
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Traffic Share per Arm")
    fig2 = px.pie(
        arm_stats,
        values="traffic_share",
        names=arm_stats["arm"].astype(str).apply(lambda x: f"Arm {x}"),
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Time-Series ─────────────────────────────────────────────────────────────

st.subheader("Cumulative Reward Over Time")

df_sorted = df.copy().reset_index(drop=True)
df_sorted["cumulative_reward"] = df_sorted["reward"].cumsum()
df_sorted["pull_index"]        = df_sorted.index + 1

fig3 = px.line(
    df_sorted, x="pull_index", y="cumulative_reward",
    labels={"pull_index": "Pull #", "cumulative_reward": "Cumulative Reward"},
)
st.plotly_chart(fig3, use_container_width=True)

# ─── Regret Curve ─────────────────────────────────────────────────────────────

st.subheader(f"Cumulative Regret Over Time  (best arm rate = {BEST_ARM_RATE})")
st.caption(
    "Regret = what you could have earned always picking the best arm "
    "minus what you actually earned. Lower is better."
)

df_sorted["regret_step"]       = BEST_ARM_RATE - df_sorted["reward"]
df_sorted["cumulative_regret"] = df_sorted["regret_step"].cumsum()

fig_regret = px.line(
    df_sorted, x="pull_index", y="cumulative_regret",
    labels={"pull_index": "Pull #", "cumulative_regret": "Cumulative Regret"},
    color_discrete_sequence=["#e74c3c"],
)
st.plotly_chart(fig_regret, use_container_width=True)

# ─── Rolling Mean ─────────────────────────────────────────────────────────────

st.subheader("Rolling Mean Reward (window = 50)")

df_sorted["rolling_mean"] = df_sorted["reward"].rolling(50, min_periods=1).mean()

fig4 = px.line(
    df_sorted, x="pull_index", y="rolling_mean",
    labels={"pull_index": "Pull #", "rolling_mean": "Rolling Mean Reward"},
)
st.plotly_chart(fig4, use_container_width=True)

# ─── Raw Data ─────────────────────────────────────────────────────────────────

with st.expander("Raw Data Table"):
    st.dataframe(arm_stats, use_container_width=True)
    st.dataframe(df.tail(200), use_container_width=True)

# ─── Auto-Refresh ─────────────────────────────────────────────────────────────

st.caption(f"Auto-refreshing every {REFRESH_INTERVAL}s...")
time.sleep(REFRESH_INTERVAL)
st.rerun()