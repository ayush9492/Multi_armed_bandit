"""
streamlit_app.py  —  Multi-Armed Bandit Dashboard
Real-time monitoring + algorithm comparison.
"""

import sqlite3
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bandit.db"
)
REFRESH_INTERVAL = 5
TRUE_PROBS = [0.3, 0.55, 0.15]   # used for demo regret baseline

ALGO_COLORS = {
    "thompson_sampling": "#636EFA",
    "ucb":               "#00CC96",
    "epsilon_greedy":    "#EF553B",
    "demo":              "#AB63FA",
    "default":           "#FFA15A",
}

ALGO_LABELS = {
    "thompson_sampling": "Thompson Sampling",
    "ucb":               "UCB",
    "epsilon_greedy":    "ε-Greedy",
}

st.set_page_config(page_title="Bandit Dashboard", page_icon="🎰", layout="wide")


# ─── Demo Data ────────────────────────────────────────────────────────────────

@st.cache_data
def generate_demo_data():
    rng = np.random.default_rng(42)
    n, n_arms = 3000, 3

    def run_thompson():
        alpha, beta_ = np.ones(n_arms), np.ones(n_arms)
        rows = []
        for _ in range(n):
            arm = int(np.argmax(rng.beta(alpha, beta_)))
            r = int(rng.random() < TRUE_PROBS[arm])
            rows.append((arm, r)); alpha[arm] += r; beta_[arm] += 1 - r
        return rows

    def run_ucb():
        counts, totals = np.zeros(n_arms), np.zeros(n_arms)
        rows = []
        for t in range(1, n + 1):
            arm = (t - 1) if t <= n_arms else int(np.argmax(
                totals / (counts + 1e-9) + np.sqrt(2 * np.log(t) / (counts + 1e-9))
            ))
            r = int(rng.random() < TRUE_PROBS[arm])
            rows.append((arm, r)); counts[arm] += 1; totals[arm] += r
        return rows

    def run_eg(eps=0.1):
        counts, totals = np.zeros(n_arms), np.zeros(n_arms)
        rows = []
        for _ in range(n):
            arm = int(rng.integers(0, n_arms)) if rng.random() < eps else int(np.argmax(totals / (counts + 1e-9)))
            r = int(rng.random() < TRUE_PROBS[arm])
            rows.append((arm, r)); counts[arm] += 1; totals[arm] += r
        return rows

    dfs = []
    for name, algo, rows in [
        ("thompson_sampling", "thompson_sampling", run_thompson()),
        ("ucb", "ucb", run_ucb()),
        ("epsilon_greedy", "epsilon_greedy", run_eg()),
    ]:
        df = pd.DataFrame(rows, columns=["arm", "reward"])
        df["experiment"] = name
        df["algorithm"]  = algo
        df["pull_index"] = np.arange(1, n + 1)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ─── DB Loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_db():
    if not os.path.exists(DB_PATH):
        return None, None
    conn = sqlite3.connect(DB_PATH)
    try:
        rewards = pd.read_sql("SELECT * FROM rewards ORDER BY id ASC", conn)
        try:
            experiments = pd.read_sql("SELECT * FROM experiments", conn)
        except Exception:
            experiments = pd.DataFrame(columns=["name", "algorithm", "n_arms", "epsilon"])
    except Exception:
        rewards = pd.DataFrame(columns=["id", "arm", "reward", "experiment"])
        experiments = pd.DataFrame(columns=["name", "algorithm", "n_arms", "epsilon"])
    conn.close()
    return rewards, experiments


# ─── Helpers ─────────────────────────────────────────────────────────────────

def algo_label(name):
    if not isinstance(name, str):
        name = str(name) if name and name == name else "unknown"  # handle NaN
    # "default" experiment uses epsilon_greedy from config
    if name == "default":
        return "ε-Greedy (default)"
    return ALGO_LABELS.get(name, name.replace("_", " ").title())

def algo_color(name):
    return ALGO_COLORS.get(name, "#999999")

def kpi_row(df):
    total_pulls  = len(df)
    total_reward = df["reward"].sum()
    mean_reward  = df["reward"].mean()
    best_arm     = df.groupby("arm")["reward"].mean().idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pulls",  f"{total_pulls:,}")
    c2.metric("Total Reward", f"{total_reward:.0f}")
    c3.metric("Mean Reward",  f"{mean_reward:.3f}")
    c4.metric("Leading Arm",  f"Arm {best_arm}")


# ─── Section: Single Experiment ───────────────────────────────────────────────

def show_single_experiment(df, exp_name, algo_name=None):
    label = f"**Algorithm:** `{algo_label(algo_name)}`" if algo_name else ""
    if label:
        st.markdown(label)

    kpi_row(df)
    st.divider()

    arm_stats = (
        df.groupby("arm")
        .agg(pulls=("reward","count"), total_reward=("reward","sum"), mean_reward=("reward","mean"))
        .reset_index()
    )
    arm_stats["traffic_share"] = arm_stats["pulls"] / arm_stats["pulls"].sum()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Mean Reward per Arm")
        fig = px.bar(arm_stats, x="arm", y="mean_reward", color="arm",
                     labels={"arm":"Arm","mean_reward":"Mean Reward"},
                     color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Traffic Share per Arm")
        fig2 = px.pie(arm_stats, values="traffic_share",
                      names=arm_stats["arm"].astype(str).apply(lambda x: f"Arm {x}"),
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    df_s = df.copy().reset_index(drop=True)
    df_s["pull_index"] = df_s.index + 1
    df_s["cumulative_reward"] = df_s["reward"].cumsum()
    df_s["rolling_mean"] = df_s["reward"].rolling(50, min_periods=1).mean()

    st.subheader("Cumulative Reward Over Time")
    st.plotly_chart(px.line(df_s, x="pull_index", y="cumulative_reward",
        labels={"pull_index":"Pull #","cumulative_reward":"Cumulative Reward"}),
        use_container_width=True)

    st.subheader("Rolling Mean Reward (window=50)")
    st.plotly_chart(px.line(df_s, x="pull_index", y="rolling_mean",
        labels={"pull_index":"Pull #","rolling_mean":"Rolling Mean"}),
        use_container_width=True)

    best_possible = df_s["pull_index"] * max(TRUE_PROBS)
    df_s["regret"] = best_possible.values - df_s["cumulative_reward"]
    st.subheader("Cumulative Regret")
    st.plotly_chart(px.line(df_s, x="pull_index", y="regret",
        color_discrete_sequence=["#EF553B"],
        labels={"pull_index":"Pull #","regret":"Cumulative Regret"}),
        use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(arm_stats, use_container_width=True)
        st.dataframe(df.tail(200), use_container_width=True)


# ─── Section: Algorithm Comparison ───────────────────────────────────────────

def show_comparison(all_df, exp_meta):
    st.header("⚔️ Algorithm Comparison")
    st.caption("Side-by-side performance across all experiments")

    # Merge algorithm name in, fallback to experiment name if not found
    if exp_meta is not None and not exp_meta.empty:
        all_df = all_df.merge(exp_meta[["name","algorithm"]], left_on="experiment", right_on="name", how="left")
    if "algorithm" not in all_df.columns:
        all_df["algorithm"] = all_df["experiment"]
    else:
        all_df["algorithm"] = all_df["algorithm"].fillna(all_df["experiment"])

    experiments = all_df["experiment"].unique().tolist()

    if len(experiments) < 2:
        st.info("Run at least 2 named experiments with different algorithms to see a comparison.")
        return

    # ── Summary table ──
    summary_rows = []
    for exp in experiments:
        sub = all_df[all_df["experiment"] == exp]
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        arm_counts = sub.groupby("arm")["reward"].count()
        top_arm = arm_counts.idxmax()
        top_pct  = arm_counts[top_arm] / len(sub) * 100
        summary_rows.append({
            "Experiment":      exp,
            "Algorithm":       algo_label(algo),
            "Total Pulls":     len(sub),
            "Mean Reward":     round(sub["reward"].mean(), 4),
            "Total Reward":    int(sub["reward"].sum()),
            "Best Arm":        f"Arm {top_arm}",
            "Best Arm Traffic":f"{top_pct:.1f}%",
        })
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cumulative Reward comparison ──
    st.subheader("📈 Cumulative Reward")
    fig_cum = go.Figure()
    for exp in experiments:
        sub = all_df[all_df["experiment"] == exp].reset_index(drop=True)
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        sub["pull_index"] = sub.index + 1
        sub["cum_reward"]  = sub["reward"].cumsum()
        fig_cum.add_trace(go.Scatter(
            x=sub["pull_index"], y=sub["cum_reward"],
            name=algo_label(algo),
            line=dict(color=algo_color(algo), width=2),
        ))
    fig_cum.update_layout(
        xaxis_title="Pull #", yaxis_title="Cumulative Reward",
        legend_title="Algorithm", hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Cumulative Regret comparison ──
    st.subheader("📉 Cumulative Regret")
    fig_reg = go.Figure()
    for exp in experiments:
        sub = all_df[all_df["experiment"] == exp].reset_index(drop=True)
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        sub["pull_index"]  = sub.index + 1
        sub["cum_reward"]  = sub["reward"].cumsum()
        sub["regret"]      = sub["pull_index"] * max(TRUE_PROBS) - sub["cum_reward"]
        fig_reg.add_trace(go.Scatter(
            x=sub["pull_index"], y=sub["regret"],
            name=algo_label(algo),
            line=dict(color=algo_color(algo), width=2),
        ))
    fig_reg.update_layout(
        xaxis_title="Pull #", yaxis_title="Cumulative Regret",
        legend_title="Algorithm", hovermode="x unified",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # ── Rolling Mean comparison ──
    st.subheader("🔄 Rolling Mean Reward (window=50)")
    fig_roll = go.Figure()
    for exp in experiments:
        sub = all_df[all_df["experiment"] == exp].reset_index(drop=True)
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        sub["pull_index"]   = sub.index + 1
        sub["rolling_mean"] = sub["reward"].rolling(50, min_periods=1).mean()
        fig_roll.add_trace(go.Scatter(
            x=sub["pull_index"], y=sub["rolling_mean"],
            name=algo_label(algo),
            line=dict(color=algo_color(algo), width=2),
        ))
    fig_roll.update_layout(
        xaxis_title="Pull #", yaxis_title="Rolling Mean Reward",
        legend_title="Algorithm", hovermode="x unified",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # ── Traffic Share per arm per algo ──
    st.subheader("🥧 Traffic Share per Arm — by Algorithm")
    pie_cols = st.columns(len(experiments))
    for i, exp in enumerate(experiments):
        sub = all_df[all_df["experiment"] == exp]
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        arm_counts = sub.groupby("arm")["reward"].count().reset_index()
        arm_counts.columns = ["arm", "pulls"]
        arm_counts["arm_label"] = arm_counts["arm"].apply(lambda x: f"Arm {x}")
        with pie_cols[i]:
            st.markdown(f"**{algo_label(algo)}**")
            fig_p = px.pie(arm_counts, values="pulls", names="arm_label",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_p.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Mean reward per arm per algo ──
    st.subheader("🏆 Mean Reward per Arm — by Algorithm")
    bar_data = []
    for exp in experiments:
        sub = all_df[all_df["experiment"] == exp]
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else exp
        stats = sub.groupby("arm")["reward"].mean().reset_index()
        stats["Algorithm"] = algo_label(algo)
        bar_data.append(stats)
    bar_df = pd.concat(bar_data)
    fig_bar = px.bar(bar_df, x="arm", y="reward", color="Algorithm",
                     barmode="group",
                     labels={"arm":"Arm","reward":"Mean Reward"},
                     color_discrete_map={algo_label(k): v for k,v in ALGO_COLORS.items()})
    fig_bar.update_layout(xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

st.title("🎰 Multi-Armed Bandit Dashboard")

rewards_df, exp_df = load_db()
is_demo = rewards_df is None

if is_demo:
    st.info(
        "**Demo Mode** — No live database found. "
        "Showing simulated data for Thompson Sampling, UCB, and ε-Greedy "
        "(3 arms · 3,000 pulls each · true win rates: 30%, 55%, 15%).",
        icon="ℹ️",
    )
    all_df = generate_demo_data()
    exp_df  = pd.DataFrame([
        {"name":"thompson_sampling","algorithm":"thompson_sampling","n_arms":3,"epsilon":None},
        {"name":"ucb",              "algorithm":"ucb",              "n_arms":3,"epsilon":None},
        {"name":"epsilon_greedy",   "algorithm":"epsilon_greedy",   "n_arms":3,"epsilon":0.1},
    ])
else:
    st.caption(f"Data source: `{DB_PATH}` · Auto-refreshes every {REFRESH_INTERVAL}s")
    all_df = rewards_df
    if "algorithm" not in all_df.columns and exp_df is not None and not exp_df.empty:
        all_df = all_df.merge(exp_df[["name","algorithm"]], left_on="experiment", right_on="name", how="left")

experiments = all_df["experiment"].unique().tolist()

# ── Tabs ──
tab_compare, *tab_exps = st.tabs(
    ["⚔️ Algorithm Comparison"] + [f"🔬 {algo_label(e)}" for e in experiments]
)

with tab_compare:
    if len(experiments) < 2:
        st.info("Run at least 2 experiments with different algorithms to see a comparison.")
    else:
        show_comparison(all_df.copy(), exp_df)

for tab, exp_name in zip(tab_exps, experiments):
    with tab:
        sub = all_df[all_df["experiment"] == exp_name].copy()
        algo = None
        if exp_df is not None and not exp_df.empty and "algorithm" in exp_df.columns:
            row = exp_df[exp_df["name"] == exp_name]
            if not row.empty:
                algo = row.iloc[0]["algorithm"]
        show_single_experiment(sub, exp_name, algo)

# ── Auto refresh ──
if not is_demo:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
else:
    st.caption("Demo mode — refresh page to regenerate.")