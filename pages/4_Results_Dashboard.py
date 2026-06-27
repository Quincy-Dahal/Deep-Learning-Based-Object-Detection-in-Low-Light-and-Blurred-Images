import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Results Dashboard", layout="wide")
st.title("Results Dashboard")

# ── Path setup ────────────────────────────────────────────────────────────────
# All CSVs stay in Results/ — where the backend scripts save them
YOLO_COCO    = "Results/yolo_coco_metrics.csv"
SSD_COCO     = "Results/ssd_coco_metrics.csv"
FRCNN_COCO   = "Results/frcnn_coco_metrics.csv"
YOLO_CUSTOM  = "Results/custom_detection_metrics.csv"
SSD_CUSTOM   = "Results/ssd_custom_detection_metrics.csv"
FRCNN_CUSTOM = "Results/frcnn_custom_detection_metrics.csv"
FINAL_CSV    = "Results/final_model_comparison.csv"
SUMMARY_CSV  = "Results/model_average_summary.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    yolo  = pd.merge(pd.read_csv(YOLO_COCO),  pd.read_csv(YOLO_CUSTOM),  on="run")
    ssd   = pd.merge(pd.read_csv(SSD_COCO),   pd.read_csv(SSD_CUSTOM),   on="run")
    frcnn = pd.merge(pd.read_csv(FRCNN_COCO), pd.read_csv(FRCNN_CUSTOM), on="run")
    final   = pd.read_csv(FINAL_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    return yolo, ssd, frcnn, final, summary

yolo_df, ssd_df, frcnn_df, final_df, summary_df = load_data()

# ── Condition ordering ────────────────────────────────────────────────────────
CONDITION_ORDER = [
    "normal",
    "low_light_mild", "low_light_medium", "low_light_severe",
    "blur_mild",      "blur_medium",      "blur_severe",
    "gamma_mild",     "gamma_medium",     "gamma_severe",
    "clahe_mild",     "clahe_medium",     "clahe_severe",
    "gamma_clahe_mild","gamma_clahe_medium","gamma_clahe_severe",
]
LABEL_MAP = {
    "normal":             "Normal",
    "low_light_mild":     "LL Mild",
    "low_light_medium":   "LL Medium",
    "low_light_severe":   "LL Severe",
    "blur_mild":          "Blur Mild",
    "blur_medium":        "Blur Medium",
    "blur_severe":        "Blur Severe",
    "gamma_mild":         "Gamma Mild",
    "gamma_medium":       "Gamma Medium",
    "gamma_severe":       "Gamma Severe",
    "clahe_mild":         "CLAHE Mild",
    "clahe_medium":       "CLAHE Medium",
    "clahe_severe":       "CLAHE Severe",
    "gamma_clahe_mild":   "G+C Mild",
    "gamma_clahe_medium": "G+C Medium",
    "gamma_clahe_severe": "G+C Severe",
}

MODELS   = ["YOLOv8", "SSD MobileNet", "Faster R-CNN"]
CONDITIONS = ["Normal", "Low-Light Severe", "Blur Severe", "CLAHE Severe"]

# ── Helper: plot individual model ─────────────────────────────────────────────
def plot_individual(df, model_name):
    df = df.copy()
    df["run"] = pd.Categorical(df["run"], categories=CONDITION_ORDER, ordered=True)
    df = df.sort_values("run")
    labels = [LABEL_MAP.get(r, r) for r in df["run"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(labels, df["AP_50"],    marker="o", label="AP@50")
    ax.plot(labels, df["F1_score"], marker="s", label="F1 Score")
    ax.set_title(f"{model_name} — AP@50 & F1 Score Across All Conditions")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=40)
    ax.legend()
    fig.tight_layout()
    return fig

# ── Helper: grouped bar chart ─────────────────────────────────────────────────
def plot_grouped_bar(pivot, title, ylabel):
    x = np.arange(len(CONDITIONS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODELS):
        vals = pivot[model].values
        bars = ax.bar(x + (i - 1) * width, vals, width=width, label=model)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=15)
    ax.legend()
    fig.tight_layout()
    return fig

# ── Helper: degradation line chart ───────────────────────────────────────────
def plot_degradation_line(df):
    deg_conds = ["Normal", "Low-Light Severe", "Blur Severe"]
    sub = df[df["Condition"].isin(deg_conds)].copy()
    sub["Condition"] = pd.Categorical(sub["Condition"], categories=deg_conds, ordered=True)
    sub = sub.sort_values(["model", "Condition"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        m_sub = sub[sub["model"] == model]
        ax.plot(m_sub["Condition"], m_sub["AP_50"], marker="o", label=model)
        for _, row in m_sub.iterrows():
            ax.text(row["Condition"], row["AP_50"] + 0.002,
                    f'{row["AP_50"]:.3f}', ha="center", va="bottom", fontsize=7)
    ax.set_title("Degradation Impact on AP@50")
    ax.set_xlabel("Condition")
    ax.set_ylabel("AP@50")
    ax.legend()
    fig.tight_layout()
    return fig

# ── Helper: average summary bar ───────────────────────────────────────────────
def plot_avg_summary(summary):
    summary = summary.copy()
    summary["Model"] = pd.Categorical(summary["Model"], categories=MODELS, ordered=True)
    summary = summary.sort_values("Model")

    x = np.arange(len(MODELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, summary["Avg_AP_50"], w, label="Avg AP@50")
    b2 = ax.bar(x + w / 2, summary["Avg_F1"],    w, label="Avg F1")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Average Summary by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Model"])
    ax.legend()
    fig.tight_layout()
    return fig

# =============================================================================
# SECTION 1 — Individual Model Graphs (1–3)
# =============================================================================
st.markdown("## Individual Model Performance")
st.caption("AP@50 and F1 Score across all 16 conditions for each model.")

tab_yolo, tab_ssd, tab_frcnn = st.tabs(["YOLOv8", "SSD MobileNet", "Faster R-CNN"])

with tab_yolo:
    st.pyplot(plot_individual(yolo_df, "YOLOv8"))

with tab_ssd:
    st.pyplot(plot_individual(ssd_df, "SSD MobileNet"))

with tab_frcnn:
    st.pyplot(plot_individual(frcnn_df, "Faster R-CNN"))

st.markdown("---")

# =============================================================================
# SECTION 2 — Cross-Model Comparison Graphs (4–6)
# =============================================================================
st.markdown("## Cross-Model Comparison")

# Prepare pivot tables
final_sorted = final_df.copy()
final_sorted["Condition"] = pd.Categorical(
    final_sorted["Condition"], categories=CONDITIONS, ordered=True
)
final_sorted["model"] = pd.Categorical(
    final_sorted["model"], categories=MODELS, ordered=True
)
final_sorted = final_sorted.sort_values(["Condition", "model"])

pivot_ap50 = final_sorted.pivot(index="Condition", columns="model", values="AP_50").loc[CONDITIONS, MODELS]
pivot_f1   = final_sorted.pivot(index="Condition", columns="model", values="F1_score").loc[CONDITIONS, MODELS]

col1, col2 = st.columns(2)
with col1:
    st.subheader("AP@50 Comparison")
    st.pyplot(plot_grouped_bar(pivot_ap50, "AP@50 Comparison Across Models", "AP@50"))
with col2:
    st.subheader("F1 Score Comparison")
    st.pyplot(plot_grouped_bar(pivot_f1, "F1 Score Comparison Across Models", "F1 Score"))

st.subheader("Degradation Impact on AP@50")
st.pyplot(plot_degradation_line(final_sorted))

st.subheader("Average Summary by Model")
st.pyplot(plot_avg_summary(summary_df))

st.markdown("---")

# =============================================================================
# SECTION 3 — Raw Data Tables
# =============================================================================
st.markdown("## Raw Data Tables")

tab_final, tab_summary = st.tabs(["Full Comparison", "Model Average Summary"])

with tab_final:
    st.dataframe(final_df, use_container_width=True)

with tab_summary:
    st.dataframe(summary_df, use_container_width=True)