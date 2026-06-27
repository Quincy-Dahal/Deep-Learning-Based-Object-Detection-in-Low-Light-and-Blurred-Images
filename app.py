import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Deep Learning Based Object Detection in Low-Light and Blurred Images",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Deep Learning Based Object Detection in Low-Light and Blurred Images")
st.subheader("Final Year Project Demo")

st.markdown("---")

# Welcome Section
st.markdown("### Welcome")
st.markdown("This application demonstrates the project:")
st.markdown("**Deep Learning Based Object Detection in Low-Light and Blurred Images**")

st.markdown("---")

# Two Column Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### What This App Includes
    - **Detection Demo** — upload an image and run detection
    - **Degradation Simulator** — apply blur and low-light conditions, then test enhancement methods
    - **Model Comparison** — compare all 3 models side by side
    - **Results Dashboard** — view graphs and result summaries
    """)

with col2:
    st.markdown("""
    ### Models Used
    - 🔵 **YOLOv8** — fast and efficient detection
    - 🟠 **SSD MobileNet** — lightweight single shot detector
    - 🟢 **Faster R-CNN** — high accuracy region based detector
    """)
st.markdown("---")

st.markdown("""
### Experimental Setup
- **Dataset:** COCO 2017 validation subset
- **Subset Size:** 1000 images
- **Degradation Conditions:** Low-light and Gaussian blur
- **Severity Levels:** Mild, Medium, Severe
- **Enhancement Methods:** Gamma correction, CLAHE, Gamma + CLAHE
- **Evaluation Metrics:** AP@50, mAP, Precision, Recall, F1 Score, Mean Confidence
""")

st.markdown("---")

# Student Details
st.markdown("""
### Student Details
- **Student:** Quincy Dahal
- **College:** The British College
- **Supervisor:** Saroj Sharma
- **Academic Year:** 2025/2026
""")

st.markdown("---")

# CSVs live in Results/ — where your backend scripts save them
summary_df = pd.read_csv("Results/model_average_summary.csv")
final_df   = pd.read_csv("Results/final_model_comparison.csv")

# Best values
best_model_row = summary_df.loc[summary_df["Avg_AP_50"].idxmax()]
best_ap50_row  = final_df.loc[final_df["AP_50"].idxmax()]
best_f1_row    = final_df.loc[final_df["F1_score"].idxmax()]

# Key Results Metrics
st.markdown("### Key Results Summary")

m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Best Model (Overall)", value=best_model_row["Model"])
m2.metric(label="Best AP@50",           value=f'{best_ap50_row["AP_50"]:.3f}')
m3.metric(label="Best F1 Score",        value=f'{best_f1_row["F1_score"]:.3f}')
m4.metric(label="Models Compared",      value=str(summary_df["Model"].nunique()))

st.caption(
    f'Best overall model: {best_model_row["Model"]} '
    f'(Avg AP@50 = {best_model_row["Avg_AP_50"]:.3f}) | '
    f'Highest AP@50: {best_ap50_row["model"]} - {best_ap50_row["Condition"]} '
    f'({best_ap50_row["AP_50"]:.3f}) | '
    f'Highest F1 Score: {best_f1_row["model"]} - {best_f1_row["Condition"]} '
    f'({best_f1_row["F1_score"]:.3f})'
)

st.markdown("---")

st.info("👈 Use the sidebar on the left to navigate through the pages. Start with Detection Demo.")