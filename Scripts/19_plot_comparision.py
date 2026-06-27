import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COMPARISON_CSV = "Results/final_model_comparison.csv"
SUMMARY_CSV = "Results/model_average_summary.csv"
OUT_DIR = "Results/figures"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(COMPARISON_CSV)
summary = pd.read_csv(SUMMARY_CSV)

conditions = ["Normal", "Low-Light Severe", "Blur Severe", "CLAHE Severe"]
models = ["YOLOv8", "SSD MobileNet", "Faster R-CNN"]

df["Condition"] = pd.Categorical(df["Condition"], categories=conditions, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
df = df.sort_values(["Condition", "model"])

# -----------------------------
# Graph 7 — AP@50 grouped bar chart
# -----------------------------
pivot_ap50 = df.pivot(index="Condition", columns="model", values="AP_50").loc[conditions, models]

x = np.arange(len(conditions))
width = 0.25

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    bars = plt.bar(x + i * width - width, pivot_ap50[model], width=width, label=model)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=7)

plt.title("Graph 7: AP@50 Comparison Across Models")
plt.xlabel("Condition")
plt.ylabel("AP@50")
plt.xticks(x, conditions, rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph7_ap50_grouped_bar.png"), dpi=300)
plt.show()

# -----------------------------
# Graph 8 — F1 grouped bar chart
# -----------------------------
pivot_f1 = df.pivot(index="Condition", columns="model", values="F1_score").loc[conditions, models]

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    bars = plt.bar(x + i * width - width, pivot_f1[model], width=width, label=model)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=7)

plt.title("Graph 8: F1 Score Comparison Across Models")
plt.xlabel("Condition")
plt.ylabel("F1 Score")
plt.xticks(x, conditions, rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph8_f1_grouped_bar.png"), dpi=300)
plt.show()

# -----------------------------
# Graph 9 — Degradation impact line graph
# -----------------------------
deg_conditions = ["Normal", "Low-Light Severe", "Blur Severe"]
df_deg = df[df["Condition"].isin(deg_conditions)].copy()
df_deg["Condition"] = pd.Categorical(df_deg["Condition"], categories=deg_conditions, ordered=True)
df_deg = df_deg.sort_values(["model", "Condition"])

plt.figure(figsize=(10, 6))
for model in models:
    sub = df_deg[df_deg["model"] == model]
    plt.plot(sub["Condition"], sub["AP_50"], marker="o", label=model)
    for _, row in sub.iterrows():
        plt.text(
            row["Condition"],
            row["AP_50"] + 0.001,
            f'{row["AP_50"]:.3f}',
            ha='center', va='bottom', fontsize=7)

plt.title("Graph 9: Degradation Impact on AP@50")
plt.xlabel("Condition")
plt.ylabel("AP@50")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph9_degradation_impact_line.png"), dpi=300)
plt.show()

# -----------------------------
# Graph 10 — Average summary bar chart
# -----------------------------
summary["Model"] = pd.Categorical(summary["Model"], categories=models, ordered=True)
summary = summary.sort_values("Model")

x2 = np.arange(len(models))
width2 = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x2 - width2/2, summary["Avg_AP_50"], width=width2, label="Avg AP@50")
bars2 = plt.bar(x2 + width2/2, summary["Avg_F1"], width=width2, label="Avg F1")

for bar in list(bars1) + list(bars2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{bar.get_height():.3f}',
        ha='center', va='bottom', fontsize=7)

plt.title("Graph 10: Average Summary by Model")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(x2, summary["Model"], rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph10_average_summary_bar.png"), dpi=300)
plt.show()

# -----------------------------
# Graph 11 — Blur severity impact on AP@50
# Normal → Blur Mild → Blur Medium → Blur Severe
# -----------------------------

# Load full COCO metric files because final_model_comparison.csv only keeps selected severe conditions
yolo_full = pd.read_csv("Results/yolo_coco_metrics.csv")
ssd_full = pd.read_csv("Results/ssd_coco_metrics.csv")
frcnn_full = pd.read_csv("Results/frcnn_coco_metrics.csv")

# Add model names
yolo_full["model"] = "YOLOv8"
ssd_full["model"] = "SSD MobileNet"
frcnn_full["model"] = "Faster R-CNN"

# Combine all models
full_df = pd.concat([yolo_full, ssd_full, frcnn_full], ignore_index=True)
full_df["model"] = pd.Categorical(full_df["model"], categories=models, ordered=True)


def plot_ap50_line_by_runs(data, run_order, label_map, title, filename):
    sub_df = data[data["run"].isin(run_order)].copy()

    sub_df["run"] = pd.Categorical(
        sub_df["run"],
        categories=run_order,
        ordered=True
    )

    sub_df["Condition"] = sub_df["run"].map(label_map)
    sub_df = sub_df.sort_values(["model", "run"])

    plt.figure(figsize=(10, 6))

    for model in models:
        model_df = sub_df[sub_df["model"] == model]

        plt.plot(
            model_df["Condition"],
            model_df["AP_50"],
            marker="o",
            label=model
        )

        for _, row in model_df.iterrows():
            plt.text(
                row["Condition"],
                row["AP_50"] + 0.001,
                f'{row["AP_50"]:.3f}',
                ha="center",
                va="bottom",
                fontsize=7
            )

    plt.title(title)
    plt.xlabel("Condition")
    plt.ylabel("AP@50")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.show()


# Graph 11: Blur
blur_runs = ["normal", "blur_mild", "blur_medium", "blur_severe"]

blur_labels = {
    "normal": "Normal",
    "blur_mild": "Blur Mild",
    "blur_medium": "Blur Medium",
    "blur_severe": "Blur Severe"
}

plot_ap50_line_by_runs(
    full_df,
    blur_runs,
    blur_labels,
    "Graph 11: AP@50 Change Across Blur Severity Levels",
    "graph11_blur_severity_ap50_line.png"
)


# -----------------------------
# Graph 12 — Low-light severity impact on AP@50
# Normal → Low-Light Mild → Low-Light Medium → Low-Light Severe
# -----------------------------

low_light_runs = [
    "normal",
    "low_light_mild",
    "low_light_medium",
    "low_light_severe"
]

low_light_labels = {
    "normal": "Normal",
    "low_light_mild": "Low-Light Mild",
    "low_light_medium": "Low-Light Medium",
    "low_light_severe": "Low-Light Severe"
}

plot_ap50_line_by_runs(
    full_df,
    low_light_runs,
    low_light_labels,
    "Graph 12: AP@50 Change Across Low-Light Severity Levels",
    "graph12_low_light_severity_ap50_line.png"
)


# -----------------------------
# Graph 13 — Enhancement impact on AP@50
# Low-Light Severe → Gamma Severe → CLAHE Severe → Gamma + CLAHE Severe
# -----------------------------

enhancement_runs = [
    "low_light_severe",
    "gamma_severe",
    "clahe_severe",
    "gamma_clahe_severe"
]

enhancement_labels = {
    "low_light_severe": "Low-Light Severe",
    "gamma_severe": "Gamma Severe",
    "clahe_severe": "CLAHE Severe",
    "gamma_clahe_severe": "Gamma + CLAHE Severe"
}

plot_ap50_line_by_runs(
    full_df,
    enhancement_runs,
    enhancement_labels,
    "Graph 13: AP@50 Change After Low-Light Enhancement",
    "graph13_enhancement_ap50_line.png"
)

print("Saved Graph 11, 12, and 13 in:", OUT_DIR)

print("Saved graphs to:", OUT_DIR)