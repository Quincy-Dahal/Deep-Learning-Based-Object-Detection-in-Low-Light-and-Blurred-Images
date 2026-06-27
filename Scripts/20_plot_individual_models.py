import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "Results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Load files
yolo_coco = pd.read_csv("Results/yolo_coco_metrics.csv")
yolo_custom = pd.read_csv("Results/custom_detection_metrics.csv")

ssd_coco = pd.read_csv("Results/ssd_coco_metrics.csv")
ssd_custom = pd.read_csv("Results/ssd_custom_detection_metrics.csv")

frcnn_coco = pd.read_csv("Results/frcnn_coco_metrics.csv")
frcnn_custom = pd.read_csv("Results/frcnn_custom_detection_metrics.csv")

# Merge official + custom metrics
yolo = pd.merge(yolo_coco, yolo_custom, on="run")
ssd = pd.merge(ssd_coco, ssd_custom, on="run")
frcnn = pd.merge(frcnn_coco, frcnn_custom, on="run")

# Order of conditions
condition_order = [
    "normal",
    "low_light_mild", "low_light_medium", "low_light_severe",
    "blur_mild", "blur_medium", "blur_severe",
    "gamma_mild", "gamma_medium", "gamma_severe",
    "clahe_mild", "clahe_medium", "clahe_severe",
    "gamma_clahe_mild", "gamma_clahe_medium", "gamma_clahe_severe"
]

# Nicer labels for x-axis
label_map = {
    "normal": "Normal",
    "low_light_mild": "LL Mild",
    "low_light_medium": "LL Medium",
    "low_light_severe": "LL Severe",
    "blur_mild": "Blur Mild",
    "blur_medium": "Blur Medium",
    "blur_severe": "Blur Severe",
    "gamma_mild": "Gamma Mild",
    "gamma_medium": "Gamma Medium",
    "gamma_severe": "Gamma Severe",
    "clahe_mild": "CLAHE Mild",
    "clahe_medium": "CLAHE Medium",
    "clahe_severe": "CLAHE Severe",
    "gamma_clahe_mild": "G+C Mild",
    "gamma_clahe_medium": "G+C Medium",
    "gamma_clahe_severe": "G+C Severe"
}

def plot_model(df, model_name, graph_no, out_name):
    df["run"] = pd.Categorical(df["run"], categories=condition_order, ordered=True)
    df = df.sort_values("run")

    x_labels = [label_map[r] for r in df["run"]]

    plt.figure(figsize=(14, 6))
    plt.plot(x_labels, df["AP_50"], marker="o", label="AP@50")
    plt.plot(x_labels, df["F1_score"], marker="o", label="F1 Score")

    plt.title(f"Graph {graph_no}: {model_name} — AP@50 and F1 Score Across All Conditions")
    plt.xlabel("Condition")
    plt.ylabel("Score")
    plt.xticks(rotation=40, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300)
    plt.show()

# Graph 4
plot_model(yolo, "YOLOv8", 4, "graph4_yolo_individual.png")

# Graph 5
plot_model(ssd, "SSD MobileNet", 5, "graph5_ssd_individual.png")

# Graph 6
plot_model(frcnn, "Faster R-CNN", 6, "graph6_frcnn_individual.png")

print("Saved Graph 4, 5, 6 in:", OUT_DIR)