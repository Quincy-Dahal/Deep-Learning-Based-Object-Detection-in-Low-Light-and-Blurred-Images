import pandas as pd

# Load COCO metrics
yolo = pd.read_csv("Results/yolo_coco_metrics.csv")
ssd = pd.read_csv("Results/ssd_coco_metrics.csv")
frcnn = pd.read_csv("Results/frcnn_coco_metrics.csv")

# Load custom metrics
yolo_c = pd.read_csv("Results/custom_detection_metrics.csv")
ssd_c = pd.read_csv("Results/ssd_custom_detection_metrics.csv")
frcnn_c = pd.read_csv("Results/frcnn_custom_detection_metrics.csv")

# Merge COCO + custom
yolo = pd.merge(yolo, yolo_c, on="run")
ssd = pd.merge(ssd, ssd_c, on="run")
frcnn = pd.merge(frcnn, frcnn_c, on="run")

# Add model names
yolo["model"] = "YOLOv8"
ssd["model"] = "SSD MobileNet"
frcnn["model"] = "Faster R-CNN"

# Combine all
df = pd.concat([yolo, ssd, frcnn], ignore_index=True)

# Keep only key conditions
runs_keep = ["normal", "low_light_severe", "blur_severe", "clahe_severe"]
df = df[df["run"].isin(runs_keep)]

# Better condition labels
condition_map = {
    "normal": "Normal",
    "low_light_severe": "Low-Light Severe",
    "blur_severe": "Blur Severe",
    "clahe_severe": "CLAHE Severe"
}

df["Condition"] = df["run"].map(condition_map)

# Keep only useful columns
df = df[[
    "model", "Condition", "AP_50_95", "AP_50", "AR_100",
    "Precision", "Recall", "F1_score", "Mean_confidence"
]]

# Save
df.to_csv("Results/final_model_comparison.csv", index=False)

print("Saved: Results/final_model_comparison.csv")
print(df)