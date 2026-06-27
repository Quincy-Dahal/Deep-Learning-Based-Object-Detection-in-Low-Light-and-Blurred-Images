import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "Results/yolo_coco_metrics.csv"
OUT_DIR = "Results/figures"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_line(df, runs, metric, title, ylabel, filename):
    sub = df[df["run"].isin(runs)].copy()
    sub["run"] = pd.Categorical(sub["run"], categories=runs, ordered=True)
    sub = sub.sort_values("run")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["run"], sub[metric], marker="o")
    plt.title(title)
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()

def plot_bar(df, runs, metric, title, ylabel, filename):
    sub = df[df["run"].isin(runs)].copy()
    sub["run"] = pd.Categorical(sub["run"], categories=runs, ordered=True)
    sub = sub.sort_values("run")

    plt.figure(figsize=(8, 5))
    plt.bar(sub["run"], sub[metric])
    plt.title(title)
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()

def main():
    ensure_dir(OUT_DIR)
    df = pd.read_csv(CSV_PATH)

    low_light_runs = ["normal", "low_light_mild", "low_light_medium", "low_light_severe"]
    blur_runs = ["normal", "blur_mild", "blur_medium", "blur_severe"]
    enhancement_runs = ["low_light_severe", "gamma_severe", "clahe_severe", "gamma_clahe_severe"]

    # 1. mAP vs low-light
    plot_line(
        df, low_light_runs, "AP_50_95",
        "YOLOv8: mAP vs Low-light Level",
        "mAP (AP@[0.5:0.95])",
        "map_low_light.png"
    )

    # 2. mAP vs blur
    plot_line(
        df, blur_runs, "AP_50_95",
        "YOLOv8: mAP vs Blur Level",
        "mAP (AP@[0.5:0.95])",
        "map_blur.png"
    )

    # 3. AP50 vs low-light
    plot_line(
        df, low_light_runs, "AP_50",
        "YOLOv8: AP50 vs Low-light Level",
        "AP50",
        "ap50_low_light.png"
    )

    # 4. AP50 vs blur
    plot_line(
        df, blur_runs, "AP_50",
        "YOLOv8: AP50 vs Blur Level",
        "AP50",
        "ap50_blur.png"
    )

    # 5. AR100 low-light + blur
    plot_line(
        df, ["normal", "low_light_mild", "low_light_medium", "low_light_severe",
             "blur_mild", "blur_medium", "blur_severe"],
        "AR_100",
        "YOLOv8: AR100 across Low-light and Blur Conditions",
        "AR100",
        "ar100_conditions.png"
    )

    # 6. Enhancement comparison (bar chart)
    plot_bar(
        df, enhancement_runs, "AP_50_95",
        "YOLOv8: Severe Low-light Enhancement Comparison",
        "mAP (AP@[0.5:0.95])",
        "enhancement_bar_severe.png"
    )

    print("Saved plots to:", OUT_DIR)

if __name__ == "__main__":
    main()