import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- SETTINGS ----
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRES = 0.25

# Output root (all predictions will be stored here)
OUT_ROOT = os.path.join(ROOT, "Outputs", "yolo")

# ---- INPUT FOLDERS  ----
runs = [
    ("normal", os.path.join(ROOT, "Dataset", "COCO", "images", "val2017_1000")),

    ("low_light_mild",   os.path.join(ROOT, "Degraded", "low_light", "mild")),
    ("low_light_medium", os.path.join(ROOT, "Degraded", "low_light", "medium")),
    ("low_light_severe", os.path.join(ROOT, "Degraded", "low_light", "severe")),

    ("blur_mild",   os.path.join(ROOT, "Degraded", "blur", "mild")),
    ("blur_medium", os.path.join(ROOT, "Degraded", "blur", "medium")),
    ("blur_severe", os.path.join(ROOT, "Degraded", "blur", "severe")),

    ("gamma_mild",   os.path.join(ROOT, "Enhanced", "low_light", "gamma", "mild")),
    ("gamma_medium", os.path.join(ROOT, "Enhanced", "low_light", "gamma", "medium")),
    ("gamma_severe", os.path.join(ROOT, "Enhanced", "low_light", "gamma", "severe")),

    ("clahe_mild",   os.path.join(ROOT, "Enhanced", "low_light", "clahe", "mild")),
    ("clahe_medium", os.path.join(ROOT, "Enhanced", "low_light", "clahe", "medium")),
    ("clahe_severe", os.path.join(ROOT, "Enhanced", "low_light", "clahe", "severe")),

    ("gamma_clahe_mild",   os.path.join(ROOT, "Enhanced", "low_light", "gamma_clahe", "mild")),
    ("gamma_clahe_medium", os.path.join(ROOT, "Enhanced", "low_light", "gamma_clahe", "medium")),
    ("gamma_clahe_severe", os.path.join(ROOT, "Enhanced", "low_light", "gamma_clahe", "severe")),
]

def check_folder(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folder not found: {path}")
    jpgs = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]
    if len(jpgs) == 0:
        raise RuntimeError(f"No .jpg images found in: {path}")
    return len(jpgs)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print("ROOT:", ROOT)
    print("OUT_ROOT:", OUT_ROOT)


    # Load model once
    model = YOLO(MODEL_WEIGHTS)

    # Run all experiments
    for name, folder in runs:
        check_folder(folder)
        print(f"\n=== Running YOLO on: {name} ===")
        print("Input:", folder)

        model.predict(
            source=folder,
            conf=CONF_THRES,
            save=True,         # saves images with boxes
            save_txt=True,     # saves predicted labels in YOLO txt format
            project=OUT_ROOT,
            name=name,
            exist_ok=True,
            verbose=False
        )

    print("\n All YOLO runs completed.")
    print("Check outputs in:", OUT_ROOT)

if __name__ == "__main__":
    main()