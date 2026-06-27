import os
import json
import time
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

ANN_PATH = "Dataset/COCO/annotations/instances_val2017_1000.json"

RUNS = [
    ("normal", "Dataset/COCO/images/val2017_1000"),

    ("low_light_mild",   "Degraded/low_light/mild"),
    ("low_light_medium", "Degraded/low_light/medium"),
    ("low_light_severe", "Degraded/low_light/severe"),

    ("blur_mild",   "Degraded/blur/mild"),
    ("blur_medium", "Degraded/blur/medium"),
    ("blur_severe", "Degraded/blur/severe"),

    ("gamma_mild",   "Enhanced/low_light/gamma/mild"),
    ("gamma_medium", "Enhanced/low_light/gamma/medium"),
    ("gamma_severe", "Enhanced/low_light/gamma/severe"),

    ("clahe_mild",   "Enhanced/low_light/clahe/mild"),
    ("clahe_medium", "Enhanced/low_light/clahe/medium"),
    ("clahe_severe", "Enhanced/low_light/clahe/severe"),

    ("gamma_clahe_mild",   "Enhanced/low_light/gamma_clahe/mild"),
    ("gamma_clahe_medium", "Enhanced/low_light/gamma_clahe/medium"),
    ("gamma_clahe_severe", "Enhanced/low_light/gamma_clahe/severe"),
]

OUT_DIR = "Results/preds_coco_ssd"
CONF_THRES = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def load_coco_maps(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    file_to_id = {img["file_name"]: img["id"] for img in coco["images"]}
    valid_cat_ids = set(cat["id"] for cat in coco["categories"])
    return file_to_id, valid_cat_ids

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    file_to_id, valid_cat_ids = load_coco_maps(ANN_PATH)

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(DEVICE)
    model.eval()

    preprocess = weights.transforms()

    print("Using device:", DEVICE)

    for run_name, folder in RUNS:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
        print(f"\n=== {run_name} ===")
        print("Images:", len(files))

        preds = []
        t0 = time.time()

        for fn in tqdm(files, desc=f"SSD {run_name}", unit="img"):
            if fn not in file_to_id:
                continue

            image_id = int(file_to_id[fn])
            img_path = os.path.join(folder, fn)

            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(DEVICE)

            with torch.no_grad():
                output = model([x])[0]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < CONF_THRES:
                    continue

                cat_id = int(label)
                if cat_id not in valid_cat_ids:
                    continue

                preds.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score)
                })

        out_path = os.path.join(OUT_DIR, f"pred_ssd_{run_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        print(f"Saved: {out_path}")
        print(f"Predictions: {len(preds)} | Time(s): {time.time() - t0:.2f}")

if __name__ == "__main__":
    main()