import os
import json
import time
from tqdm import tqdm
from ultralytics import YOLO

# ------------------ PATHS ------------------
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

OUT_DIR = "Results/preds_coco"
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRES = 0.25
IOU_NMS = 0.7

# ------------------ HELPERS ------------------
def load_coco_maps(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # file_name -> image_id
    file_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    # category name -> category_id (COCO official IDs)
    name_to_catid = {cat["name"]: cat["id"] for cat in coco["categories"]}

    return file_to_id, name_to_catid

def list_jpgs(folder: str):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

# ------------------ MAIN ------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    file_to_id, name_to_catid = load_coco_maps(ANN_PATH)

    model = YOLO(MODEL_WEIGHTS)
    yolo_names = model.names  # {0:"person", 1:"bicycle", ...}

    # Build YOLO class index -> COCO category_id by matching names
    yolo_to_coco_catid = {}
    for cls_idx, cls_name in yolo_names.items():
        if cls_name in name_to_catid:
            yolo_to_coco_catid[int(cls_idx)] = name_to_catid[cls_name]
        else:
            # If any mismatch happens, we warn once.
            yolo_to_coco_catid[int(cls_idx)] = None

    for run_name, folder in RUNS:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = list_jpgs(folder)
        if len(files) == 0:
            raise RuntimeError(f"No jpg files in: {folder}")

        preds = []
        t_start = time.time()

        for fn in tqdm(files, desc=f"Exporting {run_name}", unit="img"):
            img_path = os.path.join(folder, fn)

            # Must match COCO file_name to get image_id
            if fn not in file_to_id:
                # COCO val2017 uses the same filenames; if not found, skip
                continue
            image_id = file_to_id[fn]

            # Predict on single image (no saving)
            r = model.predict(
                source=img_path,
                conf=CONF_THRES,
                iou=IOU_NMS,
                verbose=False
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for xyxy, score, cls_idx in zip(boxes_xyxy, confs, clss):
                coco_catid = yolo_to_coco_catid.get(int(cls_idx))
                if coco_catid is None:
                    continue

                preds.append({
                    "image_id": int(image_id),
                    "category_id": int(coco_catid),
                    "bbox": xyxy_to_xywh(xyxy),
                    "score": float(score)
                })

        t_end = time.time()
        out_path = os.path.join(OUT_DIR, f"pred_{run_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        print(f"\n Saved: {out_path}")
        print(f"Images processed: {len(files)} | Predictions: {len(preds)} | Time(s): {t_end - t_start:.2f}")

if __name__ == "__main__":
    main()