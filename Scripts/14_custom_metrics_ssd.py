import os
import json
import csv
from collections import defaultdict

ANN_PATH = "Dataset/COCO/annotations/instances_val2017_1000.json"
PRED_DIR = "Results/preds_coco_ssd"
OUT_CSV = "Results/ssd_custom_detection_metrics.csv"

IOU_THRESH = 0.5

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter
    if union == 0:
        return 0.0
    return inter / union

def load_ground_truth(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append({
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    return gt_by_image

def evaluate_predictions(gt_by_image, pred_json_path, iou_thresh=0.5):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    TP = 0
    FP = 0
    FN = 0
    all_scores = []

    all_image_ids = set(gt_by_image.keys()) | set(preds_by_image.keys())

    for image_id in all_image_ids:
        gts = gt_by_image.get(image_id, [])
        prs = preds_by_image.get(image_id, [])

        prs = sorted(prs, key=lambda x: x["score"], reverse=True)
        matched_gt = [False] * len(gts)

        for pred in prs:
            all_scores.append(pred["score"])
            pred_box = xywh_to_xyxy(pred["bbox"])
            pred_cat = pred["category_id"]

            best_iou = 0.0
            best_j = -1

            for j, gt in enumerate(gts):
                if matched_gt[j]:
                    continue
                if gt["category_id"] != pred_cat:
                    continue

                gt_box = xywh_to_xyxy(gt["bbox"])
                iou = iou_xyxy(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j != -1:
                TP += 1
                matched_gt[best_j] = True
            else:
                FP += 1

        FN += matched_gt.count(False)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_conf = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1,
        "Mean_confidence": mean_conf,
        "Num_predictions": len(all_scores),
    }

def main():
    gt_by_image = load_ground_truth(ANN_PATH)

    pred_files = sorted(
        [f for f in os.listdir(PRED_DIR) if f.startswith("pred_ssd_") and f.endswith(".json")]
    )

    rows = []
    for f in pred_files:
        run_name = f.replace("pred_ssd_", "").replace(".json", "")
        path = os.path.join(PRED_DIR, f)

        print(f"Evaluating SSD custom metrics: {run_name}")
        metrics = evaluate_predictions(gt_by_image, path, IOU_THRESH)
        metrics["run"] = run_name
        rows.append(metrics)

    normal_conf = None
    for row in rows:
        if row["run"] == "normal":
            normal_conf = row["Mean_confidence"]
            break

    for row in rows:
        if normal_conf is not None:
            row["Confidence_drop"] = normal_conf - row["Mean_confidence"]
        else:
            row["Confidence_drop"] = ""

    fieldnames = [
        "run", "TP", "FP", "FN",
        "Precision", "Recall", "F1_score",
        "Mean_confidence", "Confidence_drop", "Num_predictions"
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved SSD custom metrics to:", OUT_CSV)
    for row in rows:
        print(
            f"{row['run']}: "
            f"P={row['Precision']:.4f}, "
            f"R={row['Recall']:.4f}, "
            f"F1={row['F1_score']:.4f}, "
            f"Conf={row['Mean_confidence']:.4f}"
        )

if __name__ == "__main__":
    main()