import os
import json
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANN_PATH = "Dataset/COCO/annotations/instances_val2017_1000.json"
PRED_DIR = "Results/preds_coco"
OUT_CSV = "Results/yolo_coco_metrics.csv"

def eval_one(coco_gt, pred_json_path: str):
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCOeval.stats indices:
    # 0 AP@[.5:.95], 1 AP@.5, 2 AP@.75, 3 AP small, 4 AP medium, 5 AP large,
    # 6 AR@[.5:.95] maxDets=1, 7 AR maxDets=10, 8 AR maxDets=100,
    # 9 AR small, 10 AR medium, 11 AR large
    s = coco_eval.stats.tolist()

    return {
        "AP_50_95": s[0],
        "AP_50": s[1],
        "AP_75": s[2],
        "AP_small": s[3],
        "AP_medium": s[4],
        "AP_large": s[5],
        "AR_1": s[6],
        "AR_10": s[7],
        "AR_100": s[8],
        "AR_small": s[9],
        "AR_medium": s[10],
        "AR_large": s[11],
    }

def main():
    os.makedirs("Results", exist_ok=True)
    coco_gt = COCO(ANN_PATH)

    rows = []
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.startswith("pred_") and f.endswith(".json")])

    for f in pred_files:
        run_name = f.replace("pred_", "").replace(".json", "")
        path = os.path.join(PRED_DIR, f)
        print("\n============================")
        print("Evaluating:", run_name)
        metrics = eval_one(coco_gt, path)
        metrics["run"] = run_name
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("run")
    df.to_csv(OUT_CSV, index=False)
    print("\n Saved metrics to:", OUT_CSV)
    print(df[["run", "AP_50_95", "AP_50", "AR_100"]])

if __name__ == "__main__":
    main()