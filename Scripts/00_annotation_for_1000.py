import os
import json

FULL_ANN_PATH = "Dataset/COCO/annotations/instances_val2017.json"
SUBSET_IMG_DIR = "Dataset/COCO/images/val2017_1000"
OUT_ANN_PATH = "Dataset/COCO/annotations/instances_val2017_1000.json"

subset_files = {
    f for f in os.listdir(SUBSET_IMG_DIR)
    if f.lower().endswith(".jpg")
}

with open(FULL_ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

subset_images = [img for img in coco["images"] if img["file_name"] in subset_files]
subset_image_ids = {img["id"] for img in subset_images}

subset_annotations = [
    ann for ann in coco["annotations"]
    if ann["image_id"] in subset_image_ids
]

subset_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": subset_images,
    "annotations": subset_annotations,
    "categories": coco.get("categories", [])
}

with open(OUT_ANN_PATH, "w", encoding="utf-8") as f:
    json.dump(subset_coco, f)

print("Saved:", OUT_ANN_PATH)
print("Images:", len(subset_images))
print("Annotations:", len(subset_annotations))
print("Categories:", len(subset_coco["categories"]))