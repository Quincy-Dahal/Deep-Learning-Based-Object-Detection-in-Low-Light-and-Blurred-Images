import os, json

img_dir = "Dataset/COCO/images/val2017"
ann_path = "Dataset/COCO/annotations/instances_val2017.json"

# Check images folder
print("1) Images folder exists:", os.path.isdir(img_dir))
if os.path.isdir(img_dir):
    jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    print("2) Number of .jpg images:", len(jpgs))
else:
    print(" Fix path:", img_dir)

# Check annotations file
print("3) Annotation file exists:", os.path.isfile(ann_path))
if os.path.isfile(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("4) JSON loaded successfully: True")
    print("5) JSON counts -> images:", len(data.get("images", [])))
    print("6) JSON counts -> annotations:", len(data.get("annotations", [])))
    print("7) JSON counts -> categories:", len(data.get("categories", [])))
else:
    print(" Fix path:", ann_path)