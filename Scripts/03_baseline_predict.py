from ultralytics import YOLO
import os

# 1) Load pretrained model 
model = YOLO("yolov8n.pt")

# 2) Input folder (your 1000 normal images)
img_dir = "Dataset/COCO/images/val2017_1000"

# 3) Output folder (where results will be saved)
out_dir = "outputs/baseline"

# 4) Run prediction on the entire folder
# save=True -> saves images with bounding boxes
# save_txt=True -> saves YOLO-format label text files (predictions)
model.predict(
    source=img_dir,
    save=True,
    save_txt=True,
    project=out_dir,
    name="",
    exist_ok=True,
    conf=0.25
)

print("Baseline prediction completed.")
print("Check results in:", out_dir)