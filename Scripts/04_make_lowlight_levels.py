import os
import cv2
import numpy as np

src = "Dataset/COCO/images/val2017_1000"   # input: normal subset
out_base = "Degraded/low_light"            # output base

levels = {
    "mild": 0.70,
    "medium": 0.50,
    "severe": 0.30
}

os.makedirs(out_base, exist_ok=True)

files = [f for f in os.listdir(src) if f.lower().endswith(".jpg")]
print("Total input images:", len(files))

for level, factor in levels.items():
    out_dir = os.path.join(out_base, level)
    os.makedirs(out_dir, exist_ok=True)

    for i, f in enumerate(files, start=1):
        img = cv2.imread(os.path.join(src, f))
        low = np.clip(img * factor, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f), low)

        if i % 200 == 0:
            print(f"{level}: processed {i}")

print("Low-light sets created in:", out_base)