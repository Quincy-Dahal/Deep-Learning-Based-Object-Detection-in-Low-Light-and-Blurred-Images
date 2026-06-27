import os
import cv2

src = "Dataset/COCO/images/val2017_1000"
out_base = "Degraded/blur"

kernels = {
    "mild": 5,
    "medium": 11,
    "severe": 21
}

os.makedirs(out_base, exist_ok=True)

files = [f for f in os.listdir(src) if f.lower().endswith(".jpg")]
print("Total input images:", len(files))

for level, k in kernels.items():
    out_dir = os.path.join(out_base, level)
    os.makedirs(out_dir, exist_ok=True)

    for i, f in enumerate(files, start=1):
        img = cv2.imread(os.path.join(src, f))
        blur = cv2.GaussianBlur(img, (k, k), 0)
        cv2.imwrite(os.path.join(out_dir, f), blur)

        if i % 200 == 0:
            print(f"{level}: processed {i}")

print("Blur sets created in:", out_base)