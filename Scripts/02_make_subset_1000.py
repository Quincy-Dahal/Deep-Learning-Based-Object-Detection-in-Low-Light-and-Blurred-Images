import os
import random
import shutil

src = "Dataset/COCO/images/val2017"
dst = "Dataset/COCO/images/val2017_1000"

os.makedirs(dst, exist_ok=True)

all_imgs = [f for f in os.listdir(src) if f.lower().endswith(".jpg")]
print("Total images found:", len(all_imgs))

random.seed(42)
subset = random.sample(all_imgs, 1000)

for i, f in enumerate(subset, start=1):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    if i % 100 == 0:
        print("Copied:", i)

print("Done. Subset created at:", dst)