import os
import cv2
import numpy as np

# Input low-light base folder
in_base = "Degraded/low_light"

# Output base folder
out_base = "Enhanced/low_light"

levels = ["mild", "medium", "severe"]

# Gamma values (lower gamma brightens)
gamma_values = {
    "mild": 0.8,
    "medium": 0.6,
    "severe": 0.4
}

def apply_gamma(img, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img):
    # CLAHE works on luminance channel (Y) for natural enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Create output folders
for method in ["gamma", "clahe", "gamma_clahe"]:
    for lvl in levels:
        os.makedirs(os.path.join(out_base, method, lvl), exist_ok=True)

for lvl in levels:
    in_dir = os.path.join(in_base, lvl)
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".jpg")]
    print(f"{lvl}: total images {len(files)}")

    g = gamma_values[lvl]

    for i, f in enumerate(files, start=1):
        img = cv2.imread(os.path.join(in_dir, f))

        # 1) Gamma only
        img_g = apply_gamma(img, g)
        cv2.imwrite(os.path.join(out_base, "gamma", lvl, f), img_g)

        # 2) CLAHE only
        img_c = apply_clahe(img)
        cv2.imwrite(os.path.join(out_base, "clahe", lvl, f), img_c)

        # 3) Gamma + CLAHE
        img_gc = apply_clahe(img_g)
        cv2.imwrite(os.path.join(out_base, "gamma_clahe", lvl, f), img_gc)

        if i % 200 == 0:
            print(f"{lvl}: processed {i}")

print(" Enhancement done. Check Enhanced/low_light/")