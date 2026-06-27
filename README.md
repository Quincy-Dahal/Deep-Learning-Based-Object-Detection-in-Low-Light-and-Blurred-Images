# Deep Learning Based Object Detection in Low-Light and Blurred Images

A controlled benchmark study evaluating how three pre-trained object detectors - **YOLOv8n**, **SSD MobileNet**, and **Faster R-CNN** - degrade under low-light and Gaussian-blur conditions, and whether lightweight image enhancement (Gamma, CLAHE) can recover detection performance **without retraining**. Includes a four-page interactive **Streamlit** web app for live detection, degradation simulation, model comparison, and a results dashboard.

> Final-year production project - BSc (Hons) Computing, The British College / Leeds Beckett University.

---

## Overview

Most object detectors are trained and benchmarked on clean, well-lit images, yet real deployments (night CCTV, dashcams, low-end phone cameras) routinely face darkness and blur. This project quantifies that performance gap and tests whether simple, deterministic preprocessing can close it.

The pipeline takes a 1,000-image subset of the **COCO 2017** validation set, synthesises **16 degradation conditions** per image, runs three pre-trained detectors across all of them, and scores results with the official COCO protocol (`pycocotools`) plus custom metrics.

## Key Findings

| Condition | Headline result |
|-----------|-----------------|
| **Gaussian blur** | Largest performance drop across all models - YOLOv8n AP@50 fell **49%** (0.443 → 0.226); Faster R-CNN fell **57.5%** under severe blur. |
| **Low light** | Markedly less damaging than blur for all three models (Faster R-CNN lost only ~12.3% under severe low-light). |
| **CLAHE enhancement** | Partially restored severe low-light performance for YOLOv8n and Faster R-CNN; SSD MobileNet saw little benefit from any enhancement. |

**Model summary (averaged across conditions):**

| Model | Avg AP@50 | Avg Precision | Avg Recall | Avg F1 | Best at |
|-------|:---------:|:-------------:|:----------:|:------:|---------|
| **Faster R-CNN** | **0.460** | - | **0.593** | - | Most accurate overall; best on clean images |
| **YOLOv8n** | - | **0.693** | - | **0.526** | Best-balanced; most robust on degraded images |
| **SSD MobileNet** | 0.277 | - | - | - | Lightest, but weakest under degradation |

**Takeaway:** Faster R-CNN is the most accurate on clean inputs, YOLOv8n is the most robust as quality degrades, and simple enhancement can recover meaningful accuracy without any retraining.

---

## Features

- **Detection Demo** - upload an image, run any of the three models, view bounding boxes, class labels, and confidence scores.
- **Degradation Simulator** - apply low-light (×0.70 / ×0.50 / ×0.30) or Gaussian blur (5×5 / 11×11 / 21×21) and optional Gamma / CLAHE / Gamma+CLAHE enhancement.
- **Model Comparison** - run YOLOv8n, SSD MobileNet, and Faster R-CNN side by side on the same image.
- **Results Dashboard** - pre-computed AP@50, mAP, precision, recall, and F1 across all conditions, with comparison charts.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Deep learning | PyTorch, torchvision |
| Detectors | Ultralytics YOLOv8 (`yolov8n.pt`), `ssdlite320_mobilenet_v3_large`, `fasterrcnn_resnet50_fpn` |
| Image processing | OpenCV |
| Evaluation | pycocotools (COCO AP@50 / mAP) |
| Data & viz | NumPy, pandas, Matplotlib |
| Web app | Streamlit |

---

## Methodology

1. **Dataset** - 1,000 images randomly sampled (seed = 42) from COCO 2017 val; ground-truth annotations exported to `instances_val2017_1000.json`.
2. **Degradation** - low-light (3 levels via pixel-intensity scaling) and Gaussian blur (3 odd-kernel levels), producing 6 degraded sets + original.
3. **Enhancement** - Gamma correction, CLAHE, and combined Gamma+CLAHE applied to low-light images.
4. **Detection** - all three pre-trained detectors run across every condition; **no training or fine-tuning**.
5. **Evaluation** - IoU, precision, recall, F1 (custom) + AP@50 and mAP (pycocotools, official COCO protocol).

---

### Prerequisites
- Python 3.11
- No GPU required - the full evaluation pipeline runs on CPU.

### Reproduce the experiment pipeline
```bash
python Scripts/02_make_subset_1000.py
python Scripts/00_annotation_for_1000.py
python Scripts/04_make_lowlight_levels.py
python Scripts/05_make_blur_levels.py
python Scripts/06_enhance_lowlight.py
python Scripts/08_export_preds_coco.py
python Scripts/12_export_ssd_preds_coco.py
# Faster R-CNN: run Scripts/Json_frcnn.ipynb
python Scripts/09_eval_coco.py
```

## Dataset

This project uses the **COCO 2017 validation set** (Lin et al., 2014). The 1,000-image subset and derived degraded/enhanced images are generated locally by the scripts above. COCO images and annotations are subject to their original terms - see [cocodataset.org](https://cocodataset.org).

## Author

**Quincy Dahal** - AI/ML Developer
GitHub: [@Quincy-Dahal](https://github.com/Quincy-Dahal)
