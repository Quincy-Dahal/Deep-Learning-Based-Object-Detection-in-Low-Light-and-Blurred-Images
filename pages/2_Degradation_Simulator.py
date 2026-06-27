import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from ultralytics import YOLO
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Degradation Simulator", layout="wide")
st.title("Degradation Simulator")

# ── COCO class names for torchvision ─────────────────────────────────────────
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── UI Controls ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a normal image", type=["jpg", "jpeg", "png"])

col_left, col_right = st.columns(2)
with col_left:
    degradation_type = st.selectbox("Choose degradation type", ["Blur", "Low Light"])
    severity         = st.selectbox("Choose severity", ["Mild", "Medium", "Severe"])
with col_right:
    model_choice     = st.selectbox("Select detection model", ["YOLOv8", "SSD MobileNet", "Faster R-CNN"])
    conf_threshold   = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

enhancement_option = "None"
if degradation_type == "Low Light":
    enhancement_option = st.selectbox(
        "Choose enhancement option",
        ["None", "Gamma", "CLAHE", "Gamma + CLAHE"]
    )

# ── Image Processing Functions ────────────────────────────────────────────────
def apply_blur(img, level):
    radius_map = {"Mild": 1.5, "Medium": 3.0, "Severe": 5.0}
    return img.filter(ImageFilter.GaussianBlur(radius=radius_map[level]))

def apply_low_light(img, level):
    factor_map = {"Mild": 0.7, "Medium": 0.5, "Severe": 0.3}
    return ImageEnhance.Brightness(img).enhance(factor_map[level])

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def apply_gamma(img, level):
    gamma_map = {"Mild": 0.8, "Medium": 0.6, "Severe": 0.4}
    gamma  = gamma_map[level]
    img_cv = pil_to_cv(img)
    table  = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv_to_pil(cv2.LUT(img_cv, table))

def apply_clahe(img):
    img_cv = pil_to_cv(img)
    lab    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv_to_pil(cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR))

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ssd():
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    m = ssdlite320_mobilenet_v3_large(weights=weights)
    m.to(DEVICE).eval()
    return m, weights.transforms()

@st.cache_resource
def load_frcnn():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    m = fasterrcnn_resnet50_fpn(weights=weights)
    m.to(DEVICE).eval()
    return m, weights.transforms()

# ── Detection Functions ───────────────────────────────────────────────────────
def detect_yolo(image: Image.Image, conf: float):
    model = load_yolo()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)
    try:
        result = model(tmp_path, conf=conf)[0]
        annotated = Image.fromarray(result.plot())
        count = len(result.boxes) if result.boxes else 0
        return annotated, count
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def detect_torchvision(image: Image.Image, conf: float, loader_fn):
    model, preprocess = loader_fn()
    x = preprocess(image).to(DEVICE)
    with torch.no_grad():
        output = model([x])[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    count = 0

    for box, score, label in zip(
        output["boxes"].cpu().numpy(),
        output["scores"].cpu().numpy(),
        output["labels"].cpu().numpy()
    ):
        if score < conf:
            continue
        name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        if name in ("N/A", "__background__"):
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{name} {score:.2f}", fill="blue")
        count += 1

    return annotated, count

def run_detection(image: Image.Image, conf: float, model_name: str):
    if model_name == "YOLOv8":
        return detect_yolo(image, conf)
    elif model_name == "SSD MobileNet":
        return detect_torchvision(image, conf, load_ssd)
    else:
        return detect_torchvision(image, conf, load_frcnn)

# ── Metric Card helper ────────────────────────────────────────────────────────
def show_metric(label, count_now, count_ref):
    delta = count_now - count_ref
    sign  = "+" if delta > 0 else ""
    color = "red" if delta < 0 else ("green" if delta > 0 else "gray")
    st.markdown(
        f"**{label}:** {count_now} objects &nbsp;"
        f"<span style='color:{color}'>({sign}{delta} vs original)</span>",
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN APP
# =============================================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ── Step 1: apply degradation ─────────────────────────────────────────────
    if degradation_type == "Blur":
        degraded = apply_blur(image, severity)
        enhanced = None
    else:
        degraded = apply_low_light(image, severity)
        if enhancement_option == "Gamma":
            enhanced = apply_gamma(degraded, severity)
        elif enhancement_option == "CLAHE":
            enhanced = apply_clahe(degraded)
        elif enhancement_option == "Gamma + CLAHE":
            enhanced = apply_clahe(apply_gamma(degraded, severity))
        else:
            enhanced = None

    # ── Step 2: run detection on all panels ───────────────────────────────────
    with st.spinner(f"Running {model_choice} on all panels..."):
        orig_det,    orig_count    = run_detection(image,    conf_threshold, model_choice)
        degraded_det, degraded_count = run_detection(degraded, conf_threshold, model_choice)
        if enhanced is not None:
            enhanced_det, enhanced_count = run_detection(enhanced, conf_threshold, model_choice)

    # ── Step 3: display ───────────────────────────────────────────────────────
    if enhanced is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Original")
            st.image(orig_det, use_container_width=True)
            st.success(f"Detected: **{orig_count}** objects")

        with c2:
            st.subheader(f"Degraded ({degradation_type} – {severity})")
            st.image(degraded_det, use_container_width=True)
            show_metric("Detected", degraded_count, orig_count)

        with c3:
            st.subheader(f"Enhanced ({enhancement_option})")
            st.image(enhanced_det, use_container_width=True)
            show_metric("Detected", enhanced_count, orig_count)

    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(orig_det, use_container_width=True)
            st.success(f"Detected: **{orig_count}** objects")

        with c2:
            st.subheader(f"Degraded ({degradation_type} – {severity})")
            st.image(degraded_det, use_container_width=True)
            show_metric("Detected", degraded_count, orig_count)

    # ── Step 4: summary caption + tip ────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Model: **{model_choice}** | "
        f"Effect: **{degradation_type}** | "
        f"Severity: **{severity}** | "
        f"Enhancement: **{enhancement_option}** | "
        f"Confidence threshold: **{conf_threshold}**"
    )

    if degradation_type == "Blur":
        st.info("📌 Gaussian blur simulates motion or focus blur. Notice how object count drops as blur increases.")
    elif enhancement_option == "None":
        st.info("📌 Low light reduces brightness. You can apply enhancements above to try to recover detections.")
    elif enhancement_option == "CLAHE":
        st.info("📌 CLAHE improves local contrast. Check if the detector recovers missed objects.")
    elif enhancement_option == "Gamma":
        st.info("📌 Gamma correction brightens the image globally. Compare detection counts with the degraded version.")
    elif enhancement_option == "Gamma + CLAHE":
        st.info("📌 Combined Gamma + CLAHE gives the strongest enhancement. Ideally detection count should approach the original.")

else:
    st.info("Upload an image to simulate degradation and see its effect on detection.")