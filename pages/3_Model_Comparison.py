import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
import tempfile
import os

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

# ── COCO class names for torchvision models ───────────────────────────────────
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

# ── UI ────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload one image for comparison", type=["jpg", "jpeg", "png"]
)
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

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

# ── Detection helpers ─────────────────────────────────────────────────────────
def run_yolo(image: Image.Image, conf: float):
    model = load_yolo()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    try:
        results = model(temp_path, conf=conf)
        result  = results[0]
        annotated = Image.fromarray(result.plot())
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "Label":      result.names[int(box.cls[0].item())],
                    "Confidence": round(float(box.conf[0].item()), 3)
                })
        return annotated, detections
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_torchvision(image: Image.Image, conf: float, loader_fn):
    model, preprocess = loader_fn()
    x = preprocess(image).to(DEVICE)
    with torch.no_grad():
        output = model([x])[0]

    from PIL import ImageDraw
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

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
        detections.append({"Label": name, "Confidence": round(float(score), 3)})

    return annotated, detections

# ── Summary helper ────────────────────────────────────────────────────────────
def show_model_col(col, title, annotated_img, detections):
    with col:
        st.subheader(title)
        st.image(annotated_img, use_container_width=True)
        st.write(f"Detected objects: **{len(detections)}**")
        if detections:
            avg = sum(d["Confidence"] for d in detections) / len(detections)
            st.write(f"Average confidence: **{avg:.3f}**")
        else:
            st.write("Average confidence: --")
            st.info("No objects detected above threshold.")

# ── Main ──────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running all three models — this may take a moment..."):
            yolo_img,  yolo_det  = run_yolo(image, conf_threshold)
            ssd_img,   ssd_det   = run_torchvision(image, conf_threshold, load_ssd)
            frcnn_img, frcnn_det = run_torchvision(image, conf_threshold, load_frcnn)

        col1, col2, col3 = st.columns(3)
        show_model_col(col1, "YOLOv8",       yolo_img,  yolo_det)
        show_model_col(col2, "SSD MobileNet", ssd_img,   ssd_det)
        show_model_col(col3, "Faster R-CNN",  frcnn_img, frcnn_det)

        st.markdown("---")
        st.markdown("### Detection Details")

        tab1, tab2, tab3 = st.tabs(["YOLOv8", "SSD MobileNet", "Faster R-CNN"])

        with tab1:
            if yolo_det:
                st.dataframe(yolo_det, use_container_width=True)
            else:
                st.info("No detections.")

        with tab2:
            if ssd_det:
                st.dataframe(ssd_det, use_container_width=True)
            else:
                st.info("No detections.")

        with tab3:
            if frcnn_det:
                st.dataframe(frcnn_det, use_container_width=True)
            else:
                st.info("No detections.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.info("Upload an image to compare all 3 models.")