import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
import tempfile
import os

st.set_page_config(page_title="Detection Demo", layout="wide")
st.title("Detection Demo")

# ── COCO class names used by torchvision models ──────────────────────────────
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
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

model_choice = st.selectbox(
    "Select model",
    ["YOLOv8", "SSD MobileNet", "Faster R-CNN"]
)

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ssd_model():
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(DEVICE)
    model.eval()
    return model, weights.transforms()

@st.cache_resource
def load_frcnn_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(DEVICE)
    model.eval()
    return model, weights.transforms()

# ── Detection Functions ───────────────────────────────────────────────────────
def run_yolo_detection(image: Image.Image, conf: float):
    model = load_yolo_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    results = model(temp_path, conf=conf)
    os.remove(temp_path)

    result = results[0]
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = result.names[cls_id]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"{label} {score:.2f}", fill="red")
            detections.append({"Label": label, "Confidence": round(score, 3)})

    return annotated, detections

def run_torchvision_detection(image: Image.Image, conf: float, model_fn):
    """Generic runner for SSD and Faster R-CNN (both torchvision)."""
    model, preprocess = model_fn()
    x = preprocess(image).to(DEVICE)

    with torch.no_grad():
        output = model([x])[0]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

    boxes  = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < conf:
            continue
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        if class_name == "N/A" or class_name == "__background__":
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        draw.text((x1, max(0, y1 - 15)), f"{class_name} {score:.2f}", fill="blue")
        detections.append({"Label": class_name, "Confidence": round(float(score), 3)})

    return annotated, detections

# ── Main App ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Output")

        if model_choice == "YOLOv8":
            with st.spinner("Running YOLOv8 detection..."):
                detected_image, detections = run_yolo_detection(image, conf_threshold)

        elif model_choice == "SSD MobileNet":
            with st.spinner("Running SSD MobileNet detection..."):
                detected_image, detections = run_torchvision_detection(
                    image, conf_threshold, load_ssd_model
                )

        else:  # Faster R-CNN
            with st.spinner("Running Faster R-CNN detection..."):
                detected_image, detections = run_torchvision_detection(
                    image, conf_threshold, load_frcnn_model
                )

        st.image(detected_image, use_container_width=True)

        st.markdown("### Detection Summary")
        st.write(f"Total detected objects: **{len(detections)}**")

        st.markdown("### Confidence Scores")
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.warning("No objects detected above the confidence threshold.")
else:
    st.info("Upload an image to continue.")