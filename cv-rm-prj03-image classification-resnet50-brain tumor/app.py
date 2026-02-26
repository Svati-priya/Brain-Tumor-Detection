# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Brain Tumor Classifier (ROI Crop)", layout="centered")
st.title("Brain Tumor Classification (with ROI Cropping)")
st.write("Model predicts these classes with IDs:")
st.code('{"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}')

# -----------------------
# Settings
# -----------------------
CLASS_DICT = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
class_names = ["glioma", "meningioma", "notumor", "pituitary"]
TARGET_SIZE = (224, 224)  # matches your preprocessing
THRESHOLD_VALUE = 10      # same as your threshold in preprocessing

# -----------------------
# Utilities
# -----------------------
def read_image_from_bytes(file_bytes):
    """Read image bytes into both OpenCV (BGR) and PIL (RGB)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR or None
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return img_bgr, pil_img

def crop_brain_roi_from_bgr(img_bgr, target_size=TARGET_SIZE, threshold_value=THRESHOLD_VALUE):
    """
    Crop brain ROI using the same method you used in preprocessing:
    - convert to grayscale
    - threshold at threshold_value
    - find largest contour
    - crop bounding rect and resize
    Returns cropped BGR image (uint8) or None if failed.
    """
    if img_bgr is None:
        return None

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        # if conversion fails, try interpreting as grayscale already
        gray = img_bgr if len(img_bgr.shape) == 2 else None
        if gray is None:
            return None

    # Threshold to isolate brain area (same as your code)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Clean small noise that might break contour detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Guard against very small bounding boxes
    h_img, w_img = gray.shape
    if w <= 10 or h <= 10 or w > w_img or h > h_img:
        return None

    crop = img_bgr[y:y+h, x:x+w]

    # Resize to model input
    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return crop_resized

def prepare_for_model(img_bgr):
    """Normalize and expand dims to feed the Keras model."""
    img = img_bgr.astype("float32") / 255.0
    # Ensure shape (H, W, 3)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model(path="final_resnet50_mri.keras"):
    model = tf.keras.models.load_model(path)
    return model

# We assume model.h5 exists in the same folder (as requested)
model = None
try:
    model = load_model("final_resnet50_mri.keras")
except Exception as e:
    st.warning("Could not load model.h5 automatically. Place your trained model file named 'model.h5' in this folder.")
    st.write("Model load error:", e)

# -----------------------
# UI: Upload
# -----------------------
uploaded = st.file_uploader("Upload an MRI image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an MRI image to get a prediction. The app will attempt to crop the brain ROI exactly as used in training.")
else:
    # Read bytes -> BGR + PIL
    file_bytes = uploaded.read()
    img_bgr, pil_img = read_image_from_bytes(file_bytes)

    if img_bgr is None or pil_img is None:
        st.error("Unable to read the uploaded image. Try another file.")
    else:
        st.subheader("Original Image")
        st.image(pil_img, use_column_width=True)

        # Crop ROI
        cropped_bgr = crop_brain_roi_from_bgr(img_bgr)
        if cropped_bgr is None:
            st.error("⚠️ Brain ROI not detected with the chosen threshold. Try another MRI slice or a clearer image.")
            # Optionally try a looser fallback: center crop resized to TARGET_SIZE
            st.info("Fallback: using center-crop resized to target size for prediction.")
            h, w = img_bgr.shape[:2]
            side = min(h, w)
            cx, cy = w//2, h//2
            x1 = max(cx - side//2, 0); y1 = max(cy - side//2, 0)
            x2 = x1 + side; y2 = y1 + side
            center_crop = img_bgr[y1:y2, x1:x2]
            try:
                cropped_bgr = cv2.resize(center_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            except Exception:
                st.error("Fallback also failed. Can't prepare image for prediction.")
                cropped_bgr = None

        if cropped_bgr is not None:
            # Convert BGR -> RGB for display
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            st.subheader("Cropped Brain Region (model input)")
            st.image(Image.fromarray(cropped_rgb), use_column_width=True)

            if model is None:
                st.error("Model not loaded. Place 'model.h5' next to this app and restart.")
            else:
                # Prepare and predict
                x = prepare_for_model(cropped_bgr)
                preds = model.predict(x)
                pred_idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds))
                pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

                st.subheader("Prediction")
                st.write(f"**Predicted class:** {pred_label}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.write("Full output vector (raw model softmax/probs):")
                st.json({class_names[i]: float(preds[0, i]) for i in range(len(class_names))})

                # Print mapping explicitly as you requested
                st.write("Class mapping used by this app:")
                st.code(str(CLASS_DICT))
