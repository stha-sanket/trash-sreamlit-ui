# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from PIL import Image

# -------------------------------------------------
# 1. LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        "waste_classification_mobilenetv2_pro.keras",
        compile=False,
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

model = load_model()

# -------------------------------------------------
# 2. SETTINGS
# -------------------------------------------------
CLASS_NAMES = ["metal waste", "organic waste", "paper waste", "plastic waste"]
IMG_PATH   = "img1.jpg"
INTERVAL   = 15                     # seconds

# -------------------------------------------------
# 3. PREDICTION – returns ONLY the class name
# -------------------------------------------------
def get_waste_class(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    pred = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx]

# -------------------------------------------------
# 4. UI
# -------------------------------------------------
st.title("Waste Snap – Every 15 sec")

# ---- CAMERA SELECTOR ----
cams = {}
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            cams[f"Camera {i}"] = i
        cap.release()

if not cams:
    st.error("No camera found")
    st.stop()

cam_name = st.selectbox("Select camera:", list(cams.keys()))
cam_id   = cams[cam_name]

# ---- START ----
if st.button("Start Capture"):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Cannot open selected camera")
        st.stop()

    st.success(f"Using **{cam_name}** – first snap in {INTERVAL}s")
    count_ph = st.empty()
    result_ph = st.empty()
    last = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera lost")
            break

        now = time.time()
        elapsed = now - last

        # countdown
        count_ph.markdown(f"**Next snap in:** {max(0, INTERVAL - int(elapsed))} sec")

        # ---- SNAP ----
        if elapsed >= INTERVAL:
            cv2.imwrite(IMG_PATH, frame)
            last = now

            # show picture
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"img1.jpg – {datetime.now():%H:%M:%S}", use_container_width=True)

            # predict & show ONLY the waste class
            waste = get_waste_class(IMG_PATH)
            result_ph.success(f"**Waste:** {waste}")

        time.sleep(0.1)

    cap.release()