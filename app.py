import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Traffic Density Analyzer",
    page_icon="🚦",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "traffic_sign_cnn_model.h5")
    )

    class_names = joblib.load(
        os.path.join(BASE_DIR, "class_names.pkl")
    )

    return model, class_names

model, class_names = load_artifacts()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;color:#1E88E5;'>
🚦 Smart Traffic Density Analyzer
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;color:gray;'>
CNN-Based Traffic Sign Detection & Analytics Dashboard
</h4>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📤 Upload Traffic Sign Image")

uploaded_file = st.sidebar.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# MAIN
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # ======================================
    # PREPROCESS
    # ======================================

    img = image.resize((64, 64))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ======================================
    # PREDICT
    # ======================================

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = (
        np.max(prediction) * 100
    )

    sign_name = class_names[
        predicted_class
    ]

    with col2:

        st.subheader(
            "🚦 Traffic Analysis Report"
        )

        st.success(
            f"Detected Sign: {sign_name}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

    st.divider()

    # ======================================
    # ANALYTICS DASHBOARD
    # ======================================

    st.subheader(
        "📊 Traffic Analytics Dashboard"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Detected Category",
            sign_name
        )

    with c2:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    with c3:

        if confidence > 90:

            st.metric(
                "AI Reliability",
                "High"
            )

        elif confidence > 70:

            st.metric(
                "AI Reliability",
                "Medium"
            )

        else:

            st.metric(
                "AI Reliability",
                "Low"
            )

    # ======================================
    # TOP PREDICTIONS
    # ======================================

    st.subheader(
        "📈 Top Predictions"
    )

    probs = prediction[0]

    top_indices = np.argsort(
        probs
    )[::-1][:5]

    results = pd.DataFrame({
        "Traffic Sign":
            [class_names[i]
             for i in top_indices],

        "Probability (%)":
            [round(probs[i] * 100, 2)
             for i in top_indices]
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    # ======================================
    # ALERTS
    # ======================================

    st.subheader(
        "🚨 Traffic Alert"
    )

    sign_lower = sign_name.lower()

    if "stop" in sign_lower:

        st.error(
            "STOP sign detected. Vehicles must halt completely."
        )

    elif "speed" in sign_lower:

        st.warning(
            "Speed limit sign detected. Follow the posted limit."
        )

    elif "no entry" in sign_lower:

        st.error(
            "No Entry sign detected. Restricted access zone."
        )

    else:

        st.info(
            "Traffic sign detected successfully."
        )

else:

    st.info(
        "Upload a traffic sign image to begin analysis."
    )

# ==========================================
# FEATURES
# ==========================================

st.divider()

st.subheader("⚙️ System Features")

f1, f2, f3 = st.columns(3)

with f1:
    st.success("✔ CNN Traffic Sign Detection")

with f2:
    st.success("✔ Real-Time Analytics")

with f3:
    st.success("✔ Confidence-Based Predictions")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Smart Traffic Density Analyzer | CNN-Based Traffic Sign Recognition System"
)
