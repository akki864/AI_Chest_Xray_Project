import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
from predict import predict_xray
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide"
)

# ---------------- BROWSER VOICE FUNCTION ----------------
def speak_text_browser(text):
    js_code = f"""
    <script>
    var msg = new SpeechSynthesisUtterance(`{text}`);
    window.speechSynthesis.speak(msg);
    </script>
    """
    st.components.v1.html(js_code)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main-title {
text-align:center;
font-size:42px;
font-weight:bold;
color:#00C9A7;
}

.sub-title {
text-align:center;
font-size:18px;
color:gray;
}

.card {
background-color:#1c1f26;
padding:20px;
border-radius:15px;
margin-bottom:20px;
}

.result-box {
font-size:24px;
font-weight:bold;
text-align:center;
}

.footer {
text-align:center;
color:gray;
font-size:14px;
margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🫁 AI-Based Chest X-Ray Diagnosis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Powered Pneumonia Detection</div>', unsafe_allow_html=True)

st.write("---")

# ---------------- FILE UPLOAD ----------------
st.markdown("### 📤 Upload Chest X-Ray Image")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "jfif"]
)

# ---------------- MAIN PROCESS ----------------
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    temp_path = "temp_xray.jpg"
    img.save(temp_path)

    with st.spinner("Analyzing X-ray using AI model..."):
        result = predict_xray(temp_path)

    # ---------------- IMAGE DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼 Original Chest X-Ray")
        st.image(img, use_column_width=True)

    with col2:
        if result["heatmap"]:
            st.markdown("### 🔥 Affected Lung Area (Heatmap)")
            st.image(result["heatmap"], use_column_width=True)
        else:
            st.markdown("### ✅ No Abnormal Regions Detected")

    st.write("---")

    # ---------------- RESULT CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-box">Diagnosis Result: {result["status"]}</div>',
        unsafe_allow_html=True
    )

    st.progress(result["confidence"] / 100)

    st.write(f"### Confidence Level: {result['confidence']}%")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 AI Report",
        "🫁 Affected Parts",
        "⚠ Causes",
        "🛡 Precautions"
    ])

    with tab1:
        st.markdown("### AI Generated Medical Report")
        st.write(result["report"])

    with tab2:
        st.markdown("### Lung Areas Affected")
        st.write(result["affected_parts"])

    with tab3:
        st.markdown("### Possible Causes")
        st.write(result["cause"])

    with tab4:
        st.markdown("### Recommended Precautions")
        st.write(result["precautions"])

    # ---------------- VOICE OUTPUT ----------------
    st.write("---")
    st.markdown("### 🔊 Voice Explanation")

    if st.button("Play AI Voice Report"):
        speak_text_browser(result["report"])

    # ---------------- WARNING ----------------
    st.warning("⚠ This AI system provides preliminary analysis and should not replace professional medical diagnosis.")

    os.remove(temp_path)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 About This Project")

st.sidebar.info("""
This system uses **Deep Learning** to detect pneumonia 
from chest X-ray images.

Technologies Used:

• Python  
• TensorFlow / Keras  
• MobileNetV2 (Transfer Learning)  
• Grad-CAM Heatmap  
• Streamlit Interface  

Features:

✔ Automatic X-ray analysis  
✔ Infection localization heatmap  
✔ AI-generated medical report  
✔ Voice explanation of diagnosis  
""")

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">Developed using Streamlit | AI Medical Imaging Project</div>', unsafe_allow_html=True)