import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "chest_xray_model.h5")

model = load_model(MODEL_PATH, compile=False)

# --------- CHECK XRAY ----------
def is_xray_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    avg_diff = (diff_rg + diff_rb + diff_gb) / 3
    return avg_diff < 10

# --------- HEATMAP + REGION ANALYSIS ----------
def generate_heatmap_and_analyze(img_path, output_path="heatmap.jpg"):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    if hasattr(heatmap, "numpy"):
        heatmap = heatmap.numpy()

    original = cv2.imread(img_path)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    # Region detection
    h, w = heatmap_resized.shape
    left_intensity = np.sum(heatmap_resized[:, :w//2])
    right_intensity = np.sum(heatmap_resized[:, w//2:])

    if abs(left_intensity - right_intensity) < 100:
        side = "Both lungs (bilateral involvement)"
    elif left_intensity > right_intensity:
        side = "Left lung"
    else:
        side = "Right lung"

    upper = np.sum(heatmap_resized[:h//3, :])
    middle = np.sum(heatmap_resized[h//3:2*h//3, :])
    lower = np.sum(heatmap_resized[2*h//3:, :])

    zone_values = {
        "Upper lung zone": upper,
        "Middle lung zone": middle,
        "Lower lung zone": lower
    }

    region = max(zone_values, key=zone_values.get)
    # Overlay
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, superimposed)

    return output_path, side, region


# --------- MAIN FUNCTION ----------
def predict_xray(img_path):

    if not is_xray_image(img_path):
        return {
            "status": "INVALID IMAGE",
            "confidence": 0,
            "affected_parts": "Not applicable",
            "cause": "The uploaded file does not appear to be a chest X-ray image.",
            "precautions": "Please upload a valid chest X-ray for analysis.",
            "heatmap": None,
            "report": "Analysis could not be performed because the image format is invalid."
        }

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    confidence = round(float(prediction) * 100, 2)

    # NORMAL CASE
    if prediction <= 0.5:
        return {
            "status": "NORMAL",
            "confidence": confidence,
            "affected_parts": "No abnormal lung regions detected. Lung fields appear clear.",
            "cause": "No visible infection, inflammation, or consolidation patterns detected in the lung fields.",
            "precautions": """Maintain a healthy lifestyle, avoid smoking, exercise regularly,
and undergo periodic medical checkups if respiratory symptoms occur.""",
            "heatmap": None,
            "report": f"""
AI analysis indicates that the lung structures appear normal.

The lung fields are clear without visible opacities,
fluid accumulation, or structural abnormalities.

Model Confidence: {confidence}%
"""
        }

    # SEVERITY
    if confidence < 60:
        severity = "Mild"
    elif confidence < 80:
        severity = "Moderate"
    else:
        severity = "Severe"

    heatmap_path, side, region = generate_heatmap_and_analyze(img_path)

    affected_parts_text = f"""
The infection appears to involve the {side}, primarily affecting the {region}.

Structures that may be affected include:
• Lung alveoli (air sacs responsible for oxygen exchange)
• Surrounding lung tissues and interstitial spaces
• Bronchioles and small airway passages
• Adjacent pulmonary regions showing inflammatory changes

In pneumonia, these regions may become filled with fluid or pus,
reducing the efficiency of oxygen exchange and causing visible
whitish opacities in radiographic images.
"""

    cause_text = """
Pneumonia commonly occurs when infectious microorganisms enter the respiratory tract.

Possible causes include:
• Bacterial infections (e.g., Streptococcus pneumoniae)
• Viral infections (e.g., influenza or other respiratory viruses)
• Fungal infections in rare or immunocompromised cases

These pathogens trigger an immune response that leads to inflammation,
fluid buildup, and consolidation in lung tissues, which become visible in X-ray images.
"""

    precautions_text = """
If pneumonia or abnormalities are suspected:

• Consult a qualified physician or radiologist immediately
• Follow prescribed medications and treatment plans
• Maintain hydration and proper nutrition
• Avoid exposure to pollutants, dust, or smoke
• Get adequate rest and follow up with repeat imaging if recommended

Early diagnosis and treatment significantly improve recovery outcomes.
"""

    report_text = f"""
AI-Based Chest X-ray Analysis Report

Diagnosis: Pneumonia Detected
Confidence Level: {confidence}%
Severity Level: {severity}

The analysis suggests radiological features consistent with pneumonia.
The generated heatmap highlights regions that most strongly influenced
the model’s prediction.

This AI system is designed for screening and educational purposes only.
Final diagnosis should always be confirmed by a qualified medical professional.
"""

    return {
        "status": "PNEUMONIA DETECTED",
        "confidence": confidence,
        "affected_parts": affected_parts_text,
        "cause": cause_text,
        "precautions": precautions_text,
        "heatmap": heatmap_path,
        "report": report_text
    }