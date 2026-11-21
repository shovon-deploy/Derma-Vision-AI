"""DermaVision AI - Streamlit app.   Run with:  streamlit run app.py"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from dermavision import (CLASS_INFO, eval_transform, grad_cam, load_checkpoint,
                         overlay_heatmap, predict)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "dermavision_best.pt"
METRICS_PATH = ROOT / "models" / "metrics.json"
SAMPLES_DIR = ROOT / "models" / "samples"
RISK_COLOR = {"Malignant": "#C5362D", "Pre-malignant": "#C77700", "Benign": "#2E8B57"}

st.set_page_config(page_title="DermaVision AI", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
    .stApp {font-family: 'Public Sans', system-ui, sans-serif;}
    .block-container {max-width: 1100px; padding-top: 2.5rem;}
    .dv-title {font-family: 'Newsreader', Georgia, serif; font-size: 2.8rem; font-weight: 600;
               letter-spacing: -0.01em; line-height: 1.1;}
    .dv-sub {opacity: .75; margin: .4rem 0 1.2rem; max-width: 60ch;}
    .dv-result {border-left: 5px solid var(--risk); padding: .1rem 0 .1rem 1.1rem; margin: 0 0 1.2rem;}
    .dv-risk {color: var(--risk); font-weight: 600;}
    .dv-name {font-family: 'Newsreader', Georgia, serif; font-size: 2.1rem; font-weight: 600;
              line-height: 1.15; margin: .15rem 0 .4rem;}
    .dv-note {opacity: .8; line-height: 1.5; max-width: 55ch;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, img_size = load_checkpoint(MODEL_PATH, device)
    return model, class_names, img_size, device


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    show_cam = st.toggle("Show Grad-CAM explanation", value=True)
    alpha = st.slider("Heatmap strength", 0.2, 1.0, 0.6, 0.05, disabled=not show_cam)
    threshold = st.slider("Warn when confidence is below", 0.30, 0.90, 0.60, 0.05, format="%.2f")
    st.divider()
    with st.expander("About the model"):
        st.write("EfficientNet-B0 fine-tuned on HAM10000: 10,015 dermoscopic images in 7 lesion classes. "
                 "Trained with a lesion-level train/validation/test split and a class-weighted loss.")
        if METRICS_PATH.exists():
            m = json.loads(METRICS_PATH.read_text())
            st.write(f"Held-out test set: {m['n_test_images']} images")
            st.dataframe(
                pd.DataFrame({"Metric": ["Accuracy", "Balanced accuracy", "Macro F1", "Macro AUC"],
                              "Value": [m["accuracy"], m["balanced_accuracy"], m["macro_f1"], m["macro_auc_ovr"]]}),
                hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
            st.dataframe(
                pd.DataFrame({"Class": list(m["per_class"]),
                              "Recall": [v["recall"] for v in m["per_class"].values()],
                              "F1": [v["f1-score"] for v in m["per_class"].values()]}),
                hide_index=True,
                column_config={"Recall": st.column_config.NumberColumn(format="%.2f"),
                               "F1": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------ page
st.markdown(
    '<div class="dv-title">DermaVision AI</div>'
    '<div class="dv-sub">Upload a dermoscopic image to get a predicted lesion class, a confidence score '
    'and a Grad-CAM map of the regions that drove the prediction.</div>',
    unsafe_allow_html=True,
)
st.warning("**Not a medical device.**" # DermaVision AI is a research and teaching prototype trained on HAM10000. "
           "It cannot diagnose skin cancer. See a dermatologist about any lesion that worries you.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `models/{MODEL_PATH.name}`. "
             "Run train.py first, then copy the `models` folder next to app.py.")
    st.stop()

left, right = st.columns([1, 1.15], gap="large")

with left:
    upload = st.file_uploader(
        "Dermoscopic image", type=["jpg", "jpeg", "png"],
        help="A close-up dermoscope image of a single lesion. Ordinary phone photos give unreliable results.")
    samples = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
    sample = st.selectbox("Or try a sample image", ["None"] + [p.name for p in samples]) if samples else "None"

image = None
if upload is not None:
    image = Image.open(upload)
elif sample != "None":
    image = Image.open(SAMPLES_DIR / sample)

if image is None:
    with right:
        st.info("Upload a dermoscopic image or pick a sample to see a prediction.")
    st.stop()

image = ImageOps.exif_transpose(image).convert("RGB")
with left:
    st.image(image, caption=f"{image.width} x {image.height} px")
    if min(image.size) < 200:
        st.warning("This image is very small. Predictions on low-resolution images are less reliable.")

# --------------------------------------------------------------- predict
model, class_names, img_size, device = get_model()
x = eval_transform(img_size)(image).unsqueeze(0).to(device)
probs = predict(model, x)
order = np.argsort(probs)[::-1]
top, second = int(order[0]), int(order[1])
info = CLASS_INFO[class_names[top]]
conf = float(probs[top])
suspicious = float(sum(p for c, p in zip(class_names, probs) if CLASS_INFO[c]["risk"] != "Benign"))

with right:
    st.markdown(
        f'<div class="dv-result" style="--risk:{RISK_COLOR[info["risk"]]}">'
        f'<div class="dv-risk">{info["risk"]}</div>'
        f'<div class="dv-name">{info["name"]}</div>'
        f'<div class="dv-note">{info["note"]}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{conf:.1%}")
    c2.metric(f"Runner-up: {class_names[second]}", f"{probs[second]:.1%}")
    c3.metric("Suspicious classes", f"{suspicious:.1%}",
              help="Combined probability of melanoma, basal cell carcinoma and "
                   "actinic keratosis / intraepithelial carcinoma.")

    if conf < threshold:
        st.warning(f"Confidence is below {threshold:.0%}. The model is unsure, so treat this result as unreliable.")
    if info["risk"] == "Benign":
        st.info("Predicted benign. This tool cannot rule out skin cancer; "
                "see a doctor if the lesion changes, bleeds or grows.")
    else:
        st.error("This lesion type can be serious. Have it examined by a dermatologist.")

    table = pd.DataFrame({"Class": [f"{CLASS_INFO[c]['name']} ({c})" for c in class_names],
                          "Probability (%)": probs * 100}).sort_values("Probability (%)", ascending=False)
    st.dataframe(table, hide_index=True, column_config={
        "Probability (%)": st.column_config.ProgressColumn(
            "Probability (%)", min_value=0, max_value=100, format="%.1f")})
    st.caption("Confidence is the model's softmax score. It is not a calibrated probability of disease.")

# -------------------------------------------------------------- Grad-CAM
if show_cam:
    st.divider()
    st.subheader("Why this prediction?")
    target = st.selectbox("Class to explain", class_names, index=top,
                          format_func=lambda c: f"{CLASS_INFO[c]['name']} ({c})")
    cam, _, _ = grad_cam(model, x, class_names.index(target))
    heat, overlay = overlay_heatmap(image, cam, alpha)
    g1, g2, g3 = st.columns(3)
    g1.image(image, caption="Input")
    g2.image(heat, caption="Grad-CAM heatmap")
    g3.image(overlay, caption="Overlay")
    st.caption("Warm colours mark the regions that raised the score of the selected class. "
               "The map shows where the model looked, not whether its reasoning is correct.")
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    st.download_button("Download overlay as PNG", data=buffer.getvalue(),
                       file_name="dermavision_gradcam.png", mime="image/png")
