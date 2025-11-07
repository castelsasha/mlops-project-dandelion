import os
import io
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="🌼 Dandelion vs Grass", layout="centered")
st.title("🌼 Dandelion vs Grass — Demo MLOps")
st.caption(f"API: {API_URL}")

with st.expander("ℹ️ Comment ça marche ?", expanded=False):
    st.write(
        "Upload une image (pissenlit ou herbe). "
        "La webapp l’envoie à l’API FastAPI et affiche la prédiction."
    )

uploaded = st.file_uploader("Choisis une image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Robust reading
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Impossible de lire l’image : {e}")
        st.stop()

    st.image(img, caption="Image uploadée", use_column_width=True)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    if st.button("🔮 Prédire"):
        try:
            files = {"file": ("image.jpg", data, "image/jpeg")}
            with st.spinner("Appel API…"):
                r = requests.post(API_URL, files=files, timeout=20)
            if r.ok:
                res = r.json()
                pred = res.get("prediction", "?")
                conf = float(res.get("confidence", 0.0))
                st.success(f"Résultat : **{pred.upper()}**")
                st.progress(int(conf * 100))
                st.write(f"Confiance : **{conf:.2%}**")
            else:
                st.error(f"Erreur API ({r.status_code}) : {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Appel API impossible : {e}")

st.write("---")
st.caption("Demo MLOps • FastAPI + PyTorch + MLflow + MinIO • Streamlit front.")
